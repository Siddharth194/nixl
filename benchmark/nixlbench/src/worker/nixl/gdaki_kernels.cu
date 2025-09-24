/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <nixl_types.h>
#include <gpu/ucx/nixl_device.cuh>
#include "gdaki_kernels.cuh"

// Helper function to get request index based on coordination level (from gtest)
template<nixl_gpu_level_t level>
__device__ constexpr size_t
getRequestIndex() {
    switch (level) {
    case nixl_gpu_level_t::THREAD:
        return threadIdx.x;
    case nixl_gpu_level_t::WARP:
        return threadIdx.x / warpSize;
    case nixl_gpu_level_t::BLOCK:
        return 0;
    default:
        return 0;
    }
}

nixl_gpu_level_t
getGpuLevel(const char *gdaki_level) {
    if (!strcmp(gdaki_level, "WARP"))
    	return nixl_gpu_level_t::WARP;
    if (!strcmp(gdaki_level, "BLOCK"))
	return nixl_gpu_level_t::BLOCK;
    return nixl_gpu_level_t::THREAD;
}

// GDAKI kernel for full transfers (block coordination only)
__global__ void
gdakiFullTransferKernel(nixlGpuXferReqH *req_handle,
                        int num_iterations,
                        const uint64_t signal_inc,
                        const uint64_t remote_addr) {
    __shared__ nixlGpuXferStatusH xfer_status;
    nixlGpuSignal signal = {signal_inc, remote_addr};

    // Execute transfers for the specified number of iterations
    for (int i = 0; i < num_iterations; i++) {
        // Post the GPU transfer request with signal increment of 1
        nixl_status_t status = nixlGpuPostSignalXferReq<nixl_gpu_level_t::BLOCK>(
            req_handle, 0, signal, true, &xfer_status);
        if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
            return; // Early exit on error
        }

        // Wait for transfer completion
        do {
            status = nixlGpuGetXferStatus<nixl_gpu_level_t::BLOCK>(xfer_status);
        } while (status == NIXL_IN_PROG);

        if (status != NIXL_SUCCESS) {
            return; // Early exit on error
        }
    }
}

// GDAKI kernel for partial transfers (supports thread/warp/block coordination)
template<nixl_gpu_level_t level>
__global__ void
gdakiPartialTransferKernel(nixlGpuXferReqH *req_handle,
                           int num_iterations,
                           const uint64_t signal_inc,
                           const uint64_t remote_addr) {
    constexpr size_t MAX_THREADS = 1024;
    __shared__ nixlGpuXferStatusH xfer_status[MAX_THREADS];
    nixlGpuXferStatusH *xfer_status_ptr = &xfer_status[getRequestIndex<level>()];
    nixlGpuSignal signal = {signal_inc, remote_addr};

    // Execute transfers for the specified number of iterations
    for (int i = 0; i < num_iterations; i++) {
        // Use partial transfer API which supports all coordination levels
        nixl_status_t status = nixlGpuPostPartialWriteXferReq<level>(
            req_handle, 1ULL, nullptr, nullptr, nullptr, nullptr, signal, true, xfer_status_ptr);
        if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
            return; // Early exit on error
        }

        // Wait for transfer completion
        do {
            status = nixlGpuGetXferStatus<level>(xfer_status_ptr);
        } while (status == NIXL_IN_PROG);

        if (status != NIXL_SUCCESS) {
            return; // Early exit on error
        }
    }
}

// Host-side launcher
extern "C" {

nixl_status_t
launchGdakiKernel(nixlGpuXferReqH *req_handle,
                  int num_iterations,
                  const char *level,
                  int threads_per_block,
                  int blocks_per_grid,
                  cudaStream_t stream,
                  const uint64_t signal_inc,
                  const uint64_t remote_addr) {

    nixl_gpu_level_t gpulevel = getGpuLevel(level);

    // Validate parameters
    if (num_iterations <= 0 || req_handle == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (threads_per_block < 1 || threads_per_block > 1024) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (blocks_per_grid < 1) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Use full transfer kernel for block coordination only
    if (gpulevel == nixl_gpu_level_t::BLOCK) {
        gdakiFullTransferKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle,
                                                                                   num_iterations,
										   signal_inc,
										   remote_addr);
    } else {
        // For thread/warp coordination, fall back to block coordination for full transfers
        gdakiFullTransferKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle,
                                                                                   num_iterations,
										   signal_inc,
										   remote_addr);
    }

    // Check for launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
launchGdakiPartialKernel(nixlGpuXferReqH *req_handle,
                         int num_iterations,
                         const char *level,
                         int threads_per_block,
                         int blocks_per_grid,
                         cudaStream_t stream,
                         const uint64_t signal_inc,
                         const uint64_t remote_addr) {

    nixl_gpu_level_t gpulevel = getGpuLevel(level);

    // Validate parameters
    if (num_iterations <= 0 || req_handle == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (threads_per_block < 1 || threads_per_block > 1024) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (blocks_per_grid < 1) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Launch partial transfer kernel based on coordination level
    if (gpulevel == nixl_gpu_level_t::THREAD) {
        gdakiPartialTransferKernel<nixl_gpu_level_t::THREAD>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle, num_iterations, signal_inc, remote_addr);
    } else if (gpulevel == nixl_gpu_level_t::WARP) {
        gdakiPartialTransferKernel<nixl_gpu_level_t::WARP>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle, num_iterations, signal_inc, remote_addr);
    } else if (gpulevel == nixl_gpu_level_t::BLOCK) {
        gdakiPartialTransferKernel<nixl_gpu_level_t::BLOCK>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle, num_iterations, signal_inc, remote_addr);
    }

    // Check for launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

uint64_t
readNixlGpuSignal(const void *signal_addr, const char *gpulevel) {
    const nixl_gpu_level_t level = getGpuLevel(gpulevel);
    return nixlGpuReadSignal<level>(signal_addr);
}

} // extern "C"
