# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

nixl_worker_sources = [
  'nixl_worker.cpp',
  'nixl_worker.h',
]

nixl_worker_cuda_sources = [
  'gdaki_kernels.cu',
]

ucx_gpu_device_api_available = false
nixl_worker_deps = [openmp_dep, nixl_lib, nixl_build, nixl_serdes]
if cuda_available
    nixl_worker_deps += [cuda_dep]
    cuda = meson.get_compiler('cuda')
    have_gpu_side = cuda.compiles('''
            #include <ucp/api/device/ucp_device_impl.h>
            int main() { return 0; }
        ''', dependencies : ucx_dep, args: nvcc_flags)

    have_host_side = cpp.compiles('''
            #include <ucp/api/device/ucp_host.h>
            int main() { return 0; }
        ''', dependencies: ucx_dep)

    if have_gpu_side and have_host_side
        ucx_gpu_device_api_available = true
        add_project_arguments('-DHAVE_UCX_GPU_DEVICE_API', language: ['cpp', 'cuda'])
    endif

    summary({
    'UCX GPU Device API' : ucx_gpu_device_api_available,
    'GPU-side compile'   : have_gpu_side,
    'Host-side compile'  : have_host_side,
    'nvcc available'     : nvcc_prog.found(),
  }, section: 'UCX GPU Device API', bool_yn: true)
endif

# Handle CUDA compilation if available
cuda_objects = []
if cuda_available and ucx_gpu_device_api_available
  # Find nvcc compiler
  nvcc = find_program('nvcc', required: false)
  if nvcc.found()
    # Build NVCC command arguments
    nvcc_args = [
      nvcc,
      '-c', '@INPUT@',
      '-o', '@OUTPUT@',
      '-I' + nixl_inc,
      '-I' + meson.current_source_dir() + '/../../..',
      '-I' + nixl_path + '/src',
      '-I' + nixl_path + '/src/utils',
      '--std=c++17',
      '-arch=sm_80',
      '--ptxas-options=-v',
      '--maxrregcount=32',
      '--compiler-options', '-fPIC',
      '-DHAVE_CUDA'
    ]

    # Add UCX include path if ucx_path option is provided
    ucx_path_opt = get_option('ucx_path')
    if ucx_path_opt != ''
      nvcc_args += ['-I' + ucx_path_opt + '/include']
    endif

    # Compile CUDA kernels
    nixl_cuda_obj = custom_target('nixl_gdaki_kernels_obj',
      input: nixl_worker_cuda_sources,
      output: 'gdaki_kernels.o',
      command: nvcc_args,
      install: false
    )
    cuda_objects = [nixl_cuda_obj]
  else
    warning('NVCC not found - CUDA kernels will not be available')
  endif
endif

nixl_worker_lib = static_library('nixl_worker',
  nixl_worker_sources,
  objects: cuda_objects,
  include_directories: inc_dir,
  dependencies: nixl_worker_deps,
  install: true,
)
