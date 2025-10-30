#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <cstring>
#include <string>
#include "nixl.h"

const std::string AGENT1_NAME = "Laptop1";
const std::string AGENT2_NAME = "Laptop2";
const std::string AGENT1_IP = "10.50.16.151";
const std::string AGENT2_IP = "10.50.16.151";

void printStatus(const std::string& operation, nixl_status_t status) {
    std::cout << operation << ": " << nixlEnumStrings::statusStr(status) << std::endl;
}

nixlAgent* createAgent(const std::string& name, bool isListener, int port) {
    nixlAgentConfig cfg(true);
    cfg.useListenThread = isListener;
    cfg.listenPort = port;
    return new nixlAgent(name, cfg);
}

void runAgent1() 
{
    std::cout << "=== Starting Agent1 (Sender) ===\n";
    
    nixlAgent* agent1 = createAgent(AGENT1_NAME, true, 10000);
    
    // Get LIBFABRIC backend
    nixl_b_params_t init_params;
    nixl_mem_list_t mems;
    nixl_status_t status = agent1->getPluginParams("LIBFABRIC", mems, init_params);
    assert(status == NIXL_SUCCESS);
    
    nixlBackendH* libfabric_backend = nullptr;
    status = agent1->createBackend("LIBFABRIC", init_params, libfabric_backend);
    assert(status == NIXL_SUCCESS);
    
    // Register some memory
    size_t buffer_size = 1024;
    void* buffer = malloc(buffer_size);
    assert(buffer);
    memset(buffer, 0xAA, buffer_size);
    
    nixl_reg_dlist_t dlist(DRAM_SEG);
    nixlBlobDesc desc;
    desc.addr = (uintptr_t)buffer;
    desc.len = buffer_size;
    desc.devId = 0;
    dlist.addDesc(desc);
    
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(libfabric_backend);
    status = agent1->registerMem(dlist, &extra_params);

    assert(status == NIXL_SUCCESS);
    
    std::cout << "Agent1 memory registered at: " << buffer << std::endl;

    // Dump the data Agent1 is going to send
    std::cout << "Agent1: dumping initial buffer contents (first 128 bytes):";
    uint8_t* b = reinterpret_cast<uint8_t*>(buffer);
    for (size_t i = 0; i < 128; ++i) {
        if (i % 16 == 0) std::printf("\n%08zx: ", i);
        std::printf("%02x ", b[i]);
    }
    std::printf("\n");

    
    // Send metadata to Agent2
    nixl_opt_args_t send_params;
    send_params.ipAddr = AGENT2_IP;
    send_params.port = 10001;  // Agent2's listening port
    
    std::cout << "Sending metadata to Agent2...\n";
    status = agent1->sendLocalMD(&send_params);
    assert(status == NIXL_SUCCESS);
    
    std::cout << "Metadata sent successfully!\n";
    
    // Also request metadata from Agent2 (so agent1 has remote info)
    nixl_opt_args_t fetch_params;
    fetch_params.ipAddr = AGENT2_IP;
    fetch_params.port = 10001;  // Agent2's listening port

    // Retry fetchRemoteMD until request succeeds (queued) and then try to create transfer
    int fetch_retries = 10;
    bool fetched = false;
    while (fetch_retries-- > 0) {
        status = agent1->fetchRemoteMD(AGENT2_NAME, &fetch_params);
        if (status == NIXL_SUCCESS) {
            fetched = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    if (!fetched) 
    {
        std::cerr << "Agent1: failed to queue fetchRemoteMD\n";
    }

    // Wait a short while for remote metadata to be loaded at receiver and visible to this agent.
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Query the loaded remote section for a concrete base address (first registered descriptor)
    uintptr_t remote_base = 0;
    status = agent1->getRemoteFirstAddr(AGENT2_NAME, DRAM_SEG, (nixl_backend_t)"LIBFABRIC", remote_base);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Agent1: could not obtain remote base addr: " << nixlEnumStrings::statusStr(status) << std::endl;
        // You can choose to wait/retry here instead of aborting
    } else {
        std::cout << "Agent1: remote base addr = " << (void*)remote_base << std::endl;
    }
    
    // Prepare transfer descriptors (A1 -> A2)
    size_t transfer_len = 64;
    size_t src_offset = 16;
    size_t dst_offset = 32;

    nixl_xfer_dlist_t src_descs(DRAM_SEG), dst_descs(DRAM_SEG);
    nixlBasicDesc src{};
    src.addr     = (uintptr_t) (((char*) buffer) + src_offset);
    src.len      = transfer_len;
    src.devId    = 0;
    src_descs.addDesc(src);

    nixlBasicDesc dst{};
    // Fill dst.addr from remote_base (if available). If remote_base==0, createXferReq will likely fail.
    dst.addr     = (uintptr_t)(remote_base ? (remote_base + dst_offset) : 0);
    dst.len      = transfer_len;
    dst.devId    = 0;
    dst_descs.addDesc(dst);

    std::cout << "Destination address: 0x" << std::hex << dst.addr << std::endl;

    // Request notification
    extra_params.notifMsg = "xfer_done";
    extra_params.hasNotif = true;

    nixlXferReqH* req_handle = nullptr;

    // createXferReq may fail until remote metadata is fully loaded; retry a few times
    int create_retries = 20;

    std::cout << "DEBUG SRC: addr=" << (void*)src.addr << " len=" << src.len << " devId=" << src.devId << std::endl;
    std::cout << "DEBUG DST: addr=" << (void*)dst.addr << " len=" << dst.len << " devId=" << dst.devId << std::endl;

    while (create_retries-- > 0) 
    {
        std::cout << std::endl << std::endl;
        status = agent1->createXferReq(NIXL_WRITE, src_descs, dst_descs, AGENT2_NAME, req_handle, &extra_params);
        if (status == NIXL_SUCCESS) break;
        if (status == NIXL_ERR_NOT_FOUND || status == NIXL_ERR_INVALID_PARAM) {
            // likely remote metadata or conn-info not ready yet
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        // other errors: break
        break;
    }

    if (status != NIXL_SUCCESS) {
        std::cerr << "Agent1: createXferReq failed: " << nixlEnumStrings::statusStr(status) << std::endl;
    }

    if (status != NIXL_SUCCESS || req_handle == nullptr) {
    std::cerr << "Agent1: createXferReq failed: " << nixlEnumStrings::statusStr(status) << std::endl;
    } 
    
    else 
    {
        status = agent1->postXferReq(req_handle);
        if (status < 0) {
            std::cerr << "Agent1: postXferReq error: " << nixlEnumStrings::statusStr(status) << std::endl;
        } else {
            // Status >= 0 means backend accepted the post. NIXL_IN_PROG indicates async progress.
            std::cout << "Agent1: postXferReq accepted, status: " << nixlEnumStrings::statusStr(status) << std::endl;
            // Poll until completion or error
            for (int i = 0; i < 200; ++i) {
                nixl_status_t xstatus = agent1->getXferStatus(req_handle);
                std::cout << "Agent1: transfer status: " << nixlEnumStrings::statusStr(xstatus) << std::endl;
                if (xstatus == NIXL_SUCCESS) {
                    break;
                }
                if (xstatus < 0) {
                    std::cerr << "Agent1: transfer failed: " << nixlEnumStrings::statusStr(xstatus) << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

}


void runAgent2() {
    std::cout << "=== Starting Agent2 (Receiver) ===\n";
    
    nixlAgent* agent2 = createAgent(AGENT2_NAME, true, 10001);
    
    // Get UCX backend
    nixl_b_params_t init_params;
    nixl_mem_list_t mems;
    nixl_status_t status = agent2->getPluginParams("LIBFABRIC", mems, init_params);
    assert(status == NIXL_SUCCESS);
    
    nixlBackendH* libfabric_backend = nullptr;
    status = agent2->createBackend("LIBFABRIC", init_params, libfabric_backend);
    assert(status == NIXL_SUCCESS);
    
    // Register some memory (receiver)
    size_t buffer_size = 1024;
    void* buffer = malloc(buffer_size);
    assert(buffer);
    memset(buffer, 0x00, buffer_size);
    
    nixl_reg_dlist_t dlist(DRAM_SEG);
    nixlBlobDesc desc;
    desc.addr = (uintptr_t)buffer;
    desc.len = buffer_size;
    desc.devId = 0;
    dlist.addDesc(desc);
    
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(libfabric_backend);
    status = agent2->registerMem(dlist, &extra_params);
    
    assert(status == NIXL_SUCCESS);
    
    std::cout << "Agent2 memory registered at: " << buffer << std::endl;
    
    // Wait for metadata from Agent1: request Agent1's metadata
    nixl_opt_args_t fetch_params;
    fetch_params.ipAddr = AGENT1_IP;
    fetch_params.port = 10000;  // Agent1's listening port

    int fetch_retries = 20;
    bool fetched = false;
    while (fetch_retries-- > 0) {
        status = agent2->fetchRemoteMD(AGENT1_NAME, &fetch_params);
        if (status == NIXL_SUCCESS) {
            fetched = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    if (!fetched) {
        std::cerr << "Agent2: failed to queue fetchRemoteMD\n";
    }

    // Also send our metadata to Agent1 (so both sides have each other's metadata)
    nixl_opt_args_t send_params;
    send_params.ipAddr = AGENT1_IP;
    send_params.port = 10000;
    status = agent2->sendLocalMD(&send_params);
    assert(status == NIXL_SUCCESS);
    std::cout << "Agent2: metadata sent to Agent1\n";

    // Wait for notification from Agent1 indicating a transfer completed
    std::cout << "Agent2: waiting for notification and data...\n";
    int wait_loops = 200;
    bool got_notif = false;
    while (wait_loops-- > 0) {
        nixl_notifs_t notifs;
        nixl_status_t r = agent2->getNotifs(notifs);
        if (r == NIXL_SUCCESS && !notifs.empty()) {
            auto it = notifs.find(AGENT1_NAME);
            if (it != notifs.end() && !it->second.empty()) {
                got_notif = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!got_notif) {
        std::cerr << "Agent2: did not receive notification from Agent1\n";
    } else {
        std::cout << "Agent2: notification received, verifying data...\n";
        // Verify transferred data: must match Agent1 pattern 0xAA
        size_t transfer_len = 64;
        size_t dst_offset = 32;
        uint8_t* base = (uint8_t*)buffer + dst_offset;
        bool ok = true;
        for (size_t i = 0; i < transfer_len; ++i) {
            if (base[i] != 0xAA) {
                ok = false;
                break;
            }
        }
        std::cout << "Agent2: data verification: " << (ok ? "OK" : "MISMATCH") << std::endl;

        // Dump received data for inspection
        size_t dump_off = dst_offset;
        size_t dump_len = 128;
        uint8_t* b = (uint8_t*)buffer;
        for (size_t i = 0; i < dump_len; ++i) {
            if (i % 16 == 0) std::printf("\n%08zx: ", dump_off + i);
            std::printf("%02x ", b[dump_off + i]);
        }
        std::printf("\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [1|2]" << std::endl;
        std::cerr << "  1 - Run as Agent1 (Sender)" << std::endl;
        std::cerr << "  2 - Run as Agent2 (Receiver)" << std::endl;
        return 1;
    }

    int agent_num = atoi(argv[1]);
    if (agent_num == 1) {
        runAgent1();
    } else if (agent_num == 2) {
        runAgent2();
    } else {
        std::cerr << "Invalid argument. Use 1 or 2." << std::endl;
        return 1;
    }

    return 0;
}