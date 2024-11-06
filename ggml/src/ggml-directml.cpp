#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include <ggml-directml.h>
#include <dxgi.h>
#include <dxgi1_4.h>
#include "ggml-directml/d3dx12.h"
#include <DirectML.h>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <array>
// backend interface

#define UNUSED GGML_UNUSED
using Microsoft::WRL::ComPtr;


struct dml_device_struct {
    std::mutex mutex;

    ComPtr<ID3D12Device> device;                
    D3D12_FEATURE_DATA_D3D12_OPTIONS options;   
    std::string name;                          
    uint64_t max_memory_allocation_size;
    bool fp16_support;
    ComPtr<ID3D12CommandQueue> compute_queue;   
    ComPtr<ID3D12CommandQueue> transfer_queue;  
    bool single_queue;
    uint32_t subgroup_size;
    bool uma;                                   

    size_t idx;

    std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> pipelines;
    std::unordered_map<std::string, uint64_t> pipeline_descriptor_set_requirements;

    std::vector<std::tuple<void*, size_t, ComPtr<ID3D12Resource>>> pinned_memory;

    ComPtr<ID3D12Fence> fence;                  
    ComPtr<ID3D12Resource> sync_staging;        

    ggml_backend_buffer_type buffer_type;

#ifdef GGML_DML_MEMORY_DEBUG
    std::unique_ptr<dml_memory_logger> memory_logger;
#endif
#ifdef GGML_DML_PERF
    std::unique_ptr<dml_perf_logger> perf_logger;
#endif

    ~dml_device_struct() {
        OutputDebugStringA(("Destroying device: " + name + "\n").c_str());

        if (fence) {
            fence.Reset();
        }

        if (sync_staging) {
            sync_staging.Reset();
        }

        for (auto& pipeline : pipelines) {
            pipeline.second.Reset();
        }
        pipelines.clear();
    }
};
typedef std::shared_ptr<dml_device_struct> dml_device;
typedef std::weak_ptr<dml_device_struct> dml_device_ref;

struct dml_instance_t {
    ComPtr<IDXGIFactory4> factory;
    std::vector<size_t> device_indices;
    dml_device devices[GGML_DML_MAX_DEVICES];
};
static bool dml_instance_initialized = false;
static dml_instance_t dml_instance;

struct dml_queue {
    UINT queue_family_index;              
    ComPtr<ID3D12CommandQueue> queue;       
    ComPtr<ID3D12CommandAllocator> allocator; 
    UINT cmd_buffer_idx;              
    std::vector<ComPtr<ID3D12GraphicsCommandList>> cmd_buffers;
    std::vector<ComPtr<IDMLCommandRecorder>> dml_cmd_recorder;

    D3D12_PIPELINE_STATE_FLAGS stage_flags;

    bool transfer_only;
};

struct dml_pipeline_struct {
    std::string name;
    ComPtr<ID3DBlob> shader_bytecode;
    ComPtr<ID3D12RootSignature> root_signature;
    std::vector<ComPtr<ID3D12DescriptorHeap>> descriptor_heaps;
    std::vector<D3D12_GPU_DESCRIPTOR_HANDLE> descriptor_handles;
    UINT descriptor_set_idx;
    ComPtr<ID3D12PipelineState> pipeline_state;
    UINT push_constant_size;
    UINT parameter_count;
    std::array<UINT, 3> wg_denoms;
    UINT align;
};

static int ggml_dml_get_device_count() {
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    UINT adapterCount = 0;
    if (SUCCEEDED(hr)) {
        ComPtr<IDXGIAdapter> adapter;
        while (factory->EnumAdapters(adapterCount, &adapter) != DXGI_ERROR_NOT_FOUND) {
            ++adapterCount;
            adapter.Reset(); 
        }
    }
    else {
        std::cerr << "Failed to create DXGI factory." << std::endl;
    }
    return adapterCount;
}

struct ggml_backend_dml_device_context {
    size_t device;
    std::string name;
    std::string description;
};

int ggml_backend_dml_get_device_count() {
    return ggml_dml_get_device_count();
}

static size_t ggml_backend_dml_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return ggml_backend_dml_get_device_count();
}

static const char* ggml_backend_dml_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return GGML_DML_NAME;
}

static void ggml_dml_get_device_description(int device, char* description, size_t description_size) {
    //ggml_dml_instance_init();

    //std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    //vk::PhysicalDeviceProperties props;
    //devices[device].getProperties(&props);

    snprintf(description, description_size, "%s", "fuck");
}

void ggml_backend_dml_get_device_description(int device, char* description, size_t description_size){
    GGML_ASSERT(device < (int)dml_instance.device_indices.size());
    int dev_idx = dml_instance.device_indices[device];
    ggml_dml_get_device_description(dev_idx, description, description_size);
}

//TODO: impl
static const struct ggml_backend_device_i ggml_backend_dml_device_i = {
    /* .get_name             = */ NULL,
    /* .get_description      = */ NULL,
    /* .get_memory           = */ NULL,
    /* .get_type             = */ NULL,
    /* .get_props            = */ NULL,
    /* .init_backend         = */ NULL,
    /* .get_buffer_type      = */ NULL,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ NULL,
    /* .supports_buft        = */ NULL,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static ggml_backend_dev_t ggml_backend_dml_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    static std::vector<ggml_backend_dev_t> devices;

    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            for (int i = 0; i < ggml_backend_dml_get_device_count(); i++) {
                ggml_backend_dml_device_context* ctx = new ggml_backend_dml_device_context;
                char desc[256];
                ggml_backend_dml_get_device_description(i, desc, sizeof(desc));
                ctx->device = i;
                ctx->name = GGML_DML_NAME + std::to_string(i);
                ctx->description = desc;
                devices.push_back(new ggml_backend_device{
                    /* .iface   = */ ggml_backend_dml_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                    });
            }
            initialized = true;
        }
    }

    GGML_ASSERT(device < devices.size());
    return devices[device];
}

static const struct ggml_backend_reg_i ggml_backend_dml_reg_i = {
    /* .get_name         = */ ggml_backend_dml_reg_get_name,
    /* .get_device_count = */ ggml_backend_dml_reg_get_device_count,
    /* .get_device       = */ ggml_backend_dml_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_dml_reg() {
    static ggml_backend_reg reg = {
        /* .iface   = */ ggml_backend_dml_reg_i,
        /* .context = */ nullptr,
    };

    return &reg;
}
