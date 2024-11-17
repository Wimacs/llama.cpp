#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-directx.h"
#include <dxgi.h>
#include <dxgi1_4.h>
#include "d3dx12.h"
#include <DirectML.h>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <array>
// backend interface

#define UNUSED GGML_UNUSED
using Microsoft::WRL::ComPtr;

struct dx_device_struct {
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

#ifdef GGML_DX_MEMORY_DEBUG
    std::unique_ptr<dx_memory_logger> memory_logger;
#endif
#ifdef GGML_DX_PERF
    std::unique_ptr<dx_perf_logger> perf_logger;
#endif

    ~dx_device_struct() {
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
typedef std::shared_ptr<dx_device_struct> dx_device;
typedef std::weak_ptr<dx_device_struct> dx_device_ref;

struct ggml_backend_dx_context {
    std::string name;

    dx_device device;

    size_t semaphore_idx, event_idx;
    size_t prealloc_size_x, prealloc_size_y, prealloc_size_split_k;
};


struct dx_instance_t {
    ComPtr<IDXGIFactory4> factory;
    std::vector<size_t> device_indices;
    dx_device devices[GGML_DX_MAX_DEVICES];
};
static bool dx_instance_initialized = false;
static dx_instance_t dx_instance;

struct ggml_backend_dx_device_context {
    size_t device;
    std::string name;
    std::string description;
};

struct ggml_backend_dx_buffer_type_context {
    std::string name;
    dx_device device;
};

struct dx_queue {
    UINT queue_family_index;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    UINT cmd_buffer_idx;
    std::vector<ComPtr<ID3D12GraphicsCommandList>> cmd_buffers;
    std::vector<ComPtr<IDMLCommandRecorder>> dx_cmd_recorder;

    D3D12_PIPELINE_STATE_FLAGS stage_flags;

    bool transfer_only;
};

struct dx_pipeline_struct {
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

static void* const dx_ptr_base = (void*)(uintptr_t)0x1000;  // NOLINT

static void ggml_backend_dx_buffer_free_buffer(ggml_backend_buffer_t buffer) {

}

static void* ggml_backend_dx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return dx_ptr_base;

    UNUSED(buffer);
}

static void ggml_backend_dx_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
    }
}

static void ggml_backend_dx_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, const void* data, size_t offset, size_t size) {

}

static void ggml_backend_dx_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data, size_t offset, size_t size) {

}

static bool ggml_backend_dx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {

    return false;

    UNUSED(buffer);
}

static void ggml_backend_dx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {

}

static ggml_backend_buffer_i ggml_backend_dx_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_dx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_dx_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_dx_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_dx_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_dx_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_dx_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_dx_buffer_clear,
    /* .reset           = */ NULL,
};


//TODO: fix
static const char* ggml_backend_dx_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_dx_buffer_type_context* ctx = (ggml_backend_dx_buffer_type_context*)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_dx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size)
{
    return ggml_backend_buffer_init(buft, ggml_backend_dx_buffer_interface, nullptr, size);
}
static size_t ggml_backend_dx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft)
{
    return 0;
}
static size_t ggml_backend_dx_buffer_type_get_max_size(ggml_backend_buffer_type_t buft)
{
    return 0;
}
static size_t ggml_backend_dx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor* tensor)
{
    return 0;
}
static ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_dx_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_dx_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_dx_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_dx_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_dx_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

//-----------------------------
static bool ggml_backend_buffer_is_dx(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_dx_buffer_type_name;
}



static void ggml_dx_get_device_description(int device, char* description, size_t description_size) {
    //ggml_dx_instance_init();

    //std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    //vk::PhysicalDeviceProperties props;
    //devices[device].getProperties(&props);

    snprintf(description, description_size, "%s", "fuck");
}


void ggml_dx_instance_init()
{
    if (dx_instance_initialized)
    {
        return;
    }
    dx_instance_initialized = true;

    ComPtr<IDXGIFactory1> factory = dx_instance.factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    UINT adapterCount = 0;
    if (SUCCEEDED(hr)) {
        ComPtr<IDXGIAdapter> adapter;
        while (factory->EnumAdapters(adapterCount, &adapter) != DXGI_ERROR_NOT_FOUND) {
            ++adapterCount;
            adapter.Reset();
            dx_instance.device_indices.push_back(adapterCount);
        }
    }
    else {
        std::cerr << "Failed to create DXGI factory." << std::endl;
    }
    
}

static int ggml_dx_get_device_count() {
    //ComPtr<IDXGIFactory1> factory;
    //HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    //UINT adapterCount = 0;
    //if (SUCCEEDED(hr)) {
    //    ComPtr<IDXGIAdapter> adapter;
    //    while (factory->EnumAdapters(adapterCount, &adapter) != DXGI_ERROR_NOT_FOUND) {
    //        ++adapterCount;
    //        adapter.Reset();
    //        dx_device
    //    }
    //}
    //else {
    //    std::cerr << "Failed to create DXGI factory." << std::endl;
    //}

    ggml_dx_instance_init();
    return dx_instance.device_indices.size();
}

int ggml_backend_dx_get_device_count() {
    return ggml_dx_get_device_count();
}

static size_t ggml_backend_dx_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return ggml_backend_dx_get_device_count();
}

static const char* ggml_backend_dx_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return GGML_DX_NAME;
}

// dx buffer type

void ggml_backend_dx_get_device_description(int device, char* description, size_t description_size){
    GGML_ASSERT(device < (int)dx_instance.device_indices.size());
    int dev_idx = dx_instance.device_indices[device];
    ggml_dx_get_device_description(dev_idx, description, description_size);
}

static const char* ggml_backend_dx_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_dx_device_context* ctx = (ggml_backend_dx_device_context*)dev->context;
    return ctx->name.c_str();
}

static const char* ggml_backend_dx_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_dx_device_context* ctx = (ggml_backend_dx_device_context*)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_dx_device_get_memory(ggml_backend_dev_t device, size_t* free, size_t* total) {
    ggml_backend_dx_device_context* ctx = (ggml_backend_dx_device_context*)device->context;
    ggml_backend_dx_get_device_memory(ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_dx_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_dx_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props* props) {
    props->name = ggml_backend_dx_device_get_name(dev);
    props->description = ggml_backend_dx_device_get_description(dev);
    props->type = ggml_backend_dx_device_get_type(dev);
    ggml_backend_dx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_dx_device_init(ggml_backend_dev_t dev, const char* params) {
    UNUSED(params);
    ggml_backend_dx_device_context* ctx = (ggml_backend_dx_device_context*)dev->context;
    return ggml_backend_dx_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_dx_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_dx_device_context* ctx = (ggml_backend_dx_device_context*)dev->context;
    return ggml_backend_dx_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_dx_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return ggml_backend_dx_host_buffer_type();
}

static bool ggml_backend_dx_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor* op) {
    //switch (op->op) {
    //case GGML_OP_UNARY:
    //    switch (ggml_get_unary_op(op)) {
    //    //case GGML_UNARY_OP_GELU:
    //    //case GGML_UNARY_OP_GELU_QUICK:
    //    //case GGML_UNARY_OP_SILU:
    //    //case GGML_UNARY_OP_RELU:
    //    //case GGML_UNARY_OP_TANH:
    //        return ggml_is_contiguous(op->src[0]);
    //    default:
    //        return false;
    //    }
    //    break;
    //case GGML_OP_MUL_MAT:
    //case GGML_OP_MUL_MAT_ID:
    //{
    //    switch (op->src[0]->type) {
    //    case GGML_TYPE_F32:
    //    case GGML_TYPE_F16:
    //    //case GGML_TYPE_Q4_0:
    //    //case GGML_TYPE_Q4_1:
    //    //case GGML_TYPE_Q5_0:
    //    //case GGML_TYPE_Q5_1:
    //    //case GGML_TYPE_Q8_0:
    //    //case GGML_TYPE_Q2_K:
    //    //case GGML_TYPE_Q3_K:
    //    //case GGML_TYPE_Q4_K:
    //    //case GGML_TYPE_Q5_K:
    //    //case GGML_TYPE_Q6_K:
    //    //case GGML_TYPE_IQ4_NL:
    //        break;
    //    default:
    //        return false;
    //    }
    //    struct ggml_tensor* a;
    //    struct ggml_tensor* b;
    //    if (op->op == GGML_OP_MUL_MAT) {
    //        a = op->src[0];
    //        b = op->src[1];
    //    }
    //    else {
    //        a = op->src[2];
    //        b = op->src[1];
    //    }
    //    if (a->ne[3] != b->ne[3]) {
    //        return false;
    //    }
    //    return true;
    //} break;
    //case GGML_OP_GET_ROWS:
    //{
    //    switch (op->src[0]->type) {
    //    case GGML_TYPE_F32:
    //    case GGML_TYPE_F16:
    //    case GGML_TYPE_Q4_0:
    //    case GGML_TYPE_Q4_1:
    //    case GGML_TYPE_Q5_0:
    //    case GGML_TYPE_Q5_1:
    //    case GGML_TYPE_Q8_0:
    //    case GGML_TYPE_IQ4_NL:
    //        return true;
    //    default:
    //        return false;
    //    }
    //} break;
    //case GGML_OP_CONT:
    //case GGML_OP_CPY:
    //case GGML_OP_DUP:
    //{
    //    ggml_type src0_type = op->src[0]->type;
    //    ggml_type src1_type = op->src[1] != nullptr ? op->src[1]->type : src0_type;
    //    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
    //        return true;
    //    }
    //    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
    //        return true;
    //    }
    //    if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
    //        return true;
    //    }
    //    return false;
    //} break;
    //case GGML_OP_REPEAT:
    //    return ggml_type_size(op->type) == sizeof(float) && ggml_type_size(op->src[0]->type) == sizeof(float);
    //case GGML_OP_ROPE:
    //    return ggml_is_contiguous(op->src[0]);
    //case GGML_OP_NONE:
    //case GGML_OP_RESHAPE:
    //case GGML_OP_VIEW:
    //case GGML_OP_PERMUTE:
    //case GGML_OP_TRANSPOSE:
    //case GGML_OP_NORM:
    //case GGML_OP_GROUP_NORM:
    //case GGML_OP_RMS_NORM:
    //case GGML_OP_ADD:
    //case GGML_OP_ACC:
    //case GGML_OP_MUL:
    //case GGML_OP_DIV:
    //case GGML_OP_CONCAT:
    //case GGML_OP_UPSCALE:
    //case GGML_OP_SCALE:
    //case GGML_OP_SQR:
    //case GGML_OP_SIN:
    //case GGML_OP_COS:
    //case GGML_OP_CLAMP:
    //case GGML_OP_PAD:
    //case GGML_OP_DIAG_MASK_INF:
    //case GGML_OP_SOFT_MAX:
    //case GGML_OP_ARGSORT:
    //case GGML_OP_SUM_ROWS:
    //case GGML_OP_IM2COL:
    //case GGML_OP_TIMESTEP_EMBEDDING:
    //case GGML_OP_POOL_2D:
    //case GGML_OP_LEAKY_RELU:
    //    return true;
    //default:
    //    return false;
    //}

return false;
    UNUSED(dev);
}

static bool ggml_backend_dx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_dx_buffer_type_name) {
        return false;
    }

    ggml_backend_dx_device_context* ctx = (ggml_backend_dx_device_context*)dev->context;
    ggml_backend_dx_buffer_type_context* buft_ctx = (ggml_backend_dx_buffer_type_context*)buft->context;

    return buft_ctx->device->idx == ctx->device;
}

//TODO: why?
static bool ggml_backend_dx_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor* op) {
    const int min_batch_size = 32;

    return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
        (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

    UNUSED(dev);
}

//TODO: impl
static const struct ggml_backend_device_i ggml_backend_dx_device_i = {
    /* .get_name             = */ ggml_backend_dx_device_get_name,
    /* .get_description      = */ ggml_backend_dx_device_get_description,
    /* .get_memory           = */ ggml_backend_dx_device_get_memory,
    /* .get_type             = */ ggml_backend_dx_device_get_type,
    /* .get_props            = */ ggml_backend_dx_device_get_props,
    /* .init_backend         = */ ggml_backend_dx_device_init,
    /* .get_buffer_type      = */ ggml_backend_dx_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_dx_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_dx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_dx_device_supports_buft,
    /* .offload_op           = */ ggml_backend_dx_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static dx_device ggml_dx_get_device(size_t idx) {

    if (dx_instance.devices[idx] == nullptr)
    {
        dx_device device = std::make_shared<dx_device_struct>();
        dx_instance.devices[idx] = device;

    }

    return dx_instance.devices[idx];
}

static size_t ggml_backend_dx_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 1;
    UNUSED(buft);
}

static void ggml_backend_dx_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {

}

static ggml_backend_buffer_t ggml_backend_dx_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size += 32;  // Behave like the CPU buffer type
    void* ptr = nullptr;

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_dx_host_buffer_free_buffer;

    return buffer;

    UNUSED(buft);
}

static const char* ggml_backend_dx_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_DX_NAME "_Host";
    UNUSED(buft);
}

static ggml_backend_dev_t ggml_backend_dx_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    static std::vector<ggml_backend_dev_t> devices;

    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            for (int i = 0; i < ggml_backend_dx_get_device_count(); i++) {
                ggml_backend_dx_device_context* ctx = new ggml_backend_dx_device_context;
                char desc[256];
                ggml_backend_dx_get_device_description(i, desc, sizeof(desc));
                ctx->device = i;
                ctx->name = GGML_DX_NAME + std::to_string(i);
                ctx->description = desc;
                devices.push_back(new ggml_backend_device{
                    /* .iface   = */ ggml_backend_dx_device_i,
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

static const struct ggml_backend_reg_i ggml_backend_dx_reg_i = {
    /* .get_name         = */ ggml_backend_dx_reg_get_name,
    /* .get_device_count = */ ggml_backend_dx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_dx_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_dx_reg() {
    static ggml_backend_reg reg = {
        /* .iface   = */ ggml_backend_dx_reg_i,
        /* .context = */ nullptr,
    };

    return &reg;
}

static ggml_guid_t ggml_backend_dx_guid() {
    static ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2c };
    return &guid;
}

static const char* ggml_backend_dx_name(ggml_backend_t backend){
    return "fuck";
}

static void ggml_backend_dx_free(ggml_backend_t backend) {

}

static ggml_status ggml_backend_dx_graph_compute(ggml_backend_t backend, ggml_cgraph* cgraph){
    return GGML_STATUS_SUCCESS;
}

// TODO: enable async and synchronize
static ggml_backend_i ggml_backend_dx_interface = {
    /* .get_name                = */ ggml_backend_dx_name,
    /* .free                    = */ ggml_backend_dx_free,
    /* .set_tensor_async        = */ NULL,  // ggml_backend_vk_set_tensor_async,
    /* .get_tensor_async        = */ NULL,  // ggml_backend_vk_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,  // ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ NULL,  // ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_dx_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};



ggml_backend_t ggml_backend_dx_init(size_t dev_num) {
    ggml_backend_dx_context* ctx = new ggml_backend_dx_context;
    //ggml_dx_init(ctx, dev_num);

    ggml_backend_t dx_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_dx_guid(),
        /* .interface = */ ggml_backend_dx_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_dx_reg(), dev_num),
        /* .context   = */ ctx,
    };

    return dx_backend;
}

bool ggml_backend_is_dx(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_dx_guid());
}

void ggml_backend_dx_get_device_memory(int device, size_t* free, size_t* total) {
    //GGML_ASSERT(device < (int)dx_instance.device_indices.size());

    //ComPtr<IDXGIAdapter3> adapter = dx_instance.
    ////vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];

    ////vk::PhysicalDeviceMemoryProperties memprops = vkdev.getMemoryProperties();

    ////for (const vk::MemoryHeap& heap : memprops.memoryHeaps) {
    ////    if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
    ////        *total = heap.size;
    ////        *free = heap.size;
    ////        break;
    ////    }
    ////}
    //DXGI_QUERY_VIDEO_MEMORY_INFO memoryInfo = {};
    //adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memoryInfo);

    //*total = memoryInfo.Budget;
    //*free = memoryInfo.Budget - memoryInfo.CurrentUsage;
}

ggml_backend_buffer_type_t ggml_backend_dx_buffer_type(size_t dev_num) {
    ggml_dx_instance_init();
    dx_device dev = ggml_dx_get_device(dev_num);

    return &dev->buffer_type;
}

// Should be changed to return device-specific host buffer type
// but that probably requires changes in llama.cpp
ggml_backend_buffer_type_t ggml_backend_dx_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_dx_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_dx_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_dx_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_dx_host_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_dx_reg(), 0),
        /* .context  = */ nullptr,
    };

    // Make sure device 0 is initialized
    ggml_dx_instance_init();
    ggml_dx_get_device(0);

    return &ggml_backend_dx_buffer_type_host;
}
