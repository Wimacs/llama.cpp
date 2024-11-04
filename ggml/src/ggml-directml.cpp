#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include <ggml-directml.h>
#include <dxgi.h>
#include <dxgi1_4.h>
#include "ggml-directml/d3dx12.h"
#include <DirectML.h>
#include <iostream>
// backend interface

#define UNUSED GGML_UNUSED
using Microsoft::WRL::ComPtr;

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


static const struct ggml_backend_reg_i ggml_backend_dml_reg_i = {
    /* .get_name         = */ ggml_backend_dml_reg_get_name,
    /* .get_device_count = */ ggml_backend_dml_reg_get_device_count,
    /* .get_device       = */ /*ggml_backend_vk_reg_get_device*/NULL,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_dml_reg() {
    static ggml_backend_reg reg = {
        /* .iface   = */ ggml_backend_dml_reg_i,
        /* .context = */ nullptr,
    };

    return &reg;
}
