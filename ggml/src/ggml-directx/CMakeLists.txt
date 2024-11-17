if (WIN32)
    message(STATUS "Using DirectX backend")
    file(GLOB   GGML_HEADERS_DIRECTX "*.h")
    list(APPEND GGML_HEADERS_DIRECTX "../../include/ggml-directx.h")

    file(GLOB   GGML_SOURCES_DIRECTX "*.cpp")

    add_library(ggml-directx
                ${GGML_HEADERS_DIRECTX}
                ${GGML_SOURCES_DIRECTX})
    list(APPEND GGML_DIRECTX_LIBS_PRIVATE dxgi.lib d3d12.lib)
    target_link_libraries(ggml-directx PRIVATE ggml-base ${GGML_DIRECTX_LIBS_PRIVATE})
    target_include_directories(ggml-directx PRIVATE . ..)
endif()
