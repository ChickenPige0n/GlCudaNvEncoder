add_requires("dlfcn-win32")
add_requires("cuda", {configs = {shared = true}})
add_requires("glew")

target("PhiDxRenderer")
    add_defines("WIN32")

    set_kind("binary")
    add_files("*.cpp")
    add_headerfiles("*.hpp")


    add_includedirs(".", "NvCodec")

    add_linkdirs("NvCodec/Lib/x64")
    add_links("nvcuvid", "cuda")
    add_includedirs("$(env CUDA_PATH)/include")


    -- Add NvCodec source files
    add_files("NvCodec/NvEncoder/*.cpp")
    add_files("NvCodec/NvEncoder/NvEncoderD3D11.cpp")
    add_headerfiles("NvCodec/*.h")
    add_headerfiles("NvCodec/NvEncoder/*.h")


    add_packages("cuda", "glew")
    
    
    add_syslinks("d3d11", "dxgi")