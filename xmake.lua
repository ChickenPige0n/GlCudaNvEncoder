add_rules("mode.debug", "mode.release")
add_requires("glew")
add_requires("glfw")
-- add_requires("sdl3")
target("GlCudaNvEncoder")
    add_defines("WIN32")

    set_kind("shared")
    add_files("*.cpp")

    -- Add CUDA include path
    add_includedirs(".", "NvCodec")
    add_includedirs("$(env CUDA_PATH)/include")

    add_linkdirs("NvCodec/Lib/x64", "$(env CUDA_PATH)/lib/x64")
    add_links("nvcuvid", "cuda", "cudart")
    -- CUDA headers will be included via the package and explicit include path

    -- Add NvCodec source files
    add_files("NvCodec/NvEncoder/*.cpp")
    add_files("NvCodec/NvEncoder/NvEncoderD3D11.cpp")

    add_headerfiles("NvCodec/*.h", "NvCodec/NvEncoder/*.h")
    add_packages("glew", "glfw")
    
    add_syslinks("opengl32")
    
    add_syslinks("d3d11", "dxgi")
