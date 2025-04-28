#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#include "NvCodec/NvEncoderCLIOptions.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <memory>
#include <vector>

// Define export macro for DLL
#ifdef _WIN32
#define GCNE_API __declspec(dllexport)
#else
#define GCNE_API __attribute__((visibility("default")))
#endif

// Structure to hold encoder state
struct EncoderState {
    CUcontext cuContext;
    cudaGraphicsResource_t cudaResource;
    std::unique_ptr<NvEncoderCuda> encoder;
    int width;
    int height;
    bool initialized;
};
// Export C interface
extern "C" {
// Create and initialize the encoder
GCNE_API EncoderState *gcne_create_encoder(int width, int height, int gpu_id);

// Register an existing GL texture with the encoder
GCNE_API int gcne_register_texture(EncoderState *state,
                                   unsigned int texture_id);

// Encode a frame from the registered texture and return the packet data
GCNE_API int gcne_encode_frame(EncoderState *state, unsigned char **packet_data,
                               int *packet_size);

// Finalize encoding and return any remaining packets
GCNE_API int gcne_destroy_encoder(EncoderState *state,
                                  unsigned char **packet_data,
                                  int *packet_size);

// Free packet data allocated by the encoder
GCNE_API void gcne_free_packet(unsigned char *packet_data);
}

GCNE_API EncoderState *gcne_create_encoder(int width, int height, int gpu_id) {
    EncoderState *state = new EncoderState();
    state->width = width;
    state->height = height;
    state->initialized = false;

    try {
        // Initialize CUDA
        cudaSetDevice(gpu_id);
        cuCtxCreate(&state->cuContext, 0, gpu_id);

        // Create NVENC encoder with CUDA
        state->encoder = std::make_unique<NvEncoderCuda>(
            state->cuContext, width, height, NV_ENC_BUFFER_FORMAT_ABGR);

        // Set up encoding parameters
        NvEncoderInitParam encodeCLIOptions;
        NV_ENC_INITIALIZE_PARAMS initializeParams = {
            NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;

        state->encoder->CreateDefaultEncoderParams(
            &initializeParams, encodeCLIOptions.GetEncodeGUID(),
            encodeCLIOptions.GetPresetGUID());

        // 设置帧间隔 (GOP Size)
        initializeParams.encodeConfig->gopLength = 20;
        initializeParams.encodeConfig->frameIntervalP = 3;

        encodeCLIOptions.SetInitParams(&initializeParams,
                                       NV_ENC_BUFFER_FORMAT_ABGR);
        state->encoder->CreateEncoder(&initializeParams);

        state->initialized = true;
        return state;
    } catch (const std::exception &e) {
        std::cerr << "Error creating encoder: " << e.what() << std::endl;
        delete state;
        return nullptr;
    }
}

GCNE_API int gcne_register_texture(EncoderState *state,
                                   unsigned int texture_id) {
    if (!state || !state->initialized)
        return -1;

    try {
        // Register the OpenGL texture with CUDA
        cudaError_t cudaStatus = cudaGraphicsGLRegisterImage(
            &state->cudaResource, texture_id, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsReadOnly);

        if (cudaStatus != cudaSuccess) {
            std::cerr << "Failed to register texture: "
                      << cudaGetErrorString(cudaStatus) << std::endl;
            return -1;
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error registering texture: " << e.what() << std::endl;
        return -1;
    }
}

GCNE_API int gcne_encode_frame(EncoderState *state, unsigned char **packet_data,
                               int *packet_size) {
    if (!state || !state->initialized)
        return -1;

    try {
        std::vector<std::vector<uint8_t>> vPacket;

        // Map OpenGL texture to CUDA
        cudaArray_t cudaArray;
        cudaGraphicsMapResources(1, &state->cudaResource);
        cudaGraphicsSubResourceGetMappedArray(&cudaArray, state->cudaResource,
                                              0, 0);

        // Get next input frame from encoder
        const NvEncInputFrame *encoderInputFrame =
            state->encoder->GetNextInputFrame();

        // Copy from CUDA array to encoder input
        CUDA_MEMCPY2D copyParam = {};
        copyParam.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        copyParam.srcArray = (CUarray)cudaArray;
        copyParam.srcPitch = state->width * 4; // RGBA
        copyParam.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copyParam.dstDevice =
            reinterpret_cast<CUdeviceptr>(encoderInputFrame->inputPtr);
        copyParam.dstPitch = encoderInputFrame->pitch;
        copyParam.WidthInBytes = state->width * 4;
        copyParam.Height = state->height;

        cuMemcpy2D(&copyParam);

        // Unmap resource
        cudaGraphicsUnmapResources(1, &state->cudaResource);

        // Encode frame
        state->encoder->EncodeFrame(vPacket);

        // 合并所有数据包为一个连续缓冲区
        if (vPacket.size() > 0) {
            // 计算总大小
            size_t total_size = 0;
            for (const auto &packet : vPacket) {
                total_size += packet.size();
            }

            // 分配内存
            *packet_data = new unsigned char[total_size];
            *packet_size = static_cast<int>(total_size);

            // 复制数据
            size_t offset = 0;
            for (const auto &packet : vPacket) {
                memcpy(*packet_data + offset, packet.data(), packet.size());
                offset += packet.size();
            }
        } else {
            *packet_data = nullptr;
            *packet_size = 0;
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error encoding frame: " << e.what() << std::endl;
        return -1;
    }
}

GCNE_API int gcne_destroy_encoder(EncoderState *state,
                                  unsigned char **packet_data,
                                  int *packet_size) {
    if (!state)
        return -1;

    try {
        if (state->initialized) {
            // Finish encoding
            std::vector<std::vector<uint8_t>> vPacket;
            state->encoder->EndEncode(vPacket);

            // 合并所有数据包为一个连续缓冲区
            if (packet_data && packet_size && vPacket.size() > 0) {
                // 计算总大小
                size_t total_size = 0;
                for (const auto &packet : vPacket) {
                    total_size += packet.size();
                }

                // 分配内存
                *packet_data = new unsigned char[total_size];
                *packet_size = static_cast<int>(total_size);

                // 复制数据
                size_t offset = 0;
                for (const auto &packet : vPacket) {
                    memcpy(*packet_data + offset, packet.data(), packet.size());
                    offset += packet.size();
                }
            } else if (packet_data && packet_size) {
                *packet_data = nullptr;
                *packet_size = 0;
            }

            // Clean up resources
            state->encoder->DestroyEncoder();

            // Unregister CUDA resource if needed
            if (state->cudaResource) {
                cudaGraphicsUnregisterResource(state->cudaResource);
            }

            // Destroy CUDA context
            cuCtxDestroy(state->cuContext);
        }

        delete state;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error destroying encoder: " << e.what() << std::endl;
        return -1;
    }
}


GCNE_API void gcne_free_packet(unsigned char *packet_data) {
    if (packet_data) {
        delete[] packet_data;
    }
}