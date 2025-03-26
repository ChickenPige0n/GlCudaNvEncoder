#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#include "NvCodec/NvEncoderCLIOptions.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// Structure to hold encoder state
struct EncoderState {
  CUcontext cuContext;
  cudaGraphicsResource_t cudaResource;
  std::unique_ptr<NvEncoderCuda> encoder;
  std::ofstream outputFile;
  int width;
  int height;
  bool initialized;
};

// Export C interface
extern "C" {
// Create and initialize the encoder
EncoderState *gcne_create_encoder(int width, int height,
                                  const char *output_path, int gpu_id);

// Register an existing GL texture with the encoder
int gcne_register_texture(EncoderState *state, unsigned int texture_id);

// Encode a frame from the registered texture
int gcne_encode_frame(EncoderState *state);

// Finalize encoding and clean up resources
int gcne_destroy_encoder(EncoderState *state);
}

EncoderState *gcne_create_encoder(int width, int height,
                                  const char *output_path, int gpu_id) {
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
        state->cuContext, width, height, NV_ENC_BUFFER_FORMAT_ARGB);

    // Set up encoding parameters
    NvEncoderInitParam encodeCLIOptions;
    NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
    initializeParams.encodeConfig = &encodeConfig;

    state->encoder->CreateDefaultEncoderParams(
        &initializeParams, encodeCLIOptions.GetEncodeGUID(),
        encodeCLIOptions.GetPresetGUID());

    encodeCLIOptions.SetInitParams(&initializeParams,
                                   NV_ENC_BUFFER_FORMAT_ARGB);
    state->encoder->CreateEncoder(&initializeParams);

    // Open output file
    state->outputFile.open(output_path, std::ios::out | std::ios::binary);
    if (!state->outputFile) {
      delete state;
      return nullptr;
    }

    state->initialized = true;
    return state;
  } catch (const std::exception &e) {
    std::cerr << "Error creating encoder: " << e.what() << std::endl;
    delete state;
    return nullptr;
  }
}

int gcne_register_texture(EncoderState *state, unsigned int texture_id) {
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

int gcne_encode_frame(EncoderState *state) {
  if (!state || !state->initialized)
    return -1;

  try {
    std::vector<std::vector<uint8_t>> vPacket;

    // Map OpenGL texture to CUDA
    cudaArray_t cudaArray;
    cudaGraphicsMapResources(1, &state->cudaResource);
    cudaGraphicsSubResourceGetMappedArray(&cudaArray, state->cudaResource, 0,
                                          0);

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

    // Write packets to file
    for (std::vector<uint8_t> &packet : vPacket) {
      state->outputFile.write(reinterpret_cast<char *>(packet.data()),
                              packet.size());
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error encoding frame: " << e.what() << std::endl;
    return -1;
  }
}

int gcne_destroy_encoder(EncoderState *state) {
  if (!state)
    return -1;

  try {
    if (state->initialized) {
      // Finish encoding
      std::vector<std::vector<uint8_t>> vPacket;
      state->encoder->EndEncode(vPacket);

      // Write remaining packets to file
      for (std::vector<uint8_t> &packet : vPacket) {
        state->outputFile.write(reinterpret_cast<char *>(packet.data()),
                                packet.size());
      }

      // Clean up resources
      state->encoder->DestroyEncoder();
      state->outputFile.close();

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
