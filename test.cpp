#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#include "NvCodec/NvEncoderCLIOptions.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <iostream>
#include <libloaderapi.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
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
// Replace the extern declarations with function pointers
typedef EncoderState *(*CreateEncoderFunc)(int, int, const char *, int);
typedef int (*RegisterTextureFunc)(EncoderState *, unsigned int);
typedef int (*EncodeFrameFunc)(EncoderState *);
typedef int (*DestroyEncoderFunc)(EncoderState *);

// Function pointers
CreateEncoderFunc gcne_create_encoder = nullptr;
RegisterTextureFunc gcne_register_texture = nullptr;
EncodeFrameFunc gcne_encode_frame = nullptr;
DestroyEncoderFunc gcne_destroy_encoder = nullptr;

// Create a texture with RGB columns and top half blank
GLuint createTestTexture(int width, int height) {
  GLuint textureId;
  glGenTextures(1, &textureId);
  glBindTexture(GL_TEXTURE_2D, textureId);

  // Create a pattern with left=red, middle=green, right=blue, top half blank
  unsigned char *data = (unsigned char *)malloc(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Top half is blank (black)
      if (y < height / 2) {
        data[(y * width + x) * 4 + 0] = 0; // R
        data[(y * width + x) * 4 + 1] = 0; // G
        data[(y * width + x) * 4 + 2] = 0; // B
      }
      // Bottom half has RGB columns
      else {
        // Left third is red
        if (x < width / 3) {
          data[(y * width + x) * 4 + 0] = 255; // R
          data[(y * width + x) * 4 + 1] = 0;   // G
          data[(y * width + x) * 4 + 2] = 0;   // B
        }
        // Middle third is green
        else if (x < 2 * width / 3) {
          data[(y * width + x) * 4 + 0] = 0;   // R
          data[(y * width + x) * 4 + 1] = 255; // G
          data[(y * width + x) * 4 + 2] = 0;   // B
        }
        // Right third is blue
        else {
          data[(y * width + x) * 4 + 0] = 0;   // R
          data[(y * width + x) * 4 + 1] = 0;   // G
          data[(y * width + x) * 4 + 2] = 255; // B
        }
      }
      data[(y * width + x) * 4 + 3] = 255; // A (always fully opaque)
    }
  }

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  free(data);

  return textureId;
}

// Create a texture with a simple pattern
// GLuint createTestTexture(int width, int height) {
//   GLuint textureId;
//   glGenTextures(1, &textureId);
//   glBindTexture(GL_TEXTURE_2D, textureId);

//   // Create a simple checkerboard pattern
//   unsigned char *data = (unsigned char *)malloc(width * height * 4);
//   for (int y = 0; y < height; y++) {
//     for (int x = 0; x < width; x++) {
//       unsigned char color = ((x / 32) + (y / 32)) % 2 ? 255 : 0;
//       data[(y * width + x) * 4 + 0] = color;              // R
//       data[(y * width + x) * 4 + 1] = (x * 255) / width;  // G
//       data[(y * width + x) * 4 + 2] = (y * 255) / height; // B
//       data[(y * width + x) * 4 + 3] = 255;                // A
//     }
//   }

//   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
//                GL_UNSIGNED_BYTE, data);
//   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//   free(data);

//   return textureId;
// }

int main() {
  // Initialize GLFW and create a window
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    return -1;
  }

  // In main(), add code to load the DLL
  HMODULE dllHandle = LoadLibraryA("GlCudaNvEncoder.dll");
  if (!dllHandle) {
    DWORD error = GetLastError();
    fprintf(stderr, "Failed to load DLL, error code: %lu\n", error);
    return -1;
  }

  // Get function pointers
  gcne_create_encoder =
      (CreateEncoderFunc)GetProcAddress(dllHandle, "gcne_create_encoder");
  gcne_register_texture =
      (RegisterTextureFunc)GetProcAddress(dllHandle, "gcne_register_texture");
  gcne_encode_frame =
      (EncodeFrameFunc)GetProcAddress(dllHandle, "gcne_encode_frame");
  gcne_destroy_encoder =
      (DestroyEncoderFunc)GetProcAddress(dllHandle, "gcne_destroy_encoder");

  // Check if all functions were found
  if (!gcne_create_encoder || !gcne_register_texture || !gcne_encode_frame ||
      !gcne_destroy_encoder) { // Similarly, add error code reporting when
                               // GetProcAddress fails
    DWORD error = GetLastError();
    fprintf(stderr, "Failed to get function pointers from DLL\n");
    FreeLibrary(dllHandle);
    return -1;
  }

  // Create a windowed mode window and its OpenGL context
  GLFWwindow *window =
      glfwCreateWindow(640, 480, "GlCudaNvEncoder Test", NULL, NULL);
  if (!window) {
    fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return -1;
  }

  // Make the window's context current
  glfwMakeContextCurrent(window);

  // Initialize GLEW
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    return -1;
  }

  // Video dimensions
  const int width = 640;
  const int height = 480;

  // Create a test texture
  GLuint textureId = createTestTexture(width, height);

  // Create encoder
  printf("Creating encoder...\n");
  EncoderState *encoder =
      gcne_create_encoder(width, height, "test_output.h264", 0);
  if (!encoder) {
    fprintf(stderr, "Failed to create encoder\n");
    return -1;
  }

  // Register texture with encoder
  printf("Registering texture...\n");
  if (gcne_register_texture(encoder, textureId) != 0) {
    fprintf(stderr, "Failed to register texture\n");
    gcne_destroy_encoder(encoder);
    return -1;
  }

  // Encode 100 frames
  printf("Encoding 100 frames...\n");
  std::chrono::time_point<std::chrono::high_resolution_clock> start =
      std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; i++) {
    // Update texture content (simple animation)
    glBindTexture(GL_TEXTURE_2D, textureId);
    unsigned char *data = (unsigned char *)malloc(width * height * 4);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        unsigned char color = ((x / 32) + (y / 32) + i) % 2 ? 255 : 0;
        data[(y * width + x) * 4 + 0] = color;
        data[(y * width + x) * 4 + 1] = (x * 255) / width;
        data[(y * width + x) * 4 + 2] = (y * 255) / height;
        data[(y * width + x) * 4 + 3] = 255;
      }
    }
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                    GL_UNSIGNED_BYTE, data);
    free(data);

    // Encode frame
    if (gcne_encode_frame(encoder) != 0) {
      fprintf(stderr, "Failed to encode frame %d\n", i);
      break;
    }

    printf("Frame %d encoded\r", i);
    fflush(stdout);

    // Process GLFW events to keep the window responsive
    glfwPollEvents();
    if (glfwWindowShouldClose(window)) {
      break;
    }
  }

  printf("\nFinished encoding frames\n");

  std::chrono::time_point<std::chrono::high_resolution_clock> end =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("Elapsed time: %f seconds\n", elapsed.count());
  printf("FPS: %f\n", 1000 / elapsed.count());

  // Clean up
  printf("Destroying encoder...\n");
  gcne_destroy_encoder(encoder);

  // Clean up OpenGL resources
  glDeleteTextures(1, &textureId);
  glfwTerminate();

  printf("Test completed successfully\n");

  // At the end of main()
  FreeLibrary(dllHandle);

  return 0;
}