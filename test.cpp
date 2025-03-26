#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#include "NvCodec/NvEncoderCLIOptions.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstring>
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

// Create a shader program to generate the texture pattern on GPU
GLuint createShaderProgram() {
  const char *vertexShaderSource = "#version 330 core\n"
                                   "layout (location = 0) in vec3 aPos;\n"
                                   "layout (location = 1) in vec2 aTexCoord;\n"
                                   "out vec2 TexCoord;\n"
                                   "void main() {\n"
                                   "   gl_Position = vec4(aPos, 1.0);\n"
                                   "   TexCoord = aTexCoord;\n"
                                   "}\n";

  const char *fragmentShaderSource =
      "#version 330 core\n"
      "out vec4 FragColor;\n"
      "in vec2 TexCoord;\n"
      "uniform int frameCount;\n"
      "void main() {\n"
      "   // Animation based on frameCount\n"
      "   float color = mod(floor(TexCoord.x * 32) + floor(TexCoord.y * 32) + "
      "frameCount, 2.0);\n"
      "   // Create pattern similar to the original\n"
      "   if (TexCoord.y < 0.5) {\n"
      "       FragColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
      "   } else {\n"
      "       if (TexCoord.x < 0.33) {\n"
      "           FragColor = vec4(color, TexCoord.x * 3.0, TexCoord.y, 1.0);\n"
      "       } else if (TexCoord.x < 0.66) {\n"
      "           FragColor = vec4(TexCoord.x * 1.5, color, TexCoord.y, 1.0);\n"
      "       } else {\n"
      "           FragColor = vec4(TexCoord.x, TexCoord.y, color, 1.0);\n"
      "       }\n"
      "   }\n"
      "}\n";

  // Compile vertex shader
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);

  // Check for vertex shader compilation errors
  int success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    fprintf(stderr, "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
  }

  // Compile fragment shader
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);

  // Check for fragment shader compilation errors
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    fprintf(stderr, "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n",
            infoLog);
  }

  // Link shaders into a shader program
  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);

  // Check for shader program linking errors
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    fprintf(stderr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  return shaderProgram;
}

// Setup quad for rendering to texture
void setupQuad(GLuint &VAO, GLuint &VBO, GLuint &EBO) {
  float vertices[] = {
      // positions          // texture coords
      1.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top right
      1.0f,  -1.0f, 0.0f, 1.0f, 0.0f, // bottom right
      -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
      -1.0f, 1.0f,  0.0f, 0.0f, 1.0f  // top left
  };
  unsigned int indices[] = {
      0, 1, 3, // first triangle
      1, 2, 3  // second triangle
  };

  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  // texture coord attribute
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
}

// Create a texture for rendering
GLuint createRenderTexture(int width, int height) {
  GLuint textureId;
  glGenTextures(1, &textureId);
  glBindTexture(GL_TEXTURE_2D, textureId);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  return textureId;
}

int main(int argc, char **argv) {
  bool use_ffmpeg = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-ffmpeg") == 0) {
      use_ffmpeg = true;
      printf("FFmpeg mode enabled\n");
    }
  }

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
  if (!use_ffmpeg) {
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
        !gcne_destroy_encoder) {
      DWORD error = GetLastError();
      fprintf(stderr, "Failed to get function pointers from DLL\n");
      FreeLibrary(dllHandle);
      return -1;
    }
  }

  // Set OpenGL context version
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    return -1;
  }

  // Video dimensions
  // 4k resolution
  const int width = 3840;
  const int height = 2160;
  // log video dimensions
  printf("Video dimensions: %dx%d\n", width, height);
  const int byte_size = width * height * 4;

  // Create shader program
  GLuint shaderProgram = createShaderProgram();

  // Create quad for rendering
  GLuint quadVAO, quadVBO, quadEBO;
  setupQuad(quadVAO, quadVBO, quadEBO);

  // Create a test texture
  GLuint textureId = createRenderTexture(width, height);

  // Create a framebuffer
  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         textureId, 0);

  // Check framebuffer completeness
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    fprintf(stderr, "Framebuffer is not complete!\n");
    return -1;
  }

  // Setup for encoder or FFmpeg
  EncoderState *encoder = nullptr;
  FILE *ffmpeg_pipe = nullptr;
  const int N = 60; // Number of PBOs to use
  GLuint pbos[N] = {0};

  if (use_ffmpeg) {
    // Initialize PBOs for FFmpeg mode
    glGenBuffers(N, pbos);
    for (int i = 0; i < N; i++) {
      glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[i]);
      glBufferData(GL_PIXEL_PACK_BUFFER, byte_size, NULL, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // Open pipe to FFmpeg
    char ffmpeg_cmd[512];
    snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
             "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgba "
             "-s %dx%d -r 30 -i - -c:v h264_nvenc -preset fast -crf 18 "
             "-pix_fmt yuv420p ffmpeg_output.mp4 -loglevel error -stats "
             "2>ffmpeg_log.txt",
             width, height);

    printf("Starting FFmpeg with command: %s\n", ffmpeg_cmd);
    ffmpeg_pipe = _popen(ffmpeg_cmd, "wb");
    if (!ffmpeg_pipe) {
      fprintf(stderr, "Failed to open FFmpeg pipe\n");
      return -1;
    }
  } else {
    // Create encoder
    printf("Creating encoder...\n");
    encoder = gcne_create_encoder(width, height, "test_output.h264", 0);
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
  }

  // Encode 1000 frames
  printf("Encoding 1000 frames...\n");
  std::chrono::time_point<std::chrono::high_resolution_clock> start =
      std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 1000; i++) {
    // Bind the framebuffer for rendering to texture
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Use shader and set uniforms
    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "frameCount"), i);

    // Render the quad with our texture
    glBindVertexArray(quadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // Make sure rendering is complete before encoding
    glFinish();

    if (use_ffmpeg) {
      // Read pixels to PBO
      glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[i % N]);
      glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

      // Map previous PBO and write to FFmpeg
      int prev_idx = (i + 1) % N;
      if (i >= N - 1) { // Only start reading after we've filled the pipeline
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[prev_idx]);
        void *ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
        if (ptr) {
          fwrite(ptr, 1, byte_size, ffmpeg_pipe);
          glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        }
      }

      glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    } else {
      // Encode frame using NvEncoder
      if (gcne_encode_frame(encoder) != 0) {
        fprintf(stderr, "Failed to encode frame %d\n", i);
        break;
      }
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
  if (use_ffmpeg) {
    // Process any remaining PBOs
    for (int i = 0; i < N; i++) {
      glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[i]);
      void *ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
      if (ptr) {
        fwrite(ptr, 1, byte_size, ffmpeg_pipe);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
      }
    }

    // Clean up PBOs
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glDeleteBuffers(N, pbos);

    // Close FFmpeg pipe
    printf("Closing FFmpeg pipe...\n");
    _pclose(ffmpeg_pipe);
  } else {
    // Clean up encoder
    printf("Destroying encoder...\n");
    gcne_destroy_encoder(encoder);
  }

  // Clean up OpenGL resources
  glDeleteTextures(1, &textureId);
  glDeleteProgram(shaderProgram);
  glDeleteVertexArrays(1, &quadVAO);
  glDeleteBuffers(1, &quadVBO);
  glDeleteBuffers(1, &quadEBO);
  glDeleteFramebuffers(1, &fbo);

  glfwTerminate();

  printf("Test completed successfully\n");

  // At the end of main()
  FreeLibrary(dllHandle);

  return 0;
}