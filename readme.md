# GlCudaNvEncoder NVENC Integration

This documentation explains how to use the NVIDIA hardware-accelerated video encoding integration for GlCudaNvEncoder.

## Overview

The library provides a simple C interface to encode OpenGL textures directly into video files using NVIDIA's NVENC hardware encoder. It efficiently transfers rendered textures from OpenGL to CUDA for encoding without CPU readbacks.

## Requirements

- NVIDIA GPU with NVENC support
- CUDA Toolkit
- OpenGL context
- GLFW and GLEW libraries

## API Reference

### Initialize Encoder

```c
EncoderState* gcne_create_encoder(int width, int height, const char* output_path, int gpu_id);
```

Creates and initializes the video encoder.

**Parameters:**
- `width`: Width of the output video in pixels
- `height`: Height of the output video in pixels
- `output_path`: File path for the encoded video output
- `gpu_id`: CUDA device ID to use for encoding (typically 0)

**Returns:** Encoder state handle or NULL on error

### Register OpenGL Texture

```c
int gcne_register_texture(EncoderState* state, unsigned int texture_id);
```

Registers an OpenGL texture for encoding.

**Parameters:**
- `state`: Encoder state handle
- `texture_id`: OpenGL texture ID to capture

**Returns:** 0 on success, -1 on error

### Encode Frame

```c
int gcne_encode_frame(EncoderState* state);
```

Encodes the current content of the registered texture.

**Parameters:**
- `state`: Encoder state handle

**Returns:** 0 on success, -1 on error

### Cleanup Resources

```c
int gcne_destroy_encoder(EncoderState* state);
```

Finalizes encoding and cleans up resources.

**Parameters:**
- `state`: Encoder state handle

**Returns:** 0 on success, -1 on error

## Usage Example

```c
// Create encoder (1280x720 video output to "output.h264" using GPU 0)
EncoderState* encoder = gcne_create_encoder(1280, 720, "output.h264", 0);
if (!encoder) {
	// Handle error
}

// Register your OpenGL texture
GLuint textureId = ...; // Your OpenGL texture
gcne_register_texture(encoder, textureId);

// In render loop
while (rendering) {
	// Render to texture
	// ...
	
	// Encode the current frame
	gcne_encode_frame(encoder);
}

// Clean up
gcne_destroy_encoder(encoder);
```

## Error Handling

All functions return error codes that should be checked. The library logs detailed error messages to stderr when operations fail.