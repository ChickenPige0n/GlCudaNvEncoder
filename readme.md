# GlCudaNvEncoder NVENC Integration

This documentation explains how to use the NVIDIA hardware-accelerated video encoding integration for GlCudaNvEncoder.

## Overview

The library provides a simple C interface to encode OpenGL textures directly into video streams using NVIDIA's NVENC hardware encoder. It efficiently transfers rendered textures from OpenGL to CUDA for encoding without CPU readbacks, significantly improving performance.

## Requirements

- NVIDIA GPU with NVENC support
- CUDA Toolkit
- OpenGL context
- GLFW and GLEW libraries

## API Reference

### Initialize Encoder

```c
EncoderState* gcne_create_encoder(int width, int height, int gpu_id);
```

Creates and initializes the video encoder.

**Parameters:**
- `width`: Width of the output video in pixels
- `height`: Height of the output video in pixels
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
int gcne_encode_frame(EncoderState* state, unsigned char** packet_data, int* packet_size);
```

Encodes the current content of the registered texture and returns the encoded packet.

**Parameters:**
- `state`: Encoder state handle
- `packet_data`: Output parameter pointing to the encoded data
- `packet_size`: Output parameter containing the size of the encoded data in bytes

**Returns:** 0 on success, -1 on error

### Free Packet Memory

```c
void gcne_free_packet(unsigned char* packet_data);
```

Frees memory allocated for encoded packet data.

**Parameters:**
- `packet_data`: Packet pointer allocated by `gcne_encode_frame` or `gcne_destroy_encoder`

### Cleanup Resources

```c
int gcne_destroy_encoder(EncoderState* state, unsigned char** packet_data, int* packet_size);
```

Finalizes encoding, cleans up resources, and returns any remaining packets.

**Parameters:**
- `state`: Encoder state handle
- `packet_data`: Output parameter pointing to final encoded data (if any)
- `packet_size`: Output parameter containing the size of the final encoded data in bytes

**Returns:** 0 on success, -1 on error

## Detailed Encoding Process

GlCudaNvEncoder implements an efficient OpenGL-to-H.264/HEVC encoding pipeline:

1. **Initialization** - `gcne_create_encoder` creates a CUDA context and NVENC encoder instance, setting up encoding parameters (such as GOP size and frame interval)

2. **Texture Registration** - `gcne_register_texture` registers an OpenGL texture using CUDA-OpenGL interoperability APIs, allowing CUDA to access the texture's content

3. **Frame Encoding** - `gcne_encode_frame` performs the following steps:
   - Maps the OpenGL texture to CUDA
   - Obtains the next input frame buffer from the encoder
   - Transfers texture data to the encoder buffer using CUDA memory copy (in ABGR format)
   - Calls NVENC to encode the current frame
   - Collects the encoded packet and returns it to the caller

4. **Memory Management** - `gcne_free_packet` is used to release encoded data memory to prevent memory leaks

5. **Finalization** - `gcne_destroy_encoder` completes the encoding process, flushes any buffered frames, and cleans up CUDA and NVENC resources

## Usage Example

```c
// Create encoder (1920x1080 video using GPU 0)
EncoderState* encoder = gcne_create_encoder(1920, 1080, 0);
if (!encoder) {
    // Handle error
}

// Register OpenGL texture
GLuint textureId = ...; // Your OpenGL texture
if (gcne_register_texture(encoder, textureId) != 0) {
    // Handle error
}

// Set up FFmpeg pipe for H.264 data
FILE* ffmpeg_pipe = popen("ffmpeg -y -f h264 -i - -c:v copy output.mp4", "wb");

// In render loop
for (int i = 0; i < frameCount; i++) {
    // Render to texture
    // ...
    
    // Encode current frame
    unsigned char* packet_data = nullptr;
    int packet_size = 0;
    
    if (gcne_encode_frame(encoder, &packet_data, &packet_size) == 0) {
        // Process encoded data - e.g., write to file or send over network
        if (packet_data && packet_size > 0) {
            fwrite(packet_data, 1, packet_size, ffmpeg_pipe);
        }
        
        // Free packet memory
        gcne_free_packet(packet_data);
    }
}

// Finalize encoding and get any remaining data
unsigned char* final_packet = nullptr;
int final_size = 0;
gcne_destroy_encoder(encoder, &final_packet, &final_size);

if (final_packet && final_size > 0) {
    fwrite(final_packet, 1, final_size, ffmpeg_pipe);
    gcne_free_packet(final_packet);
}

// Close FFmpeg pipe
pclose(ffmpeg_pipe);
```

## Error Handling

All functions return error codes that should be checked. The library logs detailed error messages to stderr when operations fail.

## Pixel Format

The library uses ABGR (NV_ENC_BUFFER_FORMAT_ABGR) pixel format by default. Ensure your OpenGL textures are compatible with this format, or consider using a shader for color space conversion.

## TODO:
1. Implement automatic horizontal frame inversion filter
2. Add H.265/HEVC encoding support
3. Provide more encoding parameter control options (bitrate, encoding presets, etc.)
