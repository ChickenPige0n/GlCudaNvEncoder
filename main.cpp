#include "DX11Renderer.hpp"
#include "NvCodec/NvCodecUtils.h"
#include "NvCodec/NvEncoder/NvEncoderD3D11.h"
#include "NvCodec/NvEncoderCLIOptions.h"
#include <chrono>
#include <cstring>
#include <d3d11.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <wrl.h>


using Microsoft::WRL::ComPtr;

void EncodeWithNvenc(int nWidth, int nHeight, int nFrames, int ballRadius,
                     int ballSpeed, const char *szOutFilePath,
                     NvEncoderInitParam *pEncodeCLIOptions, int iGpu) {
  ComPtr<ID3D11Device> pDevice;
  ComPtr<ID3D11DeviceContext> pContext;
  ComPtr<IDXGIFactory1> pFactory;
  ComPtr<IDXGIAdapter> pAdapter;

  ck(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                        (void **)pFactory.GetAddressOf()));
  ck(pFactory->EnumAdapters(iGpu, pAdapter.GetAddressOf()));
  ck(D3D11CreateDevice(pAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, NULL, 0, NULL,
                       0, D3D11_SDK_VERSION, pDevice.GetAddressOf(), NULL,
                       pContext.GetAddressOf()));

  DX11Renderer renderer(pDevice.Get(), pContext.Get(), nWidth, nHeight);

  NvEncoderD3D11 enc(pDevice.Get(), nWidth, nHeight, NV_ENC_BUFFER_FORMAT_ARGB);

  NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
  NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
  initializeParams.encodeConfig = &encodeConfig;
  enc.CreateDefaultEncoderParams(&initializeParams,
                                 pEncodeCLIOptions->GetEncodeGUID(),
                                 pEncodeCLIOptions->GetPresetGUID());

  pEncodeCLIOptions->SetInitParams(&initializeParams,
                                   NV_ENC_BUFFER_FORMAT_ARGB);

  enc.CreateEncoder(&initializeParams);

  std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
  if (!fpOut) {
    std::ostringstream err;
    err << "Unable to open output file: " << szOutFilePath << std::endl;
    throw std::invalid_argument(err.str());
  }

  int ballX = 0;
  int ballY = nHeight / 2;

  for (int frame = 0; frame < nFrames; ++frame) {
    std::vector<std::vector<uint8_t>> vPacket;

    renderer.DrawBall(ballX, ballY, ballRadius);

    const NvEncInputFrame *encoderInputFrame = enc.GetNextInputFrame();
    ID3D11Texture2D *pTexBgra =
        reinterpret_cast<ID3D11Texture2D *>(encoderInputFrame->inputPtr);
    pContext->CopyResource(pTexBgra, renderer.GetTexture());

    enc.EncodeFrame(vPacket);

    for (std::vector<uint8_t> &packet : vPacket) {
      fpOut.write(reinterpret_cast<char *>(packet.data()), packet.size());
    }

    ballX += ballSpeed;
    if (ballX - ballRadius > nWidth) {
      ballX = -ballRadius;
    }
  }

  std::vector<std::vector<uint8_t>> vPacket;
  enc.EndEncode(vPacket);
  enc.DestroyEncoder();

  fpOut.close();

  std::cout << "Total frames encoded with NVENC: " << nFrames << std::endl
            << "Saved in file " << szOutFilePath << std::endl;
}

void EncodeWithFFmpeg(int nWidth, int nHeight, int nFrames, int ballRadius,
                      int ballSpeed, const char *szOutFilePath, int iGpu) {
  ComPtr<ID3D11Device> pDevice;
  ComPtr<ID3D11DeviceContext> pContext;
  ComPtr<IDXGIFactory1> pFactory;
  ComPtr<IDXGIAdapter> pAdapter;

  ck(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                        (void **)pFactory.GetAddressOf()));
  ck(pFactory->EnumAdapters(iGpu, pAdapter.GetAddressOf()));
  ck(D3D11CreateDevice(pAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, NULL, 0, NULL,
                       0, D3D11_SDK_VERSION, pDevice.GetAddressOf(), NULL,
                       pContext.GetAddressOf()));

  DX11Renderer renderer(pDevice.Get(), pContext.Get(), nWidth, nHeight);

  // Setup for reading back texture to CPU
  D3D11_TEXTURE2D_DESC desc;
  ZeroMemory(&desc, sizeof(desc));
  desc.Width = nWidth;
  desc.Height = nHeight;
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_STAGING;
  desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

  ComPtr<ID3D11Texture2D> stagingTexture;
  ck(pDevice->CreateTexture2D(&desc, NULL, stagingTexture.GetAddressOf()));

  // Setup FFmpeg process
  std::string ffmpegCmd =
      "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt bgra "
      "-s " +
      std::to_string(nWidth) + "x" + std::to_string(nHeight) +
      " "
      "-r 30 -i - -c:v libx264 -preset fast -pix_fmt yuv420p " +
      szOutFilePath;

  FILE *ffmpegPipe = _popen(ffmpegCmd.c_str(), "wb");
  if (!ffmpegPipe) {
    throw std::runtime_error("Failed to open pipe to FFmpeg");
  }

  int ballX = 0;
  int ballY = nHeight / 2;
  std::vector<uint8_t> frameData(nWidth * nHeight *
                                 4); // BGRA: 4 bytes per pixel

  for (int frame = 0; frame < nFrames; ++frame) {
    renderer.DrawBall(ballX, ballY, ballRadius);

    // Copy from render texture to staging texture
    pContext->CopyResource(stagingTexture.Get(), renderer.GetTexture());

    // Map the staging texture to read the data
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    ck(pContext->Map(stagingTexture.Get(), 0, D3D11_MAP_READ, 0,
                     &mappedResource));

    // Copy data row by row
    for (UINT row = 0; row < nHeight; ++row) {
      memcpy(frameData.data() + row * nWidth * 4,
             static_cast<uint8_t *>(mappedResource.pData) +
                 row * mappedResource.RowPitch,
             nWidth * 4);
    }

    pContext->Unmap(stagingTexture.Get(), 0);

    // Send frame to FFmpeg
    fwrite(frameData.data(), 1, frameData.size(), ffmpegPipe);

    ballX += ballSpeed;
    if (ballX - ballRadius > nWidth) {
      ballX = -ballRadius;
    }
  }

  // Close the pipe to FFmpeg
  _pclose(ffmpegPipe);

  std::cout << "Total frames encoded with FFmpeg: " << nFrames << std::endl
            << "Saved in file " << szOutFilePath << std::endl;
}

void PrintHelp() {
  std::cout << "Command line options:" << std::endl;
  std::cout << "-ffmpeg       Use FFmpeg for encoding instead of NVENC"
            << std::endl;
  std::cout << "-w <width>    Set width (default: 1920)" << std::endl;
  std::cout << "-h <height>   Set height (default: 1080)" << std::endl;
  std::cout << "-n <frames>   Set number of frames (default: 3000)"
            << std::endl;
  std::cout << "-o <file>     Set output file (default: out.h264)" << std::endl;
  std::cout << "-gpu <id>     Set GPU ID (default: 0)" << std::endl;
}

int main(int argc, char **argv) {
  int nWidth = 1920, nHeight = 1080, nFrames = 3000, ballRadius = 50,
      ballSpeed = 10;
  char szOutFilePath[256] = "out.h264";
  int iGpu = 0;
  bool useFFmpeg = false;

  // Parse command line options
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-ffmpeg")) {
      useFFmpeg = true;
    } else if (!strcmp(argv[i], "-w") && i + 1 < argc) {
      nWidth = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
      nHeight = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
      nFrames = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
      strcpy(szOutFilePath, argv[++i]);
    } else if (!strcmp(argv[i], "-gpu") && i + 1 < argc) {
      iGpu = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-help") || !strcmp(argv[i], "--help")) {
      PrintHelp();
      return 0;
    }
  }

  auto start = std::chrono::system_clock::now();
  try {
    if (useFFmpeg) {
      std::cout << "Using FFmpeg encoder (CPU readback)" << std::endl;
      EncodeWithFFmpeg(nWidth, nHeight, nFrames, ballRadius, ballSpeed,
                       szOutFilePath, iGpu);
    } else {
      std::cout << "Using NVENC encoder (GPU)" << std::endl;
      NvEncoderInitParam encodeCLIOptions;
      EncodeWithNvenc(nWidth, nHeight, nFrames, ballRadius, ballSpeed,
                      szOutFilePath, &encodeCLIOptions, iGpu);
    }
  } catch (const std::exception &ex) {
    std::cout << ex.what();
    return 1;
  }

  auto end = std::chrono::system_clock::now();
  auto time_cost =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Time cost: " << time_cost << "ms" << std::endl;
  std::cout << "FPS: " << nFrames / (time_cost / 1000.0) << std::endl;

  return 0;
}