
#include "DX11Renderer.hpp"
#include "NvCodec/NvCodecUtils.h"
#include "NvCodec/NvEncoder/NvEncoderD3D11.h"
#include "NvCodec/NvEncoderCLIOptions.h"
#include <chrono>
#include <d3d11.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

// 使用示例
void Encode(int nWidth, int nHeight, int nFrames, int ballRadius, int ballSpeed,
            char *szOutFilePath, NvEncoderInitParam *pEncodeCLIOptions,
            int iGpu) {
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

  std::cout << "Total frames encoded: " << nFrames << std::endl
            << "Saved in file " << szOutFilePath << std::endl;
}

int main(int argc, char **argv) {
  int nWidth = 1920, nHeight = 1080, nFrames = 3000, ballRadius = 50,
      ballSpeed = 10;
  char szOutFilePath[256] = "out.h264";

  auto start = std::chrono::system_clock::now();
  try {
    NvEncoderInitParam encodeCLIOptions;
    int iGpu = 0;
    Encode(nWidth, nHeight, nFrames, ballRadius, ballSpeed, szOutFilePath,
           &encodeCLIOptions, iGpu);
  } catch (const std::exception &ex) {
    std::cout << ex.what();
    exit(1);
  }
  auto end = std::chrono::system_clock::now();
  auto time_cost =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Time cost: " << time_cost
            << "ms \nFPS: " << nFrames / (time_cost / 1000.0) << std::endl;

  return 0;
}