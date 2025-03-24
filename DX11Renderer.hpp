#include <d3d11.h>
#include <stdexcept>
#include <vector>
#include <wrl.h>


using Microsoft::WRL::ComPtr;

class DX11Renderer {
public:
  DX11Renderer(ID3D11Device *device, ID3D11DeviceContext *context, int width,
               int height)
      : pDevice(device), pContext(context), nWidth(width), nHeight(height) {
    CreateTexture();
  }

  void DrawBall(int ballX, int ballY, int ballRadius,
                uint32_t color = 0xFFFFFFFF) {
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    pContext->Map(pTexture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0,
                  &mappedResource);

    uint32_t *pData = reinterpret_cast<uint32_t *>(mappedResource.pData);
    memset(pData, 0, nWidth * nHeight * 4);

    for (int y = -ballRadius; y <= ballRadius; ++y) {
      for (int x = -ballRadius; x <= ballRadius; ++x) {
        if (x * x + y * y <= ballRadius * ballRadius) {
          int px = ballX + x;
          int py = ballY + y;
          if (px >= 0 && px < nWidth && py >= 0 && py < nHeight) {
            pData[py * nWidth + px] = color;
          }
        }
      }
    }

    pContext->Unmap(pTexture.Get(), 0);
  }

  ID3D11Texture2D *GetTexture() const { return pTexture.Get(); }

private:
  void CreateTexture() {
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
    desc.Width = nWidth;
    desc.Height = nHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(
            pDevice->CreateTexture2D(&desc, NULL, pTexture.GetAddressOf()))) {
      throw std::runtime_error("Failed to create texture");
    }
  }

  ComPtr<ID3D11Device> pDevice;
  ComPtr<ID3D11DeviceContext> pContext;
  ComPtr<ID3D11Texture2D> pTexture;
  int nWidth;
  int nHeight;
};