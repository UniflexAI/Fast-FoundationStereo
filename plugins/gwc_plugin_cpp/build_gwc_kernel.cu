#include <cuda_fp16.h>

extern "C" __global__ void build_gwc_f32(
    const float* __restrict__ ref, const float* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G)
{
    int w   = blockIdx.x * blockDim.x + threadIdx.x;
    int h   = blockIdx.y * blockDim.y + threadIdx.y;
    int bgd = blockIdx.z;
    if (w >= W || h >= H || bgd >= B * G * D) return;
    int d = bgd % D, g = (bgd / D) % G, b = bgd / (D * G);
    int Cg = C / G, c0 = g * Cg, tw = w - d;
    float sum = 0.0f;
    for (int c = 0; c < Cg; ++c) {
        int base = b*C*H*W + (c0+c)*H*W + h*W;
        sum += ref[base + w] * ((tw >= 0) ? tgt[base + tw] : 0.0f);
    }
    gwc[b*G*D*H*W + g*D*H*W + d*H*W + h*W + w] = __float2half(sum);
}

extern "C" __global__ void build_gwc_f16(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G)
{
    int w   = blockIdx.x * blockDim.x + threadIdx.x;
    int h   = blockIdx.y * blockDim.y + threadIdx.y;
    int bgd = blockIdx.z;
    if (w >= W || h >= H || bgd >= B * G * D) return;
    int d = bgd % D, g = (bgd / D) % G, b = bgd / (D * G);
    int Cg = C / G, c0 = g * Cg, tw = w - d;
    float sum = 0.0f;
    for (int c = 0; c < Cg; ++c) {
        int base = b*C*H*W + (c0+c)*H*W + h*W;
        sum += __half2float(ref[base + w])
             * ((tw >= 0) ? __half2float(tgt[base + tw]) : 0.0f);
    }
    gwc[b*G*D*H*W + g*D*H*W + d*H*W + h*W + w] = __float2half(sum);
}

#include "NvInfer.h"
#include <cuda_runtime.h>

void launchBuildGwc(nvinfer1::DataType type, const void* ref, const void* tgt, void* gwc,
    int B, int C, int H, int W, int D, int G, cudaStream_t stream)
{
    const int BX = 16, BY = 16;
    dim3 block(BX, BY, 1);
    dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B * G * D);
    if (type == nvinfer1::DataType::kHALF)
        build_gwc_f16<<<grid, block, 0, stream>>>(
            static_cast<__half const*>(ref), static_cast<__half const*>(tgt), static_cast<__half*>(gwc),
            B, C, H, W, D, G);
    else
        build_gwc_f32<<<grid, block, 0, stream>>>(
            static_cast<float const*>(ref), static_cast<float const*>(tgt), static_cast<__half*>(gwc),
            B, C, H, W, D, G);
}
