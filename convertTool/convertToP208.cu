#include <cuda.h>
#include <cuda_runtime.h>

#include "convertToP208.h"

__global__ static void convertToP208Kernel(uint16_t *pV210, uint16_t *dP208, int nPitch, int nWidth, int nHeight) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tidd = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
    uint4 pF;
    int nDstW = nPitch / 8;
    int nDstH = nHeight;
    if (tid < nDstW && tidd < nDstH) {
        int k = tid * 8;
        int j = tidd * nPitch;
        pF.x = (uint32_t)pV210[j + k + 0] + ((uint32_t)pV210[j + k + 1] << 16);
        pF.y = (uint32_t)pV210[j + k + 2] + ((uint32_t)pV210[j + k + 3] << 16);
        pF.z = (uint32_t)pV210[j + k + 4] + ((uint32_t)pV210[j + k + 5] << 16);
        pF.w = (uint32_t)pV210[j + k + 6] + ((uint32_t)pV210[j + k + 7] << 16);

        v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
        y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
        u0 = (uint32_t)(pF.x & 0x000003FF);
        y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20);
        u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
        y1 = (uint32_t)(pF.y & 0x000003FF);
        u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
        y3 = (uint32_t)((pF.z & 0x000FFC00) >> 10);
        v1 = (uint32_t)(pF.z & 0x000003FF);
        y5 = (uint32_t)((pF.w & 0x3FF00000) >> 20);
        v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
        y4 = (uint32_t)(pF.w & 0x000003FF);

        k = tid * 6;
        j = tidd * nPitch * 3 / 4;
        dP208[j + k + 0] = y0;
        dP208[j + k + 1] = y1;
        dP208[j + k + 2] = y2;
        dP208[j + k + 3] = y3;
        dP208[j + k + 4] = y4;
        dP208[j + k + 5] = y5;
        k = tid * 3;
        j = tidd * nPitch * 3 / 8 + nWidth * nHeight;
        dP208[j + k + 0] = u0;
        dP208[j + k + 1] = u1;
        dP208[j + k + 2] = u2;
        j = tidd * nPitch * 3 / 8 + nWidth * nHeight * 3 / 2;
        dP208[j + k + 0] = v0;
        dP208[j + k + 1] = v1;
        dP208[j + k + 2] = v2;
    }
}

void convertToP208(uint16_t *pV210, uint16_t *dP208, int nPitch, int nWidth, int nHeight, cudaStream_t stream) {
    dim3 blocks(32, 16, 1);
    dim3 grids((nPitch + blocks.x - 1) / blocks.x, (nHeight + blocks.y - 1) / blocks.y, 1);
    convertToP208Kernel << < grids, blocks, 0, stream >> > (pV210, dP208, nPitch, nWidth, nHeight);
}