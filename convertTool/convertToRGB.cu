#include <cuda.h>
#include <cuda_runtime.h>

#include "convertToRGB.h"

__global__ static void convertToRGBKernel(const uint16_t *pV210, uint16_t *tt, int nSrcWidth,
    int nDstWidth, int nDstHeight) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tidd = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
    uint4 pF;
    if (tid < (nSrcWidth / 8) && tidd < nDstHeight) {
        int j = tidd * nSrcWidth;
        int k = tid * 8;
        pF.x = (uint32_t)pV210[j + k + 0] + ((uint32_t)pV210[j + k + 1] << 16);
        pF.y = (uint32_t)pV210[j + k + 2] + ((uint32_t)pV210[j + k + 3] << 16);
        pF.z = (uint32_t)pV210[j + k + 4] + ((uint32_t)pV210[j + k + 5] << 16);
        pF.w = (uint32_t)pV210[j + k + 6] + ((uint32_t)pV210[j + k + 7] << 16);

        v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
        y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10) * 1164;
        u0 = (uint32_t)(pF.x & 0x000003FF);
        y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20) * 1164;
        u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
        y1 = (uint32_t)(pF.y & 0x000003FF) * 1164;
        u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
        y3 = (uint32_t)((pF.z & 0x000FFC00) >> 10) * 1164;
        v1 = (uint32_t)(pF.z & 0x000003FF);
        y5 = (uint32_t)((pF.w & 0x3FF00000) >> 20) * 1164;
        v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
        y4 = (uint32_t)(pF.w & 0x000003FF) * 1164;

        k = tid * 18;
        j *= 9;
        j /= 4;
        int r = 1596 * v0 - 891648, g = 813 * v0 + 392 * u0 - 542464, b = 2017 * u0 - 1107200;
        tt[j + k + 0] = (y0 + r) / 1000;
        tt[j + k + 1] = (y0 - g) / 1000;
        tt[j + k + 2] = (y0 + b) / 1000;

        tt[j + k + 3] = (y1 + r) / 1000;
        tt[j + k + 4] = (y1 - g) / 1000;
        tt[j + k + 5] = (y1 + b) / 1000;

        r = 1596 * v1 - 891648, g = 813 * v1 + 392 * u1 - 542464, b = 2017 * u1 - 1107200;
        tt[j + k + 6] = (y2 + r) / 1000;
        tt[j + k + 7] = (y2 - g) / 1000;
        tt[j + k + 8] = (y2 + b) / 1000;

        tt[j + k + 9] = (y3 + r) / 1000;
        tt[j + k + 10] = (y3 - g) / 1000;
        tt[j + k + 11] = (y3 + b) / 1000;

        r = 1596 * v2 - 891648, g = 813 * v2 + 392 * u2 - 542464, b = 2017 * u2 - 1107200;
        tt[j + k + 12] = (y4 + r) / 1000;
        tt[j + k + 13] = (y4 - g) / 1000;
        tt[j + k + 14] = (y4 + b) / 1000;

        tt[j + k + 15] = (y5 + r) / 1000;
        tt[j + k + 16] = (y5 - g) / 1000;
        tt[j + k + 17] = (y5 + b) / 1000;
    }
}

void convertToRGB(uint16_t *dpSrc, uint16_t *dpDst, int nSrcWidth, int nDstWidth, int nDstHeight, cudaStream_t stream) {
    dim3 blocks(32, 16, 1);
    dim3 grids((nSrcWidth + blocks.x - 1) / blocks.x, (nDstHeight + blocks.y - 1) / blocks.y, 1);
    convertToRGBKernel << <grids, blocks, 0, stream >> > (dpSrc, dpDst, nSrcWidth, nDstWidth, nDstHeight);
}

__global__ static void convertToRGBTestKernel(const uint16_t *pV210, uint8_t *tt, int nSrcWidth,
    int nDstWidth, int nDstHeight) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tidd = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
    uint4 pF;
    uint16_t tt1[6];
    
    if (tid < (nSrcWidth / 8) && tidd < nDstHeight) {
        int j = tidd * nSrcWidth;
        int k = tid * 8;
        pF.x = (uint32_t)pV210[j + k + 0] + ((uint32_t)pV210[j + k + 1] << 16);
        pF.y = (uint32_t)pV210[j + k + 2] + ((uint32_t)pV210[j + k + 3] << 16);
        pF.z = (uint32_t)pV210[j + k + 4] + ((uint32_t)pV210[j + k + 5] << 16);
        pF.w = (uint32_t)pV210[j + k + 6] + ((uint32_t)pV210[j + k + 7] << 16);

        v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
        y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10) * 1164;
        u0 = (uint32_t)(pF.x & 0x000003FF);
        y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20) * 1164;
        u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
        y1 = (uint32_t)(pF.y & 0x000003FF) * 1164;
        u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
        y3 = (uint32_t)((pF.z & 0x000FFC00) >> 10) * 1164;
        v1 = (uint32_t)(pF.z & 0x000003FF);
        y5 = (uint32_t)((pF.w & 0x3FF00000) >> 20) * 1164;
        v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
        y4 = (uint32_t)(pF.w & 0x000003FF) * 1164;

        k = tid * 18;
        j *= 9;
        j /= 4;
        int r = 1596 * v0 - 891648, g = 813 * v0 + 392 * u0 - 542464, b = 2017 * u0 - 1107200;
        tt[j + k + 0] = (y0 + r) * 0.249 / 1000;
        tt[j + k + 1] = (y0 - g) * 0.249 / 1000;
        tt[j + k + 2] = (y0 + b) * 0.249 / 1000;

        tt[j + k + 3] = (y1 + r) * 0.249 / 1000;
        tt[j + k + 4] = (y1 - g) * 0.249 / 1000;
        tt[j + k + 5] = (y1 + b) * 0.249 / 1000;

        r = 1596 * v1 - 891648, g = 813 * v1 + 392 * u1 - 542464, b = 2017 * u1 - 1107200;
        tt[j + k + 6] = (y2 + r) * 0.249 / 1000;
        tt[j + k + 7] = (y2 - g) * 0.249 / 1000;
        tt[j + k + 8] = (y2 + b) * 0.249 / 1000;

        tt[j + k + 9] = (y3 + r) * 0.249 / 1000;
        tt[j + k + 10] = (y3 - g) * 0.249 / 1000;
        tt[j + k + 11] = (y3 + b) * 0.249 / 1000;

        r = 1596 * v2 - 891648, g = 813 * v2 + 392 * u2 - 542464, b = 2017 * u2 - 1107200;
        tt[j + k + 12] = (y4 + r) * 0.249 / 1000;
        tt[j + k + 13] = (y4 - g) * 0.249 / 1000;
        tt[j + k + 14] = (y4 + b) * 0.249 / 1000;

        tt[j + k + 15] = (y5 + r) * 0.249 / 1000;
        tt[j + k + 16] = (y5 - g) * 0.249 / 1000;
        tt[j + k + 17] = (y5 + b) * 0.249 / 1000;
    }
}

void convertToRGBTest(uint16_t *dpSrc, uint8_t *dpDst, int nSrcWidth, int nDstWidth, int nDstHeight,
    cudaStream_t stream) {
    dim3 blocks(32, 16, 1);
    dim3 grids((nSrcWidth + blocks.x - 1) / blocks.x, (nDstHeight + blocks.y - 1) / blocks.y, 1);
    convertToRGBTestKernel << <grids, blocks, 0, stream >> > (dpSrc, dpDst, nSrcWidth, nDstWidth, nDstHeight);
}

__global__ static void convertVToRGBKernel(const uint16_t *pV210, uint8_t *tt1, int nSrcWidth,
    int nDstWidth, int nDstHeight, int *lookupTable) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tidd = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
    uint16_t tt[6];
    uint4 pF;
    int nDstH = nDstHeight;
    int nDstW = nSrcWidth / 8;

    if (tid < nDstW && tidd < nDstH) {
        int j = tidd * nSrcWidth;
        int k = tid * 8;
        pF.x = (uint32_t)pV210[j + k + 0] + ((uint32_t)pV210[j + k + 1] << 16);
        pF.y = (uint32_t)pV210[j + k + 2] + ((uint32_t)pV210[j + k + 3] << 16);
        pF.z = (uint32_t)pV210[j + k + 4] + ((uint32_t)pV210[j + k + 5] << 16);
        pF.w = (uint32_t)pV210[j + k + 6] + ((uint32_t)pV210[j + k + 7] << 16);

        v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
        y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10) * 1164;
        u0 = (uint32_t)(pF.x & 0x000003FF);
        y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20) * 1164;
        u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
        y1 = (uint32_t)(pF.y & 0x000003FF) * 1164;
        u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
        y3 = (uint32_t)((pF.z & 0x000FFC00) >> 10) * 1164;
        v1 = (uint32_t)(pF.z & 0x000003FF);
        y5 = (uint32_t)((pF.w & 0x3FF00000) >> 20) * 1164;
        v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
        y4 = (uint32_t)(pF.w & 0x000003FF) * 1164;

        k = tid * 18;
        j *= 9;
        j /= 4;
        int r = 1596 * v0 - 891648, g = 813 * v0 + 392 * u0 - 542464, b = 2017 * u0 - 1107200;
        tt[0] = (y0 + r) / 1000;
        tt[1] = (y0 - g) / 1000;
        tt[2] = (y0 + b) / 1000;

        tt[3] = (y1 + r) / 1000;
        tt[4] = (y1 - g) / 1000;
        tt[5] = (y1 + b) / 1000;

        tt1[j + k + 0] = lookupTable[tt[0]];
        tt1[j + k + 1] = lookupTable[tt[1]];
        tt1[j + k + 2] = lookupTable[tt[2]];

        tt1[j + k + 3] = lookupTable[tt[3]];
        tt1[j + k + 4] = lookupTable[tt[4]];
        tt1[j + k + 5] = lookupTable[tt[5]];

        r = 1596 * v1 - 891648, g = 813 * v1 + 392 * u1 - 542464, b = 2017 * u1 - 1107200;
        tt[0] = (y2 + r) / 1000;
        tt[1] = (y2 - g) / 1000;
        tt[2] = (y2 + b) / 1000;

        tt[3] = (y3 + r) / 1000;
        tt[4] = (y3 - g) / 1000;
        tt[5] = (y3 + b) / 1000;

        tt1[j + k + 6] = lookupTable[tt[0]];
        tt1[j + k + 7] = lookupTable[tt[1]];
        tt1[j + k + 8] = lookupTable[tt[2]];

        tt1[j + k + 9] = lookupTable[tt[3]];
        tt1[j + k + 10] = lookupTable[tt[4]];
        tt1[j + k + 11] = lookupTable[tt[5]];

        r = 1596 * v2 - 891648, g = 813 * v2 + 392 * u2 - 542464, b = 2017 * u2 - 1107200;
        tt[0] = (y4 + r) / 1000;
        tt[1] = (y4 - g) / 1000;
        tt[2] = (y4 + b) / 1000;

        tt[3] = (y5 + r) / 1000;
        tt[4] = (y5 - g) / 1000;
        tt[5] = (y5 + b) / 1000;

        tt1[j + k + 12] = lookupTable[tt[0]];
        tt1[j + k + 13] = lookupTable[tt[1]];
        tt1[j + k + 14] = lookupTable[tt[2]];

        tt1[j + k + 15] = lookupTable[tt[3]];
        tt1[j + k + 16] = lookupTable[tt[4]];
        tt1[j + k + 17] = lookupTable[tt[5]];
    }
}

__global__ static void convertPToRGBKernel(const uint16_t *dpSrc, uint8_t *tt1, int nSrcWidth,
    int nDstWidth, int nDstHeight, int *lookupTable) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tidd = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t v0, y0, u0, y1;
    uint16_t tt[6];
    int nDstH = nDstHeight;
    int nDstW = nSrcWidth / 2;
    if (tid < nDstW && tidd < nDstH) {
        int k = tid * 2;
        int j = tidd * nSrcWidth;
        y0 = (uint32_t)dpSrc[j + k + 0] * 1164;
        y1 = (uint32_t)dpSrc[j + k + 1] * 1164;
        k = tid;
        j = tidd * nSrcWidth / 2 + nDstHeight * nSrcWidth;
        u0 = (uint32_t)dpSrc[j + k + 0];
        j = tidd * nSrcWidth / 2 + nDstHeight * nSrcWidth * 3 / 2;
        v0 = (uint32_t)dpSrc[j + k + 0];

        k = tid * 6;
        j = tidd * nDstWidth * 3;
        int r = 1596 * v0 - 891648, g = 813 * v0 + 392 * u0 - 542464, b = 2017 * u0 - 1107200;
        tt[0] = (y0 + r) / 1000;
        tt[1] = (y0 - g) / 1000;
        tt[2] = (y0 + b) / 1000;
        tt[3] = (y1 + r) / 1000;
        tt[4] = (y1 - g) / 1000;
        tt[5] = (y1 + b) / 1000;

        tt1[j + k + 0] = lookupTable[tt[0]];
        tt1[j + k + 1] = lookupTable[tt[1]];
        tt1[j + k + 2] = lookupTable[tt[2]];
        tt1[j + k + 3] = lookupTable[tt[3]];
        tt1[j + k + 4] = lookupTable[tt[4]];
        tt1[j + k + 5] = lookupTable[tt[5]];
    }
}

void convertToRGB(uint16_t *dpSrc, uint8_t *dpDst, int nSrcWidth, int nDstWidth, int nDstHeight,
    int *lookupTable, yuv_format yuvFormat, cudaStream_t stream) {
    if (yuvFormat == PACKED) {
    dim3 blocks(32, 16, 1);
    dim3 grids((nSrcWidth + blocks.x - 1) / blocks.x, (nDstHeight + blocks.y - 1) / blocks.y, 1);
        convertVToRGBKernel << <grids, blocks, 0, stream >> > (dpSrc, dpDst, nSrcWidth, nDstWidth, nDstHeight, lookupTable);
    }
    else if (yuvFormat == PLANAR) {
    dim3 blocks(32, 32, 1);
    dim3 grids((nSrcWidth + blocks.x - 1) / blocks.x, (((nDstHeight * 2) + blocks.y) - 1) / blocks.y, 1);
        convertPToRGBKernel << <grids, blocks, 0, stream >> > (dpSrc, dpDst, nSrcWidth, nDstWidth, nDstHeight, lookupTable);
    }
}

__global__ static void convertToNppiKernel(uint16_t *dSrc, uint8_t *dDst,
	int nSrcWidth, int nDstWidth, int nDstHeight, int *lookupTable) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tidd = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
	uint16_t tt[6];
	uint4 pF;
	int nDstH = nDstHeight;
	int nDstW = nSrcWidth / 8;

	if (tid < nDstW && tidd < nDstH) {
		int j = tidd * nSrcWidth;
		int k = tid * 8;
		pF.x = (uint32_t)dSrc[j + k + 0] + ((uint32_t)dSrc[j + k + 1] << 16);
		pF.y = (uint32_t)dSrc[j + k + 2] + ((uint32_t)dSrc[j + k + 3] << 16);
		pF.z = (uint32_t)dSrc[j + k + 4] + ((uint32_t)dSrc[j + k + 5] << 16);
		pF.w = (uint32_t)dSrc[j + k + 6] + ((uint32_t)dSrc[j + k + 7] << 16);

		v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
		y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10) * 1164;
		u0 = (uint32_t)(pF.x & 0x000003FF);
		y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20) * 1164;
		u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
		y1 = (uint32_t)(pF.y & 0x000003FF) * 1164;
		u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
		y3 = (uint32_t)((pF.z & 0x000FFC00) >> 10) * 1164;
		v1 = (uint32_t)(pF.z & 0x000003FF);
		y5 = (uint32_t)((pF.w & 0x3FF00000) >> 20) * 1164;
		v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
		y4 = (uint32_t)(pF.w & 0x000003FF) * 1164;

		k = tid * 18;
		j *= 9;
		j /= 4;
		int r = 1596 * v0 - 891648, g = 813 * v0 + 392 * u0 - 542464, b = 2017 * u0 - 1107200;
		tt[0] = (y0 + r) / 1000;
		tt[1] = (y0 - g) / 1000;
		tt[2] = (y0 + b) / 1000;

		tt[3] = (y1 + r) / 1000;
		tt[4] = (y1 - g) / 1000;
		tt[5] = (y1 + b) / 1000;

		dDst[j + k + 0] = lookupTable[tt[0]];
		dDst[j + k + 1] = lookupTable[tt[1]];
		dDst[j + k + 2] = lookupTable[tt[2]];

		dDst[j + k + 3] = lookupTable[tt[3]];
		dDst[j + k + 4] = lookupTable[tt[4]];
		dDst[j + k + 5] = lookupTable[tt[5]];

		r = 1596 * v1 - 891648, g = 813 * v1 + 392 * u1 - 542464, b = 2017 * u1 - 1107200;
		tt[0] = (y2 + r) / 1000;
		tt[1] = (y2 - g) / 1000;
		tt[2] = (y2 + b) / 1000;

		tt[3] = (y3 + r) / 1000;
		tt[4] = (y3 - g) / 1000;
		tt[5] = (y3 + b) / 1000;

		dDst[j + k + 6] = lookupTable[tt[0]];
		dDst[j + k + 7] = lookupTable[tt[1]];
		dDst[j + k + 8] = lookupTable[tt[2]];

		dDst[j + k + 9] = lookupTable[tt[3]];
		dDst[j + k + 10] = lookupTable[tt[4]];
		dDst[j + k + 11] = lookupTable[tt[5]];

		r = 1596 * v2 - 891648, g = 813 * v2 + 392 * u2 - 542464, b = 2017 * u2 - 1107200;
		tt[0] = (y4 + r) / 1000;
		tt[1] = (y4 - g) / 1000;
		tt[2] = (y4 + b) / 1000;

		tt[3] = (y5 + r) / 1000;
		tt[4] = (y5 - g) / 1000;
		tt[5] = (y5 + b) / 1000;

		dDst[j + k + 12] = lookupTable[tt[0]];
		dDst[j + k + 13] = lookupTable[tt[1]];
		dDst[j + k + 14] = lookupTable[tt[2]];

		dDst[j + k + 15] = lookupTable[tt[3]];
		dDst[j + k + 16] = lookupTable[tt[4]];
		dDst[j + k + 17] = lookupTable[tt[5]];
	}
}

void convertToRGBNpp(uint16_t *dSrc, uint8_t *dDst, int nSrcW, int nDstW, int nDstH,
	int *lookupTable, cudaStream_t stream) {
	dim3 blocks(32, 16, 1);
	dim3 grids((nSrcW + blocks.x - 1) / blocks.x, (nDstH + blocks.y - 1) / blocks.y, 1);
	convertToNppiKernel <<<grids, blocks, 0, stream >> > (dSrc, dDst, nSrcW, nDstW, nDstH, lookupTable);
}