#include "mapToGL.h"

__global__ static void mapToGLKernel(uint8_t *dSrc, uint8_t *dDst, int nWidth, int nHeight) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tidd = blockIdx.y * blockDim.y + threadIdx.y;
	if (tid < nWidth && tidd < nHeight) {
		int j = tidd * nWidth * 3;
		int k = tid * 3;
		dDst[j + k + 0] = dSrc[j + k + 0];
		dDst[j + k + 1] = dSrc[j + k + 1];
		dDst[j + k + 2] = dSrc[j + k + 2];
	}
}

void mapToGL(unsigned char *dSrc, unsigned char *dDst, int nWidth, int nHeight) {
	dim3 blocks(32, 16, 1);
	dim3 grids((nWidth + blocks.x - 1) / blocks.x, (((nHeight * 3) + blocks.y) - 1) / blocks.y, 1);
	mapToGLKernel << <grids, blocks >> > (dSrc, dDst, nWidth, nHeight);
}