#ifndef __H_CONVERTTORGB__
#define __H_CONVERTTORGB__

#include <iostream>
#include <helper_cuda.h>

using namespace std;
typedef enum {
	PACKED = 0,
	PLANAR = 1
} yuv_format;

// v210 to rgb
extern "C"
void convertToRGB(uint16_t *dpSrc, uint16_t *dpDst, int nSrcWidth, int nDstWidth, int nDstHeight,
	cudaStream_t stream = 0);

void convertToRGB(uint16_t *dpSrc, uint8_t *dpDst, int nSrcWidth, int nDstWidth, int nDstHeight,
	int nBatch, int *lookupTable, cudaStream_t stream = 0);

void convertToRGB(uint16_t *dpSrc, uint8_t *dpDst, int nSrcWidth, int nDstWidth, int nDstHeight,
	int nBatch, int *lookupTable, yuv_format yuvFormat, cudaStream_t stream = 0);

#endif // !__H_CONVERTTORGB__