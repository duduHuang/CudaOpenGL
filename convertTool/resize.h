#ifndef __H_RESIZE__
#define __H_RESIZE__

#include <iostream>
#include <helper_cuda.h>

using namespace std;

extern "C"
void resizeBatch(uint8_t *dpSrc, int nSrcPitch, int nSrcHeight, uint8_t *dpDst, int nDstWidth, int nDstHeight,
    cudaStream_t stram = 0);

void resizeBatch(uint16_t *dpSrc, int nSrcPitch, int nSrcHeight, uint16_t *dpDst, int nDstWidth, int nDstHeight,
    cudaStream_t stram = 0);

void resizeBatch(uint16_t *dpSrc, int nSrcPitch, int nSrcHeight, uint8_t *dpDst, int nDstWidth, int nDstHeight,
    int *lookupTable_cuda, cudaStream_t stram = 0);

void resizeBatch(uint16_t *dpSrc, int nSrcPitch, int nSrcHeight, uint8_t *dpDst0, uint8_t *dpDst1, uint8_t *dpDst2, int nDstWidth, int nDstHeight,
    int *lookupTable_cuda, cudaStream_t stram = 0);

#endif // !__H_RESIZE__