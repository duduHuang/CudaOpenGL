#ifndef __H_RGBTOV210__
#define __H_RGBTOV210__

#include <iostream>
#include <helper_cuda.h>

using namespace std;

extern "C"
void rgbToV210(uint16_t *dpSrc, uint16_t *dpDst, int nPitch, int nHeight, cudaStream_t stream = 0);

#endif // __H_RGBTOV210__
