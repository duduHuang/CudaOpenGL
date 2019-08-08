#ifndef __H_CONVERTTOP208__
#define __H_CONVERTTOP208__

#include <iostream>
#include <helper_cuda.h>

using namespace std;

// v210 to p208
extern "C"
void convertToP208(uint16_t *dpSrc, uint16_t *dpDst, int nPitch, int nWidth, int nHeight, cudaStream_t stream = 0);

#endif // !__H_CONVERTTOP208__