#ifndef __H_CONVERTTOP210__
#define __H_CONVERTTOP210__

#include <iostream>
#include <helper_cuda.h>

using namespace std;

// v210 to p210
extern "C"
void convertToP210(uint16_t *dpSrc, uint16_t *dpDst, int nPitch, int nWidth, int nHeight, cudaStream_t stream = 0);

#endif // !__H_CONVERTTOP210__