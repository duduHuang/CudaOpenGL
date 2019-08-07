#ifndef __H_MAPTOGL__
#define __H_MAPTOGL__

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;

// map to openGL
extern "C"
void mapToGL(unsigned char *dSrc, unsigned char *dDst, int nWidth, int nHeight);

#endif // !__H_MAPTOGL__