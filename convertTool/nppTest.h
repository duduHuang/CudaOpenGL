#ifndef __H_NPPTEST__
#define __H_NPPTEST__

#include <npp.h>
#include <cuda_runtime.h>
//#include <Exceptions.h>
#include <iostream>

#include <helper_string.h>
#include <helper_cuda.h>

using namespace std;

typedef struct _nppStruct_
{
	Npp16u *pSrcData16b;
	Npp8u *pSrcData8b;
	Npp8u *pDstData8b;
	NppiSize srcSize;
	NppiSize dstSize;
	NppiRect srcRect;
	NppiRect dstRect;
	size_t nSrcPitch;
	size_t nDstPitch;
	Npp32s nSrcStep;
	Npp32s nDstStep;
	Npp32s eInterpolation;
	NppStatus status;
} nppStruct;

class nppTest {
private:
	nppStruct *nppT;
	int type;
	cudaError cudaStatus;
public:
	nppTest();
	~nppTest();
	void nppTestInitial();
	void nppTestDestroy();
	void nppTestSetType(int type);
	void nppTestSetSrcSize(int w, int h);
	void nppTestSetDstSize(int w, int h);
	void nppTestSetSrcData(unsigned char *pSrcData);
	void nppTestProcess(unsigned char *oDstData);
};

#endif // !__H_NPPTEST__
