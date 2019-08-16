#include "nppTest.h"

nppTest::nppTest() {
	nppT = new nppStruct();
}

void nppTest::nppTestInitial() {
	nppT->eInterpolation = NPPI_INTER_SUPER;
	// Allocate Memory
	cudaStatus = cudaMalloc(&nppT->pSrcData16b, nppT->srcSize.width * nppT->srcSize.height * type * sizeof(Npp16u));
	cudaStatus = cudaMalloc(&nppT->pSrcData8b, nppT->srcSize.width * nppT->srcSize.height * type);
	nppT->nSrcStep = nppT->srcSize.width * type;
	cudaStatus = cudaMalloc(&nppT->pDstData8b, nppT->dstSize.width * nppT->dstSize.height * type);
	nppT->nDstStep = nppT->dstSize.width * type;
}

void nppTest::nppTestSetType(int n) {
	type = n;
}

void nppTest::nppTestSetSrcSize(int nWidth, int nHeight) {
	nppT->srcSize.width = nWidth;
	nppT->srcSize.height = nHeight;
	nppT->srcRect = { 0, 0, nppT->srcSize.width, nppT->srcSize.height };
}

void nppTest::nppTestSetDstSize(int nWidth, int nHeight) {
	nppT->dstSize.width = nWidth;
	nppT->dstSize.height = nHeight;
	nppT->dstRect = { 0, 0, nppT->dstSize.width, nppT->dstSize.height };
}

void nppTest::nppTestSetSrcData(unsigned char *pSrcData) {
	checkCudaErrors(cudaMemcpy(nppT->pSrcData8b, pSrcData,
		nppT->srcSize.width * nppT->srcSize.height * type * sizeof(Npp8u), cudaMemcpyDeviceToDevice));
}

void nppTest::nppTestProcess(unsigned char *oDstData) {
	//nppT->status = nppiConvert_16u8u_C3R(nppT->pSrcData16b, nppT->nSrcStep * sizeof(Npp16u),
	//	nppT->pSrcData8b, nppT->nSrcStep, nppT->srcSize);
	nppT->status = nppiResize_8u_C3R(nppT->pSrcData8b, nppT->nSrcStep, nppT->srcSize, nppT->srcRect,
		nppT->pDstData8b, nppT->nDstStep, nppT->dstSize, nppT->dstRect, nppT->eInterpolation);
	checkCudaErrors(
		cudaMemcpy(oDstData, nppT->pDstData8b, nppT->dstSize.width * nppT->dstSize.height * type, cudaMemcpyDeviceToDevice));
}

void nppTest::nppTestDestroy() {
	cudaStatus = cudaFree(nppT->pSrcData16b);
	cudaStatus = cudaFree(nppT->pSrcData8b);
	cudaStatus = cudaFree(nppT->pDstData8b);
}

nppTest::~nppTest() {
	delete nppT;
}