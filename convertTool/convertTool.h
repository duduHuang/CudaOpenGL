#ifndef __H_CONVERTTOOL__
#define __H_CONVERTTOOL__
#ifndef WIN32
#include "nvjpeg.h"
#endif // !WIN32
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

#define TEST_LOOP 1
#define RGB_SIZE 3
#define YUV422_PLANAR_SIZE 2
#define DEFAULT_PINNED_GENERIC_MEMORY true

#ifndef WIN32
#include <sys/mman.h> // for mmap() / munmap()
#endif

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x, size) ( ((size_t)x + (size - 1)) & (~(size - 1)) )

typedef struct _nv210_context_t {
    int img_width;
    int img_height;
    int device;  // cuda device ID
    int img_rowByte; // the value will be suitable for Texture memroy.
    int batch;

    int dst_width;
    int dst_height;
    char *input_v210_file;
} nv210_context_t;

void multiStream(unsigned short *d_src, char *argv, nv210_context_t *g_ctx, int device_sync_method, bool bPinGenericMemory);

#ifndef WIN32
typedef struct _encode_params_t {
    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    nvjpegImage_t nv_image;
    nvjpegStatus_t err;

    unsigned short *t_16;
    unsigned char *t_8;
} encode_params_t;
#endif // !WIN32

class ConverterTool {
private:
    int argc, v210Size, nstreams;
    char **argv;
    unsigned short *v210Src, *v210SrcAligned, *dev_v210Src;
    unsigned char *dev_display8bit;
    nv210_context_t *g_ctx;
#ifndef WIN32
    encode_params_t *en_params;
#endif // !WIN32
    // allocate generic memory and pin it laster instead of using cudaHostAlloc()
    bool bPinGenericMemory; // we want this to be the default behavior
    int device_sync_method; // by default we use BlockingSync
    cudaError_t cudaStatus;
    cudaEvent_t start_event, stop_event;
    cudaStream_t *streams;
    int *lookupTable, *lookupTable_cuda;
public:
    ConverterTool();
    ConverterTool(int argcc, char **argvv);
    bool isGPUEnable();
    void initialCuda();
#ifndef WIN32
    void convertToP208ThenResize(unsigned short *src, int nSrcW, int nSrcH,
        unsigned char *p208Dst, int nDstW, int nDstH, int *nJPEGSize);
    void testFunction();
#endif // !WIN32
    int preprocess();
    void lookupTableF();
    void convertToRGBThenResize(unsigned char *rgb_8bit);
    void resizeThenConvertToRGB(unsigned char *rgb_8bit);
    void convertToP208ThenResize(unsigned char *p208);
	void convertToV210(unsigned short *v210);
	void callNppTest();
    void display();
    void freeMemory();
    void destroyCudaEvent();
    ~ConverterTool();
};

#endif // !__H_CONVERTTOOL__