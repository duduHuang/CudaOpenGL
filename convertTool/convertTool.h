#ifndef __H_CONVERTTOOL__
#define __H_CONVERTTOOL__
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
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

static const char *sSyncMethod[] =
{
	"0 (Automatic Blocking)",
	"1 (Spin Blocking)",
	"2 (Yield Blocking)",
	"3 (Undefined Blocking Method)",
	"4 (Blocking Sync Event) = low CPU utilization",
	NULL
};

const char *sDeviceSyncMethod[] =
{
	"cudaDeviceScheduleAuto",
	"cudaDeviceScheduleSpin",
	"cudaDeviceScheduleYield",
	"INVALID",
	"cudaDeviceScheduleBlockingSync",
	NULL
};

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

/*typedef struct _encode_params_t {
	nvjpegHandle_t nv_handle;
	nvjpegEncoderState_t nv_enc_state;
	nvjpegEncoderParams_t nv_enc_params;
	nvjpegImage_t nv_image;
	nvjpegStatus_t err;

	unsigned short *t_16;
	unsigned char *t_8;
} encode_params_t;*/

int convertTool(int argc, char* argv[]);

#endif // !__H_CONVERTTOOL__