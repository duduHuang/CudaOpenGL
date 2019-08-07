#include "convertTool.h"
#include "convertToP210.h"
#include "convertToRGB.h"
#include "resize.h"
#include "writeFile.h"

void runStdProgram(int argc, char **argv, unsigned char *disData, int nWidth, int nHeight, int nBytes);

int parseCmdLine(int argc, char *argv[], nv210_context_t *g_ctx) {
    int w = 7680, h = 4320;
    string s;
    //if (argc == 3) {
        // Run using default arguments
        g_ctx->input_v210_file = "v210.yuv";
        if (g_ctx->input_v210_file == NULL) {
            cout << "Cannot find input file" << argv[1] << "\n Exiting\n";
            return -1;
        }
        g_ctx->img_width = 7680;
		g_ctx->img_rowByte = (7680 + 47) / 48 * 128 / 2;
        g_ctx->img_height = 4320;
        g_ctx->batch = 1;
        cout << "Output resolution: (7680 4320)\n"
            << "                   (3840 2160)\n"
            << "                   (1920 1080)\n"
            << "                   default (1280 720) ";
        getline(cin, s);
        if (!s.empty()) {
            istringstream ss(s);
            ss >> w >> h;
        }
        g_ctx->dst_width = w;
        g_ctx->dst_height = h;
    //}
    if (g_ctx->img_width == 0 || g_ctx->img_height == 0 || !g_ctx->input_v210_file) {
        cout << "Usage: " << argv[0] << " inputf outputf\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/*
  Load v210 yuvfile data into host memory
*/
static int loadV210Frame(unsigned short *dSrc, nv210_context_t *g_ctx) {
    int frameSize = g_ctx->img_rowByte * g_ctx->img_height;
    ifstream v210File(g_ctx->input_v210_file, ifstream::in | ios::binary);
    if (!v210File.is_open()) {
        cerr << "Can't open files\n";
        return -1;
    }
    v210File.read((char *)dSrc, frameSize * sizeof(unsigned short));
    if (v210File.gcount() < frameSize * sizeof(unsigned short)) {
        cerr << "can't get one frame\n";
        return -1;
    }
    v210File.close();
    return EXIT_SUCCESS;
}

void lookupTableF(int *lookupTable) {
	for (int i = 0; i < 1024; ++i) {
		lookupTable[i] = round(i * 0.249);
	}
}

int checkGPU(int argc, char *argv[], int device_sync_method, bool bPinGenericMemory, nv210_context_t *g_ctx) {
	float scale_factor = 1.0f;
	int v210Size = g_ctx->img_rowByte * g_ctx->img_height;

	if ((device_sync_method = getCmdLineArgumentInt(argc, (const char **)argv, "sync_method")) >= 0) {
		if (device_sync_method == 0 || device_sync_method == 1 || device_sync_method == 2 || device_sync_method == 4) {
			cout << "Device synchronization method set to = " << sSyncMethod[device_sync_method] << "\n";
			//printf("Setting reps to 100 to demonstrate steady state\n");
			//nreps = 100;
		}
		else {
			cout << "Invalid command line option sync_method=\"" << device_sync_method << "\\\n";
			return EXIT_FAILURE;
		}
	}
	else {
		return EXIT_SUCCESS;
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "use_generic_memory")) {
#if defined(__APPLE__) || defined(MACOSX)
		bPinGenericMemory = false;  // Generic Pinning of System Paged memory not currently supported on Mac OSX
#else
		bPinGenericMemory = true;
#endif
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "use_cuda_malloc_host")) {
		bPinGenericMemory = false;
	}

	printf("\n> ");
	g_ctx->device = findCudaDevice(argc, (const char **)argv);

	// check the compute capability of the device
	int num_devices = 0;
	checkCudaErrors(cudaGetDeviceCount(&num_devices));

	if (0 == num_devices) {
		printf("your system does not have a CUDA capable device, waiving test...\n");
		return EXIT_WAIVED;
	}

	// check if the command-line chosen device ID is within range, exit if not
	if (g_ctx->device >= num_devices) {
		printf("cuda_device=%d is invalid, must choose device ID between 0 and %d\n", g_ctx->device, num_devices - 1);
		return EXIT_FAILURE;
	}

	checkCudaErrors(cudaSetDevice(g_ctx->device));

	// Checking for compute capabilities
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, g_ctx->device));

	// Check if GPU can map host memory (Generic Method), if not then we override bPinGenericMemory to be false
	if (bPinGenericMemory) {
		printf("Device: <%s> canMapHostMemory: %s\n", deviceProp.name, deviceProp.canMapHostMemory ? "Yes" : "No");

		if (deviceProp.canMapHostMemory == 0)
		{
			printf("Using cudaMallocHost, CUDA device does not support mapping of generic host memory\n");
			bPinGenericMemory = false;
		}
	}

	// Anything that is less than 32 Cores will have scaled down workload
	scale_factor = max((32.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)), 1.0f);
	v210Size = (int)rint((float)v210Size / scale_factor);

	printf("> CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);
	printf("> %d Multiprocessor(s) x %d (Cores/Multiprocessor) = %d (Cores)\n",
		deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

	printf("> scale_factor = %1.4f\n", 1.0f / scale_factor);
	printf("> array_size   = %d\n\n", v210Size);

	// enable use of blocking sync, to reduce CPU usage
	printf("> Using CPU/GPU Device Synchronization method (%s)\n", sDeviceSyncMethod[device_sync_method]);
}

inline void AllocateHostMemory(bool bPinGenericMemory, unsigned short **pp_a, unsigned short **ppAligned_a, int nbytes) {
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
    if (bPinGenericMemory) {
        // allocate a generic page-aligned chunk of system memory
#ifdef WIN32
        cout << "> VirtualAlloc() allocating " << (float)nbytes / 1048576.0f
            << " Mbytes of (generic page-aligned system memory)\n";
        *pp_a = (unsigned short *)VirtualAlloc(NULL, (nbytes + MEMORY_ALIGNMENT), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        cout << "> mmap() allocating " << (float)nbytes / 1048576.0f << " Mbytes (generic page-aligned system memory)\n";
        *pp_a = (unsigned short *)mmap(NULL, (nbytes + MEMORY_ALIGNMENT), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#endif
        *ppAligned_a = (unsigned short *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);
        cout << "> cudaHostRegister() registering " << (float)nbytes / 1048576.0f
            << " Mbytes of generic allocated system memory\n";
        // pin allocate memory
        checkCudaErrors(cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped));
    }
    else {
#endif
#endif
        cout << "> cudaMallocHost() allocating " << (float)nbytes / 1048576.0f << " Mbytes of system memory\n";
        // allocate host memory (pinned is required for achieve asynchronicity)
        checkCudaErrors(cudaMallocHost((void **)pp_a, nbytes));
        *ppAligned_a = *pp_a;
    }
}

inline void AllocateHostMemory(bool bPinGenericMemory, unsigned char **pp_a, unsigned char **ppAligned_a, int nbytes) {
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
	if (bPinGenericMemory) {
		// allocate a generic page-aligned chunk of system memory
#ifdef WIN32
		cout << "> VirtualAlloc() allocating " << (float)nbytes / 1048576.0f
			<< " Mbytes of (generic page-aligned system memory)\n";
		*pp_a = (unsigned char *)VirtualAlloc(NULL, (nbytes + MEMORY_ALIGNMENT), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
		cout << "> mmap() allocating " << (float)nbytes / 1048576.0f << " Mbytes (generic page-aligned system memory)\n";
		*pp_a = (unsigned short *)mmap(NULL, (nbytes + MEMORY_ALIGNMENT), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#endif
		*ppAligned_a = (unsigned char *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);
		cout << "> cudaHostRegister() registering " << (float)nbytes / 1048576.0f
			<< " Mbytes of generic allocated system memory\n";
		// pin allocate memory
		checkCudaErrors(cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped));
	}
	else {
#endif
#endif
		cout << "> cudaMallocHost() allocating " << (float)nbytes / 1048576.0f << " Mbytes of system memory\n";
		// allocate host memory (pinned is required for achieve asynchronicity)
		checkCudaErrors(cudaMallocHost((void **)pp_a, nbytes));
		*ppAligned_a = *pp_a;
	}
}

inline void FreeHostMemory(bool bPinGenericMemory, unsigned short **pp_a, unsigned short **ppAligned_a, int nbytes) {
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
    // CUDA 4.0 support pinning of generic host memory
    if (bPinGenericMemory)  {
        // unpin and delete host memory
        checkCudaErrors(cudaHostUnregister(*ppAligned_a));
#ifdef WIN32
        VirtualFree(*pp_a, 0, MEM_RELEASE);
#else
        munmap(*pp_a, nbytes);
#endif
    }
    else {
#endif
#endif
        cudaFreeHost(*pp_a);
    }
}

inline void FreeHostMemory(bool bPinGenericMemory, unsigned char **pp_a, unsigned char **ppAligned_a, int nbytes) {
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
	// CUDA 4.0 support pinning of generic host memory
	if (bPinGenericMemory) {
		// unpin and delete host memory
		checkCudaErrors(cudaHostUnregister(*ppAligned_a));
#ifdef WIN32
		VirtualFree(*pp_a, 0, MEM_RELEASE);
#else
		munmap(*pp_a, nbytes);
#endif
	}
	else {
#endif
#endif
		cudaFreeHost(*pp_a);
	}
}

/*
  Convert v210 to other type
*/
void multiStream(unsigned short *d_src, char *argv, nv210_context_t *g_ctx, int device_sync_method, bool bPinGenericMemory) {
    // create CUDA event handles
    // use blocking sync
    cudaEvent_t start_event, stop_event;
    unsigned short *dV210Device, *dP210, *dAligned_P210, *dP210Device;
    const int nStreams = 4;
    int v210Size = 0, byteChunkSize = 0, byteOffset = 0, p210Size = 0, p210ByteChunkSize = 0, p210ByteOffset = 0;
    float elapsed_time = 0.0f;
    cout << " \n";
    cout << "Computing results using GPU, using " << nStreams << " streams.\n";
    cout << " \n";
    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));
    cout << "    Creating " << nStreams << " CUDA streams.\n";
    for (int i = 0; i < nStreams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }
    int eventflags = ((device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync : cudaEventDefault);
    checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
    checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));
    v210Size = g_ctx->img_rowByte * g_ctx->img_height;
    p210Size = g_ctx->img_width * g_ctx->img_height * YUV422_PLANAR_SIZE;
    byteChunkSize = (v210Size * sizeof(unsigned short)) / nStreams;
    p210ByteChunkSize = (p210Size * sizeof(unsigned short)) / nStreams;

    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &dP210, &dAligned_P210, p210Size * sizeof(unsigned short));
    // allocate device memory
    checkCudaErrors(cudaMalloc((void **)&dV210Device, v210Size * sizeof(unsigned short))); // pointers to data in the device memory
    checkCudaErrors(cudaMalloc((void **)&dP210Device, p210Size * sizeof(unsigned short))); // pointers to data in the device memory
    checkCudaErrors(cudaEventRecord(start_event, 0));
    checkCudaErrors(cudaMemcpyAsync(dV210Device, d_src, v210Size * sizeof(unsigned short), cudaMemcpyHostToDevice, streams[0]));
    convertToP210(dV210Device, dP210Device, g_ctx->img_rowByte, g_ctx->img_width, g_ctx->img_height, streams[0]);
    checkCudaErrors(cudaMemcpyAsync(dP210, dP210Device, p210Size * sizeof(unsigned short)   , cudaMemcpyDeviceToHost, streams[0]));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    cout << fixed << "  CUDA v210 to p210(" << g_ctx->img_width << "x" << g_ctx->img_height << " --> "
        << g_ctx->img_width << "x" << g_ctx->img_height << "), "
        << "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << "ms"
        << " ==> " << (elapsed_time / (TEST_LOOP * 1.0f)) / g_ctx->batch << " ms/frame\n";
    checkCudaErrors(cudaEventRecord(start_event, 0));
    for (int i = 0; i < nStreams; i++) {
        cout << "        Launching stream " << i << ".\n";
        byteOffset = v210Size * i / nStreams;
        p210ByteOffset = p210Size * i / nStreams;
        checkCudaErrors(cudaMemcpyAsync(dV210Device + byteOffset, d_src + byteOffset,
            byteChunkSize, cudaMemcpyHostToDevice, streams[i]));
        convertToP210(dV210Device + byteOffset, dP210Device + p210ByteOffset, g_ctx->img_rowByte,
            g_ctx->img_width, g_ctx->img_height / nStreams, streams[i]);
        checkCudaErrors(cudaMemcpyAsync(dP210 + p210ByteOffset, dP210Device + p210ByteOffset,
            p210ByteChunkSize, cudaMemcpyDeviceToHost, streams[i]));
    }
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    cout << fixed << "  CUDA v210 to p210(" << g_ctx->img_width << "x" << g_ctx->img_height << " --> "
        << g_ctx->img_width << "x" << g_ctx->img_height << "), "
        << "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << "ms"
        << " ==> " << (elapsed_time / (TEST_LOOP * 1.0f)) / g_ctx->batch << " ms/frame\n";
    // release resources
    // Free cudaMallocHost or Generic Host allocated memory (from CUDA 4.0)
    FreeHostMemory(bPinGenericMemory, &dP210, &dAligned_P210, p210Size * sizeof(unsigned short));
    checkCudaErrors(cudaFree(dV210Device));
    checkCudaErrors(cudaFree(dP210Device));
    for (int i = 0; i < nStreams; i++)
    {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));
}

cudaError_t doConvert(int argc, char* argv[], unsigned short *v210Src, char *fileName, nv210_context_t *g_ctx,
	int device_sync_method, bool bPinGenericMemory) {
	cudaError_t cudaStatus;
	int nstreams = 1;
	// allocate and initialize an array of stream handles
	cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
	cudaEvent_t start_event, stop_event;
	int *lookupTable, *lookupTable_cuda;
	unsigned short *dev_v210Src;
	unsigned char *dev_rgbDst, *rgbDst, *rgbDstAligned, *dev_rgb8bit, *rgb8bit, *rgb8bitAligned;
	int v210Size = g_ctx->img_rowByte * g_ctx->img_height, rgbSize = g_ctx->img_width * g_ctx->img_height * RGB_SIZE,
		rgb8bitSize = g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE;
	float elapsed_time = 0.0f;
	
	int eventflags = ((device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync : cudaEventDefault);
	checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
	checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));
	for (int i = 0; i < nstreams; i++) {
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}

	// Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
	AllocateHostMemory(bPinGenericMemory, &rgbDst, &rgbDstAligned, rgbSize * sizeof(unsigned char));
	AllocateHostMemory(bPinGenericMemory, &rgb8bit, &rgb8bitAligned, rgb8bitSize * sizeof(unsigned char));
	lookupTable = (int *)malloc(sizeof(int) * 1024);
	checkCudaErrors(cudaMalloc((void**)&lookupTable_cuda, sizeof(int) * 1024));
	
	// Allocate GPU buffers.
	cudaStatus = cudaMalloc((void**)&dev_v210Src, v210Size * sizeof(unsigned short));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc failed!\n";
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_rgbDst, rgbSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc failed!\n";
		goto Error;
	}

	//rgbSize = g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE;
	cudaStatus = cudaMalloc((void**)&dev_rgb8bit, rgbSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc failed!\n";
		goto Error;
	}

	checkCudaErrors(cudaEventRecord(start_event, 0));
	checkCudaErrors(cudaMemcpyAsync(dev_v210Src, v210Src, v210Size * sizeof(unsigned short), cudaMemcpyHostToDevice, streams[0]));
	// Do convert
	lookupTableF(lookupTable);
	cudaMemcpy(lookupTable_cuda, lookupTable, sizeof(int) * 1024, cudaMemcpyHostToDevice);
	convertToRGB(dev_v210Src, dev_rgbDst, g_ctx->img_rowByte, g_ctx->dst_width, g_ctx->dst_height, 0, lookupTable_cuda, streams[0]);
	resizeBatch(dev_rgbDst, g_ctx->img_width, g_ctx->img_height, dev_rgb8bit, g_ctx->dst_width, g_ctx->dst_height, 0, streams[0]);
	checkCudaErrors(cudaMemcpyAsync(rgb8bit, dev_rgb8bit, rgb8bitSize * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[0]));
	checkCudaErrors(cudaEventRecord(stop_event, 0));
	checkCudaErrors(cudaEventSynchronize(stop_event));
	checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
	cout << fixed << "  CUDA v210 to other type(" << g_ctx->img_width << "x" << g_ctx->img_height << " --> "
		<< g_ctx->img_width << "x" << g_ctx->img_height << "), "
		<< "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << " ms/frame\n";
	bmp_w("r.bmp", g_ctx->dst_width, g_ctx->dst_height, rgb8bitSize, rgb8bit);
	dumpYUV(rgb8bit, g_ctx->dst_width, g_ctx->dst_height, "r.rgb", RGB_SIZE);
	runStdProgram(argc, argv, dev_rgb8bit, g_ctx->dst_width, g_ctx->dst_height, g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE);

Error:
	// release resources
	checkCudaErrors(cudaFree(lookupTable_cuda));
	free(lookupTable);
	cudaFree(dev_v210Src);
	cudaFree(dev_rgbDst);
	cudaFree(dev_rgb8bit);
	FreeHostMemory(bPinGenericMemory, &rgbDst, &rgbDstAligned, rgbSize * sizeof(unsigned char));
	FreeHostMemory(bPinGenericMemory, &rgb8bit, &rgb8bitAligned, rgb8bitSize * sizeof(unsigned char));
	for (int i = 0; i < nstreams; i++)
	{
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	}
	checkCudaErrors(cudaEventDestroy(start_event));
	checkCudaErrors(cudaEventDestroy(stop_event));

	return cudaStatus;
}

int convertTool(int argc, char* argv[]) {
    nv210_context_t *g_ctx;
	unsigned short *v210Src, *v210SrcAligned;
	int v210Size = 0;
	g_ctx = (nv210_context_t *)malloc(sizeof(nv210_context_t));
	v210Size = g_ctx->img_rowByte * g_ctx->img_height;
    // allocate generic memory and pin it laster instead of using cudaHostAlloc()
    bool bPinGenericMemory = DEFAULT_PINNED_GENERIC_MEMORY; // we want this to be the default behavior
    int  device_sync_method = cudaDeviceScheduleAuto; // by default we use BlockingSync
	
	//checkGPU(argc, argv, device_sync_method, bPinGenericMemory, g_ctx);
    checkCudaErrors(cudaSetDeviceFlags(device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0)));
    
    if (parseCmdLine(argc, argv, g_ctx) < 0) {
        return EXIT_FAILURE;
    }
    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &v210Src, &v210SrcAligned, v210Size * sizeof(unsigned short));
    // Load v210 yuv data into d_inputV210.
    if (loadV210Frame(v210Src, g_ctx)) {
        cerr << "failed to load data!\n";
        return EXIT_FAILURE;
    }
    cout << "Begin to convert....\n";
    //multiStream(v210Src, argv[2], g_ctx, device_sync_method, bPinGenericMemory);
	checkCudaErrors(doConvert(argc, argv, v210Src, argv[2], g_ctx, device_sync_method, bPinGenericMemory));
    FreeHostMemory(bPinGenericMemory, &v210Src, &v210SrcAligned, v210Size * sizeof(unsigned short));
    return EXIT_SUCCESS;
}