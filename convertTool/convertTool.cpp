#include "convertTool.h"
#include "convertToP208.h"
#include "convertToRGB.h"
#include "rgbToV210.h"
#include "resize.h"
#include "writeFile.h"
#include "nppTest.h"
#ifdef WIN32
#include "nvGL.h"
#endif // !WIN32

static const char *sSyncMethod[] = {
    "0 (Automatic Blocking)",
    "1 (Spin Blocking)",
    "2 (Yield Blocking)",
    "3 (Undefined Blocking Method)",
    "4 (Blocking Sync Event) = low CPU utilization",
    NULL
};

const char *sDeviceSyncMethod[] = {
    "cudaDeviceScheduleAuto",
    "cudaDeviceScheduleSpin",
    "cudaDeviceScheduleYield",
    "INVALID",
    "cudaDeviceScheduleBlockingSync",
    NULL
};

int parseCmdLine(nv210_context_t *g_ctx) {
    int w = 7680, h = 4320;
    string s;
    // Run using default arguments
    g_ctx->input_v210_file = "v210.yuv";
    if (g_ctx->input_v210_file == NULL) {
        cout << "Cannot find input file\n Exiting\n";
        return -1;
    }
    g_ctx->img_width = 7680;
    g_ctx->img_rowByte = (7680 + 47) / 48 * 128 / 2;
    g_ctx->img_height = 4320;
    g_ctx->batch = 1;
#ifdef WIN32
    cout << "Output resolution: (7680 4320)\n"
        << "                   (3840 2160)\n"
        << "                   (1920 1080)\n"
        << "                   default (1280 720) ";
    getline(cin, s);
    if (!s.empty()) {
        istringstream ss(s);
        ss >> w >> h;
    }
#endif // WIN32
    g_ctx->dst_width = w;
    g_ctx->dst_height = h;
    if (g_ctx->img_width == 0 || g_ctx->img_height == 0 || !g_ctx->input_v210_file) {
        cout << "Error: inputf outputf\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

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

void ConverterTool::lookupTableF() {
    lookupTable = new int[1024];

    checkCudaErrors(cudaMalloc((void**)&lookupTable_cuda, sizeof(int) * 1024));

    for (int i = 0; i < 1024; ++i) {
        lookupTable[i] = round(i * 0.249);
    }
    checkCudaErrors(cudaMemcpy(lookupTable_cuda, lookupTable, sizeof(int) * 1024, cudaMemcpyHostToDevice));
}

bool ConverterTool::isGPUEnable() {
    float scale_factor = 1.0f;
    v210Size = ((7680 + 47) / 48 * 128 / 2) * 4320;

    if ((device_sync_method = getCmdLineArgumentInt(argc, (const char **)argv, "sync_method")) >= 0) {
        if (device_sync_method == 0 || device_sync_method == 1 || device_sync_method == 2 || device_sync_method == 4) {
            cout << "Device synchronization method set to = " << sSyncMethod[device_sync_method] << "\n";
            //printf("Setting reps to 100 to demonstrate steady state\n");
            //nreps = 100;
        }
        else {
            cout << "Invalid command line option sync_method=\"" << device_sync_method << "\\\n";
            return false;
        }
    }
    else {
        return false;
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

    // check the compute capability of the device
    int num_devices = 0;
    checkCudaErrors(cudaGetDeviceCount(&num_devices));

    if (0 == num_devices) {
        printf("your system does not have a CUDA capable device, waiving test...\n");
        return false;
    }

    // check if the command-line chosen device ID is within range, exit if not
	g_ctx->device = findCudaDevice(argc, (const char **)argv);
    if (g_ctx->device >= num_devices) {
        printf("cuda_device=%d is invalid, must choose device ID between 0 and %d\n", g_ctx->device, num_devices - 1);
        return false;
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
    return true;
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
        *pp_a = (unsigned char *)mmap(NULL, (nbytes + MEMORY_ALIGNMENT), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
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

//  Convert v210 to other type
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
    convertToP208(dV210Device, dP210Device, g_ctx->img_rowByte, g_ctx->img_width, g_ctx->img_height, streams[0]);
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
        convertToP208(dV210Device + byteOffset, dP210Device + p210ByteOffset, g_ctx->img_rowByte,
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

ConverterTool::ConverterTool() {
    argc = 0;
    argv = 0;
    v210Size = 0;
    nstreams = 1;
    bPinGenericMemory = DEFAULT_PINNED_GENERIC_MEMORY; // we want this to be the default behavior
    device_sync_method = cudaDeviceScheduleAuto; // by default we use BlockingSync
    g_ctx = new nv210_context_t();
#ifndef WIN32
    en_params = new encode_params_t();
#endif // !WIN32
}

ConverterTool::ConverterTool(int argcc, char **argvv) {
    argc = argcc;
    argv = argvv;
    v210Size = 0;
    nstreams = 1;
    bPinGenericMemory = DEFAULT_PINNED_GENERIC_MEMORY; // we want this to be the default behavior
    device_sync_method = cudaDeviceScheduleAuto; // by default we use BlockingSync
    g_ctx = new nv210_context_t();
#ifndef WIN32
    en_params = new encode_params_t();
#endif // !WIN32
}

ConverterTool::~ConverterTool() {
    freeMemory();
    destroyCudaEvent();
}

void ConverterTool::initialCuda() {
    checkCudaErrors(cudaSetDeviceFlags(device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0)));
    // allocate and initialize an array of stream handles
    int eventflags = ((device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync : cudaEventDefault);
    checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
    checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));
    //streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    streams = new cudaStream_t[nstreams];
    for (int i = 0; i < nstreams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }
#ifndef WIN32
    // initialize nvjpeg structures
    checkCudaErrors(nvjpegCreateSimple(&en_params->nv_handle));
    checkCudaErrors(nvjpegEncoderStateCreate(en_params->nv_handle, &en_params->nv_enc_state, streams[0]));
    checkCudaErrors(nvjpegEncoderParamsCreate(en_params->nv_handle, &en_params->nv_enc_params, streams[0]));
#endif // !WIN32
}

#ifndef WIN32
void ConverterTool::convertToP208ThenResize(unsigned short *src, int nSrcW, int nSrcH,
    unsigned char *p208Dst, int nDstW, int nDstH, int *nJPEGSize) {
    int rowByte = (nSrcW + 47) / 48 * 128 / 2;
    size_t length = 0;

    checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(en_params->nv_enc_params, NVJPEG_CSS_422, streams[0]));

    checkCudaErrors(cudaMalloc((void **)&en_params->t_16, rowByte * nSrcH * sizeof(unsigned short)));

    en_params->nv_image.pitch[0] = nDstW * sizeof(unsigned char);
    en_params->nv_image.pitch[1] = nDstW / 2 * sizeof(unsigned char);
    en_params->nv_image.pitch[2] = nDstW / 2 * sizeof(unsigned char);
    checkCudaErrors(cudaMalloc(&en_params->nv_image.channel[0], nDstW * nDstH * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&en_params->nv_image.channel[1], nDstW * nDstH / 2 * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&en_params->nv_image.channel[2], nDstW * nDstH / 2 * sizeof(unsigned char)));

    checkCudaErrors(
        cudaMemcpy((void *)en_params->t_16, (void *)src, rowByte * nSrcH * sizeof(unsigned short), cudaMemcpyHostToDevice));
    resizeBatch(en_params->t_16, rowByte, nSrcH,
        en_params->nv_image.channel[0], en_params->nv_image.channel[1], en_params->nv_image.channel[2],
        nDstW, nDstH, lookupTable_cuda, streams[0]);

    checkCudaErrors(nvjpegEncodeYUV(en_params->nv_handle, en_params->nv_enc_state, en_params->nv_enc_params,
        &en_params->nv_image, NVJPEG_CSS_422, nDstW, nDstH, streams[0]));

    checkCudaErrors(cudaStreamSynchronize(streams[0]));
    // get compressed stream size
    checkCudaErrors(
        nvjpegEncodeRetrieveBitstream(en_params->nv_handle, en_params->nv_enc_state, NULL, &length, streams[0]));

    // get stream itself
    checkCudaErrors(cudaStreamSynchronize(streams[0]));
    vector<unsigned char> jpeg(length);
    checkCudaErrors(
        nvjpegEncodeRetrieveBitstream(en_params->nv_handle, en_params->nv_enc_state, jpeg.data(), &length, streams[0]));

    // write stream to file
    checkCudaErrors(cudaStreamSynchronize(streams[0]));
    memcpy(p208Dst, jpeg.data(), length);
    *nJPEGSize = length;
}

void ConverterTool::testFunction() {
    unsigned char *p208;
    p208 = new unsigned char[1280 * 720 * 2];
    int nJPEGSize = 0;
    convertToP208ThenResize(v210Src, 7680, 4320, p208, 1280, 720, &nJPEGSize);
    ofstream output_file("r.jpg", ios::out | ios::binary);
    output_file.write((char *)p208, nJPEGSize);
    output_file.close();
    delete[] p208;
}
#endif // !WIN32

int ConverterTool::preprocess() {
    if (parseCmdLine(g_ctx) < 0) {
        return EXIT_FAILURE;
    }

    v210Size = g_ctx->img_rowByte * g_ctx->img_height;
    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &v210Src, &v210SrcAligned, v210Size * sizeof(unsigned short));

    cudaStatus = cudaMalloc((void**)&dev_v210Src, v210Size * sizeof(unsigned short));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_v210Src cudaMalloc failed!\n";
        freeMemory();
        return EXIT_FAILURE;
    }

    // Load v210 yuv data into v210Src.
    cout << "Load image " << g_ctx->input_v210_file << " " << g_ctx->img_width << "x" << g_ctx->img_height << "\n";
    if (loadV210Frame(v210Src, g_ctx)) {
        cerr << "failed to load data!\n";
        return EXIT_FAILURE;
    }

    checkCudaErrors(cudaEventRecord(start_event, 0));
    checkCudaErrors(
        cudaMemcpyAsync(dev_v210Src, v210Src, v210Size * sizeof(unsigned short), cudaMemcpyHostToDevice, streams[0]));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
}

void ConverterTool::convertToRGBThenResize(unsigned char *rgb_8bit) {
    unsigned char *dev_rgbDst, *rgbDst, *rgbDstAligned, *dev_rgb8bit, *rgb8bit, *rgb8bitAligned;
    int rgbSize = g_ctx->img_width * g_ctx->img_height * RGB_SIZE,
        rgb8bitSize = g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE;
    float elapsed_time = 0.0f;

    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &rgbDst, &rgbDstAligned, rgbSize * sizeof(unsigned char));
    AllocateHostMemory(bPinGenericMemory, &rgb8bit, &rgb8bitAligned, rgb8bitSize * sizeof(unsigned char));

    cudaStatus = cudaMalloc((void**)&dev_rgbDst, rgbSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_rgbDst cudaMalloc failed!\n";
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rgb8bit, rgb8bitSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_rgb8bit cudaMalloc failed!\n";
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_display8bit, rgb8bitSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_rgb8bit cudaMalloc failed!\n";
        goto Error;
    }

    checkCudaErrors(cudaEventRecord(start_event, 0));
    convertToRGB(dev_v210Src, dev_rgbDst, g_ctx->img_rowByte, g_ctx->img_width, g_ctx->img_height,
        lookupTable_cuda, PACKED, streams[0]);
    resizeBatch(dev_rgbDst, g_ctx->img_width, g_ctx->img_height,
        dev_rgb8bit, g_ctx->dst_width, g_ctx->dst_height, streams[0]);
    checkCudaErrors(
        cudaMemcpyAsync(rgb8bit, dev_rgb8bit, rgb8bitSize * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[0]));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    cout << fixed << "  CUDA v210 to rgb 8 bit type(" << g_ctx->img_width << "x" << g_ctx->img_height << " --> "
        << g_ctx->img_width << "x" << g_ctx->img_height << "), "
        << "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << " ms/frame\n";
    rgb_8bit = rgb8bit;
    checkCudaErrors(cudaMemcpyAsync(dev_display8bit, dev_rgb8bit, rgb8bitSize * sizeof(unsigned char),
        cudaMemcpyDeviceToDevice, streams[0]));
	dumpRGB(rgb8bit, g_ctx->img_width, g_ctx->img_height, "r.rgb");
Error:
    cudaFree(dev_rgbDst);
    cudaFree(dev_rgb8bit);
    FreeHostMemory(bPinGenericMemory, &rgbDst, &rgbDstAligned, rgbSize * sizeof(unsigned char));
    FreeHostMemory(bPinGenericMemory, &rgb8bit, &rgb8bitAligned, rgb8bitSize * sizeof(unsigned char));
}

void ConverterTool::resizeThenConvertToRGB(unsigned char *rgb_8bit) {
    unsigned short *dev_p210Dst, *p210Dst, *p210DstAligned;
    unsigned char *dev_rgb8bit, *rgb8bit, *rgb8bitAligned;
    int p210Size = g_ctx->dst_width * g_ctx->dst_height * YUV422_PLANAR_SIZE,
        rgb8bitSize = g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE;
    float elapsed_time = 0.0f;

    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &p210Dst, &p210DstAligned, p210Size * sizeof(unsigned short));
    AllocateHostMemory(bPinGenericMemory, &rgb8bit, &rgb8bitAligned, rgb8bitSize * sizeof(unsigned char));

    cudaStatus = cudaMalloc((void**)&dev_p210Dst, p210Size * sizeof(unsigned short));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_rgbDst cudaMalloc failed!\n";
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rgb8bit, rgb8bitSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_rgb8bit cudaMalloc failed!\n";
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_display8bit, rgb8bitSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_display8bit cudaMalloc failed!\n";
        goto Error;
    }

    checkCudaErrors(cudaEventRecord(start_event, 0));
    resizeBatch(dev_v210Src, g_ctx->img_rowByte, g_ctx->img_height,
        dev_p210Dst, g_ctx->dst_width, g_ctx->dst_height, streams[0]);
    convertToRGB(dev_p210Dst, dev_rgb8bit, g_ctx->dst_width, g_ctx->dst_width, g_ctx->dst_height,
        lookupTable_cuda, PLANAR, streams[0]);

    checkCudaErrors(
        cudaMemcpyAsync(rgb8bit, dev_rgb8bit, rgb8bitSize * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[0]));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    cout << fixed << "  CUDA v210 resize and convert to rgb 8 bit type(" << g_ctx->img_width << "x" << g_ctx->img_height
        << " --> " << g_ctx->img_width << "x" << g_ctx->img_height << "), "
        << "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << " ms/frame\n";
    rgb_8bit = rgb8bit;
    bmp_w("r.jpg", g_ctx->dst_width, g_ctx->dst_height, g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE, rgb_8bit);
    checkCudaErrors(cudaMemcpyAsync(dev_display8bit, dev_rgb8bit, rgb8bitSize * sizeof(unsigned char),
        cudaMemcpyDeviceToDevice, streams[0]));
Error:
    cudaFree(dev_p210Dst);
    cudaFree(dev_rgb8bit);
    FreeHostMemory(bPinGenericMemory, &p210Dst, &p210DstAligned, p210Size * sizeof(unsigned short));
    FreeHostMemory(bPinGenericMemory, &rgb8bit, &rgb8bitAligned, rgb8bitSize * sizeof(unsigned char));
}

void ConverterTool::convertToP208ThenResize(unsigned char *o_p208) {
    unsigned char *dev_p208, *p208, *p208Aligned;
    int p208Size = g_ctx->dst_width * g_ctx->dst_height * YUV422_PLANAR_SIZE;
    float elapsed_time = 0.0f;

    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &p208, &p208Aligned, p208Size * sizeof(unsigned char));

    cudaStatus = cudaMalloc((void**)&dev_p208, p208Size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_p208 cudaMalloc failed!\n";
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_display8bit, p208Size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_display8bit cudaMalloc failed!\n";
        goto Error;
    }
    checkCudaErrors(cudaEventRecord(start_event, 0));
    resizeBatch(dev_v210Src, g_ctx->img_rowByte, g_ctx->img_height,
        dev_p208, g_ctx->dst_width, g_ctx->dst_height, lookupTable_cuda, streams[0]);
    checkCudaErrors(
        cudaMemcpyAsync(p208, dev_p208, p208Size * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[0]));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    cout << fixed << "  CUDA v210 resize and convert to rgb 8 bit type(" << g_ctx->img_width << "x" << g_ctx->img_height
        << " --> " << g_ctx->img_width << "x" << g_ctx->img_height << "), "
        << "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << " ms/frame\n";
    o_p208 = p208;
Error:
    cudaFree(dev_p208);
    FreeHostMemory(bPinGenericMemory, &p208, &p208Aligned, p208Size * sizeof(unsigned char));
}

void ConverterTool::convertToV210(unsigned short *vv210) {
	unsigned short *dev_rgbDst,
		*dev_v210, *v210, *v210Aligned;
	int rgbSize = g_ctx->img_width * g_ctx->img_height * RGB_SIZE;
	float elapsed_time = 0.0f;

	// Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
	AllocateHostMemory(bPinGenericMemory, &v210, &v210Aligned, v210Size * sizeof(unsigned short));

	cudaStatus = cudaMalloc((void**)&dev_rgbDst, rgbSize * sizeof(unsigned short));
	if (cudaStatus != cudaSuccess) {
		cerr << "dev_rgbDst cudaMalloc failed!\n";
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_v210, v210Size * sizeof(unsigned short));
	if (cudaStatus != cudaSuccess) {
		cerr << "dev_rgb8bit cudaMalloc failed!\n";
		goto Error;
	}

	checkCudaErrors(cudaEventRecord(start_event, 0));
	convertToRGB(dev_v210Src, dev_rgbDst, g_ctx->img_rowByte, g_ctx->img_width, g_ctx->img_height,
		streams[0]);
	rgbToV210(dev_rgbDst, dev_v210, g_ctx->img_width * 3, g_ctx->img_height, streams[0]);
	checkCudaErrors(
		cudaMemcpyAsync(v210, dev_v210, v210Size * sizeof(unsigned short), cudaMemcpyDeviceToHost, streams[0]));
	checkCudaErrors(cudaEventRecord(stop_event, 0));
	checkCudaErrors(cudaEventSynchronize(stop_event));
	checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
	cout << fixed << "  CUDA v210 to rgb and to V210 type(" << g_ctx->img_width << "x" << g_ctx->img_height << " --> "
		<< g_ctx->img_width << "x" << g_ctx->img_height << "), "
		<< "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << " ms/frame\n";
	vv210 = v210;
	dumpYUV(v210, g_ctx->img_rowByte, g_ctx->img_height, "vv.yuv");

Error:
	cudaFree(dev_rgbDst);
	cudaFree(dev_v210);
	FreeHostMemory(bPinGenericMemory, &v210, &v210Aligned, v210Size * sizeof(unsigned short));
}

void ConverterTool::display() {
#ifdef  WIN32
    runStdProgram(argc, argv, dev_display8bit, g_ctx->dst_width, g_ctx->dst_height,
        g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE);
#endif // ! WIN32
}

void ConverterTool::freeMemory() {
    cout << "Free memory...\n";
    checkCudaErrors(cudaFree(lookupTable_cuda));
    delete[] lookupTable;
    checkCudaErrors(cudaFree(dev_v210Src));
    checkCudaErrors(cudaFree(dev_display8bit));
    FreeHostMemory(bPinGenericMemory, &v210Src, &v210SrcAligned, v210Size * sizeof(unsigned short));
    delete g_ctx;

#ifndef WIN32
    checkCudaErrors(cudaFree(en_params->t_16));
    checkCudaErrors(cudaFree(en_params->nv_image.channel[0]));
    checkCudaErrors(cudaFree(en_params->nv_image.channel[1]));
    checkCudaErrors(cudaFree(en_params->nv_image.channel[2]));
#endif // !WIN32
}

void ConverterTool::destroyCudaEvent() {
    for (int i = 0; i < nstreams; i++) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));

#ifndef WIN32
    checkCudaErrors(nvjpegEncoderStateDestroy(en_params->nv_enc_state));
    checkCudaErrors(nvjpegEncoderParamsDestroy(en_params->nv_enc_params));
    checkCudaErrors(nvjpegDestroy(en_params->nv_handle));
    delete en_params;
#endif // !WIN32
}

void ConverterTool::callNppTest() {
    nppTest *nT = new nppTest();
    nT->nppTestSetType(RGB_SIZE);
    nT->nppTestSetSrcSize(g_ctx->img_width, g_ctx->img_height);
    nT->nppTestSetDstSize(g_ctx->dst_width, g_ctx->dst_height);
    nT->nppTestInitial();

    unsigned char *dev_rgbDst, *dev_rgb8bit, *rgb8bit, *rgb8bitAligned;
    int rgbSize = g_ctx->img_width * g_ctx->img_height * RGB_SIZE,
        rgb8bitSize = g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE;
    float elapsed_time = 0.0f;

    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &rgb8bit, &rgb8bitAligned, rgb8bitSize * sizeof(unsigned char));

    cudaStatus = cudaMalloc((void**)&dev_rgbDst, rgbSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_rgbDst cudaMalloc failed!\n";
        cerr << "CUDA error at " << cudaGetErrorName(cudaStatus) << "\n";
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rgb8bit, rgb8bitSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_rgb8bit cudaMalloc failed!\n";
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_display8bit, rgb8bitSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        cerr << "dev_display8bit cudaMalloc failed!\n";
        goto Error;
    }

    checkCudaErrors(cudaEventRecord(start_event, 0));
    convertToRGBNpp(dev_v210Src, dev_rgbDst, g_ctx->img_rowByte, g_ctx->img_width, g_ctx->img_height,
        lookupTable_cuda, streams[0]);
    nT->nppTestSetSrcData(dev_rgbDst);
    nT->nppTestProcess(dev_rgb8bit);
    checkCudaErrors(
        cudaMemcpyAsync(rgb8bit, dev_rgb8bit, rgb8bitSize * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[0]));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
	cout << fixed << "  CUDA NPP test v210 convert to rgb 8 bit type(" << g_ctx->img_width << "x" << g_ctx->img_height
		<< " --> " << g_ctx->img_width << "x" << g_ctx->img_height << "), "
		<< "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << " ms/frame\n";

    bmp_w("r.jpg", g_ctx->dst_width, g_ctx->dst_height, g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE, rgb8bit);
    checkCudaErrors(cudaMemcpyAsync(dev_display8bit, dev_rgb8bit, rgb8bitSize * sizeof(unsigned char),
        cudaMemcpyDeviceToDevice, streams[0]));

Error:
    cudaFree(dev_rgbDst);
    cudaFree(dev_rgb8bit);
    FreeHostMemory(bPinGenericMemory, &rgb8bit, &rgb8bitAligned, rgb8bitSize * sizeof(unsigned char));
    nT->nppTestDestroy();
    delete nT;
}