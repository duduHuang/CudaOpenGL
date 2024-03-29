#include "nvGL.h"
#include "mapToGL.h"

const char *sSDKname = "RGB 8 bit";

// constants / global variables
unsigned int window_width = 7680;
unsigned int window_height = 4320;
int image_width = 7680;
int image_height = 4320;
int iGLUTWindowHandle = 0;          // handle to the GLUT window
unsigned char *imageSrc;

// pbo and fbo variables
#ifdef USE_TEXSUBIMAGE2D
GLuint pbo_dest;
struct cudaGraphicsResource *cuda_pbo_dest_resource;
#else
unsigned char *cuda_dest_resource;
GLubyte shDrawTex;  // draws a texture
struct cudaGraphicsResource *cuda_tex_result_resource;
#endif

unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// (offscreen) render target fbo variables
GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // where we will copy the CUDA result

bool enable_cuda = true;

// Timer
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

//! Initialize GL
bool initGL(int *argc, char **argv) {
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow(sSDKname);

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported(2, 0)) {
        cerr << "ERROR: Support for necessary OpenGL extensions missing.\n";
        fflush(stderr);
        return false;
    }

    glDisable(GL_DEPTH_TEST);

    // viewport
    //glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1f, 10.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

    SDK_CHECK_ERROR_GL();

    return true;
}

// copy image and process using CUDA
void processImage() {
    // run the Cuda kernel
    unsigned char *out_data;
#ifdef USE_TEXSUBIMAGE2D
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&out_data, &num_bytes, cuda_pbo_dest_resource));
    //printf("CUDA mapped pointer of pbo_out: May access %ld bytes, expected %d\n", num_bytes, size_tex_data);
#else
    out_data = cuda_dest_resource;
#endif
    //mapToGL(imageSrc, out_data, image_width, image_height);
    checkCudaErrors(cudaMemcpy((void *)out_data, (void *)imageSrc, size_tex_data, cudaMemcpyDeviceToDevice));
    // CUDA generated data in cuda memory or in a mapped PBO made of BGRA 8 bits
    // 2 solutions, here :
    // - use glTexSubImage2D(), there is the potential to loose performance in possible hidden conversion
    // - map the texture and blit the result thanks to CUDA API
#ifdef USE_TEXSUBIMAGE2D
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);

    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // We want to copy cuda_dest_resource data to the texture
    // map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));

    int num_texels = image_width * image_height;
    int num_values = num_texels * 3;
    int size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));
#endif
}

// display image to the screen as textured quad
void displayImage(GLuint texture) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    //glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //glViewport(0, 0, image_width, image_height);

    // if the texture is a 8 bits UI, scale the fetch with a GLSL shader
#ifndef USE_TEXSUBIMAGE2D
    GLint id = glGetUniformLocation(shDrawTex, "texImage");
    glUniform1i(id, 0); // texture unit 0 to "texImage"
    SDK_CHECK_ERROR_GL();
#endif

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0); glVertex2f(-1.0f, -1.0f);

    glTexCoord2f(1.0, 1.0); glVertex2f(1.0f, -1.0f);

    glTexCoord2f(1.0, 0.0); glVertex2f(1.0f, 1.0f);

    glTexCoord2f(0.0, 0.0); glVertex2f(-1.0f, 1.0f);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

#ifndef USE_TEXSUBIMAGE2D
    glUseProgram(0);
#endif
    SDK_CHECK_ERROR_GL();
}

//! Display callback
void display() {
    sdkStartTimer(&timer);
    if (enable_cuda) {
        processImage();
        displayImage(tex_cudaResult);
    }

    // NOTE: I needed to add this call so the timing is consistent.
    // Need to investigate why
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    // flip backbuffer
    glutSwapBuffers();

    // Update fps counter, fps/title display and log
    if (++fpsCount == fpsLimit) {
        char cTitle[256];
        float fps = 1000.0f / sdkGetAverageTimerValue(&timer);
        sprintf(cTitle, "%s (%d x %d): %.1f fps", sSDKname, window_width, window_height, fps);
        glutSetWindowTitle(cTitle);
        fpsCount = 0;
        fpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
        sdkResetTimer(&timer);
    }
}

void timerEvent(int value) {
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {
    case (27):
        Cleanup(EXIT_SUCCESS);
        break;
    case (GLUT_KEY_RIGHT):
        break;
    case (GLUT_KEY_LEFT):
        break;
    }
}

void reshape(int w, int h) {
    window_width = w;
    window_height = h;
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat)w / (GLfloat)h, 1.0, 100.0);
	glMatrixMode(GL_MODELVIEW);
}

#ifdef USE_TEXSUBIMAGE2D
void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource) {
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 3;
    size_tex_data = sizeof(GLubyte) * num_values;
    void *data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone));

    SDK_CHECK_ERROR_GL();
}

void deletePBO(GLuint *pbo) {
    glDeleteBuffers(1, pbo);
    SDK_CHECK_ERROR_GL();
    *pbo = 0;
}
#endif

void createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y) {
    // create a texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // set basic parameters
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

#ifdef USE_TEXSUBIMAGE2D
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size_x, size_y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
#else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult,
        GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
#endif
}

void deleteTexture(GLuint *tex) {
    glDeleteTextures(1, tex);
    SDK_CHECK_ERROR_GL();

    *tex = 0;
}

void initGLBuffers() {
    // create pbo
#ifdef USE_TEXSUBIMAGE2D
    createPBO(&pbo_dest, &cuda_pbo_dest_resource);
#endif
    // create texture that will receive the result of CUDA
    createTextureDst(&tex_cudaResult, image_width, image_height);
    SDK_CHECK_ERROR_GL();
}

#ifndef USE_TEXSUBIMAGE2D
void initCUDABuffers() {
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 3;
    size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMalloc((void **)&cuda_dest_resource, size_tex_data));
}
#endif

void FreeResource() {
    sdkDeleteTimer(&timer);

    // unregister this buffer object with CUDA
    //checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_result_resource));
#ifdef USE_TEXSUBIMAGE2D
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_dest_resource));
    deletePBO(&pbo_dest);
#else
    cudaFree(cuda_dest_resource);
#endif
    deleteTexture(&tex_screen);
    deleteTexture(&tex_cudaResult);

    if (iGLUTWindowHandle) {
        glutDestroyWindow(iGLUTWindowHandle);
    }

    // finalize logs and leave
    printf("postProcessGL.exe Exiting...\n");
}

void Cleanup(int iExitCode) {
    FreeResource();
    printf("Images are %s\n", (iExitCode == EXIT_SUCCESS) ? "Matching" : "Not Matching");
    exit(EXIT_SUCCESS);
}

//! Run standard demo loop with or without GL verification
void runStdProgram(int argc, char **argv, unsigned char *disData, int nWidth, int nHeight, int nBytes) {
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    image_width = nWidth, image_height = nHeight, window_width = nWidth, window_height = nHeight;
    imageSrc = disData;
    if (false == initGL(&argc, argv)) {
        return;
    }
    // Now initialize CUDA context (GL context has been created already)
    findCudaDevice(argc, (const char **)argv);
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    initGLBuffers();
#ifndef USE_TEXSUBIMAGE2D
    initCUDABuffers();
#endif

    // start rendering mainloop
    glutMainLoop();

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
}