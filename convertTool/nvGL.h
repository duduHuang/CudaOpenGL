#ifndef __H_NVGL__
#define __H_NVGL__

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#pragma warning(disable:4996)
#endif
// USE_TEXSUBIMAGE2D uses glTexSubImage2D() to update the final result
// commenting it will make the sample use the other way :
// map a texture in CUDA and blit the result into it
#define USE_TEXSUBIMAGE2D

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
// Sorry for Apple : unsigned int sampler is not available to you, yet...
// Let's switch to the use of PBO and glTexSubImage
#define USE_TEXSUBIMAGE2D
#else
#include <GL/freeglut.h>
#endif

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>

#include <iostream>

// Shared Library Test Functions
#define MAX_EPSILON 10
#define REFRESH_DELAY     10 //ms

using namespace std;

// Forward declarations
void runStdProgram(int argc, char **argv, unsigned char *disData, int nWidth, int nHeight, int nBytes);
void FreeResource();
void Cleanup(int iExitCode);

#endif // !__H_NVGL__