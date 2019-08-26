#ifndef __H_WRITEFILE__
#define __H_WRITEFILE__

#include <fstream>
#include <iostream>
using namespace std;

void bmp_w(char *cFileName, int nWidth, int nHeight, int nSize, unsigned char *data);

void dumpYUV(unsigned short *d_srcNv12, int width, int height, string filename);

void dumpRGB(unsigned char *dSrc, int w, int h, string fileName);

#endif // !__H_WRITEFILE__