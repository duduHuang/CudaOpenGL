#ifndef __H_WRITEFILE__
#define __H_WRITEFILE__

#include <fstream>
#include <iostream>
using namespace std;

void bmp_w(char *cFileName, int nWidth, int nHeight, int nSize, unsigned char *data);

void dumpYUV(unsigned char *d_srcNv12, int width, int height, string filename, int coloer_space);

#endif // !__H_WRITEFILE__