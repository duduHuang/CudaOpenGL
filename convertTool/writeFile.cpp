#include "writeFile.h"

void bmp_w(char *cFileName, int nWidth, int nHeight, int nSize, unsigned char *data) {
    ofstream ofile(cFileName, ostream::out | ostream::binary);
    unsigned char header[54] = {
        0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
        54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0
    };
    header[2] = nSize & 0x000000ff;
    header[3] = (nSize >> 8) & 0x000000ff;
    header[4] = (nSize >> 16) & 0x000000ff;
    header[5] = (nSize >> 24) & 0x000000ff;
    header[18] = nWidth & 0x000000ff;
    header[19] = (nWidth >> 8) & 0x000000ff;
    header[20] = (nWidth >> 16) & 0x000000ff;
    header[21] = (nWidth >> 24) & 0x000000ff;
    header[22] = nHeight & 0x000000ff;
    header[23] = (nHeight >> 8) & 0x000000ff;
    header[24] = (nHeight >> 16) & 0x000000ff;
    header[25] = (nHeight >> 24) & 0x000000ff;
    ofile.write((char *)header, 54);
    ofile.write((char *)data, nSize);
    ofile.close();
}

void dumpYUV(unsigned char *d_srcNv12, int width, int height, string filename, int coloer_space) {
    int size = width * height * coloer_space * sizeof(unsigned char);
    ofstream nv12File(filename, ostream::out | ostream::binary);
    if (nv12File) {
        nv12File.write((char *)d_srcNv12, size);
        nv12File.close();
    }
}