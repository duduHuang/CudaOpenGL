#include <iostream>
#include "convertTool.h"

using namespace std;

int main(int argc, char* argv[]) {
    unsigned char *r = new unsigned char[1280 * 720 * 3];
    ConverterTool *converterTool = new ConverterTool(argc, argv);
    converterTool->preprocess();
    converterTool->lookupTableF();
    //converterTool->convertToRGBThenResize(r);
    converterTool->resizeThenConvertToRGB(r);
    converterTool->display();
    converterTool->destroyCudaEvent();
    delete[] r;
    return 0;
}