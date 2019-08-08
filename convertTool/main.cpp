#include <iostream>
#include "convertTool.h"

using namespace std;

int main(int argc, char* argv[]) {
    unsigned char *r = new unsigned char[1280 * 720 * 3];
    ConverterTool *converterTool;
    converterTool = new ConverterTool();
    if (converterTool->isGPUEnable()) {
#ifdef WIN32
        cout << "device has cuda !!!\n";
        converterTool->initialCuda();
        converterTool->preprocess();
        converterTool->lookupTableF();
        //converterTool->convertToRGBThenResize(r);
        converterTool->resizeThenConvertToRGB(r);
        converterTool->display();
#endif // !WIN32
#ifndef WIN32
        int i = 1;
        converterTool->initialCuda();
        converterTool->preprocess();
        converterTool->lookupTableF();
        while (i) {
            converterTool->testFunction();
            cout << "continue ? ";
            cin >> i;
        }
#endif
    }
    else {
        cout << "device hasn't cuda !!!\n";
    }
    converterTool->freeMemory();
    converterTool->destroyCudaEvent();
    delete[] r;
    return 0;
}