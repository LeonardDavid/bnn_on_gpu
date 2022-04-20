#define BATCH_SIZE 2
#define NR_CHANNELS 3   // number of channels of input image
#define IMG_HEIGHT 32   // original input image
#define IMG_WIDTH 32    // original input image
#define OUT_SIZE 10     // number of classes

#pragma once 
inline int index3D(const int x, const int y, const int z, const int sizey, const int sizez) {
    return x*sizey*sizez + y*sizez + z;
}

#pragma once 
inline int index4D(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
    return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}