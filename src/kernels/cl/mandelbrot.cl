#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void mandelbrot(__global float* results,
                     unsigned int width, unsigned int height,
                     float fromX, float fromY,
                     float sizeX, float sizeY,
                     unsigned int iters, unsigned int isSmoothing)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    // TODO
    if (i >= width || j >= height) {
        return;
    }
    float threshold = 256.0f;
    float threshold2 = threshold * threshold;
    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;
    float x = x0;
    float y = y0;
    uint iter = 0;
    for (; iter < iters; iter++) {
        float xPrev = x;
        float yPrev = y;
        x = xPrev * xPrev - yPrev * yPrev + x0;
        y = 2.0f * xPrev * yPrev + y0;
        if (x * x + y * y > threshold2) {
            break;
        }
    }
    float res = iter;
    if (isSmoothing && iter != iters) {
        res = iter + 1 - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }
    results[j * width + i] = res / iters;
}
