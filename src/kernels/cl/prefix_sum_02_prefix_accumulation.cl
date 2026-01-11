#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2,
    unsigned int pow2_sum_len)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    if (pow2 >= 31u) return; // защита
    uint step = 1u << pow2;  // длина блока

    // номер блока, в котором находится i
    uint block = i >> pow2; // = i / step

    // прибавляем лишь тогда, когда блок нечётный (бит pow2 у i == 1)
    if ((block & 1u) != 0u) {
        // индекс левого блока в S_cur = block - 1
        uint left = block - 1u;
        if (left < pow2_sum_len) {
            // S_cur[left] — sum of elements in that left block
            prefix_sum_accum[i] += pow2_sum[left];
        }
    }
}