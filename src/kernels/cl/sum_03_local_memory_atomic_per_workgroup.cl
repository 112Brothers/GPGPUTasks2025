#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);
    // TODO
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    __local uint local_data[GROUP_SIZE];

    if (gid < n) {
        local_data[lid] = a[gid];
    } else {
       local_data[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            local_data[lid] += local_data[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        atomic_add(sum, local_data[0]);
    }
}
