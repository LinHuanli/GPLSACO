#pragma once

#include <cmath>

#ifdef __CUDACC__
#define GPFACO_HD __host__ __device__
#else
#define GPFACO_HD
#endif

namespace gp_faco {

GPFACO_HD inline float apply_operation(int opcode, float left, float right) {
    float result = 0.0f;
    switch (opcode) {
        case 2: result = left + right; break;
        case 3: result = left - right; break;
        case 4: result = left * right; break;
        case 5: result = fminf(left, right); break;
        case 6: result = fmaxf(left, right); break;
        case 7: result = fabsf(left); break;
        case 8:
#ifdef __CUDA_ARCH__
            result = left * rsqrtf(1.0f + right * right);
#else
            result = left * (1.0f / sqrtf(1.0f + right * right));
#endif
            break;
    }
    return fminf(8.0f, fmaxf(-8.0f, result));
}

}  // namespace gp_faco

#undef GPFACO_HD
