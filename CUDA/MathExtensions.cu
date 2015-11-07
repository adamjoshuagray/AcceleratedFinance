
#include "MathExtensions.h"
#include "math.h"


__device__ __host__
float af_normpdff(float x) {
  return rsqrtf(2. * M_PI) * exp(-af_powf(x,2.) / 2.);
}

__device__ __host__
float af_powf(float x, float y) {
  #ifdef __CUDA_ARCH__
    return __powf(x, y);
  #else
    return powf(x, y);
  #endif
}
