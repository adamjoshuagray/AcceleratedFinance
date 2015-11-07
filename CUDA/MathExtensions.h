#ifndef AF_MATH_EXTENSIONS_H
#define AF_MATH_EXTENSIONS_H

//
// Computes the value of the standard normal pdf.
//
// x      - The value to calculate it for.
//
__device__ __host__
float af_normpdff(float x);

__device__ __host__
float af_powf(float x, float y);

#endif // AF_MATH_EXTENSIONS_H
