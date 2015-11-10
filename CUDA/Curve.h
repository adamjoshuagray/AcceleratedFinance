
#ifndef AF_CURVE_H
#define AF_CURVE_H

#include "time.h"
#include "stdbool.h"

//
// This defines the style that will be used for interpolation.
//
#define AF_INTERPOLATION_STYLE_PREVIOUS 1
#define AF_INTERPOLATION_STYLE_LINEAR 2

//
// A type for which is used to define interpolation style.
//
typedef int afInterpolationStyle_t;

//
// A pair which has the time and the value associated with that time.
//
typedef struct afTimeCurvePair_t {
  time_t      time;
  float       value;
} afTimeCurvePair_t;

//
// A curve is essentially just an array of pairs.
//
typedef struct afTimeCurve_t {
  // The number of pairs in the curve.
  int                 count;
  // The actual array of pairs.
  afTimeCurvePair_t*  pairs;
} afTimeCurve_t;

//
// Calculates a * x + b * y
//
// This will create a curve with points defined by x and y.
//
// a - The value to multiply x by.
// x - One of the curves to add.
// b - The value to multiply y by.
// y - One of the curves to add.
//
__device__ __host__
afTimeCurve_t* af_TimeCuveSumAxBy(float a, afTimeCurve_t* x, float b, afTimeCurve_t* y);

//
// Calculates a * x and returns the value as a new curve.
//
// a - The value to multiply x by.
// x - The curve to multiply.
//
__device__ __host__
afTimeCurve_t* af_TimeCurveMult(float a, afTimeCurve_t* x);

//
// Returns the last time point in the curve.
//
// x - The curve to find the last time point of.
//
__device__ __host__
time_t af_TimeCurveLastTime(afTimeCurve_t* x);

//
// Modifies in place the curve by multipling each
// point by a.
//
// a - The value to multiply the curve by.
// x - The curve to modify in place.
//
__device__ __host__
void af_TimeCurveMultInPlace(float a, afTimeCurve_t* x);

//
// Gets the firm time point in the curve.
//
__device__ __host__
time_t af_TimeCurveFirstTime(afTimeCurve_t* x);

__device__ __host__
float af_TimeCurveInterpolate(afTimeCurve_t* x, time_t t, afInterpolationStyle_t style);

__device__ __host__
float af_TimeCurveInterpolatePrevious(afTimeCurve_t* x, time_t t);

__device__ __host__
float af_TimeCurveInterpolateLinear(afTimeCurve_t* x, time_t t);

//
// This deallocates the memory associated with a curve.
//
// curve - The curve to delete.
//
__device__ __host__
void af_TimeCurveDelete(afTimeCurve_t* curve);

//
// This copies a curve from the host to the device.
//
// src    - The location on the host to copy from.
// dst    - A pointer to a pointer to the location in device memory where the
//          curve will be stored.
__host__
cudaError_t af_TimeCurveCopyToDevice(afTimeCurve_t* src, afTimeCurve_t** dst);
//
// This copies a curve from the device to the host.
//
// src    - The device pointer to the curve which we want to copy to the host.
// dst    - A pointer to a pointer to the location where we will copy the
//          curve to.
// assume_contiguous - Whether to first copy a contiguous block of memory
//          and check that it is indeed contiguous or to use two memcpys
// count  - Only used of assume_contiguous == true. This defines the number
//          of pairs to copy.
//
__host__
cudaError_t af_TimeCurveCopyToHost(afTimeCurve_t* src, afTimeCurve_t** dst, bool assume_contiguous, int count);


#endif // AF_CURVE_H
