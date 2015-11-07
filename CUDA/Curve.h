
#ifndef AF_CURVE_H
#define AF_CURVE_H

#include "time.h"

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

__device__ __host__
afTimeCurve_t* af_TimeCurveMult(float a, afTimeCurve_t* x);

__device__ __host__
time_t af_TimeCurveLastTime(afTimeCurve_t* x);

__device__ __host__
time_t af_TimeCurveFirstTime(afTimeCurve_t* x);

__device__ __host__
float af_TimeCurveInterpolate(afTimeCurve_t* x, time_t t, afInterpolationStyle_t style);

__device__ __host__
float af_TimeCurveInterpolatePrevious(afTimeCurve_t* x, time_t t);

__device__ __host__
float af_TimeCurveInterpolateLinear(afTimeCurve_t* x, time_t t);

__device__ __host__
void af_TimeCurveMultInPlace(float a, afTimeCurve_t* x);

//
// This deallocates the memory associated with a curve.
//
// curve - The curve to delete.
//
__device__ __host__
void af_TimeCurveDelete(afTimeCurve_t* curve);


#endif // AF_CURVE_H
