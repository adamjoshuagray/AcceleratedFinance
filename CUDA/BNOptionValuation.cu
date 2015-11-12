
#include "Options.h"
#include "BNOptionValuation.h"
#include "Result.h"

__device__ __host__
float af_BNOptionPrice(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  // This gets a bit complicated.
  // If this function is called from the host
  // we need to copy the option to the device
  // and invoke the pricing functions.
  // Otherwise we just invoke the pricing functions.
  #ifdef __CUDA_ARCH__
    // Just price
  #else
    // Copy and then price.
  #endif
}

__device__
float af_BNOptionDelta(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  float S         = option->S;
  option->S       = S + settings->epsilon;
  float v1        = af_BNOptionPrice(option, settings);
  option->S       = S - settings->epsilon;
  float v2        = af_BNOptionPrice(option, settings);
  option->S       = S;
  return (v1 - v2) / (2. * settings->epsilon);
}

__device__
float af_BNOptionGamma(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  // This can be made faster by doing three price evaluations
  // instead of 4. - FIXME
  float S         = option->S;
  option->S       = S + settings->epsilon;
  float v1        = af_BNOptionDelta(option, settings);
  option->S       = S - settings->epsilon;
  float v2        = af_BNOptionDelta(option, settings);
  option->S       = S;
  return (v1 - v2) / (2. * settings->epsilon);
}

__device__
float af_BNOptionVega(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  if (option->sigma_style == AF_OPTION_SIGMA_STYLE_SCALAR) {
    float sigma     = option->sigma;
    option->sigma   += sigma + settings->epsilon;
    float v1        = af_BNOptionPrice(option, settings);
    option->sigma   -= sigma - settings->epsilon;
    float v2        = af_BNOptionPrice(option, settings);
    option->sigma   = sigma;
    return (v1 - v2) / (2. * settings->epsilon);
  }
  if (option->sigma_style == AF_OPTION_SIGMA_STYLE_CURVE) {
    // We have an anoying choice here between speed an numerical stability.
    // On the one hand we can copy the curve and put new curves back in with +/- epsilon
    // or we can add epsilon and then subtract 2 epsilon.
    // For the moment we just use the copy method.
    afTimeCurve_t* sigma_curve = option->sigma_curve;
    option->sigma_curve   = af_TimeCurveAdd(settings->epsilon, sigma_curve);
    float v1              = af_BNOptionPrice(option, settings);
    af_TimeCurveDelete(option->sigma_curve);
    option->sigma_curve   = af_TimeCurveAdd(-settings->epsilon, simga_curve);
    float v2              = af_BNOptionPrice(option, settings);
    af_TimeCurveDelete(option->sigma_curve);
    option->sigma_curve   = sigma_curve;
    return (v1 - v2) / (2. * settings->epsilon);
  }
  return AF_UNKNOWN_FLOAT;
}

__device__
float af_BNOptionRho(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  if (option->r_style == AF_OPTION_YIELD_STYLE_SCALAR) {
    float r         = option->r;
    option->r       += r + settings->epsilon;
    float v1        = af_BNOptionPrice(option, settings);
    option->r       -= r - settings->epsilon;
    float v2        = af_BNOptionPrice(option, settings);
    return (v1 - v2) / (2. * settings->epsilon);
  }
  if (option->r_style == AF_OPTION_YIELD_STYLE_CURVE) {
    afTimeCurve_t* r_curve  = option->r_curve;
    option->r_curve         = af_TimeCurveAdd(settings->epsilon, r_curve);
    float v1                = af_BNOptionPrice(option, settings);
    af_TimeCurveDelete(option->r_curve);
    option->r_curve         = af_TimeCurveAdd(-settings->epsilon, r_rurve);
    float v2                = af_BNOptionPrice(option, settings);
    af_TimeCurveDelete(option->r_curve);
    option->r_curve         = r_curve;
    return (v1 - v2) / (2. * settings->epsilon);
  }
  return AF_UNKNOWN_FLOAT;
}

__device__
float af_BNOptionTheta(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  float tau     = option->tau;
  option->tau   = tau + settings->epsilon;
  float v1      = af_BNOptionPrice(option, settings);
  option->tau   = tau - settings->epsilon;
  float v2      = af_BNOptionPrice(option, settings);
  option->tau   = tau;
  return (v2 - v1) / (2. * settings->epsilon);
}
