
#include "Options.h"
#include "BNOptionValuation.h"
#include "Result.h"

__device__
float af_BNOptionPrice(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  return AF_UNKNOWN_FLOAT;
}

__device__
float af_BNOptionDelta(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  float S         = option->S;
  option->S       = S + settings->epsilon;
  float v1        = af_BNOptionPrice(option, settings);
  option->S       = S - settings->epsilon;
  float v2        = af_BNOptionPrice(option, settings);
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
  return (v1 - v2) / (2. * settings->epsilon);
}

__device__
float af_BNOptionVega(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  float sigma     = option->sigma;
  option->sigma   += sigma + settings->epsilon;
  float v1        = af_BNOptionPrice(option, settings);
  option->sigma   -= sigma - settings->epsilon;
  float v2        = af_BNOptionPrice(option, settings);
  return (v1 - v2) / (2. * settings->epsilon);
}

__device__
float af_BNOptionRho(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  float r         = option->r;
  option->r       += r + settings->epsilon;
  float v1        = af_BNOptionPrice(option, settings);
  option->r       -= r - settings->epsilon;
  float v2        = af_BNOptionPrice(option, settings);
  return (v1 - v2) / (2. * settings->epsilon);
}

__device__
float af_BNOptionTheta(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings) {
  float tau     = option->tau;
  option->tau   = tau + settings->epsilon;
  float v1      = af_BNOptionPrice(option, settings);
  option->tau   = tau - settings->epsilon;
  float v2      = af_BNOptionPrice(option, settings);
  return (v2 - v1) / (2. * settings->epsilon);
}
