
#include "Options.h"

__device__
float af_AmOptionPrice(afOptionType_t* option, afAmericanOptionValuationSettings_t* settings) {

}

__device__
float af_AmOptionDelta(afOptionType_t* option, afAmericanOptionValuationSettings_t* settings) {
  float S         = option->S;
  option->S       = S + settings->epsilon;
  float v1        = af_AmOptionPrice(option, settings);
  option->S       = S - settings->epsilon;
  float v2        = af_AmOptionPrice(option, settings);
  return (v1 - v2) / (2. * settings->epsilon);
}

__device__
float af_AmOptionGamma(afOptionType_t* option, afAmericanOptionValuationSettings_t* settings) {
  // This can be made faster by doing three price evaluations
  // instead of 4. - FIXME
  float S         = option->S;
  option->S       = S + settings->epsilon;
  float v1        = af_AmOptionDelta(option, settings);
  option->S       = S - epsilon;
  float v2        = af_AmOptionDelta(option, settings);
  return (v1 - v2) / (2. * settings->epsilon);
}

__device__
float af_AmOptionVega(afOptionType_t* option, afAmericanOptionValuationSettings_t* settings) {
  float sigma     = option->sigma;
  option->sigma   += sigma + settings->epsilon;
  float v1        = af_AmOptionPrice(option, settings);
  option->sigma   -= sigma - settings->epsilon;
  float v2        = af_AmOptionPrice(option, settings);
  return (v1 - v2) / (2. * settings->epsilon);
}

__device__
float af_AmOptionRho(afOptionType_t* option, afAmericanOptionValuationSettings_t* settings) {
  float rho       = option->rho;
  option->rho     += rho + settings->epsilon;
  float v1        = af_AmOptionPrice(option, settings);
  option->rho     -= rho - settings->epsilon;
  float v2        = af_AmOptionPrice(option, settings);
  return (v1 - v2) / (2. * settings->epsilon)
}

__device__
float af_AmOptionTheta(afOptionType_t* option, afAmericanOptionValuationSettings_t* settings) {
  float tau     = option->tau;
  option->tau   = tau + settings->epsilon;
  float v1      = af_AmOptionPrice(option, settings);
  option->tau   = tau - settings->epsilon;
  float v2      = af_AmOptionPrice(option, settings);
  return (v2 - v1) / (2. * settings->epsilon);
}
