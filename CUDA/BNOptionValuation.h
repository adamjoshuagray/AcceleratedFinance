
#ifndef AF_BS_OPTION_VALUATION_H
#define AF_BS_OPTION_VALUATION_H

#include "Options.h"

typedef struct afBNOptionValuationSettings_t {
  float epsilon;
  float tau_steps;
} afBNOptionValuationSettings_t;

__device__
float af_BNOptionPrice(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

__device__
float af_BNOptionDelta(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

__device__
float af_BNOptionGamma(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

__device__
float af_BNOptionVega(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

__device__
float af_BNOptionRho(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

__device__
float af_BNOptionTheta(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

#endif // AF_BS_OPTION_VALUATION_H
