
#ifndef AF_CUDA_H
#define AF_CUDA_H

#include "time.h"

//
// Exports from Result.h
// See comments in Result.h for details.
//

#define AF_RESULT_SUCCESS 1
#define AF_RESULT_ERROR 0

#define AF_UNKNOWN_FLOAT NAN
#define AF_UNKNOWN_TIME (time_t) 0;

extern bool af_ResultIsUnknownFloat(float x);

extern bool af_ResultIsUnknownTime(time_t x);

//
// Exports from MathExtensions.h
// See comments in MathExtensions.h for details.
//

extern float af_normpdff(float x);

extern float af_powf(float x, float y);

//
// Exports from Curve.h
// See comments in Curve.h for details.
//

typedef int afInterpolationStyle_t;

typedef struct afTimeCurvePair_t {
  time_t      time;
  float       value;
} afTimeCurvePair_t;


typedef struct afTimeCurve_t {
  int                 count;
  afTimeCurvePair_t*  pairs;
} afTimeCurve_t;

extern afTimeCurve_t* af_TimeCuveSumAxBy(float a, afTimeCurve_t* x, float b, afTimeCurve_t* y);

extern afTimeCurve_t* af_TimeCurveMult(float a, afTimeCurve_t* x);

extern time_t af_TimeCurveLastTime(afTimeCurve_t* x);

extern time_t af_TimeCurveFirstTime(afTimeCurve_t* x);

extern float af_TimeCurveInterpolate(afTimeCurve_t* x, time_t t, afInterpolationStyle_t style);

extern float af_TimeCurveInterpolatePrevious(afTimeCurve_t* x, time_t t);

extern float af_TimeCurveInterpolateLinear(afTimeCurve_t* x, time_t t);

extern void af_TimeCurveMultInPlace(float a, afTimeCurve_t* x);

extern void af_TimeCurveDelete(afTimeCurve_t* curve);

//
// Exports from Options.h
// See comments in Options.h for details.
//

#define AF_OPTION_TYPE_CALL 1
#define AF_OPTION_TYPE_PUT 2

#define AF_OPTION_STYLE_EUROPEAN 1
#define AF_OPTION_STYLE_AMERICAN 2

#define AF_OPTION_YIELD_STYLE_SCALAR 1
#define AF_OPTION_YIELD_STYLE_CURVE 2

#define AF_OPTION_DIVIDEND_STYLE_SCALAR 1
#define AF_OPTION_DIVIDEND_STYLE_CURVE 2

#define AF_OPTION_SIGMA_STYLE_SCALAR 1
#define AF_OPTION_SIGMA_STYLE_CURVE 2

typedef int afOptionType_t;
typedef int afOptionStyle_t;
typedef int afOptionSigmaStyle_t;
typedef int afOptionYieldStyle_t;
typedef int afOptionDividendStyle_t;

typedef struct afOptionInfo_t {
    afOptionStyle_t style;
    afOptionType_t  type;
    afOptionSigmaStyle_t      sigma_style;
    afOptionDividendStyle_t   q_style;
    afOptionYieldStyle_t      r_style;
    float           S;
    float           K;
    float           tau;
    float           r;
    float           sigma;
    float           price;
    float           q;
    afTimeCurve_t*  r_curve;
    afTimeCurve_t*  q_curve;
    afTimeCurve_t*  sigma_curve;

} afOptionInfo_t;

extern void af_OptionInfoDelete(afOptionInfo_t* option);

//
// Exports from BNOptionValuation.h
// See comments in BNOoptionValuation.h for details.
//

typedef struct afBNOptionValuationSettings_t {
  float epsilon;
  float tau_steps;
} afBNOptionValuationSettings_t;

extern float af_BNOptionPrice(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

extern float af_BNOptionDelta(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

extern float af_BNOptionGamma(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

extern float af_BNOptionVega(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

extern float af_BNOptionRho(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

extern float af_BNOptionTheta(afOptionInfo_t* option, afBNOptionValuationSettings_t* settings);

//
// Exports from BSOptionValuation.h
// See comments in BSOptionValuation.h for details.
//

extern bool af_BSOptionValidate(afOptionInfo_t* option);

extern float af_BSOptionD_1(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q);

extern float af_BSOptionD_2(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q);

extern float af_BSOptionPrice(afOptionType_t type, float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q, float S, float K, float disc_r, float disc_q);

extern float af_BSOptionPrice(afOptionInfo_t* option);

extern float af_BSOptionDelta(afOptionInfo_t* option);

extern float af_BSOptionVega(afOptionInfo_t* option);

extern float af_BSOptionGamma(afOptionInfo_t* option);

extern float af_BSOptionRho(afOptionInfo_t* option);

extern float af_EuropOptionTheta(afOptionInfo_t* option);

extern float af_BSOptionImpliedSigma(afOptionInfo_t* option, float min_sigma, float max_sigma, float tol, int max_iter);

#endif // AF_CUDA_H
