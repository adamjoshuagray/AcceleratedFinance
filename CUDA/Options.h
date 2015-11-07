
#ifndef AF_OPTIONS_H
#define AF_OPTIONS_H

#include "Curve.h"

/**
 * These define the type of option.
 * That is, put or call.
 */
#define AF_OPTION_TYPE_CALL 1
#define AF_OPTION_TYPE_PUT 2

/**
 * These define the style of option.
 * That is european or american.
 * Support may be added for more
 * styles in the future.
 */
#define AF_OPTION_STYLE_EUROPEAN 1
#define AF_OPTION_STYLE_AMERICAN 2

// Whether we use r for r_curve for
// the risk free rate.
#define AF_OPTION_YIELD_STYLE_SCALAR 1
#define AF_OPTION_YIELD_STYLE_CURVE 2

// Whether we use q or q_curve for the
// dividend yeilds.
#define AF_OPTION_DIVIDEND_STYLE_SCALAR 1
#define AF_OPTION_DIVIDEND_STYLE_CURVE 2

// Whether we use sigma or sigma_curve
// for the volatility of the stock.
#define AF_OPTION_SIGMA_STYLE_SCALAR 1
#define AF_OPTION_SIGMA_STYLE_CURVE 2
/*
 * Just typedefs for the type and
 * style of option.
 */
typedef int afOptionType_t;
typedef int afOptionStyle_t;
typedef int afOptionSigmaStyle_t;
typedef int afOptionYieldStyle_t;
typedef int afOptionDividendStyle_t;

/**
 * This defines the info associated
 * with an option.
 * These are passed to functions for
 * valuation or calculating missing
 * parameters.
 */
typedef struct afOptionInfo_t {
    // The style of the option - American or European.
    afOptionStyle_t style;
    // The type of the option - Put or Call.
    afOptionType_t  type;

    afOptionSigmaStyle_t      sigma_style;

    afOptionDividendStyle_t   q_style;

    afOptionYieldStyle_t      r_style;
    // The price of the underlying stock.
    float           S;
    // The strike price of the option.
    float           K;
    // The time to expiry of the option.
    float           tau;
    // The risk free rate associated with the option.
    float           r;
    // The volatility of the underlying stock.
    float           sigma;
    // The price of the option.
    float           price;
    // The dividend yield of the underlying stock.
    float           q;
    //
    afTimeCurve_t*  r_curve;

    afTimeCurve_t*  q_curve;
    //
    afTimeCurve_t*  sigma_curve;

} afOptionInfo_t;

__device__ __host__
void af_OptionInfoDelete(afOptionInfo_t* option);


#endif // AF_OPTIONS_H
