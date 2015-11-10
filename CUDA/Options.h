
#ifndef AF_OPTIONS_H
#define AF_OPTIONS_H

#include "Curve.h"
#include "Result.h"

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
    // The way that the volatility is specified.
    afOptionSigmaStyle_t      sigma_style;
    // The way that the dividends are specified.
    afOptionDividendStyle_t   q_style;
    // The way that the discount rate is specified.
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
    // The yield curve used in the option calculations.
    afTimeCurve_t*  r_curve;
    // The curve used to define when dividends are paid.
    afTimeCurve_t*  q_curve;
    // The curve used to define the volatility throughout time.
    afTimeCurve_t*  sigma_curve;
} afOptionInfo_t;

//
// Copies an option from the host to the device.
// Note that this is not part of the "CUDA API" and shouldn't be exported.
//
__host__
cudaError_t __af_OptionCopyToDevice(afOptionInfo_t* src, afOptionInfo_t** dst);

//
// Copies an option from the host to the device.
//
__host__
cudaError_t af_OptionCopyArrayToDevice(afOptionInfo_t* src, int count, afOptionInfo_t** dst);

//
// Copies an option from the device to the host.
//
__host__
cudaError_t af_OptionCopyToHost(afOptionInfo_t* src, afOptionInfo_t** dst);

//
// Deallocates the memory associated with the option.
//
__device__ __host__
void af_OptionInfoDelete(afOptionInfo_t* option);


#endif // AF_OPTIONS_H
