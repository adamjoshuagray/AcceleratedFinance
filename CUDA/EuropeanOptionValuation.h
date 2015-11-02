
#ifndef AF_EUROPEAN_OPTION_VALUATION_H
#define AF_EUROPEAN_OPTION_VALUATION_H

#include "Options.h"

//
// Calculates d_1 as described by https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
//
// rsqrt_tau    - The reciprocal square root of tau
// tau          - The time to expiry of the option
// sigma        - The volatility of the underlying stock
// ln_S_K       - ln(S/K)
// r            - The risk free rate.
// q            - The continuous dividend yeild.
//
__device__
float af_EuroOptionD_1(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q);

//
// Calculates d_2 as described by https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
//
// rsqrt_tau    - The reciprocal square root of tau
// tau          - The time to expiry of the option
// sigma        - The volatility of the underlying stock
// ln_S_K       - ln(S/K)
// r            - The risk free rate.
// q            - The continuous dividend yeild.
//
__device__
float af_EuroOptionD_2(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q);

//
// Calculates the price of a European stock option using the Black Scholes method.
// We use this method in the implied volatility calculator because we can cache
// values such as rsqrt_tau and not recalculate them on ever iteration.
//
// type         - The type of the option (put or call)
// rsqrt_tau    - The reciprocal square root of tau
// tau          - The time to expiry of the option
// sigma        - The volatility of the underlying stock
// ln_S_K       - ln(S/K)
// r            - The risk free rate.
// q            - The continuous dividend yeild.
// S            - The underlying stock price.
// K            - The option strike price.
// disc_r       - The risk free discount factor (exp(-r * tau))
// disc_q       - The continuous dividend discount factor (exp(-r * tau))
//
__device__
float af_EuroOptionPrice(afOptionType_t type, float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q, float S, float K, float disc_r, float disc_q);


//
// Calculates the price of a European stock option using the Black Scholes method.
//
// option       - The option to calculate the price of. The price field will be ignored.
//
// This will return AF_UNKNOWN if the option has style != AF_OPTION_STYLE_EUROPEAN
//
__device__
float af_EuroOptionPrice(afOptionInfo_t* option);

//
// Calculates the delta of a European stock option.
//
// option       - The option to calculate the delta of. The price field will be ignored.
//
// This will return AF_UNKNOWN if the option has style != AF_OPTION_STYLE_EUROPEAN
//
__device__
float af_EuroOptionDelta(afOptionInfo_t* option);

//
// Calculates the vega of a European stock option.
//
// option       - The option to calculate the vega of. The price field will be ignored.
//
// This will return AF_UNKNOWN if the option has style != AF_OPTION_STYLE_EUROPEAN
//
__device__
float af_EuroOptionVega(afOptionInfo_t* option);

//
// Calculates the gamma of a European stock option.
//
// option       - The option to calculate the gamma of. The price field will be ignored.
//
// This will return AF_UNKNOWN if the option has style != AF_OPTION_STYLE_EUROPEAN
//
__device__
float af_EuroOptionGamma(afOptionInfo_t* option);

//
// Calculates the rho of a European stock option.
//
// option       - The option to calculate the rho of. The price field will be ignored.
//
// This will return AF_UNKNOWN if the option has style != AF_OPTION_STYLE_EUROPEAN
//
__device__
float af_EuroOptionRho(afOptionInfo_t* option);

//
// Calculates the theta of a European stock option.
//
// option       - The option to calculate the theta of. The price field will be ignored.
//
// This will return AF_UNKNOWN if the option has style != AF_OPTION_STYLE_EUROPEAN
//
__device__
float af_EuropOptionTheta(afOptionInfo_t* option);

//
// Calculates the implied volatility of a European stock option.
//
// option       - The option to calculate the implied volatility of.
// min_sigma    - The minimum value that we suspect sigma to have.
// max_sigma    - The maximum value that we suspect sigma to have.
// tol          - The tolerance in the price to require before returning a value.
// max_iter     - The maximum number of iterations before we MUST return a value.
//                This value takes precedence over tol.
//
// This will return AF_UNKNOWN if the volatility lies outside of [min_sigma, max_sigma]
// or if the option has style != AF_OPTION_STYLE_EUROPEAN
//
__device__
float af_EuroOptionImpliedSigma(afOptionInfo_t* option, float min_sigma, float max_sigma, float tol, int max_iter);

#endif // AF_EUROPEAN_OPTION_VALUATION_H
