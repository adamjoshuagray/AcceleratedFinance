/**
 * EuropeanVolSurface.cuh
 *
 * This deals with simple calculations for European options.
 * We try and precompute quite a few values so that this can
 * run a quickly as possible.
 *
 * We assume that there are no dividends for the underlying stock
 * for these calculations.
 */

#ifndef AF_EUROPEANVOLSURFACE_H
#define AF_EUROPEANVOLSURFACE_H
#include "Result.h"
#include "Options.h"

/**
 * If no volatility is known already
 * then this is the default max volatility that is used.
 */
#define AF_DEFAULT_MAX_SIGMA 100
/**
 * If no volatility is known already
 * then this is the default min volatility that is used.
 */
#define AF_DEFAULT_MIN_SIGMA 0
/**
 * If a volatility is already known then this is the
 * the value that is added to get the max vol.
 */
#define AF_SIGMA_UPPER_BRACKET 0.1
/**
 * If a volatiltiy is already known then this is the
 * value that is subtracted to get the min vol.
 */
#define AF_SIGMA_LOWER_BRACKET 0.05
/**
 * The tolerance to use when calculating the implied volatility.
 */
#define AF_TOL 0.0001
/**
 * The is set to but an upper bound on the number of iterations that the
 * implied volatility solver can use.
 *
 * This is really just used to stop the funciton from jamming if someone
 * specifies tol = 0 or some other unachievable value.
 */
#define AF_MAX_ITER 100

/**
 * A common function used in calculating the price of call and put options.
 *
 * This is defined in much the same way as it is on wikipedia.
 */
__device__
float af_Eurd_1(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r);

/**
 * A common function used in calculating the price of call and put options.
 *
 * This is defined in much the same way as it is on wikipedia.
 */
__device__
float af_Eurd_2(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r);

/**
 * Calculates the price of a call option.
 *
 * rsqrt_tau    - The reciprocal square root of tau where tau is the time to expiry.
 * tau          - The time to expiry.
 * sigma        - The volatility of the of the underlying.
 * ln_S_K       - The natural logarithm of S/K where S is the underlying price and K is
 *                the strike price of the option.
 * r            - The risk free rate.
 * S            - The underlying price.
 * K            - The strike price of the option.
 * disc         - The discount factor to use, i.e. expf(-r * tau)
 */
__device__
float af_EurCall(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float S, float K, float disc);

/**
 * Calculates the price of a put option.
 *
 * rsqrt_tau    - The reciprocal square root of tau where tau is the time to expiry.
 * tau          - The time to expiry.
 * sigma        - The volatility of the of the underlying.
 * ln_S_K       - The natural logarithm of S/K where S is the underlying price and K is
 *                the strike price of the option.
 * r            - The risk free rate.
 * S            - The underlying price.
 * K            - The strike price of the option.
 * disc         - The discount factor to use, i.e. expf(-r * tau)
 */
__device__
float af_EurPut(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float S, float K, float disc);
/**
 * This calculates the implied volatility of a european call option.
 *
 * K            - The strike price of the option.
 * S            - The price of the underlying.
 * tau          - The time to expiry.
 * r            - The risk free rate.
 * call         - The price of the call option.
 * min_vol      - A lower bound to be assumed on the volatility.
 * max_vol      - An upper bound to be assumed on the volatility.
 * tol          - A tolerance in the resultant price for each iteration.
 *                Once the difference between the price of the call option
 *                in an iteration and the price of the call option passed
 *                to this function fall below this value then the function
 *                will returned the implied volatility.
 */
__device__
float af_EurCallImpliedVol(float K, float S, float tau, float r, float call, float min_vol, float max_vol, float tol);


/**
 * This calculates the implied volatility of a european put option.
 *
 * K            - The strike price of the option.
 * S            - The price of the underlying.
 * tau          - The time to expiry.
 * r            - The risk free rate.
 * call         - The price of the put option.
 * min_vol      - A lower bound to be assumed on the volatility.
 * max_vol      - An upper bound to be assumed on the volatility.
 * tol          - A tolerance in the resultant price for each iteration.
 *                Once the difference between the price of the put option
 *                in an iteration and the price of the put option passed
 *                to this function fall below this value then the function
 *                will returned the implied volatility.
 */
__device__
float af_EurPutImpliedVol(float K, float S, float tau, float r, float put, float min_vol, float max_vol, float tol);

/**
 * This calculates the implied volatility for many options in parallel
 *
 * options      - An array of options which we wish to calculate the implied volatility for.
 *                The result of the calculation will be stored in the sigma component.
 * n_options    - The size of the array options.
 */
__global__
void af_EurCalcVolMany(afOptionInfo_t* options, int n_options);

/**
 * This will allocate a block of memory on the device and copy accross an array
 * of options.
 *
 * options      - The options that we wish to copy across.
 */
__host__
afResult_t af_EurInit(afOptionInfo_t* options, int n_options, void* dev_options);
__host__
afResult_t af_EurDeinit(void* dev_options);

__host__
afResult_t af_EurUpdate(afOptionInfo_t* options, int n_options, void* dev_options);

__host__
afResult_t af_EurDownload(afOptionInfo_t* options, int n_options, void* dev_options);


#endif //AF_EUROPEANVOLSURFACE_H
