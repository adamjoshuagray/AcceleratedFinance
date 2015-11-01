
#ifndef AF_OPTIONS_H
#define AF_OPTIONS_H

/**
 * These define the type of option.
 * That is, put or call.
 */
#define AF_OT_CALL 1
#define AF_OT_PUT 2

#define AF_UNKOWN_PRICE -1
#define AF_UNKNOWN_SIGMA -1

/**
 * These define the style of option.
 * That is european or american.
 * Support may be added for more
 * styles in the future.
 */
#define AF_OS_EUROPEAN 1
#define AF_OS_AMERICAN 2

/*
 * Just typedefs for the type and
 * style of option.
 */
typedef int afOptionType_t;
typedef int afOptionStyle_t;

/**
 * This defines the info associated
 * with an option.
 * These are passed to functions for
 * valuation or calculating missing
 * parameters.
 */
typedef struct afOptionInfo_t {
    /**
     * The style of the option - American or European.
     */
    afOptionStyle_t style;
    /**
     * The type of the option - Put or Call.
     */
    afOptionType_t  type;
    /**
     * The price of the underlying stock.
     */
    float           S;
    /**
     * The strike price of the option.
     */
    float           K;
    /**
     * The time to expiry of the option.
     */
    float           tau;
    /**
     * The risk free rate associated with the option.
     */
    float           r;
    /**
     * The volatility of the underlying stock.
     */
    float           sigma;
    /**
     * The price of the option.
     */
    float           price;
} afOptionInfo_t;

/**
 * This defines the info associated
 * with a volatility surface.
 */
typedef struct afVolSurface_t {
    /**
     * The number of strikes in the surface.
     */
    int             n_K;
    /**
     * The number of expiries in the surface.
     */
    int             n_tau;
    /**
     * The actual surface itself.
     * This is an n_K * n_tau array
     * containing the volatilties.
     */
    float*          surf;
} afVolSurface_t;

#endif // AF_OPTIONS_H