
#include "BSOptionValuation.h"
#include "Result.h"
#include "MathExtensions.h"
#include "math.h"
#include "stdbool.h"
#include "stdio.h"

__device__ __host__
bool af_BSOptionValidate(afOptionInfo_t* option) {
  return (option->style == AF_OPTION_STYLE_EUROPEAN &&
    option->sigma_style == AF_OPTION_SIGMA_STYLE_SCALAR &&
    option->r_style == AF_OPTION_YIELD_STYLE_SCALAR &&
    option->q_style == AF_OPTION_DIVIDEND_STYLE_SCALAR);
}

__device__ __host__
float af_BSOptionD_1(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q) {
  float a = 1. / sigma * rsqrt_tau;
  float b = ln_S_K + (r - q + af_powf(sigma, 2.) / 2.) * tau;
  float d_1 = a * b;
  return d_1;
}

__device__ __host__
float af_BSOptionD_2(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q) {
  float a = 1. / sigma * rsqrt_tau;
  float b = ln_S_K + (r - q - af_powf(sigma, 2.) / 2.) * tau;
  float d_1 = a * b;
  return d_1;
}

__device__ __host__
float af_BSOptionPrice(afOptionType_t type, float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q, float S, float K, float disc_r, float disc_q) {
  float d_1 = af_BSOptionD_1(rsqrt_tau, tau, sigma, ln_S_K, r, q);
  float d_2 = af_BSOptionD_2(rsqrt_tau, tau, sigma, ln_S_K, r, q);
  float a;
  float b;
  if (type == AF_OPTION_TYPE_CALL) {
    a = normcdff(d_1) * S * disc_q;
    b = normcdff(d_2) * K * disc_r;
    return a - b;
  }
  if (type == AF_OPTION_TYPE_PUT) {
    a = normcdff(-d_1) * S * disc_q;
    b = normcdff(-d_2) * K * disc_r;
    return b - a;
  }
  return AF_UNKNOWN_FLOAT;
}

__device__ __host__
float af_BSOptionPrice(afOptionInfo_t* option) {
  if (af_BSOptionValidate(option)) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_r    = expf(-option->r * option->tau);
    float disc_q    = expf(-option->q * option->tau);
    #ifndef __CUDA_ARCH__
      printf("rsqrt_tau: %f ln_S_K: %f disc_r: %f disc_q: %f\n", rsqrt_tau, ln_S_K, disc_r, disc_q);
    #endif
    return af_BSOptionPrice(option->type, rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q, option->S, option->K, disc_r, disc_q);
  }
  return AF_UNKNOWN_FLOAT;
}

__device__ __host__
float af_BSOptionDelta(afOptionInfo_t* option) {
  if (af_BSOptionValidate(option)) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_q    = expf(-option->q * option->tau);
    float d_1       = af_BSOptionD_1(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    if (option->type == AF_OPTION_TYPE_CALL) {
      return disc_q * normcdff(d_1);
    }
    if (option->type == AF_OPTION_TYPE_PUT) {
      return -disc_q * normcdff(-d_1);
    }
  }
  return AF_UNKNOWN_FLOAT;
}

__device__ __host__
float af_BSOptionVega(afOptionInfo_t* option) {
  if (af_BSOptionValidate(option)) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_q    = expf(-option->q * option->tau);
    float d_1       = af_BSOptionD_1(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    return option->S * disc_q * normcdff(d_1) * sqrtf(option->tau);
  }
  return AF_UNKNOWN_FLOAT;
}

__device__ __host__
float af_BSOptionGamma(afOptionInfo_t* option) {
  if (af_BSOptionValidate(option)) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_q    = expf(-option->q * option->tau);
    float d_1       = af_BSOptionD_1(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    return disc_q * af_normpdff(d_1) * sqrtf(option->tau) * rsqrt_tau / (option->S * option->sigma);
  }
  return AF_UNKNOWN_FLOAT;
}

__device__ __host__
float af_BSOptionRho(afOptionInfo_t* option) {
  if (af_BSOptionValidate(option)) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_r    = expf(-option->r * option->tau);
    float d_2       = af_BSOptionD_2(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    if (option->type == AF_OPTION_TYPE_CALL) {
      return option->K * option->tau * disc_r * normcdff(d_2);
    } else {
      return -option->K * option->tau * disc_r * normcdff(-d_2);
    }
  }
  return AF_UNKNOWN_FLOAT;
}

__device__ __host__
float af_EuropOptionTheta(afOptionInfo_t* option) {
  if (af_BSOptionValidate(option)) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_r    = expf(-option->r * option->tau);
    float disc_q    = expf(-option->q * option->tau);
    float d_1       = af_BSOptionD_1(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    float d_2       = af_BSOptionD_2(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    float a         = disc_q * option->S * af_normpdff(d_1) * option->sigma * rsqrt_tau / 2.;
    float b;
    float c;
    if (option->type == AF_OPTION_TYPE_CALL) {
      b             = option->r * option->K * disc_r * normcdff(d_2);
      c             = option->q * option->S * disc_q * normcdff(d_1);
      return c - b - a;
    }
    if (option->type == AF_OPTION_TYPE_PUT) {
      b             = option->r * option->K * disc_r * normcdff(-d_2);
      c             = option->q * option->S * disc_q * normcdff(-d_1);
      return b - c - a;
    }
  }
  return AF_UNKNOWN_FLOAT;
}


__device__ __host__
float af_BSOptionImpliedSigma(afOptionInfo_t* option, float min_sigma, float max_sigma, float tol, int max_iter) {
  if (af_BSOptionValidate(option)) {
    float rsqrt_tau   = rsqrt(option->tau);
    float ln_S_K      = logf(option->S / option->K);
    float disc_r      = expf(-option->r * option->tau);
    float disc_q      = expf(-option->q * option->tau);
    float a           = min_sigma;
    float b           = max_sigma;
    float diff        = tol + 1.;
    float value;
    float mid;
    if (af_BSOptionPrice(option->type, rsqrt_tau, option->tau, b, ln_S_K, option->r, option->q, option->S, option->K, disc_r, disc_q) > option->price) {
        // The volatility was above the max value given.
        return AF_UNKNOWN_FLOAT;
    }
    if(af_BSOptionPrice(option->type, rsqrt_tau, option->tau, a, ln_S_K, option->r, option->q, option->S, option->K, disc_r, disc_q) < option->price) {
        // The volatility was below the min value given.
        return AF_UNKNOWN_FLOAT;
    }
    // Run a simple bisection method.
    for (int i = 0; i < max_iter && diff > tol; i++) {
        mid = (a + b) / 2;
        value = af_BSOptionPrice(option->type, rsqrt_tau, option->tau, mid, ln_S_K, option->r, option->q, option->S, option->K, disc_r, disc_q);
        diff = abs(value - option->price);
        if (value > option->price) {
            b = mid;
        } else {
            a = mid;
        }
    }
    return mid;
  }
  return AF_UNKNOWN_FLOAT;
}
