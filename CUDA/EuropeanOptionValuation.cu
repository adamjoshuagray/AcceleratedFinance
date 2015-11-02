
#include "EuropeanOptionValuation.h"
#include "math.h"

__device__
float af_EuroOptionD_1(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q) {
  float a = 1. / sigma * rsqrt_tau;
  float b = ln_S_K + (r - q + pow(sigma, 2.) / 2.) * tau;
  float d_1 = a * b;
  return d_1;
}

__device__
float af_EuroOptionD_2(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q) {
  float a = 1. / sigma * rsqrt_tau;
  float b = ln_S_K + (r - q - pow(sigma, 2.) / 2.) * tau;
  float d_1 = a * b;
  return d_1;
}

__device__
float af_EuroOptionPrice(afOptionType_t type, float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float q, float S, float K, float disc_r, float disc_q) {
  float d_1 = af_EuroOptionD_1(rsqrt_tau, tau, sigma, ln_S_K, r, q);
  float d_2 = af_EuroOptionD_2(rsqrt_tau, tau, sigma, ln_S_K, r, q);
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
  return AF_UNKNOWN;
}

__device__
float af_EuroOptionPrice(afOptionInfo_t* option) {
  if (option->style == AF_OPTION_STYLE_EUROPEAN) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_r    = expf(-option->r * option->tau);
    float disc_q    = expf(-option->q * option->tau);
    return af_EuroOptionPrice(option->type, rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q, option->S, option->K, disc_r, disc_q);
  }
  return AF_UNKNOWN;
}

__device__
float af_EuroOptionDelta(afOptionInfo_t* option) {
  if (option->style == AF_OPTION_STYLE_EUROPEAN) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_q    = expf(-option->q * option->tau);
    float d_1       = af_EuroOptionD_1(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    if (option->type == AF_OPTION_TYPE_CALL) {
      return disc_q * normcdff(d_1);
    }
    if (option->type == AF_OPTION_TYPE_PUT) {
      return -disc_q * normcdff(-d_1);
    }
  }
  return AF_UNKNOWN;
}

__device__
float af_EuroOptionVega(afOptionInfo_t* option) {
  if (option->style == AF_OPTION_STYLE_EUROPEAN) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_q    = expf(-option->q * option->tau);
    float d_1       = af_EuroOptionD_1(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    return S * disc_q * normcdff(d_1) * sqrtf(option->tau);
  }
  return AF_UNKNOWN;
}

__device__
float af_EuroOptionGamma(afOptionInfo_t* option) {
  if (option->style == AF_OPTION_STYLE_EUROPEAN) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_q    = expf(-option->q * option->tau);
    float d_1       = af_EuroOptionD_1(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    return disc_q * normpdff(d_1) * sqrtf(option->tau) * rsqrt_tau / (S * option->sigma);
  }
  return AF_UNKNOWN;
}

__device__
float af_EuroOptionRho(afOptionInfo_t* option) {
  if (option->style == AF_OPTION_STYLE_EUROPEAN) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_r    = expf(-option->r * option->tau);
    float d_2       = af_EuroOptionD_2(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    if (option->type == AF_OPTION_TYPE_CALL) {
      return option->K * option->tau * disc_r * normcdff(d_2);
    } else {
      return -option->K * option->tau * disc_r * normcdff(-d_2);
    }
  }
  return AF_UNKNOWN;
}

__device__
float af_EuropOptionTheta(afOptionInfo_t* option) {
  if (option->style == AF_OPTION_STYLE_EUROPEAN) {
    float rsqrt_tau = rsqrt(option->tau);
    float ln_S_K    = logf(option->S / option->K);
    float disc_r    = expf(-option->r * option->tau);
    float disc_q    = expf(-option->q * option->tau);
    float d_1       = af_EuroOptionD_1(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    float d_2       = af_EuroOptionD_2(rsqrt_tau, option->tau, option->sigma, ln_S_K, option->r, option->q);
    float a         = disc_q * S * normpdff(d_1) * option->sigma * rsqrt_tau / 2.;
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
  return AF_UNKNOWN;
}


__device__
float af_EuroOptionImpliedSigma(afOptionInfo_t* option, float min_sigma, float max_sigma, float tol, int max_iter) {
  if (option->style == AF_OPTION_STYLE_EUROPEAN) {
    float rsqrt_tau   = rsqrt(option->tau);
    float ln_S_K      = logf(option->S / option->K);
    float disc_r      = expf(-option->r * option->tau);
    float disc_q      = expf(-option->q * option->tau);
    float a           = min_sigma;
    float b           = max_sigma;
    float diff;
    float value;
    float mid;
    if (af_EuroOptionPrice(rsqrt_tau, option->tau, b, ln_S_K, option->r, option->q, option->S, option->K, disc_r, disc_q) > option->price) {
        // The volatility was above the max value given.
        return AF_UNKNOWN_SIGMA;
    }
    if(af_EuroOptionPrice(rsqrt_tau, option->tau, a, ln_S_K, option->r, option->q, option->S, option->K, disc_r, disc_q) < option->price) {
        // The volatility was below the min value given.
        return AF_UNKNOWN_SIGMA;
    }
    for (int i = 0; i < max_iter && diff > tol; i++) {
        mid = (a + b) / 2;
        value = af_EuroOptionPrice(rsqrt_tau, option->tau, mid, ln_S_K, option->r, option->q, option->S, option->K, disc_r, disc_q);
        diff = abs(value - option->price);
        if (value > option->price) {
            b = mid;
        } else {
            a = mid;
        }
    }
    return mid;
  }
  return AF_UNKNOWN;
}
