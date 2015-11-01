/**
 * EuropeanVolSurface.cu
 *
 * This deals with simple calculations for European options.
 * We try and precompute quite a few values so that this can
 * run a quickly as possible.
 *
 * We assume that there are no dividends for the underlying stock
 * for these calculations.
 */

#include <math.h>
#include <cuda_runtime.h>
#include "Options.h"
#include "EuropeanVolSurface.h"
#include "Result.h"

__device__
float af_Eurod_1(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r) {
    float a = 1 / sigma * rsqrt_tau;
    float b = ln_S_K + (r + pow(sigma, 2) / 2 ) * tau;
    float d_1 = a * b;
    return d_1;
}

__device__
float af_Eurod_2(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r) {
    float a = 1 / sigma * rsqrt_tau;
    float b = ln_S_K + (r - pow(sigma, 2) / 2 ) * tau;
    float d_1 = a * b;
    return d_1;
}


__device__
float af_EurCall(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float S, float K, float disc) {
    float a = normcdff(af_Eurd_1(rsqrt_tau, tau, sigma, ln_S_K, r)) * S;
    float b = normcdff(af_Eurd_2(rsqrt_tau, tau, sigma, ln_S_K, r)) * K * disc;
    float call = a - b;
    return call;
}

__device__
float af_EurPut(float rsqrt_tau, float tau, float sigma, float ln_S_K, float r, float S, float K, float disc) {
    float a = normcdff(-af_Eurd_1(rsqrt_tau, tau, sigma, ln_S_K, r)) * S;
    float b = normcdff(-af_Eurd_2(rsqrt_tau, tau, sigma, ln_S_K, r)) * K * disc;
    float put = b - a;
    return put;
}

__device__
float af_EurCallImpliedVol(float K, float S, float tau, float r, float call, float min_vol, float max_vol, float tol) {
    // We just use a simple bisection method.
    // We'll speed this up if this becomes a bottleneck
    float rsqrt_tau = rsqrt(tau);
    float ln_S_K = logf(S / K);
    float disc = expf(-r * tau);
    float a = min_vol;
    float b = max_vol;
    float diff = 100;
    float value;
    float mid;
    if (af_EurCall(rsqrt_tau, tau, b, ln_S_K, r, S, K, disc) > call) {
        // The volatility was above the max value given.
        return AF_UNKNOWN_SIGMA;
    }
    if(af_EurCall(rsqrt_tau, tau, a, ln_S_K, r, S, K, disc) < call) {
        // The volatility was below the min value given.
        return AF_UNKNOWN_SIGMA;
    }
    for (int i = 0; i < AF_MAX_ITER && diff > tol; i++) {
        mid = (a + b) / 2;
        value = af_EurCall(rsqrt_tau, tau, mid, ln_S_K, r, S, K, disc);
        diff = abs(value - call);
        if (value > call) {
            b = mid;
        } else {
            a = mid;
        }
    }
    return mid;
}


__device__
float af_EurPutImpliedVol(float K, float S, float tau, float r, float put, float min_vol, float max_vol, float tol) {
    // We just use a simple bisection method.
    // We'll speed this up if this becomes a bottleneck
    float rsqrt_tau = rsqrt(tau);
    float ln_S_K = logf(S / K);
    float disc = expf(-r * tau);
    float a = min_vol;
    float b = max_vol;
    float diff = 100;
    float value;
    float mid;
    if (af_EurPut(rsqrt_tau, tau, b, ln_S_K, r, S, K, disc) > put) {
        // The volatility was above the max value given.
        return AF_UNKNOWN_SIGMA;
    }
    if(af_EurPut(rsqrt_tau, tau, a, ln_S_K, r, S, K, disc) < put) {
        // The volatility was below the min value given.
        return AF_UNKNOWN_SIGMA;
    }
    for (int i = 0; i < AF_MAX_ITER && diff > tol; i++) {
        mid = (a + b) / 2;
        value = af_EurPut(rsqrt_tau, tau, mid, ln_S_K, r, S, K, disc);
        diff = abs(value - put);
        if (value > put) {
            b = mid;
        } else {
            a = mid;
        }
    }
    return mid;
}

__global__
void af_EurCalcVolMany(afOptionInfo_t* options, int n_options) {
    // This untangles the index of the array that we should be looking at.
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float min_vol;
    float max_vol;
    float sigma;
    if (index >= n_options) {
        return; // Nothing to do
    }
    afOptionInfo_t info = options[index];
    // Setup the max and min values.
    if (info.sigma != AF_UNKNOWN_SIGMA) {
        min_vol = max(info.sigma - AF_SIGMA_LOWER_BRACKET, 0.);
        max_vol = info.sigma + AF_SIGMA_UPPER_BRACKET;
    } else {
        min_vol = AF_DEFAULT_MIN_SIGMA;
        max_vol = AF_DEFAULT_MAX_SIGMA;
    }
    // Do the actual computation.
    if (info.type == AF_OT_CALL) {
        sigma = af_EurCallImpliedVol(info.K, info.S, info.tau, info.r, info.price, min_vol, max_vol, AF_TOL);
        // Vol has moved a lot, lets try again with default max and min
        if (sigma == AF_UNKNOWN_SIGMA) {
            min_vol = AF_DEFAULT_MIN_SIGMA;
            max_vol = AF_DEFAULT_MAX_SIGMA;
            sigma = af_EurCallImpliedVol(info.K, info.S, info.tau, info.r, info.price, min_vol, max_vol, AF_TOL);
        }
    }
    if (info.type == AF_OT_PUT) {
        sigma = af_EurPutImpliedVol(info.K, info.S, info.tau, info.r, info.price, min_vol, max_vol, AF_TOL);
        // Vol has moved a lot, lets try again with default max and min
        if (sigma == AF_UNKNOWN_SIGMA) {
            min_vol = AF_DEFAULT_MIN_SIGMA;
            max_vol = AF_DEFAULT_MAX_SIGMA;
            sigma = af_EurPutImpliedVol(info.K, info.S, info.tau, info.r, info.price, min_vol, max_vol, AF_TOL);
        }
    }
    options[index].sigma = sigma;
}

__host__
afResult_t af_EurInit(afOptionInfo_t* options, int n_options, void** dev_options) {
    cudaError_t c_res;
    c_res = cudaMalloc(dev_options, sizeof(afOptionInfo_t) * n_options);
    if (c_res != cudaSuccess) {
        return AF_RESULT_ERROR;
    }
    afResult_t result = af_EurUpdate(options, n_options, *dev_options);
    return result;
}

__host__
afResult_t af_EurDeinit(void* dev_options) {
    cudaError_t c_res;
    c_res = cudaFree(dev_options);
    if (c_res != cudaSuccess) {
        return AF_RESULT_ERROR;
    }
    return AF_RESULT_SUCCESS;
}

__host__
afResult_t af_EurUpdate(afOptionInfo_t* options, int n_options, void* dev_options) {
    cudaError_t c_res;
    c_res = cudaMemcpy(dev_options, options, sizeof(afOptionInfo_t) * n_options, cudaMemcpyHostToDevice);
    if (c_res != cudaSuccess) {
        return AF_RESULT_ERROR;
    }
    return AF_RESULT_SUCCESS;
}

__host__
afResult_t af_EurDownload(afOptionInfo_t* options, int n_options, void* dev_options) {
    cudaError_t c_res;
    c_res = cudaMemcpy(options, dev_options, sizeof(afOptionInfo_t) * n_options, cudaMemcpyDeviceToHost);
    if (c_res != cudaSuccess) {
        return AF_RESULT_ERROR;
    }
    return AF_RESULT_SUCCESS;
}
