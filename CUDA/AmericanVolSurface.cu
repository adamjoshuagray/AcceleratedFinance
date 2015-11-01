
#include "AmericanVolSurface.cuh"
#include "cublas_v2.h"

struct fdOptions_t {
    /**
     * The maximum stock price.
     *
     * We can't solve to an infinite stock price
     * so we much choose a maximum stock price.
     * This price should be large with respect
     */
    float S_max;

    /**
     * The number of steps to use in the division of
     * the stock price.
     */
    int S_n;
    /**
     * The number of time divisions to use.
     */
    int tau_n;
}


__global__
void set_put_boundary(float K, float* boundary, fdOptions_t* options) {

}

__global__
void set_put_boundary(float K, float* boundary, fdOptions_t* options) {

}

__device__
float am_fd_put(float S, float K, float r, float sigma, float tau, fdOptions_t* options, cublasHandle_t* handle) {
    // We need to set up the linear system to solve
    float dS    = options->S_max / options->S_n;
    float dtau  = tau / options->tau_n;
}