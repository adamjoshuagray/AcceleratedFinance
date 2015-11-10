
#include "Options.h"

__device__ __host__
void af_OptionInfoDelete(afOptionInfo_t* option) {
  if (option->r_curve != NULL) {
    free(option->r_curve);
  }
  if (option->q_curve != NULL) {
    free(option->q_curve);
  }
  if (option->sigma_curve != NULL) {
    free(option->sigma_curve);
  }
  free(option);
}

__host__
afCUDACopyResult_t af_OptionCopyToDevice(afOptionInfo_t* src, afOptionInfo_t** dst) {
  // FIXME: We can have resource leakage if this fails halfway through.
  void* dev_mem;
  cudaError_t result;
  result      = cudaMalloc(&dev_mem, sizeof(afOptionInfo_t));
  if (result != cudaSuccess) {
    return result;
  }
  if (src->r_style == AF_OPTION_YIELD_STYLE_CURVE) {
    // Copy
    void* dev_mem_r;
  }
  if (src->q_style == AF_OPTION_DIVIDEND_STYLE_CURVE) {
    // Copy
  }
  if (src->sigma_style == AF_OPTION_SIGMA_STYLE_CURVE) {
    // Copy
  }
  result      = cudaMemcpy(dev_mem, src, sizeof(afOptionInfo_t));
}
