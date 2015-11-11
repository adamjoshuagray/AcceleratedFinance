
#include "Options.h"
#include "Curve.h"

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
cudaError_t af_OptionCopyToDevice(afOptionInfo_t* src, afOptionInfo_t** dst) {
  cudaError_t result;
  void* dev_mem;
  afTimeCurve_t* dev_mem_r      = NULL;
  afTimeCurve_t* dev_mem_q      = NULL;
  afTimeCurve_t* dev_mem_sigma  = NULL;
  // Need to allocate a block of memory on the device.
  result = cudaMalloc(&dev_mem, sizeof(afOptionInfo_t));
  if (result != cudaSuccess) {
    // Something went wrong allocating the memory.
    return result;
  }
  // Try and copy the curves that are needed.
  if (src->r_style == AF_OPTION_YIELD_STYLE_CURVE) {
    result = af_TimeCurveCopyToDevice(src->r_curve, &dev_mem_r);
    if (result != cudaSuccess) {
      // Something went wrong copying the curve.
      // Lets try and back out.
      cudaFree(dev_mem);
      return result;
    }
  }
  if (src->q_style == AF_OPTION_DIVIDEND_STYLE_CURVE) {
    result = af_TimeCurveCopyToDevice(src->q_curve, &dev_mem_q);
    if (result != cudaSuccess) {
      // Something went wrong copying the curve.
      // Lets try and back out.
      cudaFree(dev_mem_r);
      cudaFree(dev_mem);
      return result;
    }
  }
  if (src->sigma_style == AF_OPTION_SIGMA_STYLE_CURVE) {
    result = af_TimeCurveCopyToDevice(src->sigma_curve, &dev_mem_sigma);
    if (result != cudaSuccess) {
      // Something went wrong copying the curve.
      // Lets try and back out.
      cudaFree(dev_mem_r);
      cudaFree(dev_mem_q);
      cudaFree(dev_mem);
      return result;
    }
  }
  // Now lets update the option we want to copy
  // so that it has the device pointers in it
  // when we copy.
  afTimeCurve_t* r_tmp      = src->r_curve;
  src->r_curve              = dev_mem_r;
  afTimeCurve_t* q_tmp      = src->q_curve;
  src->q_curve              = dev_mem_q;
  afTimeCurve_t* sigma_tmp  = src->sigma_curve;
  src->sigma_curve          = dev_mem_sigma;

  result = cudaMemcpy(dev_mem, src, sizeof(afOptionInfo_t), cudaMemcpyHostToDevice);
  // Now lets revert the changes in the option.
  src->r_curve              = r_tmp;
  src->q_curve              = q_tmp;
  src->sigma_curve          = sigma_tmp;
  if (result != cudaSuccess) {
    // Something went wrong copying the memory.
    // Try and free the allocated memory.
    cudaFree(dev_mem_r);
    cudaFree(dev_mem_q);
    cudaFree(dev_mem_sigma);
    cudaFree(dev_mem);
    return result;
  }
  (*dst) = (afOptionInfo_t*) dev_mem;
  return result;
}
