
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
