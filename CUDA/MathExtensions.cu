
#include "math.h"

__device__
float af_normpdff(float x) {
  return rsqrtf(2. * M_PI) * exp(-__powf(x,2.) / 2.);
}
