
#include "Result.h"

bool af_ResultIsUnknownFloat(float x) {
  return isnan(x);
}

bool af_ResultIsUnknownTime(time_t x) {
  return x == AF_UNKNOWN_TIME;
}
