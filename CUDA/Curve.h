
#ifndef AF_CURVE_H
#define AF_CURVE_H

#include "ctime.h"

typedef struct afCurvePair_t {
  time_t      time;
  float       value;
}

typedef struct afCurve_t {
  int             count;
  afCurvePair_t*  pairs;
} afCurve_t;

#endif // AF_CURVE_H
