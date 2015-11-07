
#include "math.h"
#include "Result.h"
#include "Curve.h"

__device__ __host__
afTimeCurve_t* af_TimeCuveSumAxBy(float a, afTimeCurve_t* x, float b, afTimeCurve_t* y) {
  // FIXME - There is a bug if one of the curves has count == 0
  int count     = max(x->count, y->count);
  // need to alloc new curve
  afTimeCurve_t* new_curve  = (afTimeCurve_t*)malloc(sizeof(afTimeCurve_t));
  new_curve->count          = count;
  new_curve->pairs          = (afTimeCurvePair_t*)malloc(count * sizeof(afTimeCurvePair_t));
  if (x->count != 0 && y->count != 0) {
    int j = 0;
    int k = 0;
    for(int i = 0; i < count; i++) {
      new_curve->pairs[i].value = a * x->pairs[j].value + b * y->pairs[k].value;
      if(x->pairs[j].time < y->pairs[k].time) {
        new_curve->pairs[i].time  = x->pairs[k].time;
        k++;
      } else {
        new_curve->pairs[i].time  = x->pairs[j].time;
        j++;
      }
    }
    return new_curve;
  }
  if (x->count != 0) {
    return af_TimeCurveMult(a, x);
  }
  if (y-> count != 0) {
    return af_TimeCurveMult(b, y);
  }
  return NULL;
}

__device__ __host__
afTimeCurve_t* af_TimeCurveMult(float a, afTimeCurve_t* x) {
  afTimeCurve_t* new_curve  = (afTimeCurve_t*)malloc(sizeof(afTimeCurve_t));
  new_curve->pairs          = (afTimeCurvePair_t*)malloc(x->count * sizeof(afTimeCurvePair_t));
  new_curve->count          = x->count;
  for(int i = 0; i < x->count; i++) {
    new_curve->pairs[i].time  = x->pairs[i].time;
    new_curve->pairs[i].value = a * x->pairs[i].value;
  }
  return new_curve;
}

__device__ __host__
void af_TimeCurveMultInPlace(float a, afTimeCurve_t* x) {
  for(int i = 0; i < x->count; i++) {
    x->pairs[i].value *= a;
  }
}

__device__ __host__
time_t af_TimeCurveLastTime(afTimeCurve_t* x) {
  if (x->count != 0) {
    return x->pairs[x->count - 1].time;
  }
  return AF_UNKNOWN_TIME;
}

__device__ __host__
time_t af_TimeCurveFirstTime(afTimeCurve_t* x) {
  if (x->count != 0) {
    return x->pairs[0].time;
  }
  return AF_UNKNOWN_TIME;
}

__device__ __host__
float af_TimeCurveInterpolatePrevious(afTimeCurve_t* x, time_t t) {
  if (af_TimeCurveFirstTime(x) <= t && af_TimeCurveLastTime(x) >= t) {
    int a = 0;
    int b = x->count;
    int m = (a + b) / 2;
    while (b - a > 1) {
      if (x->pairs[m].time < t) {
        a = m;
      }
      if (x->pairs[m].time > t) {
        b = m;
      }
      if (x->pairs[m].time == t) {
        return x->pairs[m].value;
      }
      m = (a + b) / 2;
    }
    return x->pairs[a].value;
  }
  return AF_UNKNOWN_FLOAT;
}

__device__ __host__
float af_TimeCurveInterpolateLinear(afTimeCurve_t* x, time_t t) {
  if (af_TimeCurveFirstTime(x) <= t && af_TimeCurveLastTime(x) >= t) {
    int a = 0;
    int b = x->count;
    int m = (a + b) / 2;
    while (b - a > 1) {
      if (x->pairs[m].time < t) {
        a = m;
      }
      if (x->pairs[m].time > t) {
        b = m;
      }
      if (x->pairs[m].time == t) {
        return x->pairs[m].value;
      }
      m = (a + b) / 2;
    }
    float t_diff  = (int)x->pairs[b].time - (int)x->pairs[a].time;
    float a_diff  = (int)t - (int)x->pairs[a].time;
    float b_diff  = (int)x->pairs[b].time - (int)t;
    return (x->pairs[a].value * a_diff + x->pairs[b].value * b_diff) / t_diff;
  }
  return AF_UNKNOWN_FLOAT;
}

__device__ __host__
float af_TimeCurveInterpolate(afTimeCurve_t* x, time_t t, afInterpolationStyle_t style) {
  if (style == AF_INTERPOLATION_STYLE_PREVIOUS) {
    return af_TimeCurveInterpolatePrevious(x, t);
  }
  if (style == AF_INTERPOLATION_STYLE_LINEAR) {
    return af_TimeCurveInterpolateLinear(x, t);
  }
  return AF_UNKNOWN_FLOAT;
}

__device__ __host__
void af_TimeCurveDelete(afTimeCurve_t* curve) {
  free(curve->pairs);
  free(curve);
}
