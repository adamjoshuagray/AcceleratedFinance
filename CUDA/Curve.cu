
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


afTimeCurve_t* af_TimeCurveAdd(float a, afTimeCurve_t* x) {
  afTimeCurve_t* new_curve = (afTimeCurve_t*)malloc(sizeof(afTimeCurve_t));
  new_curve->pairs          = (afTimeCurvePair_t*)malloc(x->count * sizeof(afTimeCurvePair_t));
  new_curve->count          = x->count;
  for(int i = 0; i < x->count; i++) {
    new_curve->pairs[i].time  = x->pairs[i].time;
    new_curve->pairs[i].value = a + x->pairs[i].value;
  }
  return new_curve;
}

void af_TimeCurveAddInPlace(float a, afTimeCurve_t* x) {
  for(int i = 0; i < x->count; i++) {
    x->pairs[i].value += a;
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
  // We can interpolate and (extrapolate) so long as the time
  // is greater than the time for the first knot in the curve.
  if (af_TimeCurveFirstTime(x) <= t) {
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
  // We can only interpolate between the knots of the curve.
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
    return (x->pairs[a].value * b_diff + x->pairs[b].value * a_diff) / t_diff;
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

__host__
cudaError_t af_TimeCurveCopyToDevice(afTimeCurve_t* src, afTimeCurve_t** dst) {
  void* dev_mem;
  cudaError_t result;
  // We attempt to put a whole curve in a contiguous block of memory.
  result      = cudaMalloc(&dev_mem, sizeof(afTimeCurve_t) + src->count * sizeof(afTimeCurvePair_t));
  if (result != cudaSuccess) {
    return result;
  }
  afTimeCurvePair_t* host_pairs = src->pairs;
  src->pairs = (afTimeCurvePair_t*) dev_mem + sizeof(afTimeCurve_t);
  if (host_pairs == (afTimeCurvePair_t*) src + sizeof(afTimeCurve_t)) {
    // We can do an contiguous copy.
    result      = cudaMemcpy(dev_mem, src, sizeof(afTimeCurve_t) + src->count * sizeof(afTimeCurvePair_t), cudaMemcpyHostToDevice);
    src->pairs  = host_pairs;
    if (result != cudaSuccess) {
      // Try to free the allocated memory.
      cudaFree(dev_mem);
      return result;
    }
    (*dst) = (afTimeCurve_t*) dev_mem;
    return result;
  } else {
    // We have to do a non-contiguous copy.
    result      = cudaMemcpy(dev_mem, src, sizeof(afTimeCurve_t), cudaMemcpyHostToDevice);
    src->pairs  = host_pairs;
    if (result != cudaSuccess) {
      // Try to free the allocated memory.
      cudaFree(dev_mem);
      return result;
    }
    result      = cudaMemcpy(dev_mem + sizeof(afTimeCurve_t), src->pairs, src->count * sizeof(afTimeCurvePair_t), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      // Try and free the allocated memory.
      cudaFree(dev_mem);
      return result;
    }
    (*dst) = (afTimeCurve_t*) dev_mem;
    return result;
  }
}

__host__
cudaError_t af_TimeCurveCopyToHost(afTimeCurve_t* src, afTimeCurve_t** dst, bool assume_contiguous, int count) {
  void* host_mem;
  cudaError_t result;
  // If it's not contiguous then we'll assume it has a count of 0
  // which means we'll only download the curve struct.
  // From there we'll be able to download the correct array for the pairs.
  if (!assume_contiguous) {
    return af_TimeCurveCopyToHost(src, dst, true, 0);
  }
  // We try and pull down the whole block of memory in one go.
  host_mem  = malloc(sizeof(afTimeCurve_t) + count * sizeof(afTimeCurvePair_t));
  // We don't check that it's not NULL because the next operation will do
  // that for us an give us a cudaError_t return code.
  result    = cudaMemcpy(host_mem, src, sizeof(afTimeCurve_t) + count * sizeof(afTimeCurvePair_t), cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    // Try and free host allocated memory
    free(host_mem);
    return result;
  }
  afTimeCurve_t* host_curve = (afTimeCurve_t*) host_mem;
  // Check that the memory copied actually makes sense.
  if (host_curve->pairs == src + sizeof(afTimeCurve_t) && host_curve->count == count) {
    // Copy worked!
    host_curve->pairs = (afTimeCurvePair_t*) host_curve + sizeof(afTimeCurve_t);
    (*dst) = host_curve;
    return result;
  } else if (host_curve->pairs == src + sizeof(afTimeCurve_t) && host_curve->count < count ) {
    // We over allocated on the host. Lets deallocate some memory.
    afTimeCurve_t* new_host_curve = realloc(host_curve, sizeof(afTimeCurve_t) + host_curve->count * sizeof(afTimeCurvePair_t));
    if (new_host_curve == NULL) {
      // Try and free host allocated memory.
      free(host_mem);
      return cudaErrorUnknown;
    }
    host_curve->pairs = (afTimeCurvePair_t*) host_curve + sizeof(afTimeCurve_t);
    (*dst) = new_host_curve;
    return result;
  } else if (host_curve->pairs == src + sizeof(afTimeCurve_t) && host_curve->count > count) {
    // Extend the memory allocated a bit because we under allocated on the host.
    afTimeCurve_t* new_host_curve = realloc(host_curve, sizeof(afTimeCurve_t) + host_curve->count * sizeof(afTimeCurvePair_t));
    if (new_host_curve == NULL) {
      free(host_mem);
      return cudaErrorUnknown;
    }
    int offset =
  } else {
    // We copied memory that was unrelated. Lets reallocate what we have and try again.
  }
}
