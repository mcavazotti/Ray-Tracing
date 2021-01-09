#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <curand_kernel.h>

#include <cmath>
#include <cstdlib>
//#include <limits>
//#include <memory>

// usings

// constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef REC_MAX_DEPTH
#define REC_MAX_DEPTH 50
#endif
const float infinity = INFINITY;

// utiliy functions
__host__ __device__ inline float degree_to_radian(float degrees) {
  return degrees * M_PI / 180;
}

__host__ inline float random_float() { return rand() / (float(RAND_MAX) + 1); }

__device__ inline float random_float(curandState *localRandState) {
  return curand_uniform(localRandState);
}

__host__ inline float random_float(float min, float max) {
  return min + (max - min) * random_float();
}

__device__ inline float random_float(float min, float max,
                                     curandState *localRandState) {
  return min + (max - min) * curand_uniform(localRandState);
}

__host__ __device__ inline float clamp(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

// commom headers
#include "ray.h"

#endif