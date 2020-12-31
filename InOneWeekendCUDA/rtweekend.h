#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

// usings
using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// utiliy functions
__host__ __device__ inline double degree_to_radian(double degrees) {
  return degrees * pi / 180;
}

/*__host__ __device__ inline double random_double() {
  return rand() / (double(RAND_MAX) + 1);
}

__host__ __device__ inline double random_double(double min, double max) {
  return min + (max - min) * random_double();
}

__host__ __device__ inline double clamp(double x, double min, double max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}*/

// commom headers
//#include "ray.h"

#endif