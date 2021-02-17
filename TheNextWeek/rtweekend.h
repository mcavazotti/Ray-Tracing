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
inline double degree_to_radian(double degrees) { return degrees * pi / 180; }

inline double random_double() { return rand() / (double(RAND_MAX) + 1); }

inline double random_double(double min, double max) {
  return min + (max - min) * random_double();
}

inline double clamp(double x, double min, double max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}
inline int random_int(int min, int max) {
  return static_cast<int>(random_double(min, max + 1));
}
// commom headers
#include "ray.h"

#endif