#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

// usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// utiliy functions
inline double degree_to_radian(double degrees) {
    return degrees * pi / 180;
}

// commom headers
#include "ray.h"
#include "vec3.h"


#endif