#ifndef VEC3_H
#define VEC3_H

#include <curand_kernel.h>

#include <cmath>
#include <iostream>

#include "rtweekend.h"

using std::sqrt;

class vec3 {
 public:
  __host__ __device__ vec3() : e{0, 0, 0} {}
  __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

  __host__ inline static vec3 random() {
    return vec3(random_float(), random_float(), random_float());
  }
  __device__ inline static vec3 random(curandState *localRandState) {
    return vec3(random_float(localRandState), random_float(localRandState),
                random_float(localRandState));
  }
  __host__ inline static vec3 random(float min, float max) {
    return vec3(random_float(min, max), random_float(min, max),
                random_float(min, max));
  }

  __device__ inline static vec3 random(float min, float max,
                                       curandState *localRandState) {
    return vec3(random_float(min, max, localRandState),
                random_float(min, max, localRandState),
                random_float(min, max, localRandState));
  }

  __host__ __device__ float x() const { return e[0]; }
  __host__ __device__ float y() const { return e[1]; }
  __host__ __device__ float z() const { return e[2]; }

  __host__ __device__ vec3 operator-() const {
    return vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ float operator[](int i) const { return e[i]; }
  __host__ __device__ float &operator[](int i) { return e[i]; }

  __host__ __device__ vec3 &operator+=(const vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  __host__ __device__ vec3 &operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }

  __host__ __device__ vec3 &operator/=(const float t) { return *this *= 1 / t; }

  __host__ __device__ float length() const { return sqrt(length_squared()); }

  __host__ __device__ float length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }

 public:
  float e[3];
};

typedef vec3 point3;  // 3D point
typedef vec3 color;   // RGB color

inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
  return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
  return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
  return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
              u.e[2] * v.e[0] - u.e[0] * v.e[2],
              u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ vec3 reflect(const vec3 &v, const vec3 &n) {
  return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ vec3 refract(const vec3 &uv, const vec3 &n,
                                 float etai_over_etat) {
  float cos_theta = dot(-uv, n);
  vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
  vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
  return r_out_perp + r_out_parallel;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) { return v / v.length(); }

__host__ vec3 random_in_unit_sphere() {
  while (true) {
    vec3 p = vec3::random(-1, 1);
    if (p.length_squared() >= 1.0f) continue;
    return p;
  }
}

__device__ vec3 random_in_unit_sphere(curandState *localRandState) {
  while (true) {
    vec3 p = vec3::random(-1, 1, localRandState);
    if (p.length_squared() >= 1.0f) continue;
    return p;
  }
}

__host__ vec3 random_unit_vector() {
  float a = random_float(0, 2.0f * M_PI);
  float z = random_float(-1, 1);
  float r = sqrt(1 - z * z);
  return vec3(r * cos(a), r * sin(a), z);
}

__device__ vec3 random_unit_vector(curandState *localRandState) {
  float a = random_float(0, 2.0f * M_PI);
  float z = random_float(-1, 1);
  float r = sqrt(1 - z * z);
  return vec3(r * cos(a), r * sin(a), z);
}

__host__ vec3 random_in_hemisfere(const vec3 &normal) {
  vec3 in_unit_sphere = random_in_unit_sphere();
  if (dot(in_unit_sphere, normal) > 0.0f)
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}

__device__ vec3 random_in_hemisfere(const vec3 &normal, curandState *localRandState) {
  vec3 in_unit_sphere = random_in_unit_sphere(localRandState);
  if (dot(in_unit_sphere, normal) > 0.0f)
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}

__host__ vec3 random_in_unit_disk() {
  while (true) {
    vec3 p = vec3(random_float(-1, 1), random_float(-1, 1), 0);
    if (p.length_squared() >= 1) continue;
    return p;
  }
}

__device__ vec3 random_in_unit_disk(curandState *localRandState) {
  while (true) {
    vec3 p = vec3(random_float(-1, 1, localRandState),
                  random_float(-1, 1, localRandState), 0);
    if (p.length_squared() >= 1.f) continue;
    return p;
  }
}

#endif