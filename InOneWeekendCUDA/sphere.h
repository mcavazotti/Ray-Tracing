#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
 public:
  __device__ sphere(){};
  __device__ sphere(point3 cen, float r, material *m)
      : center(cen), radius(r), mat_ptr(m){};

  __device__ virtual bool hit(const ray& r, float t_min, float t_max,
                              hit_record& rec) const;

 public:
  point3 center;
  float radius;
  material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max,
                            hit_record& rec) const {
  vec3 oc = r.origin() - center;
  float a = r.direction().length_squared();
  float half_b = dot(oc, r.direction());
  float c = oc.length_squared() - radius * radius;
  float discriminant = half_b * half_b - a * c;

  if (discriminant > 0.f) {
    float root = sqrtf(discriminant);
    float tmp = (-half_b - root) / a;
    if (tmp < t_max && tmp > t_min) {
      rec.t = tmp;
      rec.p = r.at(rec.t);  
      vec3 outward_normal = (rec.p - center) / radius;
      rec.set_face_normal(r, outward_normal);
      rec.mat_ptr = mat_ptr;
      return true;
    }

    tmp = (-half_b + root) / a;
    if (tmp < t_max && tmp > t_min) {
      rec.t = tmp;
      rec.p = r.at(rec.t);
      vec3 outward_normal = (rec.p - center) / radius;
      rec.set_face_normal(r, outward_normal);
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }
  return false;
}

#endif