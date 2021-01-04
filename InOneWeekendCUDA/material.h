#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "rtweekend.h"

struct hit_record;

__device__ float schlick(float cosine, float ref_idx) {
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
}

class material {
 public:
  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
                                  color& attenuation, ray& scattered,
                                  curandState* localRandState) const = 0;
};

class lambertian : public material {
 public:
  __device__ lambertian(const color& a) : albedo(a) {}

  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
                                  color& attenuation,
                                  ray& scattered, curandState *localRandState) const override {
    vec3 scatter_direction = rec.normal + random_unit_vector(localRandState);
    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  }

 public:
  color albedo;
};

class metal : public material {
 public:
  __device__ metal(const color& a, float f) : albedo(a), fuzz(f < 1.0f ? f : 1.0f) {}

  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
                                 color& attenuation,
                                 ray& scattered, curandState *localRandState) const override {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(localRandState));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }

 public:
  color albedo;
  float fuzz;
};

class dielectric : public material {
 public:
  __device__ dielectric(float ri) : ref_idx(ri) {}

  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
                                  color& attenuation,
                                  ray& scattered, curandState *localRandState) const override {
    attenuation = color(1, 1, 1);
    float etai_over_etat = rec.front_face ? (1.0f / ref_idx) : ref_idx;

    vec3 unit_direction = unit_vector(r_in.direction());

    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1);
    float sin_theta = sqrtf(1.0 - cos_theta * cos_theta);
    if (etai_over_etat * sin_theta > 1.0f) {
      vec3 reflected = reflect(unit_direction, rec.normal);
      scattered = ray(rec.p, reflected);
      return true;
    }

    float reflect_prob = schlick(cos_theta, etai_over_etat);
    if (random_float(localRandState) < reflect_prob) {
      vec3 reflected = reflect(unit_direction, rec.normal);
      scattered = ray(rec.p, reflected);
      return true;
    }

    vec3 refracted = refract(unit_direction, rec.normal, etai_over_etat);
    scattered = ray(rec.p, refracted);
    return true;
  }

 public:
  float ref_idx;
};


#endif