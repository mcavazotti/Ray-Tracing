#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <memory>
#include <vector>

#include "aabb.h"
#include "hittable.h"

using std::make_shared;
using std::shared_ptr;

class hittable_list : public hittable {
 public:
  hittable_list() {}
  hittable_list(shared_ptr<hittable> object) { add(object); }

  void clear() { objects.clear(); }
  void add(shared_ptr<hittable> object) { objects.push_back(object); }

  virtual bool hit(const ray& r, double tmin, double tmax,
                   hit_record& rec) const;
  virtual bool bounding_box(double time0, double time1,
                            aabb& output_box) const override;

 public:
  std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray& r, double t_min, double t_max,
                        hit_record& rec) const {
  hit_record tmp_rec;
  bool hit_anything = false;
  auto closest = t_max;

  for (const auto& object : objects) {
    if (object->hit(r, t_min, closest, tmp_rec)) {
      hit_anything = true;
      if (tmp_rec.t < closest) {
        closest = tmp_rec.t;
        rec = tmp_rec;
      }
    }
  }

  return hit_anything;
}

bool hittable_list::bounding_box(double time0, double time1,
                                 aabb& output_box) const {
  if (objects.empty()) return false;

  aabb tmp_box;
  bool first_box = true;

  for (const auto& object : objects) {
    if (!object->bounding_box(time0, time1, tmp_box)) return false;
    output_box = first_box ? tmp_box : surrounding_box(output_box, tmp_box);
    first_box = false;
  }
  return true;
}

#endif