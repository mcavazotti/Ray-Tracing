#include <omp.h>

#include <iostream>

#include "bvh.h"
#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "moving_sphere.h"
#include "rtweekend.h"
#include "sphere.h"
#include "scene_generator.h"

void create_scene(int scene, hittable_list &world, color &background, point3 &lookfrom, point3 &lookat, double *vfov, double *aperture)
{
  *vfov = 10;
  *aperture = 0;
  background = color(0, 0, 0);

  switch (scene)
  {
  default:
  case 1:
    world = random_scene();
    background = color(0.70, 0.80, 1.00);
    lookfrom = point3(13, 2, 3);
    lookat = point3(0, 0, 0);
    *vfov = 20.0;
    *aperture = 0.1;
    break;

  case 2:
    world = two_spheres();
    background = color(0.70, 0.80, 1.00);
    lookfrom = point3(13, 2, 3);
    lookat = point3(0, 0, 0);
    *vfov = 20.0;
    break;

  case 3:
    world = two_perlin_spheres();
    background = color(0.70, 0.80, 1.00);
    lookfrom = point3(13, 2, 3);
    lookat = point3(0, 0, 0);
    *vfov = 20.0;
    break;
  case 4:
    world = earth();
    background = color(0.70, 0.80, 1.00);
    lookfrom = point3(13, 2, 3);
    lookat = point3(0, 0, 0);
    *vfov = 20.0;
    break;
  case 5:
    world = simple_light();
    background = color(0, 0, 0);
    lookfrom = point3(26, 3, 6);
    lookat = point3(0, 2, 0);
    *vfov = 20.0;
    break;
  case 6:
    world = cornell_box();
    background = color(0, 0, 0);
    lookfrom = point3(278, 278, -800);
    lookat = point3(278, 278, 0);
    *vfov = 40.0;
    break;
  case 7:
    world = cornell_smoke();
    lookfrom = point3(278, 278, -800);
    lookat = point3(278, 278, 0);
    *vfov = 40.0;
    break;

  case 8:
    world = final_scene();
    background = color(0, 0, 0);
    lookfrom = point3(478, 278, -600);
    lookat = point3(278, 278, 0);
    *vfov = 40.0;
    break;
  }
}

color ray_color(const ray &r, const color &background, const hittable &world, int depth)
{
  hit_record rec;

  if (depth <= 0)
    return color(0, 0, 0);

  if (!world.hit(r, 0.001, infinity, rec))
    return background;

  ray scattered;
  color attenuation;
  color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

  if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
    return emitted;

  return emitted + attenuation * ray_color(scattered, background, world, depth - 1);
}

int main(int argc, char *argv[])
{
  if (argc < 5)
  {
    std::cerr << "Missing arguments." << std::endl
              << "Usage: " << argv[0] << " <scene> <w> <h> <samples>" << std::endl;
    exit(-1);
  }

  int scene = atoi(argv[1]);
  int image_width = atoi(argv[2]);
  int image_height = atoi(argv[3]);
  int samples_per_pixel = atoi(argv[4]);
  double aspect_ratio = double(image_width) / double(image_height);
  int max_depth = 50;

  color *pixel_matrix =
      (color *)malloc(image_width * image_height * sizeof(color));
  if (pixel_matrix == NULL)
  {
    std::cerr << "Error allocating image buffer." << std::endl;
    exit(-1);
  }
  // Image

  // omp_set_num_threads(4);
  std::cerr << "max threads " << omp_get_max_threads() << std::endl;
  /*std::cerr << "num threads" << omp_get_num_threads() << std::endl;*/
  std::cout << "P3\n"
            << image_width << ' ' << image_height << "\n255\n";

  // Setup Scene

  hittable_list world;
  color background;
  point3 lookfrom;
  point3 lookat;
  vec3 vup(0, 1, 0);
  double dist_to_focus = 10.0;
  double aperture, vfov;
  create_scene(scene, world, background, lookfrom, lookat, &vfov, &aperture);

  bvh_node world_bvh(world, 0, 1);

  camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture,
             dist_to_focus);

  std::cerr << "Aperture: " << aperture << "\n";
  std::cerr << "Vertical field of View: " << vfov << "\n";

  // Render
  struct timespec start, stop;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
#pragma omp parallel for
  for (int j = image_height - 1; j >= 0; j--)
  {
    /*std::cerr.precision(3);
    std::cerr << "\rRendering: "
              << double(image_height - j) * 100.0 / double(image_height)
              << "%"
              << std::flush;*/
    for (int i = 0; i < image_width; i++)
    {
      color pixel_color = color(0, 0, 0);
      for (int s = 0; s < samples_per_pixel; s++)
      {
        auto u = (i + random_double()) / (image_width - 1);
        auto v = (j + random_double()) / (image_height - 1);
        ray r = cam.get_ray(u, v);
        pixel_color += ray_color(r, background, world_bvh, max_depth);
      }
      pixel_matrix[j * image_width + i] = pixel_color;
      // write_color(std::cout, pixel_color, samples_per_pixel);
    }
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
  double timer_milisecs =
      ((stop.tv_sec * 1000 * 1000 * 1000 + stop.tv_nsec) -
       (start.tv_sec * 1000 * 1000 * 1000 + start.tv_nsec)) /
      (1000 * 1000);
  std::cerr << "Elapsed time " << timer_milisecs << "ms.\n";
  for (int j = image_height - 1; j >= 0; j--)
  {
    for (int i = 0; i < image_width; i++)
    {
      write_color(std::cout, pixel_matrix[j * image_width + i],
                  samples_per_pixel);
    }
  }
  std::cerr << "\nDone.\n";
  return 0;
}
