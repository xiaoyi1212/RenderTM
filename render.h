#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include "vector.h"

struct Object
{
    std::vector<vector> vertices;
    std::vector<int64_t> indices; // 每3个索引组成一个三角形
    vector position; // 物体在世界中的位置
    vector rotation; // 物体自身的旋转
    uint32_t color; // 物体的基础颜色
};

struct Point2D { int x, y; };

struct PixelPoint {
    int x, y;
    double depth; // Z 值
};

struct Vec3
{
    double x, y, z;
};

struct Vec2
{
    double x, y;
};

enum class RenderScene
{
    CubeOnly
};

void render_update_array(uint32_t* framebuffer, size_t width, size_t height);
void render_set_paused(bool paused);
bool render_is_paused();
void render_toggle_pause();
void render_set_rotation(float angle);
float render_get_rotation();
void render_set_scene(RenderScene scene);
RenderScene render_get_scene();
void render_set_light_direction(Vec3 dir);
Vec3 render_get_light_direction();
void render_set_light_intensity(double intensity);
double render_get_light_intensity();
void render_set_sun_orbit_enabled(bool enabled);
bool render_get_sun_orbit_enabled();
void render_set_sun_orbit_angle(double angle);
double render_get_sun_orbit_angle();
void render_set_moon_direction(Vec3 dir);
Vec3 render_get_moon_direction();
void render_set_moon_intensity(double intensity);
double render_get_moon_intensity();
void render_set_sky_top_color(uint32_t color);
uint32_t render_get_sky_top_color();
void render_set_sky_bottom_color(uint32_t color);
uint32_t render_get_sky_bottom_color();
void render_set_sky_light_intensity(double intensity);
double render_get_sky_light_intensity();
void render_set_ambient_occlusion_enabled(bool enabled);
bool render_get_ambient_occlusion_enabled();
void render_set_shadow_enabled(bool enabled);
bool render_get_shadow_enabled();
size_t render_get_shadow_map_resolution();
int render_get_shadow_pcf_kernel();
bool render_get_terrain_top_ao(int x, int z, float out_ao[4]);
bool render_get_shadow_factor_at_point(Vec3 world, Vec3 normal, float* out_factor);
void render_set_camera_position(Vec3 pos);
Vec3 render_get_camera_position();
void render_move_camera(Vec3 delta);
void render_move_camera_local(Vec3 delta);
void render_set_camera_rotation(Vec2 rot);
Vec2 render_get_camera_rotation();
void render_rotate_camera(Vec2 delta);
Vec2 render_project_point(Vec3 world, size_t width, size_t height);
