#pragma once

#include <cstdint>
#include <cstddef>

struct Vec3
{
    double x, y, z;
};

struct Vec2
{
    double x, y;
};

struct Mat4
{
    double m[4][4];
};

void render_update_array(uint32_t* framebuffer, size_t width, size_t height);
void render_set_paused(bool paused);
bool render_is_paused();
void render_toggle_pause();
void render_set_light_direction(Vec3 dir);
Vec3 render_get_light_direction();
void render_set_light_intensity(double intensity);
double render_get_light_intensity();
void render_set_sun_orbit_enabled(bool enabled);
bool render_get_sun_orbit_enabled();
void render_set_sun_orbit_angle(double angle);
double render_get_sun_orbit_angle();
void render_set_moon_direction(Vec3 dir);
void render_set_moon_intensity(double intensity);
void render_set_sky_top_color(uint32_t color);
uint32_t render_get_sky_top_color();
void render_set_sky_bottom_color(uint32_t color);
uint32_t render_get_sky_bottom_color();
void render_set_sky_light_intensity(double intensity);
double render_get_sky_light_intensity();
void render_set_exposure(double exposure);
double render_get_exposure();
void render_set_taa_enabled(bool enabled);
bool render_get_taa_enabled();
void render_set_taa_blend(double blend);
double render_get_taa_blend();
void render_set_taa_clamp_enabled(bool enabled);
bool render_get_taa_clamp_enabled();
void render_set_gi_enabled(bool enabled);
bool render_get_gi_enabled();
void render_set_gi_strength(double strength);
double render_get_gi_strength();
void render_reset_taa_history();
void render_set_ambient_occlusion_enabled(bool enabled);
void render_set_shadow_enabled(bool enabled);
bool render_get_shadow_factor_at_point(Vec3 world, Vec3 normal, float* out_factor);
bool render_get_terrain_vertex_sky_visibility(int x, int y, int z, int face, int corner, float* out_visibility);
void render_set_camera_position(Vec3 pos);
Vec3 render_get_camera_position();
void render_move_camera(Vec3 delta);
void render_move_camera_local(Vec3 delta);
void render_set_camera_rotation(Vec2 rot);
Vec2 render_get_camera_rotation();
void render_rotate_camera(Vec2 delta);
Vec2 render_project_point(Vec3 world, size_t width, size_t height);
Vec3 render_unproject_point(Vec3 screen, size_t width, size_t height);
Vec2 render_reproject_point(Vec3 world, size_t width, size_t height);
bool render_debug_depth_at_sample(Vec3 v0, Vec3 v1, Vec3 v2, Vec2 p, float* out_depth);
double render_debug_eval_specular(double ndoth, double vdoth, double ndotl,
                                  double shininess, double f0);
Vec3 render_debug_tonemap_reinhard(Vec3 color, double exposure);
Vec3 render_debug_sample_history_bilinear(const Vec3* buffer, size_t width, size_t height, Vec2 screen_coord);
void render_debug_set_sky_colors_raw(uint32_t top, uint32_t bottom);
Mat4 render_debug_get_current_vp();
Mat4 render_debug_get_previous_vp();
Mat4 render_debug_get_inverse_current_vp();
double render_debug_get_taa_sharpen_strength();
double render_debug_get_taa_sharpen_percent();
bool render_debug_shadow_factor_with_frame(Vec3 world, Vec3 normal, Vec3 light_dir,
                                           int pixel_x, int pixel_y, int frame,
                                           float* out_factor);
float render_debug_shadow_filter(const float* mask, const float* depth, const Vec3* normals);
size_t render_debug_get_terrain_block_count();
size_t render_debug_get_terrain_visible_face_count();
size_t render_debug_get_terrain_triangle_count();
bool render_should_rasterize_triangle(Vec3 v0, Vec3 v1, Vec3 v2);
double render_get_near_plane();
size_t render_clip_triangle_to_near_plane(Vec3 v0, Vec3 v1, Vec3 v2, Vec3* out_vertices, size_t max_vertices);
