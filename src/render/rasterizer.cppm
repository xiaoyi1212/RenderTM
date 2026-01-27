module;

#include "../prelude.hpp"

module render;

auto Rasterizer::clip_lerp(const ClipVertex& a, const ClipVertex& b, const double t) -> ClipVertex
{
    const Vec3 view{
        std::lerp(a.view.x, b.view.x, t),
        std::lerp(a.view.y, b.view.y, t),
        std::lerp(a.view.z, b.view.z, t)
    };
    const Vec3 world{
        std::lerp(a.world.x, b.world.x, t),
        std::lerp(a.world.y, b.world.y, t),
        std::lerp(a.world.z, b.world.z, t)
    };
    Vec3 normal{
        std::lerp(a.normal.x, b.normal.x, t),
        std::lerp(a.normal.y, b.normal.y, t),
        std::lerp(a.normal.z, b.normal.z, t)
    };
    normal = normal.normalize();
    const float visibility = std::lerp(a.sky_visibility, b.sky_visibility, static_cast<float>(t));
    return {view, world, normal, visibility};
}

auto Rasterizer::clip_triangle_to_near_plane(double near_plane,
                                             std::span<const ClipVertex> input,
                                             std::span<ClipVertex> output) const -> size_t
{
    if (input.empty() || output.empty())
    {
        return 0;
    }
    size_t out_count = 0;
    ClipVertex prev = input[input.size() - 1];
    bool prev_inside = prev.view.z >= near_plane;

    for (size_t i = 0; i < input.size(); ++i)
    {
        const ClipVertex cur = input[i];
        const bool cur_inside = cur.view.z >= near_plane;

        if (cur_inside)
        {
            if (!prev_inside)
            {
                const double t = (near_plane - prev.view.z) / (cur.view.z - prev.view.z);
                if (out_count < output.size())
                {
                    output[out_count++] = clip_lerp(prev, cur, t);
                }
            }
            if (out_count < output.size())
            {
                output[out_count++] = cur;
            }
        }
        else if (prev_inside)
        {
            const double t = (near_plane - prev.view.z) / (cur.view.z - prev.view.z);
            if (out_count < output.size())
            {
                output[out_count++] = clip_lerp(prev, cur, t);
            }
        }

        prev = cur;
        prev_inside = cur_inside;
    }

    return out_count;
}

auto Rasterizer::edge_function(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c) -> float
{
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

auto Rasterizer::draw_shaded_triangle(float* zbuffer, LinearColor* sample_ambient,
                                      LinearColor* sample_direct_sun, LinearColor* sample_direct_moon,
                                      float* shadow_mask_sun, float* shadow_mask_moon,
                                      Vec3* sample_normals, LinearColor* sample_albedo, float* sample_ao,
                                      Vec3* world_positions, uint32_t* world_stamp, uint32_t frame_index,
                                      size_t width, size_t height,
                                      const ScreenVertex& v0, const ScreenVertex& v1, const ScreenVertex& v2,
                                      const Vec3& wp0, const Vec3& wp1, const Vec3& wp2,
                                      const Vec3& n0, const Vec3& n1, const Vec3& n2,
                                      const float vis0, const float vis1, const float vis2,
                                      const Terrain& terrain,
                                      const LightingEngine& lighting,
                                      const ShadingContext& ctx,
                                      const float jitter_x, const float jitter_y,
                                      const std::array<Vec3, 2>& lights_right_scaled,
                                      const std::array<Vec3, 2>& lights_up_scaled,
                                      const std::array<BlueNoise::Shift, 2>& shadow_shift_u,
                                      const std::array<BlueNoise::Shift, 2>& shadow_shift_v) const -> void
{
    float min_x = std::min({v0.x, v1.x, v2.x});
    float max_x = std::max({v0.x, v1.x, v2.x});
    float min_y = std::min({v0.y, v1.y, v2.y});
    float max_y = std::max({v0.y, v1.y, v2.y});

    const int x0 = std::max(0, static_cast<int>(std::floor(min_x)));
    const int x1 = std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(max_x)));
    const int y0 = std::max(0, static_cast<int>(std::floor(min_y)));
    const int y1 = std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(max_y)));

    const float area = edge_function(v0, v1, v2);
    if (area == 0.0f) return;

    const bool area_positive = area > 0.0f;
    const float inv_area = 1.0f / area;
    const float inv_z0 = 1.0f / v0.z;
    const float inv_z1 = 1.0f / v1.z;
    const float inv_z2 = 1.0f / v2.z;

    const float w0_a = v2.y - v1.y;
    const float w0_b = v1.x - v2.x;
    const float w0_c = v1.y * v2.x - v1.x * v2.y;

    const float w1_a = v0.y - v2.y;
    const float w1_b = v2.x - v0.x;
    const float w1_c = v2.y * v0.x - v2.x * v0.y;

    const float w2_a = v1.y - v0.y;
    const float w2_b = v0.x - v1.x;
    const float w2_c = v0.y * v1.x - v0.x * v1.y;

    const float start_x = static_cast<float>(x0) + 0.5f + jitter_x;
    const float start_y = static_cast<float>(y0) + 0.5f + jitter_y;

    float w0_row = w0_a * start_x + w0_b * start_y + w0_c;
    float w1_row = w1_a * start_x + w1_b * start_y + w1_c;
    float w2_row = w2_a * start_x + w2_b * start_y + w2_c;

    const float w0_a_i = w0_a * inv_z0 * inv_area;
    const float w1_a_i = w1_a * inv_z1 * inv_area;
    const float w2_a_i = w2_a * inv_z2 * inv_area;
    const float w0_b_i = w0_b * inv_z0 * inv_area;
    const float w1_b_i = w1_b * inv_z1 * inv_area;
    const float w2_b_i = w2_b * inv_z2 * inv_area;

    float w0i_row = w0_row * inv_z0 * inv_area;
    float w1i_row = w1_row * inv_z1 * inv_area;
    float w2i_row = w2_row * inv_z2 * inv_area;

    const bool ao_enabled = ctx.ambient_occlusion_enabled;
    const bool direct_lighting_enabled = ctx.direct_lighting_enabled;
    const bool shadows_enabled = ctx.shadows_enabled;
    const double ambient = direct_lighting_enabled ? ctx.ambient_light * ctx.material.ambient : 0.0;
    const bool use_sky = ctx.sky_scale > 0.0f;
    const float sky_scale = ctx.sky_scale;
    const LinearColor albedo = ctx.albedo;
    const LinearColor hemi_ground = ctx.hemi_ground;
    const LinearColor sky_top = ctx.sky_top;
    const Vec3 camera_pos = ctx.camera_pos;
    const double f0 = std::clamp(ctx.material.specular, 0.0, 1.0);
    const bool has_specular = f0 > 0.0;
    const double diffuse_coeff = ctx.material.diffuse;
    const double shininess = ctx.material.shininess;

    for (int y = y0; y <= y1; ++y)
    {
        float w0 = w0_row;
        float w1 = w1_row;
        float w2 = w2_row;
        float w0i = w0i_row;
        float w1i = w1i_row;
        float w2i = w2i_row;

        for (int x = x0; x <= x1; ++x)
        {
            if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f && area_positive) ||
                (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f && !area_positive))
            {
                const float inv_z = w0i + w1i + w2i;
                if (inv_z > 0.0f)
                {
                    const float depth = 1.0f / inv_z;
                    const size_t idx = static_cast<size_t>(y) * width + static_cast<size_t>(x);
                    if (depth < zbuffer[idx])
                    {
                        zbuffer[idx] = depth;
                        const float w0p = w0i / inv_z;
                        const float w1p = w1i / inv_z;
                        const float w2p = w2i / inv_z;
                        const float visibility = ao_enabled
                                                     ? std::clamp(w0p * vis0 + w1p * vis1 + w2p * vis2, 0.0f, 1.0f)
                                                     : 1.0f;
                        Vec3 normal{
                            w0p * n0.x + w1p * n1.x + w2p * n2.x,
                            w0p * n0.y + w1p * n1.y + w2p * n2.y,
                            w0p * n0.z + w1p * n1.z + w2p * n2.z
                        };
                        normal = normal.normalize();

                        Vec3 world{
                            w0p * wp0.x + w1p * wp1.x + w2p * wp2.x,
                            w0p * wp0.y + w1p * wp1.y + w2p * wp2.y,
                            w0p * wp0.z + w1p * wp1.z + w2p * wp2.z
                        };
                        if (world_positions && world_stamp)
                        {
                            world_positions[idx] = world;
                            world_stamp[idx] = frame_index;
                        }

                        LinearColor ambient_color{
                            static_cast<float>(albedo.r * ambient),
                            static_cast<float>(albedo.g * ambient),
                            static_cast<float>(albedo.b * ambient)
                        };
                        if (use_sky)
                        {
                            float sky_t = static_cast<float>((-normal.y) * 0.5 + 0.5);
                            sky_t = std::clamp(sky_t, 0.0f, 1.0f);
                            const LinearColor sky = LinearColor::lerp(hemi_ground, sky_top, sky_t);
                            ambient_color.r = sky.r * sky_scale * albedo.r;
                            ambient_color.g = sky.g * sky_scale * albedo.g;
                            ambient_color.b = sky.b * sky_scale * albedo.b;
                        }
                        ambient_color.r *= visibility;
                        ambient_color.g *= visibility;
                        ambient_color.b *= visibility;

                        LinearColor direct_sun{0.0f, 0.0f, 0.0f};
                        LinearColor direct_moon{0.0f, 0.0f, 0.0f};
                        float shadow_sun = 1.0f;
                        float shadow_moon = 1.0f;
                        if (direct_lighting_enabled)
                        {
                            const Vec3 view_vec{
                                camera_pos.x - world.x,
                                camera_pos.y - world.y,
                                camera_pos.z - world.z
                            };
                            const Vec3 view_dir = view_vec.normalize();

                            auto eval_light = [&](const ShadingContext::DirectionalLightInfo& light,
                                                  const int light_idx, const int shadow_salt,
                                                  LinearColor& out_direct, float& out_shadow) {
                                out_direct = {0.0f, 0.0f, 0.0f};
                                out_shadow = 1.0f;
                                if (light.intensity <= 0.0)
                                {
                                    return;
                                }
                                const double ndotl = std::max(0.0, normal.dot(light.dir));
                                if (ndotl <= 0.0)
                                {
                                    return;
                                }
                                if (shadows_enabled)
                                {
                                    const Vec3 shadow_dir = lighting.jitter_shadow_direction(light.dir,
                                                            lights_right_scaled[light_idx],
                                                            lights_up_scaled[light_idx],
                                                            x, y,
                                                            shadow_shift_u[light_idx],
                                                            shadow_shift_v[light_idx]);
                                    out_shadow = lighting.compute_shadow_factor(terrain, shadow_dir, world, normal);
                                }

                                const Vec3 half_vec = (light.dir + view_dir).normalize();
                                const double vdoth = std::max(0.0, view_dir.dot(half_vec));
                                const double fresnel = schlick_fresnel(vdoth, f0);
                                const double diffuse_scale = std::clamp(1.0 - fresnel, 0.0, 1.0);
                                const double diffuse = ndotl * light.intensity * diffuse_coeff * diffuse_scale;
                                LinearColor light_color{
                                    static_cast<float>(albedo.r * diffuse),
                                    static_cast<float>(albedo.g * diffuse),
                                    static_cast<float>(albedo.b * diffuse)
                                };
                                light_color.r *= light.color.r;
                                light_color.g *= light.color.g;
                                light_color.b *= light.color.b;
                                if (has_specular)
                                {
                                    const double spec_dot = std::max(0.0, normal.dot(half_vec));
                                    double spec = eval_specular_term(spec_dot, vdoth, ndotl,
                                                                     shininess, f0);
                                    spec *= light.intensity;
                                    spec = std::clamp(spec, 0.0, 1.0);
                                    light_color.r += light.color.r * static_cast<float>(spec);
                                    light_color.g += light.color.g * static_cast<float>(spec);
                                    light_color.b += light.color.b * static_cast<float>(spec);
                                }
                                out_direct = light_color;
                            };

                            eval_light(ctx.lights[0], 0, kSunShadowSalt, direct_sun, shadow_sun);
                            eval_light(ctx.lights[1], 1, kMoonShadowSalt, direct_moon, shadow_moon);
                        }

                        sample_ambient[idx] = ambient_color;
                        sample_direct_sun[idx] = direct_sun;
                        sample_direct_moon[idx] = direct_moon;
                        shadow_mask_sun[idx] = shadow_sun;
                        shadow_mask_moon[idx] = shadow_moon;
                        sample_normals[idx] = normal;
                        if (sample_albedo)
                        {
                            sample_albedo[idx] = ctx.albedo;
                        }
                        if (sample_ao)
                        {
                            sample_ao[idx] = visibility;
                        }
                    }
                }
            }

            w0 += w0_a;
            w1 += w1_a;
            w2 += w2_a;
            w0i += w0_a_i;
            w1i += w1_a_i;
            w2i += w2_a_i;
        }

        w0_row += w0_b;
        w1_row += w1_b;
        w2_row += w2_b;
        w0i_row += w0_b_i;
        w1i_row += w1_b_i;
        w2i_row += w2_b_i;
    }
}

auto Rasterizer::render_quad(float* zbuffer, LinearColor* sample_ambient,
                             LinearColor* sample_direct_sun, LinearColor* sample_direct_moon,
                             float* shadow_mask_sun, float* shadow_mask_moon, Vec3* sample_normals,
                             LinearColor* sample_albedo, float* sample_ao,
                             Vec3* world_positions, uint32_t* world_stamp, uint32_t frame_index,
                             size_t width, size_t height,
                             const RenderQuad& quad, const double proj_scale_x, const double proj_scale_y,
                             const Vec3& camera_pos, const ViewRotation& view_rot,
                             const double near_plane,
                             const Terrain& terrain,
                             const LightingEngine& lighting, const ShadingContext& ctx,
                             const float jitter_x, const float jitter_y,
                             const std::array<Vec3, 2>& lights_right_scaled,
                             const std::array<Vec3, 2>& lights_up_scaled,
                             const std::array<BlueNoise::Shift, 2>& shadow_shift_u,
                             const std::array<BlueNoise::Shift, 2>& shadow_shift_v) const -> void
{
    std::array<Vec3, 4> view_space{};
    for (int i = 0; i < 4; ++i)
    {
        Vec3 view{
            quad.v[i].x - camera_pos.x,
            quad.v[i].y - camera_pos.y,
            quad.v[i].z - camera_pos.z
        };
        view_space[i] = rotate_yaw_pitch_cached(view, view_rot.cy, view_rot.sy, view_rot.cp, view_rot.sp);
    }

    const Vec3 ab{
        quad.v[1].x - quad.v[0].x,
        quad.v[1].y - quad.v[0].y,
        quad.v[1].z - quad.v[0].z
    };
    const Vec3 ac{
        quad.v[2].x - quad.v[0].x,
        quad.v[2].y - quad.v[0].y,
        quad.v[2].z - quad.v[0].z
    };
    const Vec3 normal_raw{
        ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z,
        ab.x * ac.y - ab.y * ac.x
    };

    const Vec3 center{
        (quad.v[0].x + quad.v[1].x + quad.v[2].x + quad.v[3].x) * 0.25,
        (quad.v[0].y + quad.v[1].y + quad.v[2].y + quad.v[3].y) * 0.25,
        (quad.v[0].z + quad.v[1].z + quad.v[2].z + quad.v[3].z) * 0.25
    };
    const Vec3 view_vec{
        camera_pos.x - center.x,
        camera_pos.y - center.y,
        camera_pos.z - center.z
    };
    const double facing = normal_raw.x * view_vec.x + normal_raw.y * view_vec.y + normal_raw.z * view_vec.z;
    if (facing <= 0.0)
    {
        return;
    }

    auto project_vertex = [&](const Vec3& view) {
        const double invZ = 1.0 / view.z;
        return ScreenVertex{
            static_cast<float>(view.x * invZ * proj_scale_x + width / 2.0),
            static_cast<float>(view.y * invZ * proj_scale_y + height / 2.0),
            static_cast<float>(view.z)
        };
    };

    auto draw_triangle = [&](const ClipVertex& a, const ClipVertex& b, const ClipVertex& c) {
        const ScreenVertex sv0 = project_vertex(a.view);
        const ScreenVertex sv1 = project_vertex(b.view);
        const ScreenVertex sv2 = project_vertex(c.view);
        draw_shaded_triangle(zbuffer, sample_ambient,
                             sample_direct_sun, sample_direct_moon,
                             shadow_mask_sun, shadow_mask_moon, sample_normals, sample_albedo, sample_ao,
                             world_positions, world_stamp, frame_index,
                             width, height,
                             sv0, sv1, sv2,
                             a.world, b.world, c.world,
                             a.normal, b.normal, c.normal,
                             a.sky_visibility, b.sky_visibility, c.sky_visibility,
                             terrain,
                             lighting,
                             ctx,
                             jitter_x, jitter_y,
                             lights_right_scaled, lights_up_scaled,
                             shadow_shift_u, shadow_shift_v);
    };

    auto draw_clipped = [&](int i0, int i1, int i2) {
        const std::array<ClipVertex, 3> input{{
            {view_space[i0], quad.v[i0], quad.n[i0], quad.sky_visibility[i0]},
            {view_space[i1], quad.v[i1], quad.n[i1], quad.sky_visibility[i1]},
            {view_space[i2], quad.v[i2], quad.n[i2], quad.sky_visibility[i2]}
        }};
        std::array<ClipVertex, 4> clipped{};
        const size_t clipped_count = clip_triangle_to_near_plane(near_plane, input, clipped);
        if (clipped_count < 3)
        {
            return;
        }
        if (clipped_count == 3)
        {
            draw_triangle(clipped[0], clipped[1], clipped[2]);
        }
        else if (clipped_count == 4)
        {
            draw_triangle(clipped[0], clipped[1], clipped[2]);
            draw_triangle(clipped[0], clipped[2], clipped[3]);
        }
    };

    draw_clipped(0, 1, 2);
    draw_clipped(0, 2, 3);
}

auto Rasterizer::should_rasterize_triangle(const Vec3 v0, const Vec3 v1, const Vec3 v2,
                                           const double near_plane) const -> bool
{
    return v0.z >= near_plane || v1.z >= near_plane || v2.z >= near_plane;
}

auto RenderEngine::should_rasterize_triangle(const Vec3 v0, const Vec3 v1, const Vec3 v2) const -> bool
{
    return rasterizer.should_rasterize_triangle(v0, v1, v2, Camera::near_plane);
}

auto RenderEngine::clip_triangle_to_near_plane(const Vec3 v0, const Vec3 v1, const Vec3 v2,
                                               Vec3* out_vertices, const size_t max_vertices) const -> size_t
{
    if (!out_vertices || max_vertices == 0)
    {
        return 0;
    }
    const std::array<ClipVertex, 3> input{{
        {v0, v0, {0.0, 0.0, 0.0}, 0.0f},
        {v1, v1, {0.0, 0.0, 0.0}, 0.0f},
        {v2, v2, {0.0, 0.0, 0.0}, 0.0f}
    }};
    std::array<ClipVertex, 4> clipped{};
    const size_t count = rasterizer.clip_triangle_to_near_plane(Camera::near_plane, input, clipped);
    const size_t out_count = std::min(count, max_vertices);
    for (size_t i = 0; i < out_count; ++i)
    {
        out_vertices[i] = clipped[i].view;
    }
    return out_count;
}
