module;

#include "../prelude.hpp"

export module rasterizer;

import math;
import noise;
import terrain;
import settings;
import lighting;

export struct ClipVertex
{
    Vec3 view;
    Vec3 world;
    Vec3 normal;
    float sky_visibility;
};

export struct ScreenVertex
{
    float x, y, z;
};

export struct ViewRotation
{
    double cy;
    double sy;
    double cp;
    double sp;

    [[nodiscard]]
    static auto from_yaw_pitch(const double yaw, const double pitch) -> ViewRotation
    {
        return {std::cos(yaw), std::sin(yaw), std::cos(pitch), std::sin(pitch)};
    }
};

export struct RasterTarget
{
    float* zbuffer;
    LinearColor* sample_colors;
    LinearColor* sample_direct_sun;
    LinearColor* sample_direct_moon;
    float* shadow_mask_sun;
    float* shadow_mask_moon;
    Vec3* sample_normals;
    LinearColor* sample_albedo;
    float* sample_ao;
    Vec3* world_positions;
    uint32_t* world_stamp;
    uint32_t frame_index;
    size_t width;
    size_t height;
};

export struct RasterInputs
{
    const Terrain& terrain;
    const LightingEngine& lighting;
    const ShadowSettings& shadow_settings;
    const ShadingContext& ctx;
    float jitter_x;
    float jitter_y;
    std::array<Vec3, 2> lights_right_scaled;
    std::array<Vec3, 2> lights_up_scaled;
    std::array<BlueNoise::Shift, 2> shadow_shift_u;
    std::array<BlueNoise::Shift, 2> shadow_shift_v;
};

export struct RasterQuadInput
{
    const RenderQuad& quad;
    double proj_scale_x;
    double proj_scale_y;
    Vec3 camera_pos;
    ViewRotation view_rot;
    double near_plane;
};

struct RasterTriangleInput
{
    ScreenVertex v0;
    ScreenVertex v1;
    ScreenVertex v2;
    Vec3 wp0;
    Vec3 wp1;
    Vec3 wp2;
    Vec3 n0;
    Vec3 n1;
    Vec3 n2;
    float vis0;
    float vis1;
    float vis2;
};

export struct Rasterizer
{
    auto clip_to_near(double near_plane,
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
                        output[out_count++] = lerp_clip(prev, cur, t);
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
                    output[out_count++] = lerp_clip(prev, cur, t);
                }
            }

            prev = cur;
            prev_inside = cur_inside;
        }

        return out_count;
    }

    auto render_quad(const RasterTarget& target,
                     const RasterQuadInput& quad_input,
                     const RasterInputs& inputs) const -> void
    {
        const auto& quad = quad_input.quad;
        const auto proj_scale_x = quad_input.proj_scale_x;
        const auto proj_scale_y = quad_input.proj_scale_y;
        const auto& camera_pos = quad_input.camera_pos;
        const auto& view_rot = quad_input.view_rot;
        const auto near_plane = quad_input.near_plane;
        const auto width = target.width;
        const auto height = target.height;

        std::array<Vec3, 4> view_space{};
        for (int i = 0; i < 4; ++i)
        {
            Vec3 view{
                quad.v[i].x - camera_pos.x,
                quad.v[i].y - camera_pos.y,
                quad.v[i].z - camera_pos.z
            };
            view_space[i] = rotate_cached(view, view_rot.cy, view_rot.sy, view_rot.cp, view_rot.sp);
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
        const Vec3 face_normal{
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
        const double facing_dot = face_normal.x * view_vec.x + face_normal.y * view_vec.y + face_normal.z * view_vec.z;
        if (facing_dot <= 0.0)
        {
            return;
        }

        auto project_vertex = [&](const Vec3& view) {
            const double inv_z = 1.0 / view.z;
            return ScreenVertex{
                static_cast<float>(view.x * inv_z * proj_scale_x + width / 2.0),
                static_cast<float>(-view.y * inv_z * proj_scale_y + height / 2.0),
                static_cast<float>(view.z)
            };
        };

        auto draw_triangle = [&](const ClipVertex& a, const ClipVertex& b, const ClipVertex& c) {
            const ScreenVertex sv0 = project_vertex(a.view);
            const ScreenVertex sv1 = project_vertex(b.view);
            const ScreenVertex sv2 = project_vertex(c.view);
            const RasterTriangleInput tri{
                .v0 = sv0,
                .v1 = sv1,
                .v2 = sv2,
                .wp0 = a.world,
                .wp1 = b.world,
                .wp2 = c.world,
                .n0 = a.normal,
                .n1 = b.normal,
                .n2 = c.normal,
                .vis0 = a.sky_visibility,
                .vis1 = b.sky_visibility,
                .vis2 = c.sky_visibility
            };
            shade_triangle(target, tri, inputs);
        };

        auto draw_clipped = [&](const std::array<int, 3>& idx) {
            const std::array<ClipVertex, 3> input{{
                {view_space[idx[0]], quad.v[idx[0]], quad.n[idx[0]], quad.sky_visibility[idx[0]]},
                {view_space[idx[1]], quad.v[idx[1]], quad.n[idx[1]], quad.sky_visibility[idx[1]]},
                {view_space[idx[2]], quad.v[idx[2]], quad.n[idx[2]], quad.sky_visibility[idx[2]]}
            }};
            std::array<ClipVertex, 4> clipped{};
            const size_t clipped_count = clip_to_near(near_plane, input, clipped);
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

        draw_clipped(std::array<int, 3>{0, 1, 2});
        draw_clipped(std::array<int, 3>{0, 2, 3});
    }

private:
    static auto lerp_clip(const ClipVertex& a, const ClipVertex& b, const double t) -> ClipVertex
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

    static auto rotate_cached(const Vec3& v, double cy, double sy,
                                        double cp, double sp) -> Vec3
    {
        const double x1 = v.x * cy + v.z * sy;
        const double z1 = -v.x * sy + v.z * cy;
        const double y1 = v.y;

        const double y2 = y1 * cp - z1 * sp;
        const double z2 = y1 * sp + z1 * cp;

        return {x1, y2, z2};
    }

    auto shade_triangle(const RasterTarget& target,
                              const RasterTriangleInput& tri,
                              const RasterInputs& inputs) const -> void
    {
        auto* zbuffer = target.zbuffer;
        auto* ambient_buf = target.sample_colors;
        auto* sun_buf = target.sample_direct_sun;
        auto* moon_buf = target.sample_direct_moon;
        auto* sun_shadow = target.shadow_mask_sun;
        auto* moon_shadow = target.shadow_mask_moon;
        auto* normal_buf = target.sample_normals;
        auto* albedo_buf = target.sample_albedo;
        auto* ao_buf = target.sample_ao;
        auto* world_buf = target.world_positions;
        auto* stamp_buf = target.world_stamp;
        const auto frame_index = target.frame_index;
        const auto width = target.width;
        const auto height = target.height;

        const auto& v0 = tri.v0;
        const auto& v1 = tri.v1;
        const auto& v2 = tri.v2;
        const auto& wp0 = tri.wp0;
        const auto& wp1 = tri.wp1;
        const auto& wp2 = tri.wp2;
        const auto& n0 = tri.n0;
        const auto& n1 = tri.n1;
        const auto& n2 = tri.n2;
        const auto vis0 = tri.vis0;
        const auto vis1 = tri.vis1;
        const auto vis2 = tri.vis2;

        const auto& terrain = inputs.terrain;
        const auto& lighting = inputs.lighting;
        const auto& shadow = inputs.shadow_settings;
        const auto& ctx = inputs.ctx;
        const auto jitter_x = inputs.jitter_x;
        const auto jitter_y = inputs.jitter_y;
        const auto& light_right = inputs.lights_right_scaled;
        const auto& light_up = inputs.lights_up_scaled;
        const auto& shadow_u = inputs.shadow_shift_u;
        const auto& shadow_v = inputs.shadow_shift_v;

        auto edge = [&](const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c) {
            return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
        };

        float min_x = std::min({v0.x, v1.x, v2.x});
        float max_x = std::max({v0.x, v1.x, v2.x});
        float min_y = std::min({v0.y, v1.y, v2.y});
        float max_y = std::max({v0.y, v1.y, v2.y});

        const int x0 = std::max(0, static_cast<int>(std::floor(min_x)));
        const int x1 = std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(max_x)));
        const int y0 = std::max(0, static_cast<int>(std::floor(min_y)));
        const int y1 = std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(max_y)));

        const float area = edge(v0, v1, v2);
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

        const auto ao_on = ctx.ambient_occlusion_enabled;
        const auto direct_on = ctx.direct_lighting_enabled;
        const auto shadows_on = ctx.shadows_enabled;
        const auto use_sky = ctx.sky_scale > 0.0f;
        const auto& albedo = ctx.albedo;

        for (int y = y0; y <= y1; ++y, w0_row += w0_b, w1_row += w1_b, w2_row += w2_b,
             w0i_row += w0_b_i, w1i_row += w1_b_i, w2i_row += w2_b_i)
        {
            float w0 = w0_row;
            float w1 = w1_row;
            float w2 = w2_row;
            float w0i = w0i_row;
            float w1i = w1i_row;
            float w2i = w2i_row;

            for (int x = x0; x <= x1; ++x, w0 += w0_a, w1 += w1_a, w2 += w2_a,
                 w0i += w0_a_i, w1i += w1_a_i, w2i += w2_a_i)
            {
                const bool inside = area_positive
                                        ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                                        : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
                if (!inside) continue;

                const float inv_z = w0i + w1i + w2i;
                if (inv_z <= 0.0f) continue;

                const size_t idx = static_cast<size_t>(y) * width + static_cast<size_t>(x);
                const float depth = 1.0f / inv_z;
                if (depth >= zbuffer[idx]) continue;

                zbuffer[idx] = depth;
                const float w0p = w0i / inv_z;
                const float w1p = w1i / inv_z;
                const float w2p = w2i / inv_z;
                const float ao_vis = ao_on
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
                if (world_buf && stamp_buf)
                {
                    world_buf[idx] = world;
                    stamp_buf[idx] = frame_index;
                }

                const double ambient_scale = ctx.ambient_light * ctx.material.ambient;
                LinearColor ambient{
                    static_cast<float>(albedo.r * ambient_scale),
                    static_cast<float>(albedo.g * ambient_scale),
                    static_cast<float>(albedo.b * ambient_scale)
                };
                if (use_sky)
                {
                    float sky_t = static_cast<float>(normal.y * 0.5 + 0.5);
                    sky_t = std::clamp(sky_t, 0.0f, 1.0f);
                    const LinearColor sky = LinearColor::lerp(ctx.hemi_ground, ctx.sky_top, sky_t);
                    ambient.r = sky.r * ctx.sky_scale * albedo.r;
                    ambient.g = sky.g * ctx.sky_scale * albedo.g;
                    ambient.b = sky.b * ctx.sky_scale * albedo.b;
                }
                ambient.r *= ao_vis;
                ambient.g *= ao_vis;
                ambient.b *= ao_vis;

                LinearColor direct_sun{0.0f, 0.0f, 0.0f};
                LinearColor direct_moon{0.0f, 0.0f, 0.0f};
                float shadow_sun = 1.0f;
                float shadow_moon = 1.0f;
                if (direct_on)
                {
                    const auto& material = ctx.material;
                    const double f0 = std::clamp(material.specular, 0.0, 1.0);
                    const bool has_specular = f0 > 0.0;
                    const double diffuse_coeff = material.diffuse;
                    const double shininess = material.shininess;

                    const Vec3 view_vec{
                        ctx.camera_pos.x - world.x,
                        ctx.camera_pos.y - world.y,
                        ctx.camera_pos.z - world.z
                    };
                    const Vec3 view_dir = view_vec.normalize();

                    auto eval_light = [&](const int light_idx) {
                        auto* out_direct = &direct_sun;
                        auto* out_shadow = &shadow_sun;
                        if (light_idx == 1)
                        {
                            out_direct = &direct_moon;
                            out_shadow = &shadow_moon;
                        }

                        *out_direct = {0.0f, 0.0f, 0.0f};
                        *out_shadow = 1.0f;

                        const auto& light = ctx.lights[light_idx];
                        if (light.intensity <= 0.0) return;

                        const double ndotl = std::max(0.0, normal.dot(light.dir));
                        if (ndotl <= 0.0) return;

                        if (shadows_on)
                        {
                            const Vec3 shadow_dir = lighting.jitter_shadow(
                                light.dir,
                                light_right[light_idx],
                                light_up[light_idx],
                                x, y,
                                shadow_u[light_idx],
                                shadow_v[light_idx]
                            );
                            *out_shadow = lighting.shadow_factor(terrain, shadow_dir, world, normal,
                                                                 shadow);
                        }

                        const Vec3 half_vec = (light.dir + view_dir).normalize();
                        const double vdoth = std::max(0.0, view_dir.dot(half_vec));
                        const double fresnel = lighting.fresnel(vdoth, f0);
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
                            double spec = lighting.specular_term(spec_dot, vdoth, ndotl,
                                                                shininess, f0);
                            spec *= light.intensity;
                            spec = std::clamp(spec, 0.0, 1.0);
                            light_color.r += light.color.r * static_cast<float>(spec);
                            light_color.g += light.color.g * static_cast<float>(spec);
                            light_color.b += light.color.b * static_cast<float>(spec);
                        }

                        *out_direct = light_color;
                    };

                    eval_light(0);
                    eval_light(1);
                }

                ambient_buf[idx] = ambient;
                sun_buf[idx] = direct_sun;
                moon_buf[idx] = direct_moon;
                sun_shadow[idx] = shadow_sun;
                moon_shadow[idx] = shadow_moon;
                normal_buf[idx] = normal;
                if (albedo_buf)
                {
                    albedo_buf[idx] = ctx.albedo;
                }
                if (ao_buf)
                {
                    ao_buf[idx] = ao_vis;
                }
            }
        }
    }
};
