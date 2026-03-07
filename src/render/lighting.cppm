module;

#include "../prelude.hpp"

export module lighting;

import math;
import camera;
import noise;
import terrain;
import settings;
import framebuffer;

export struct Material
{
    uint32_t color;
    double ambient;
    double diffuse;
    double specular;
    double shininess;
};

export struct ShadingContext
{
    LinearColor albedo;
    LinearColor sky_top;
    LinearColor sky_bottom;
    LinearColor hemi_ground;
    float sky_scale;
    Vec3 camera_pos;
    double ambient_light;
    Material material;
    bool direct_lighting_enabled;
    bool ambient_occlusion_enabled;
    bool shadows_enabled;

    struct DirectionalLightInfo
    {
        Vec3 dir;
        double intensity;
        LinearColor color;
        double angular_radius;
    };
    std::array<DirectionalLightInfo, 2> lights;
};

export struct GiHit
{
    Vec3 position;
    Vec3 normal;
    LinearColor albedo;
    float sky_visibility;
};

export struct DirectFrame
{
    size_t width = 0;
    size_t height = 0;
    size_t samples = 0;
    float depth_max = 0.0f;
    bool shadows_on = false;
};

export struct GiFrame
{
    size_t width = 0;
    size_t height = 0;
    double width_d = 0.0;
    double height_d = 0.0;
    float depth_max = 0.0f;
    uint32_t frame_index = 0;
    double jitter_x_d = 0.0;
    double jitter_y_d = 0.0;
    double proj_a = 0.0;
    double proj_b = 0.0;
    int gi_bounces = 0;
    float gi_scale = 0.0f;
    float sky_scale = 0.0f;
    LinearColor sky_top{};
    LinearColor hemi_ground{};
    std::array<ShadingContext::DirectionalLightInfo, 2> lights{};
    Mat4 inv_current_vp = Mat4::identity();
};

export struct LightingEngine
{
    using Light = ShadingContext::DirectionalLightInfo;
    using Lights = std::array<Light, 2>;
    using FloatSpan = std::span<const float>;
    using FloatSpanMut = std::span<float>;
    using VecSpan = std::span<const Vec3>;

    [[nodiscard]]
    static auto fresnel(const double vdoth, const double f0) -> double
    {
        if (f0 <= 0.0)
        {
            return 0.0;
        }
        const double f0_clamped = std::clamp(f0, 0.0, 1.0);
        const double vdoth_clamped = std::clamp(vdoth, 0.0, 1.0);
        const double one_minus = 1.0 - vdoth_clamped;
        const double one_minus2 = one_minus * one_minus;
        const double one_minus4 = one_minus2 * one_minus2;
        const double pow5 = one_minus4 * one_minus;
        return f0_clamped + (1.0 - f0_clamped) * pow5;
    }

    [[nodiscard]]
    static auto spec_norm(const double shininess) -> double
    {
        return (shininess + 8.0) / (8.0 * std::numbers::pi_v<double>);
    }

    [[nodiscard]]
    static auto specular_term(const double ndoth, const double vdoth, const double ndotl,
                              const double shininess, const double f0) -> double
    {
        if (ndoth <= 0.0 || ndotl <= 0.0 || f0 <= 0.0)
        {
            return 0.0;
        }
        const double ndoth_clamped = std::clamp(ndoth, 0.0, 1.0);
        const double ndotl_clamped = std::clamp(ndotl, 0.0, 1.0);
        const double power = std::pow(ndoth_clamped, shininess);
        const double f = fresnel(vdoth, f0);
        return spec_norm(shininess) * power * f * ndotl_clamped;
    }

    [[nodiscard]]
    auto hemi_ground(const LinearColor& base,
                     const Lights& lights,
                     const LightingSettings& lighting) const -> LinearColor
    {
        const double strength = lighting.hemisphere_bounce_strength;
        const LinearColor& bounce_color = lighting.hemisphere_bounce_color;

        double energy = 0.0;
        for (const auto& light : lights)
        {
            if (light.intensity > 0.0)
            {
                const double height_factor = std::clamp(light.dir.y, 0.0, 1.0);
                energy += light.intensity * height_factor;
            }
        }

        const double bounce_factor = std::clamp(energy * strength, 0.0, 1.0);
        if (bounce_factor <= std::numeric_limits<double>::epsilon())
        {
            return base;
        }

        return LinearColor::lerp(base, bounce_color, static_cast<float>(bounce_factor));
    }

    [[nodiscard]]
    auto jitter_shadow(const Vec3& light_dir,
                       const Vec3& right,
                       const Vec3& up,
                       const int px, const int py,
                       const BlueNoise::Shift& shift_u,
                       const BlueNoise::Shift& shift_v) const -> Vec3
    {
        if (right.x == 0.0 && right.y == 0.0 && right.z == 0.0)
        {
            return light_dir;
        }

        const float u = BlueNoise::sample(px, py, shift_u);
        const float v = BlueNoise::sample(px, py, shift_v);

        const size_t ix = static_cast<size_t>(u * 8.0f) & 7;
        const size_t iy = static_cast<size_t>(v * 8.0f) & 7;
        const size_t idx = (iy << 3) | ix;

        const Vec2 sample = disk_samples()[idx];
        return light_dir + (right * sample.x) + (up * sample.y);
    }

    [[nodiscard]]
    auto gi_hit(const Terrain& terrain,
                const Vec3& world_pos, const Vec3& normal, const Vec3& ray_dir,
                const GiSettings& gi, GiHit* out_hit) const -> bool
    {
        if (!out_hit) return false;

        const auto& topo = terrain.topology;
        const auto& cfg = terrain.config;
        const int size = terrain.chunk_size;
        const int max_height = topo.max_height;

        if (size <= 0 || max_height <= 0) return false;

        const double block_size = cfg.block_size;
        const double inv_block = 1.0 / block_size;

        const Vec3 origin_world = world_pos + (normal * gi.ray_bias);

        const double half_block = block_size * 0.5;
        const double start_x = -(static_cast<double>(size) - 1.0) * half_block;

        const Vec3 origin_grid{
            (origin_world.x - start_x + half_block) * inv_block,
            (origin_world.y - cfg.base_y + half_block) * inv_block,
            (origin_world.z - cfg.start_z + half_block) * inv_block
        };

        const Vec3 dir_grid = ray_dir * inv_block;

        if (dir_grid.x == 0.0 && dir_grid.y == 0.0 && dir_grid.z == 0.0) return false;

        int x = static_cast<int>(std::floor(origin_grid.x));
        int y = static_cast<int>(std::floor(origin_grid.y));
        int z = static_cast<int>(std::floor(origin_grid.z));

        if (x < 0 || x >= size || z < 0 || z >= size || y < 0 || y >= max_height)
        {
            return false;
        }

        const auto get_step = [](double d) -> int { return d > 0.0 ? 1 : (d < 0.0 ? -1 : 0); };
        const int step_x = get_step(dir_grid.x);
        const int step_y = get_step(dir_grid.y);
        const int step_z = get_step(dir_grid.z);

        const double inf = std::numeric_limits<double>::infinity();
        const double t_delta_x = step_x != 0 ? 1.0 / std::abs(dir_grid.x) : inf;
        const double t_delta_y = step_y != 0 ? 1.0 / std::abs(dir_grid.y) : inf;
        const double t_delta_z = step_z != 0 ? 1.0 / std::abs(dir_grid.z) : inf;

        const auto get_t_max = [&](double pos, int step, double dir) {
            if (step == 0) return inf;
            const double next_boundary = std::floor(pos) + (step > 0 ? 1.0 : 0.0);
            return (next_boundary - pos) / dir;
        };

        double t_max_x = get_t_max(origin_grid.x, step_x, dir_grid.x);
        double t_max_y = get_t_max(origin_grid.y, step_y, dir_grid.y);
        double t_max_z = get_t_max(origin_grid.z, step_z, dir_grid.z);

        const double max_t_limit = gi.max_distance * inv_block;
        const int max_steps = (size + size + max_height) * 4;
        
        bool is_first_block = true;
        double dist_traveled = 0.0;
        int last_step_axis = -1;

        for (int i = 0; i < max_steps; ++i)
        {
            if (!is_first_block)
            {
                if (const auto block_opt = topo.block_at(terrain.blocks, x, y, z))
                {
                    const VoxelBlock& voxel = block_opt->get();
                    Vec3 hit_normal{0.0, 0.0, 0.0};
                    int face_idx = BlockGeometry::FaceTop;

                    switch (last_step_axis)
                    {
                        case 0: // X-axis hit
                            hit_normal = {-static_cast<double>(step_x), 0.0, 0.0};
                            face_idx = (step_x > 0) ? BlockGeometry::FaceLeft : BlockGeometry::FaceRight;
                            break;
                        case 1: // Y-axis hit
                            hit_normal = {0.0, -static_cast<double>(step_y), 0.0};
                            face_idx = (step_y > 0) ? BlockGeometry::FaceBottom : BlockGeometry::FaceTop;
                            break;
                        case 2: // Z-axis hit
                            hit_normal = {0.0, 0.0, -static_cast<double>(step_z)};
                            face_idx = (step_z > 0) ? BlockGeometry::FaceBack : BlockGeometry::FaceFront;
                            break;
                    }

                    float visibility = 1.0f;
                    if (face_idx >= 0 && face_idx < 6)
                    {
                        const auto& corners = voxel.sky_visibility[static_cast<size_t>(face_idx)];
                        float sum = 0.0f;
                        for (float v : corners) sum += v;
                        visibility = sum * 0.25f;
                    }

                    const double hit_dist_world = dist_traveled * block_size;
                    out_hit->position = origin_world + (ray_dir * hit_dist_world);
                    out_hit->normal = hit_normal;
                    out_hit->albedo = voxel.albedo_linear;
                    out_hit->sky_visibility = std::clamp(visibility, 0.0f, 1.0f);
                    return true;
                }
            }
            
            is_first_block = false;

            if (t_max_x < t_max_y)
            {
                if (t_max_x < t_max_z)
                {
                    x += step_x; dist_traveled = t_max_x;
                    t_max_x += t_delta_x; last_step_axis = 0;
                }
                else
                {
                    z += step_z; dist_traveled = t_max_z;
                    t_max_z += t_delta_z; last_step_axis = 2;
                }
            }
            else
            {
                if (t_max_y < t_max_z)
                {
                    y += step_y; dist_traveled = t_max_y;
                    t_max_y += t_delta_y; last_step_axis = 1;
                }
                else
                {
                    z += step_z; dist_traveled = t_max_z;
                    t_max_z += t_delta_z; last_step_axis = 2;
                }
            }

            if (dist_traveled > max_t_limit) return false;
            if (x < 0 || x >= size || z < 0 || z >= size || y < 0 || y >= max_height) return false;
        }

        return false;
    }

    [[nodiscard]]
    auto shadow_factor(const Terrain& terrain,
                       const Vec3& light_dir, const Vec3& world_pos, const Vec3& normal,
                       const ShadowSettings& shadow) const -> float
    {
        const double ndotl = normal.dot(light_dir);
        if (ndotl <= 0.0)
        {
            return 1.0f;
        }
        
        return shadow_hit(terrain, world_pos, normal, light_dir, shadow) ? 0.0f : 1.0f;
    }

    [[nodiscard]]
    auto shadow_filter(FloatSpan mask, FloatSpan depth,
                       VecSpan normals, const size_t width, const size_t height,
                       const int px, const int py, const float depth_max,
                       const ShadowSettings& shadow) const -> float
    {
        if (width == 0 || height == 0)
        {
            return 0.0f;
        }

        const int ix = std::clamp(px, 0, static_cast<int>(width) - 1);
        const int iy = std::clamp(py, 0, static_cast<int>(height) - 1);
        const size_t center_idx = static_cast<size_t>(iy) * width + static_cast<size_t>(ix);

        const float center_depth = depth[center_idx];
        if (center_depth >= depth_max) return mask[center_idx];

        const Vec3 center_normal = normals[center_idx];
        if (center_normal.dot(center_normal) <= 1e-6) return mask[center_idx];

        float weighted_mask_sum = mask[center_idx] * shadow.filter_center_weight;
        float weight_sum = shadow.filter_center_weight;

        auto accumulate_neighbor = [&](int nx, int ny) {
            if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height)) return;
            
            const size_t idx = static_cast<size_t>(ny) * width + static_cast<size_t>(nx);
            const float neighbor_depth = depth[idx];
            if (neighbor_depth >= depth_max) return;

            const Vec3 neighbor_normal = normals[idx];
            if (neighbor_normal.dot(neighbor_normal) <= 1e-6) return;
            if (std::abs(neighbor_depth - center_depth) > shadow.filter_depth_threshold) return;

            const float dot = static_cast<float>(center_normal.dot(neighbor_normal));
            if (std::clamp(dot, -1.0f, 1.0f) < shadow.filter_normal_threshold) return;
            weighted_mask_sum += mask[idx] * shadow.filter_neighbor_weight;
            weight_sum += shadow.filter_neighbor_weight;
        };

        accumulate_neighbor(ix - 1, iy);
        accumulate_neighbor(ix + 1, iy);
        accumulate_neighbor(ix, iy - 1);
        accumulate_neighbor(ix, iy + 1);

        if (weight_sum <= std::numeric_limits<float>::epsilon()) return mask[center_idx];

        return std::clamp(weighted_mask_sum / weight_sum, 0.0f, 1.0f);
    }

    auto filter_shadows(FloatSpan mask_a, FloatSpan mask_b,
                        FloatSpanMut out_a, FloatSpanMut out_b,
                        FloatSpan depth, VecSpan normals,
                        const size_t width, const size_t height, const float depth_max,
                        const ShadowSettings& shadow) const -> void
    {
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                const size_t center_idx = y * width + x;
                const float center_depth = depth[center_idx];

                const Vec3 center_normal = normals[center_idx];
                const bool is_background = center_depth >= depth_max;
                const bool is_invalid_normal = center_normal.dot(center_normal) <= 1e-6;

                if (is_background || is_invalid_normal)
                {
                    out_a[center_idx] = mask_a[center_idx];
                    out_b[center_idx] = mask_b[center_idx];
                    continue;
                }

                float sum_a = mask_a[center_idx] * shadow.filter_center_weight;
                float sum_b = mask_b[center_idx] * shadow.filter_center_weight;
                float weight_sum = shadow.filter_center_weight;

                auto process_neighbor = [&](size_t idx) {
                    const float neighbor_depth = depth[idx];
                    if (neighbor_depth >= depth_max) return;

                    const Vec3 neighbor_normal = normals[idx];
                    if (neighbor_normal.dot(neighbor_normal) <= 1e-6) return;

                    if (std::abs(neighbor_depth - center_depth) > shadow.filter_depth_threshold) return;

                    const float dot = static_cast<float>(center_normal.dot(neighbor_normal));
                    if (std::clamp(dot, -1.0f, 1.0f) < shadow.filter_normal_threshold) return;

                    const float w = shadow.filter_neighbor_weight;
                    sum_a += mask_a[idx] * w;
                    sum_b += mask_b[idx] * w;
                    weight_sum += w;
                };

                if (x > 0) process_neighbor(center_idx - 1);
                if (x + 1 < width) process_neighbor(center_idx + 1);
                if (y > 0) process_neighbor(center_idx - width);
                if (y + 1 < height) process_neighbor(center_idx + width);

                if (weight_sum <= std::numeric_limits<float>::epsilon())
                {
                    out_a[center_idx] = mask_a[center_idx];
                    out_b[center_idx] = mask_b[center_idx];
                }
                else
                {
                    const float inv_weight = 1.0f / weight_sum;
                    out_a[center_idx] = std::clamp(sum_a * inv_weight, 0.0f, 1.0f);
                    out_b[center_idx] = std::clamp(sum_b * inv_weight, 0.0f, 1.0f);
                }
            }
        }
    }

    auto resolve_direct(const DirectFrame& frame,
                        RenderBuffers& buffers,
                        const ShadowSettings& shadow) const -> void
    {
        if (frame.shadows_on)
        {
            const float* shadow_sun = buffers.shadow_mask_sun.data();
            const float* shadow_moon = buffers.shadow_mask_moon.data();
            if (shadow.filter_enabled)
            {
                const std::span<const float> shadow_sun_span(buffers.shadow_mask_sun.data(), frame.samples);
                const std::span<const float> shadow_moon_span(buffers.shadow_mask_moon.data(), frame.samples);
                const std::span<float> shadow_sun_out_span(buffers.shadow_mask_filtered_sun.data(), frame.samples);
                const std::span<float> shadow_moon_out_span(buffers.shadow_mask_filtered_moon.data(), frame.samples);
                const std::span<const float> depth_span(buffers.zbuffer.data(), frame.samples);
                const std::span<const Vec3> normals_span(buffers.sample_normals.data(), frame.samples);
                filter_shadows(shadow_sun_span, shadow_moon_span,
                               shadow_sun_out_span, shadow_moon_out_span,
                               depth_span, normals_span,
                               frame.width, frame.height, frame.depth_max,
                               shadow);
                shadow_sun = buffers.shadow_mask_filtered_sun.data();
                shadow_moon = buffers.shadow_mask_filtered_moon.data();
            }
            for (size_t i = 0; i < frame.samples; ++i)
            {
                const LinearColor sun = buffers.sample_direct_sun[i];
                const LinearColor moon = buffers.sample_direct_moon[i];
                const LinearColor sun_shadowed = sun * shadow_sun[i];
                const LinearColor moon_shadowed = moon * shadow_moon[i];
                buffers.sample_direct[i] = sun_shadowed + moon_shadowed;
            }
        }
        else
        {
            for (size_t i = 0; i < frame.samples; ++i)
            {
                buffers.sample_direct[i] = buffers.sample_direct_sun[i] + buffers.sample_direct_moon[i];
            }
        }
    }

    auto gi_pass(const GiFrame& frame,
                 RenderBuffers& buffers,
                 const Terrain& terrain,
                 const GiSettings& gi) const -> void
    {
        const size_t shift_count = static_cast<size_t>(frame.gi_bounces) * gi.sample_count * 2;
        std::vector<BlueNoise::Shift> shifts(shift_count);
        for (int bounce = 0; bounce < frame.gi_bounces; ++bounce)
        {
            for (int sample_idx = 0; sample_idx < gi.sample_count; ++sample_idx)
            {
                const int salt_index = bounce * gi.sample_count + sample_idx;
                const int base_salt = gi.noise_salt + salt_index * 2;
                const size_t base_idx = static_cast<size_t>(salt_index) * 2;
                shifts[base_idx + 0] = BlueNoise::shift(static_cast<int>(frame.frame_index), base_salt);
                shifts[base_idx + 1] = BlueNoise::shift(static_cast<int>(frame.frame_index), base_salt + 1);
            }
        }

        for (size_t y = 0; y < frame.height; ++y)
        {
            const double screen_y = static_cast<double>(y) + 0.5 + frame.jitter_y_d;
            for (size_t x = 0; x < frame.width; ++x)
            {
                const size_t idx = y * frame.width + x;
                if (buffers.zbuffer[idx] >= frame.depth_max)
                {
                    continue;
                }
                buffers.sample_indirect[idx] = {0.0f, 0.0f, 0.0f};
                Vec3 normal = buffers.sample_normals[idx];
                const double n_len_sq = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
                if (n_len_sq <= 1e-6)
                {
                    continue;
                }
                const LinearColor albedo = buffers.sample_albedo[idx];

                const double screen_x = static_cast<double>(x) + 0.5 + frame.jitter_x_d;
                const Vec3 world_pos = Camera::screen_to_world(screen_x, screen_y, buffers.zbuffer[idx],
                                                               frame.inv_current_vp, frame.width_d, frame.height_d,
                                                               frame.proj_a, frame.proj_b);
                buffers.world_positions[idx] = world_pos;
                buffers.world_stamp[idx] = frame.frame_index;

                LinearColor gi_sum{0.0f, 0.0f, 0.0f};
                for (int sample_idx = 0; sample_idx < gi.sample_count; ++sample_idx)
                {
                    LinearColor gi_sample{0.0f, 0.0f, 0.0f};
                    Vec3 cur_world = world_pos;
                    Vec3 cur_normal = normal;
                    LinearColor throughput = albedo;
                    bool hit_any = false;

                    for (int bounce = 0; bounce < frame.gi_bounces; ++bounce)
                    {
                        const int salt_index = bounce * gi.sample_count + sample_idx;
                        const size_t base_idx = static_cast<size_t>(salt_index) * 2;
                        GiHit hit{};
                        double cos_theta = 0.0;
                        const bool hit_found = trace_bounce(cur_world,
                                                            cur_normal,
                                                            cur_normal,
                                                            shifts[base_idx + 0],
                                                            shifts[base_idx + 1],
                                                            static_cast<int>(x),
                                                            static_cast<int>(y),
                                                            terrain,
                                                            gi,
                                                            hit,
                                                            cos_theta);
                        if (!hit_found)
                        {
                            break;
                        }

                        const LinearColor incoming = eval_incoming(hit, frame.lights,
                                                                   frame.sky_top, frame.hemi_ground,
                                                                   frame.sky_scale);
                        if (incoming.r > 0.0f || incoming.g > 0.0f || incoming.b > 0.0f)
                        {
                            LinearColor bounced = incoming * hit.albedo;
                            bounced = bounced * throughput;
                            bounced = bounced * static_cast<float>(cos_theta);
                            gi_sample = gi_sample + bounced;
                            hit_any = true;
                        }

                        LinearColor next_throughput = throughput * hit.albedo;
                        next_throughput = next_throughput * static_cast<float>(cos_theta);
                        if (next_throughput.r <= 0.0f && next_throughput.g <= 0.0f && next_throughput.b <= 0.0f)
                        {
                            break;
                        }

                        throughput = next_throughput;
                        cur_world = hit.position;
                        cur_normal = hit.normal;
                    }

                    if (hit_any)
                    {
                        gi_sum.r += gi_sample.r;
                        gi_sum.g += gi_sample.g;
                        gi_sum.b += gi_sample.b;
                    }
                }

                if (gi_sum.r > 0.0f || gi_sum.g > 0.0f || gi_sum.b > 0.0f)
                {
                    const float inv_samples = 1.0f / static_cast<float>(gi.sample_count);
                    LinearColor gi_sample{
                        gi_sum.r * inv_samples,
                        gi_sum.g * inv_samples,
                        gi_sum.b * inv_samples
                    };
                    gi_sample = gi_sample * frame.gi_scale;
                    gi_sample.r = std::clamp(gi_sample.r, 0.0f, gi.clamp);
                    gi_sample.g = std::clamp(gi_sample.g, 0.0f, gi.clamp);
                    gi_sample.b = std::clamp(gi_sample.b, 0.0f, gi.clamp);
                    buffers.sample_indirect[idx] = gi_sample;
                }
            }
        }
    }

private:
    static constexpr double kPi = std::numbers::pi_v<double>;

    static auto sincos_double(const double angle, double* out_sin, double* out_cos) -> void
    {
#if defined(__GNUC__)
        __builtin_sincos(angle, out_sin, out_cos);
#else
        *out_sin = std::sin(angle);
        *out_cos = std::cos(angle);
#endif
    }

    [[nodiscard]]
    static auto sample_dir(const Vec3& hemi_normal,
                           const BlueNoise::Shift& shift_u,
                           const BlueNoise::Shift& shift_v,
                           const int px, const int py,
                           Vec3& out_dir,
                           double& out_cos) -> bool
    {
        const float u1 = BlueNoise::sample(px, py, shift_u);
        const float u2 = BlueNoise::sample(px, py, shift_v);
        const double r = std::sqrt(static_cast<double>(u1));
        const double theta = 2.0 * kPi * static_cast<double>(u2);
        double sin_theta = 0.0;
        double cos_theta = 0.0;
        sincos_double(theta, &sin_theta, &cos_theta);
        const double local_x = r * cos_theta;
        const double local_y = r * sin_theta;
        const double local_z = std::sqrt(std::max(0.0, 1.0 - r * r));

        auto [tangent, bitangent, forward] = Vec3::get_basis(hemi_normal);
        out_dir = {
            tangent.x * local_x + bitangent.x * local_y + forward.x * local_z,
            tangent.y * local_x + bitangent.y * local_y + forward.y * local_z,
            tangent.z * local_x + bitangent.z * local_y + forward.z * local_z
        };
        out_cos = local_z;
        return out_cos > 0.0;
    }

    [[nodiscard]]
    static auto eval_incoming(const GiHit& hit,
                              const std::array<ShadingContext::DirectionalLightInfo, 2>& lights,
                              const LinearColor& sky_top,
                              const LinearColor& hemi_ground,
                              const float sky_scale) -> LinearColor
    {
        LinearColor incoming{0.0f, 0.0f, 0.0f};
        for (const auto& light : lights)
        {
            if (light.intensity <= 0.0)
            {
                continue;
            }
            const double ndotl = std::max(0.0, hit.normal.dot(light.dir));
            if (ndotl <= 0.0)
            {
                continue;
            }
            const float scale = static_cast<float>(light.intensity * ndotl);
            incoming.r += light.color.r * scale;
            incoming.g += light.color.g * scale;
            incoming.b += light.color.b * scale;
        }

        if (sky_scale > 0.0f)
        {
            float sky_t = static_cast<float>(hit.normal.y * 0.5 + 0.5);
            sky_t = std::clamp(sky_t, 0.0f, 1.0f);
            const LinearColor sky = LinearColor::lerp(hemi_ground, sky_top, sky_t);
            const float vis = hit.sky_visibility;
            incoming.r += sky.r * sky_scale * vis;
            incoming.g += sky.g * sky_scale * vis;
            incoming.b += sky.b * sky_scale * vis;
        }

        return incoming;
    }

    [[nodiscard]]
    auto trace_bounce(const Vec3& world_origin,
                      const Vec3& surface_normal,
                      const Vec3& hemi_normal,
                      const BlueNoise::Shift& shift_u,
                      const BlueNoise::Shift& shift_v,
                      const int px, const int py,
                      const Terrain& terrain,
                      const GiSettings& gi,
                      GiHit& out_hit,
                      double& out_cos) const -> bool
    {
        Vec3 dir{};
        if (!sample_dir(hemi_normal, shift_u, shift_v, px, py, dir, out_cos))
        {
            return false;
        }
        return gi_hit(terrain, world_origin, surface_normal, dir, gi, &out_hit);
    }

    [[nodiscard]]
    auto shadow_hit(const Terrain& terrain,
                    const Vec3& world_pos, const Vec3& normal, const Vec3& light_dir,
                    const ShadowSettings& shadow) const -> bool
    {
        const auto& topo = terrain.topology;
        const auto& cfg = terrain.config;
        const int size = terrain.chunk_size;
        const int max_height = topo.max_height;

        if (size <= 0 || max_height <= 0) return false;

        const double block_size = cfg.block_size;
        const double inv_block = 1.0 / block_size;

        const Vec3 origin_world = world_pos + (normal * shadow.ray_bias);
        const double half_block = block_size * 0.5;
        const double start_x = -(static_cast<double>(size) - 1.0) * half_block;
        
        const Vec3 origin_grid{
            (origin_world.x - start_x + half_block) * inv_block,
            (origin_world.y - cfg.base_y + half_block) * inv_block,
            (origin_world.z - cfg.start_z + half_block) * inv_block
        };

        const Vec3 dir_grid = light_dir * inv_block;
        
        if (dir_grid.x == 0.0 && dir_grid.y == 0.0 && dir_grid.z == 0.0) return false;

        int x = static_cast<int>(std::floor(origin_grid.x));
        int y = static_cast<int>(std::floor(origin_grid.y));
        int z = static_cast<int>(std::floor(origin_grid.z));

        if (x < 0 || x >= size || z < 0 || z >= size || y < 0 || y >= max_height) return false;

        const double inf = std::numeric_limits<double>::infinity();
        const auto get_step = [](double d) -> int { return d > 0.0 ? 1 : (d < 0.0 ? -1 : 0); };
        const int step_x = get_step(dir_grid.x);
        const int step_y = get_step(dir_grid.y);
        const int step_z = get_step(dir_grid.z);

        const double t_delta_x = step_x != 0 ? 1.0 / std::abs(dir_grid.x) : inf;
        const double t_delta_y = step_y != 0 ? 1.0 / std::abs(dir_grid.y) : inf;
        const double t_delta_z = step_z != 0 ? 1.0 / std::abs(dir_grid.z) : inf;

        const auto get_t_max = [&](double pos, int step, double dir) {
            if (step == 0) return inf;
            const double next_boundary = std::floor(pos) + (step > 0 ? 1.0 : 0.0);
            return (next_boundary - pos) / dir;
        };

        double t_max_x = get_t_max(origin_grid.x, step_x, dir_grid.x);
        double t_max_y = get_t_max(origin_grid.y, step_y, dir_grid.y);
        double t_max_z = get_t_max(origin_grid.z, step_z, dir_grid.z);

        const int max_steps = (size + size + max_height) * 4;
        bool is_first_block = true;

        for (int i = 0; i < max_steps; ++i)
        {
            if (!is_first_block)
            {
                if (topo.has_block(x, y, z))
                {
                    return true;
                }
            }
            is_first_block = false;

            if (t_max_x < t_max_y)
            {
                if (t_max_x < t_max_z)
                {
                    x += step_x; t_max_x += t_delta_x;
                }
                else
                {
                    z += step_z; t_max_z += t_delta_z;
                }
            }
            else
            {
                if (t_max_y < t_max_z)
                {
                    y += step_y; t_max_y += t_delta_y;
                }
                else
                {
                    z += step_z; t_max_z += t_delta_z;
                }
            }

            if (x < 0 || x >= size || z < 0 || z >= size || y < 0 || y >= max_height) return false;
        }

        return false;
    }

    [[nodiscard]]
    static auto disk_samples() -> const std::array<Vec2, 64>&
    {
        static const auto samples = [] {
            std::array<Vec2, 64> points{};
            constexpr double golden_angle = 2.39996322972865332;
            constexpr double count_inv = 1.0 / static_cast<double>(points.size());
            
            for (size_t i = 0; i < points.size(); ++i)
            {
                const double u = (static_cast<double>(i) + 0.5) * count_inv;
                const double r = std::sqrt(u);
                const double theta = static_cast<double>(i) * golden_angle;
                points[i] = {r * std::cos(theta), r * std::sin(theta)};
            }
            return points;
        }();
        return samples;
    }
};
