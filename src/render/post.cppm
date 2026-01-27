module;

#include "../prelude.hpp"

module render;

namespace {
auto reproject_point(const Mat4& prev_vp, const double near_plane,
                     const Vec3& world, const size_t width, const size_t height) -> Vec2
{
    if (width == 0 || height == 0)
    {
        return {0.0, 0.0};
    }

    const double clip_x = prev_vp.m[0][0] * world.x + prev_vp.m[0][1] * world.y + prev_vp.m[0][2] * world.z + prev_vp.m[0][3];
    const double clip_y = prev_vp.m[1][0] * world.x + prev_vp.m[1][1] * world.y + prev_vp.m[1][2] * world.z + prev_vp.m[1][3];
    const double clip_w = prev_vp.m[3][0] * world.x + prev_vp.m[3][1] * world.y + prev_vp.m[3][2] * world.z + prev_vp.m[3][3];

    if (clip_w <= near_plane)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }

    const double inv_w = 1.0 / clip_w;
    const double ndc_x = clip_x * inv_w;
    const double ndc_y = clip_y * inv_w;
    const double screen_x = (ndc_x * 0.5 + 0.5) * static_cast<double>(width);
    const double screen_y = (ndc_y * 0.5 + 0.5) * static_cast<double>(height);
    return {screen_x, screen_y};
}
} // namespace

auto PostProcessor::update_taa_state(bool taa_on, size_t width, size_t height, size_t sample_count,
                                     uint64_t state_version,
                                     const Vec3& camera_pos, double yaw, double pitch) -> float
{
    if (taa_on)
    {
        if (width != taa_width || height != taa_height || !taa_was_enabled || taa_state_version != state_version)
        {
            taa_width = width;
            taa_height = height;
            taa_history[0].assign(sample_count, {0.0f, 0.0f, 0.0f});
            taa_history[1].assign(sample_count, {0.0f, 0.0f, 0.0f});
            taa_history_valid = false;
            taa_state_version = state_version;
            taa_ping_pong = 0;
        }
    }
    else
    {
        taa_history_valid = false;
    }
    taa_was_enabled = taa_on;

    float taa_sharpen_strength = 0.0f;
    if (!taa_on)
    {
        taa_motion_activity = 0.0f;
        last_camera_valid = false;
    }
    else
    {
        if (taa_motion_state_version != state_version)
        {
            taa_motion_state_version = state_version;
            taa_motion_activity = 0.0f;
            last_camera_valid = false;
        }
        double motion_factor = 0.0;
        if (last_camera_valid)
        {
            const double dx = camera_pos.x - last_camera_pos.x;
            const double dy = camera_pos.y - last_camera_pos.y;
            const double dz = camera_pos.z - last_camera_pos.z;
            const double move_dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double dyaw = yaw - last_camera_yaw;
            const double dpitch = pitch - last_camera_pitch;
            const double rot_dist = std::sqrt(dyaw * dyaw + dpitch * dpitch);
            const double move_factor = kTaaSharpenMoveThreshold > 0.0
                ? (move_dist / kTaaSharpenMoveThreshold) * kTaaSharpenMoveGain
                : 0.0;
            const double rot_factor = kTaaSharpenRotThreshold > 0.0
                ? (rot_dist / kTaaSharpenRotThreshold) * kTaaSharpenRotGain
                : 0.0;
            motion_factor = std::sqrt(move_factor * move_factor + rot_factor * rot_factor);
            motion_factor = std::clamp(motion_factor, 0.0, 1.0);
        }

        const float target = static_cast<float>(motion_factor);
        if (!last_camera_valid)
        {
            taa_motion_activity = target;
        }
        else
        {
            const float rate = (target > taa_motion_activity) ? kTaaSharpenAttack : kTaaSharpenRelease;
            taa_motion_activity = taa_motion_activity + (target - taa_motion_activity) * rate;
        }

        taa_sharpen_strength = static_cast<float>(taa_motion_activity * kTaaSharpenMax);
    }

    last_camera_pos = camera_pos;
    last_camera_yaw = yaw;
    last_camera_pitch = pitch;
    last_camera_valid = true;

    return taa_sharpen_strength;
}

auto PostProcessor::resolve_frame(uint32_t* framebuffer, size_t width, size_t height, size_t sample_count,
                                  float depth_max, const RenderBuffers& buffers,
                                  const LinearColor& sky_top_linear, const LinearColor& sky_bottom_linear,
                                  bool taa_on, bool clamp_history, float taa_factor, float taa_sharpen_strength,
                                  bool gi_active, uint32_t frame_index,
                                  float jitter_x, float jitter_y, double jitter_x_d, double jitter_y_d,
                                  double width_d, double height_d, double proj_a, double proj_b,
                                  const Mat4& inverse_current_vp, const Mat4& previous_vp, double camera_near_plane,
                                  float exposure_factor) -> void
{
    static constexpr std::array<std::array<int, 4>, 4> bayer4 = {{
        {{0, 8, 2, 10}},
        {{12, 4, 14, 6}},
        {{3, 11, 1, 9}},
        {{15, 7, 13, 5}}
    }};
    const float dither_strength = 2.0f;
    const float dither_scale = dither_strength / 16.0f;
    const bool use_history = taa_on && taa_history_valid;

    const int read_idx = (taa_ping_pong + 1) % 2;
    const int write_idx = taa_ping_pong;
    const LinearColor* history_read_ptr = taa_history[read_idx].data();
    LinearColor* history_write_ptr = taa_history[write_idx].data();
    const std::span<const LinearColor> history_read_span(history_read_ptr, sample_count);

    for (size_t y = 0; y < height; ++y)
    {
        const float sky_t = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
        const LinearColor sky_row_linear = LinearColor::lerp(sky_top_linear, sky_bottom_linear, sky_t);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            if (buffers.zbuffer[idx] >= depth_max)
            {
                current_linear_buffer[idx] = sky_row_linear;
                continue;
            }
            LinearColor accum = buffers.sample_colors[idx];
            const LinearColor direct = buffers.sample_direct[idx];
            accum.r += direct.r;
            accum.g += direct.g;
            accum.b += direct.b;
            if (gi_active)
            {
                const LinearColor indirect = buffers.sample_indirect[idx];
                const float gi_ao = std::min(1.0f, buffers.sample_ao[idx] + kGiAoLift);
                const LinearColor indirect_scaled = indirect * gi_ao;
                accum.r += indirect_scaled.r;
                accum.g += indirect_scaled.g;
                accum.b += indirect_scaled.b;
            }
            current_linear_buffer[idx] = accum;
        }
    }

    auto current_linear_at = [&](int ix, int iy) -> LinearColor {
        ix = std::clamp(ix, 0, static_cast<int>(width) - 1);
        iy = std::clamp(iy, 0, static_cast<int>(height) - 1);
        const size_t idx = static_cast<size_t>(iy) * width + static_cast<size_t>(ix);
        return current_linear_buffer[idx];
    };

    for (size_t y = 0; y < height; ++y)
    {
        const double screen_y = static_cast<double>(y) + 0.5 + jitter_y_d;
        for (size_t x = 0; x < width; ++x)
        {
            const size_t pixel = y * width + x;
            const float depth = buffers.zbuffer[pixel];
            const bool is_sky = depth >= depth_max;
            const LinearColor current_linear = current_linear_buffer[pixel];

            LinearColor blended = current_linear;
            bool history_used = false;
            if (taa_on)
            {
                bool history_valid = use_history;
                LinearColor prev = current_linear;
                if (history_valid)
                {
                    prev = history_read_ptr[pixel];
                    if (!is_sky)
                    {
                        const double screen_x = static_cast<double>(x) + 0.5 + jitter_x_d;
                        Vec3 world;
                        if (buffers.world_stamp[pixel] == frame_index)
                        {
                            world = buffers.world_positions[pixel];
                        }
                        else
                        {
                        world = unproject_fast(screen_x, screen_y, depth,
                                               inverse_current_vp, width_d, height_d,
                                               proj_a, proj_b);
                        }

                        Vec2 prev_screen = reproject_point(previous_vp, camera_near_plane, world, width, height);

                        if (std::isfinite(prev_screen.x) && std::isfinite(prev_screen.y))
                        {
                            prev_screen.x -= static_cast<double>(jitter_x);
                            prev_screen.y -= static_cast<double>(jitter_y);
                        }
                        if (std::isfinite(prev_screen.x) && std::isfinite(prev_screen.y) &&
                            prev_screen.x >= 0.0 && prev_screen.x <= static_cast<double>(width) &&
                            prev_screen.y >= 0.0 && prev_screen.y <= static_cast<double>(height))
                        {
                            prev = sample_bilinear_history(history_read_span, width, height,
                                                          prev_screen.x, prev_screen.y);
                        }
                        else
                        {
                            history_valid = false;
                        }
                    }
                }

                if (history_valid && clamp_history)
                {
                    LinearColor minc = current_linear;
                    LinearColor maxc = current_linear;
                    for (int ny = -1; ny <= 1; ++ny)
                    {
                        for (int nx = -1; nx <= 1; ++nx)
                        {
                            const LinearColor neighbor = current_linear_at(static_cast<int>(x) + nx,
                                                                       static_cast<int>(y) + ny);
                            minc.r = std::min(minc.r, neighbor.r);
                            minc.g = std::min(minc.g, neighbor.g);
                            minc.b = std::min(minc.b, neighbor.b);
                            maxc.r = std::max(maxc.r, neighbor.r);
                            maxc.g = std::max(maxc.g, neighbor.g);
                            maxc.b = std::max(maxc.b, neighbor.b);
                        }
                    }
                    prev.r = std::clamp(prev.r, minc.r, maxc.r);
                    prev.g = std::clamp(prev.g, minc.g, maxc.g);
                    prev.b = std::clamp(prev.b, minc.b, maxc.b);
                }

                if (history_valid)
                {
                    blended.r = prev.r + (current_linear.r - prev.r) * taa_factor;
                    blended.g = prev.g + (current_linear.g - prev.g) * taa_factor;
                    blended.b = prev.b + (current_linear.b - prev.b) * taa_factor;
                    history_used = true;
                }
                else
                {
                    blended = current_linear;
                }

                history_write_ptr[pixel] = blended;
            }

            taa_resolved[pixel] = blended;
            taa_history_mask[pixel] = static_cast<uint8_t>(history_used && !is_sky);
        }
    }

    auto resolved_at = [&](int ix, int iy) -> LinearColor {
        ix = std::clamp(ix, 0, static_cast<int>(width) - 1);
        iy = std::clamp(iy, 0, static_cast<int>(height) - 1);
        const size_t idx = static_cast<size_t>(iy) * width + static_cast<size_t>(ix);
        return taa_resolved[idx];
    };

    const bool apply_sharpen = taa_sharpen_strength > 0.0f;

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            const size_t pixel = y * width + x;
            const bool is_sky = buffers.zbuffer[pixel] >= depth_max;
            LinearColor resolved = taa_resolved[pixel];

            if (apply_sharpen && taa_history_mask[pixel])
            {
                const LinearColor center = resolved;
                const LinearColor north = resolved_at(static_cast<int>(x), static_cast<int>(y) - 1);
                const LinearColor south = resolved_at(static_cast<int>(x), static_cast<int>(y) + 1);
                const LinearColor west = resolved_at(static_cast<int>(x) - 1, static_cast<int>(y));
                const LinearColor east = resolved_at(static_cast<int>(x) + 1, static_cast<int>(y));
                const float inv = 1.0f / 8.0f;
                const LinearColor blur{
                    (center.r * 4.0f + north.r + south.r + west.r + east.r) * inv,
                    (center.g * 4.0f + north.g + south.g + west.g + east.g) * inv,
                    (center.b * 4.0f + north.b + south.b + west.b + east.b) * inv
                };
                resolved.r = center.r + (center.r - blur.r) * taa_sharpen_strength;
                resolved.g = center.g + (center.g - blur.g) * taa_sharpen_strength;
                resolved.b = center.b + (center.b - blur.b) * taa_sharpen_strength;
                resolved.r = std::max(0.0f, resolved.r);
                resolved.g = std::max(0.0f, resolved.g);
                resolved.b = std::max(0.0f, resolved.b);
            }

            const LinearColor mapped = tonemap_reinhard(resolved, exposure_factor);
            ColorSrgb srgb = mapped.to_srgb();
            if (!is_sky)
            {
                const float dither = (static_cast<float>(bayer4[y & 3][x & 3]) - 7.5f) * dither_scale;
                srgb.r += dither;
                srgb.g += dither;
                srgb.b += dither;
            }

            framebuffer[pixel] = pack_color(srgb);
        }
    }

    if (taa_on)
    {
        taa_history_valid = true;
        taa_ping_pong = (taa_ping_pong + 1) % 2;
    }
}
