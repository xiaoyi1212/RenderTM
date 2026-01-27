module;

#include "../prelude.hpp"

module render;

auto RenderEngine::update(uint32_t* framebuffer, size_t width, size_t height) -> void
{
    auto& camera = this->camera;
    auto& world = this->world;
    auto& terrain = this->terrain;
    auto& rotationPaused = this->rotationPaused;
    auto& renderFrameIndex = this->renderFrameIndex;
    auto& renderStateVersion = this->renderStateVersion;
    auto& taaEnabled = this->taaEnabled;
    auto& taaBlend = this->taaBlend;
    auto& taaClampEnabled = this->taaClampEnabled;
    auto& taaSharpenStrength = this->taaSharpenStrength;
    auto& giEnabled = this->giEnabled;
    auto& giStrength = this->giStrength;
    auto& giBounceCount = this->giBounceCount;
    auto& ambientOcclusionEnabled = this->ambientOcclusionEnabled;
    auto& shadowEnabled = this->shadowEnabled;
    auto& ambientLight = this->ambientLight;
    auto& currentVP = this->currentVP;
    auto& previousVP = this->previousVP;
    auto& inverseCurrentVP = this->inverseCurrentVP;
    auto& buffers = this->buffers;
    auto& rasterizer = this->rasterizer;
    auto& lighting = this->lighting;
    auto& post = this->post;

    const size_t sample_count = width * height;
    const double width_d = static_cast<double>(width);
    const double height_d = static_cast<double>(height);
    const float depth_max = std::numeric_limits<float>::max();

    const bool taa_on = taaEnabled.load(std::memory_order_relaxed);
    const float base_blend = static_cast<float>(std::clamp(taaBlend.load(std::memory_order_relaxed), 0.0, 1.0));
    const float taa_factor = base_blend;
    const bool clamp_history = taaClampEnabled.load(std::memory_order_relaxed);
    const bool gi_on = giEnabled.load(std::memory_order_relaxed);
    const float gi_scale = static_cast<float>(std::max(0.0, giStrength.load(std::memory_order_relaxed)));
    const uint64_t state_version = renderStateVersion.load(std::memory_order_relaxed);

    const bool resized = buffers.resize(width, height, depth_max);
    if (resized)
    {
        post.resize_buffers(sample_count);
    }

    const bool paused = rotationPaused.load(std::memory_order_relaxed);
    const uint32_t frame_index = renderFrameIndex.fetch_add(1u, std::memory_order_relaxed);
    previousVP = currentVP;

    world.update_orbits(paused);

    const Vec3 camera_pos_snapshot = camera.position;
    const double yaw_snapshot = camera.rotation.x;
    const double pitch_snapshot = camera.rotation.y;

    const float taa_sharpen_strength = post.update_taa_state(taa_on, width, height, sample_count,
                                                             state_version,
                                                             camera_pos_snapshot, yaw_snapshot, pitch_snapshot);
    taaSharpenStrength.store(static_cast<double>(taa_sharpen_strength), std::memory_order_relaxed);

    const double view_yaw = -yaw_snapshot;
    const double view_pitch = -pitch_snapshot;
    const Mat4 view = camera.view_matrix();
    const double proj_scale_y = height_d * 0.8;
    const double proj_scale_x = proj_scale_y;
    const Mat4 proj = make_projection_matrix(width_d, height_d, proj_scale_x, proj_scale_y, camera);
    currentVP = proj * view;
    if (const auto inv = currentVP.invert())
    {
        inverseCurrentVP = *inv;
    }
    else
    {
        inverseCurrentVP = Mat4::identity();
    }

    const double inv_dist = 1.0 / (Camera::far_plane - Camera::near_plane);
    const double proj_a = Camera::far_plane * inv_dist;
    const double proj_b = -Camera::near_plane * Camera::far_plane * inv_dist;

    float jitter_x = 0.0f;
    float jitter_y = 0.0f;
    if (taa_on)
    {
        const float jitter_scale = 1.0f;
        const float u = BlueNoise::sample(0, 0, static_cast<int>(frame_index), kTaaJitterSalt);
        const float v = BlueNoise::sample(1, 0, static_cast<int>(frame_index), kTaaJitterSalt + 1);
        jitter_x = (u - 0.5f) * jitter_scale;
        jitter_y = (v - 0.5f) * jitter_scale;
    }
    const double jitter_x_d = static_cast<double>(jitter_x);
    const double jitter_y_d = static_cast<double>(jitter_y);
    const ViewRotation view_rot = make_view_rotation(view_yaw, view_pitch);

    const double mat_ambient = 0.25;
    const double mat_diffuse = 1.0;
    const Material cube_material{0xFFFFFFFF, mat_ambient, mat_diffuse, 0.15, 24.0};

    terrain.generate();

    const bool sun_orbit = world.sun.orbit_enabled;
    Vec3 sun_dir = world.sun.direction.normalize();
    double sun_intensity = world.sun.intensity;
    if (sun_orbit)
    {
        sun_dir = world.sun.orbit_direction(world.sun.orbit_angle);
        const double visibility = world.sun.orbit_height(sun_dir);
        sun_intensity = world.sun.intensity * visibility;
    }
    sun_intensity *= kSunIntensityBoost;

    const auto sky_gradient = world.sky_gradient();
    const LinearColor sky_top_linear = sky_gradient.zenith;
    const LinearColor sky_bottom_linear = sky_gradient.horizon;
    double sun_height = 1.0;
    if (sun_orbit)
    {
        sun_height = world.sun.orbit_height(sun_dir);
    }
    const double moon_intensity = world.moon.intensity;
    double effective_sky_intensity = world.sky.sky_intensity;
    if (sun_orbit)
    {
        const double sun_factor = std::pow(std::clamp(sun_height, 0.0, 1.0), world.sky.sun_height_power);
        effective_sky_intensity *= sun_factor;

        const double moon_factor = std::clamp(moon_intensity, 0.0, 1.0) * world.sky.moon_ambient_floor * (1.0 - sun_factor);
        effective_sky_intensity = std::min(1.0, effective_sky_intensity + moon_factor);
    }

    const Vec3 moon_dir = world.moon.direction.normalize();
    const bool shadows_on = shadowEnabled.load(std::memory_order_relaxed);
    const std::array<ShadingContext::DirectionalLightInfo, 2> lights = {
        ShadingContext::DirectionalLightInfo{sun_dir, sun_intensity, world.sun.color, world.sun.angular_radius},
        ShadingContext::DirectionalLightInfo{moon_dir, moon_intensity, world.moon.color, world.moon.angular_radius}
    };

    std::array<Vec3, 2> lights_right_scaled{};
    std::array<Vec3, 2> lights_up_scaled{};

    for (int i = 0; i < 2; ++i)
    {
        if (lights[i].angular_radius > 0.0 && lights[i].intensity > 0.0)
        {
            auto [right, up, forward] = Vec3::get_basis(lights[i].dir);
            const double scale = std::tan(lights[i].angular_radius);

            lights_right_scaled[i] = {right.x * scale, right.y * scale, right.z * scale};
            lights_up_scaled[i]    = {up.x * scale, up.y * scale, up.z * scale};
        }
        else
        {
            lights_right_scaled[i] = {0.0, 0.0, 0.0};
            lights_up_scaled[i]    = {0.0, 0.0, 0.0};
        }
    }

    const std::array<BlueNoise::Shift, 2> shadow_shift_u{
        BlueNoise::shift(static_cast<int>(frame_index), kSunShadowSalt),
        BlueNoise::shift(static_cast<int>(frame_index), kMoonShadowSalt)
    };
    const std::array<BlueNoise::Shift, 2> shadow_shift_v{
        BlueNoise::shift(static_cast<int>(frame_index), kSunShadowSalt + 1),
        BlueNoise::shift(static_cast<int>(frame_index), kMoonShadowSalt + 1)
    };

    const bool direct_lighting_enabled = (lights[0].intensity > 0.0) || (lights[1].intensity > 0.0);
    const bool ao_enabled = ambientOcclusionEnabled.load(std::memory_order_relaxed);
    const float sky_scale = static_cast<float>(std::clamp(effective_sky_intensity, 0.0, 1.0));
    const LinearColor hemi_ground = lighting.compute_hemisphere_ground(sky_bottom_linear, lights);
    const double ambient_value = ambientLight.load(std::memory_order_relaxed);

    struct CachedContext
    {
        uint32_t color = 0;
        ShadingContext ctx{};
    };
    std::array<CachedContext, 8> ctx_cache{};
    size_t ctx_cache_size = 0;

    auto get_ctx = [&](const uint32_t color) -> ShadingContext& {
        for (size_t i = 0; i < ctx_cache_size; ++i)
        {
            if (ctx_cache[i].color == color)
            {
                return ctx_cache[i].ctx;
            }
        }
        Material material = cube_material;
        material.color = color;
        const ColorSrgb albedo_srgb = ColorSrgb::from_hex(material.color);
        const LinearColor albedo = albedo_srgb.to_linear();
        CachedContext entry{};
        entry.color = color;
        entry.ctx = {
            albedo,
            sky_top_linear,
            sky_bottom_linear,
            hemi_ground,
            sky_scale,
            camera_pos_snapshot,
            ambient_value,
            material,
            direct_lighting_enabled,
            ao_enabled,
            shadows_on,
            lights
        };
        if (ctx_cache_size < ctx_cache.size())
        {
            ctx_cache[ctx_cache_size] = entry;
            return ctx_cache[ctx_cache_size++].ctx;
        }
        ctx_cache[0] = entry;
        return ctx_cache[0].ctx;
    };

    for (const auto& quad : terrain.mesh)
    {
        ShadingContext& ctx = get_ctx(quad.color);
        rasterizer.render_quad(buffers.zbuffer.data(), buffers.sample_colors.data(),
                               buffers.sample_direct_sun.data(), buffers.sample_direct_moon.data(),
                               buffers.shadow_mask_sun.data(), buffers.shadow_mask_moon.data(),
                               buffers.sample_normals.data(), buffers.sample_albedo.data(), buffers.sample_ao.data(),
                               buffers.world_positions.data(), buffers.world_stamp.data(), frame_index,
                               width, height, quad, proj_scale_x, proj_scale_y,
                               camera_pos_snapshot, view_rot, Camera::near_plane, terrain,
                               lighting, ctx, jitter_x, jitter_y,
                               lights_right_scaled, lights_up_scaled,
                               shadow_shift_u, shadow_shift_v);
    }

    if (shadows_on)
    {
        const float* shadow_sun = buffers.shadow_mask_sun.data();
        const float* shadow_moon = buffers.shadow_mask_moon.data();
        if (kShadowFilterEnabled)
        {
            const std::span<const float> shadow_sun_span(buffers.shadow_mask_sun.data(), sample_count);
            const std::span<const float> shadow_moon_span(buffers.shadow_mask_moon.data(), sample_count);
            const std::span<float> shadow_sun_out_span(buffers.shadow_mask_filtered_sun.data(), sample_count);
            const std::span<float> shadow_moon_out_span(buffers.shadow_mask_filtered_moon.data(), sample_count);
            const std::span<const float> depth_span(buffers.zbuffer.data(), sample_count);
            const std::span<const Vec3> normals_span(buffers.sample_normals.data(), sample_count);
            lighting.filter_shadow_masks(shadow_sun_span, shadow_moon_span,
                                         shadow_sun_out_span, shadow_moon_out_span,
                                         depth_span, normals_span,
                                         width, height, depth_max);
            shadow_sun = buffers.shadow_mask_filtered_sun.data();
            shadow_moon = buffers.shadow_mask_filtered_moon.data();
        }
        for (size_t i = 0; i < sample_count; ++i)
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
        for (size_t i = 0; i < sample_count; ++i)
        {
            buffers.sample_direct[i] = buffers.sample_direct_sun[i] + buffers.sample_direct_moon[i];
        }
    }

    const int raw_bounce_count = giBounceCount.load(std::memory_order_relaxed);
    const int gi_bounce_count = raw_bounce_count > 0 ? raw_bounce_count : 0;
    const bool gi_active = gi_on && gi_scale > 0.0f && gi_bounce_count > 0;
    if (gi_active)
    {
        const size_t gi_shift_count = static_cast<size_t>(gi_bounce_count) * kGiSampleCount * 2;
        std::vector<BlueNoise::Shift> gi_shifts(gi_shift_count);
        for (int bounce = 0; bounce < gi_bounce_count; ++bounce)
        {
            for (int sample_idx = 0; sample_idx < kGiSampleCount; ++sample_idx)
            {
                const int salt_index = bounce * kGiSampleCount + sample_idx;
                const int base_salt = kGiNoiseSalt + salt_index * 2;
                const size_t base_idx = static_cast<size_t>(salt_index) * 2;
                gi_shifts[base_idx + 0] = BlueNoise::shift(static_cast<int>(frame_index), base_salt);
                gi_shifts[base_idx + 1] = BlueNoise::shift(static_cast<int>(frame_index), base_salt + 1);
            }
        }
        for (size_t y = 0; y < height; ++y)
        {
            const double screen_y = static_cast<double>(y) + 0.5 + jitter_y_d;
            for (size_t x = 0; x < width; ++x)
            {
                const size_t idx = y * width + x;
                if (buffers.zbuffer[idx] >= depth_max)
                {
                    continue;
                }
                buffers.sample_indirect[idx] = {0.0f, 0.0f, 0.0f};
                Vec3 normal = buffers.sample_normals[idx];
                const double normal_len_sq = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
                if (normal_len_sq <= 1e-6)
                {
                    continue;
                }
                const LinearColor origin_albedo = buffers.sample_albedo[idx];

                const double screen_x = static_cast<double>(x) + 0.5 + jitter_x_d;
                const Vec3 world = unproject_fast(screen_x, screen_y, buffers.zbuffer[idx],
                                                  inverseCurrentVP, width_d, height_d,
                                                  proj_a, proj_b);
                buffers.world_positions[idx] = world;
                buffers.world_stamp[idx] = frame_index;

                auto sample_gi_dir = [&](const Vec3& hemi_normal,
                                         const BlueNoise::Shift& shift_u,
                                         const BlueNoise::Shift& shift_v,
                                         Vec3& out_dir,
                                         double& out_cos_theta) -> bool {
                    const float u1 = BlueNoise::sample(static_cast<int>(x), static_cast<int>(y), shift_u);
                    const float u2 = BlueNoise::sample(static_cast<int>(x), static_cast<int>(y), shift_v);
                    const double r = std::sqrt(static_cast<double>(u1));
                    const double theta = 2.0 * kPi * static_cast<double>(u2);
                    double sin_theta_angle = 0.0;
                    double cos_theta_angle = 0.0;
                    sincos_double(theta, &sin_theta_angle, &cos_theta_angle);
                    const double local_x = r * cos_theta_angle;
                    const double local_y = r * sin_theta_angle;
                    const double local_z = std::sqrt(std::max(0.0, 1.0 - r * r));

                    auto [tangent, bitangent, forward] = Vec3::get_basis(hemi_normal);
                    out_dir = {
                        tangent.x * local_x + bitangent.x * local_y + forward.x * local_z,
                        tangent.y * local_x + bitangent.y * local_y + forward.y * local_z,
                        tangent.z * local_x + bitangent.z * local_y + forward.z * local_z
                    };
                    out_cos_theta = local_z;
                    return out_cos_theta > 0.0;
                };

                auto eval_gi_incoming = [&](const GiHit& hit) -> LinearColor {
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
                        float sky_t = static_cast<float>((-hit.normal.y) * 0.5 + 0.5);
                        sky_t = std::clamp(sky_t, 0.0f, 1.0f);
                        const LinearColor sky = LinearColor::lerp(hemi_ground, sky_top_linear, sky_t);
                        const float vis = hit.sky_visibility;
                        incoming.r += sky.r * sky_scale * vis;
                        incoming.g += sky.g * sky_scale * vis;
                        incoming.b += sky.b * sky_scale * vis;
                    }

                    return incoming;
                };

                auto trace_gi_bounce = [&](const Vec3& world_origin,
                                           const Vec3& surface_normal,
                                           const Vec3& hemi_normal,
                                           const BlueNoise::Shift& shift_u,
                                           const BlueNoise::Shift& shift_v,
                                           GiHit& out_hit,
                                           double& out_cos_theta) -> bool {
                    Vec3 dir;
                    if (!sample_gi_dir(hemi_normal, shift_u, shift_v, dir, out_cos_theta))
                    {
                        return false;
                    }
                    return lighting.gi_raymarch_hit(terrain, world_origin, surface_normal, dir,
                                                    kGiMaxDistance, &out_hit);
                };

                LinearColor gi_sum{0.0f, 0.0f, 0.0f};
                for (int sample_idx = 0; sample_idx < kGiSampleCount; ++sample_idx)
                {
                    LinearColor gi_sample{0.0f, 0.0f, 0.0f};
                    Vec3 current_world = world;
                    Vec3 current_normal = normal;
                    LinearColor throughput = origin_albedo;
                    bool hit_any = false;

                    for (int bounce = 0; bounce < gi_bounce_count; ++bounce)
                    {
                        const int salt_index = bounce * kGiSampleCount + sample_idx;
                        const size_t base_idx = static_cast<size_t>(salt_index) * 2;
                        GiHit hit{};
                        double cos_theta = 0.0;
                        bool hit_found = trace_gi_bounce(current_world,
                                                         current_normal,
                                                         current_normal,
                                                         gi_shifts[base_idx + 0],
                                                         gi_shifts[base_idx + 1],
                                                         hit,
                                                         cos_theta);
                        if (!hit_found)
                        {
                            break;
                        }

                        const LinearColor incoming = eval_gi_incoming(hit);
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
                        current_world = hit.position;
                        current_normal = hit.normal;
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
                    const float inv_samples = 1.0f / static_cast<float>(kGiSampleCount);
                    LinearColor gi_sample{
                        gi_sum.r * inv_samples,
                        gi_sum.g * inv_samples,
                        gi_sum.b * inv_samples
                    };
                    gi_sample = gi_sample * gi_scale;
                    gi_sample.r = std::clamp(gi_sample.r, 0.0f, kGiClamp);
                    gi_sample.g = std::clamp(gi_sample.g, 0.0f, kGiClamp);
                    gi_sample.b = std::clamp(gi_sample.b, 0.0f, kGiClamp);
                    buffers.sample_indirect[idx] = gi_sample;
                }
            }
        }
    }

    const float exposure_factor = static_cast<float>(std::max(0.0, world.sky.exposure));
    post.resolve_frame(framebuffer, width, height, sample_count,
                       depth_max, buffers,
                       sky_top_linear, sky_bottom_linear,
                       taa_on, clamp_history, taa_factor, taa_sharpen_strength,
                       gi_active, frame_index,
                       jitter_x, jitter_y, jitter_x_d, jitter_y_d,
                       width_d, height_d, proj_a, proj_b,
                       inverseCurrentVP, previousVP, Camera::near_plane,
                       exposure_factor);
}

Vec2 RenderEngine::reproject_point(const Vec3 world, const size_t width, const size_t height) const
{
    if (width == 0 || height == 0)
    {
        return {0.0, 0.0};
    }

    const Mat4& vp = previousVP;
    const double clip_x = vp.m[0][0] * world.x + vp.m[0][1] * world.y + vp.m[0][2] * world.z + vp.m[0][3];
    const double clip_y = vp.m[1][0] * world.x + vp.m[1][1] * world.y + vp.m[1][2] * world.z + vp.m[1][3];
    const double clip_w = vp.m[3][0] * world.x + vp.m[3][1] * world.y + vp.m[3][2] * world.z + vp.m[3][3];

    if (clip_w <= Camera::near_plane)
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

void RenderEngine::set_camera_position(const Vec3 pos)
{
    camera.position = pos;
}

Vec3 RenderEngine::get_camera_position() const
{
    return camera.position;
}

void RenderEngine::move_camera(const Vec3 delta)
{
    camera.position = camera.position + delta;
}

void RenderEngine::move_camera_local(const Vec3 delta)
{
    camera.move_local(delta);
}

void RenderEngine::set_camera_rotation(const Vec2 rot)
{
    camera.set_rotation(rot);
}

Vec2 RenderEngine::get_camera_rotation() const
{
    return camera.rotation;
}

void RenderEngine::rotate_camera(const Vec2 delta)
{
    camera.rotate(delta);
}

double RenderEngine::get_near_plane() const
{
    return Camera::near_plane;
}

Vec2 RenderEngine::project_point(const Vec3 world, const size_t width, const size_t height) const
{
    if (width == 0 || height == 0)
    {
        return {0.0, 0.0};
    }

    const Vec3 view = camera.to_camera_space(world);
    if (view.z <= Camera::near_plane)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }

    const double inv_z = 1.0 / view.z;
    const double proj_scale = static_cast<double>(height) * 0.8;
    const double proj_x = view.x * inv_z * proj_scale;
    const double proj_y = view.y * inv_z * proj_scale;

    return {proj_x + static_cast<double>(width) / 2.0,
            proj_y + static_cast<double>(height) / 2.0};
}

Vec3 RenderEngine::unproject_point(const Vec3 screen, const size_t width, const size_t height) const
{
    if (width == 0 || height == 0)
    {
        return {0.0, 0.0, 0.0};
    }

    const double proj_scale = static_cast<double>(height) * 0.8;
    const double half_w = static_cast<double>(width) / 2.0;
    const double half_h = static_cast<double>(height) / 2.0;

    const double view_x = (screen.x - half_w) / proj_scale * screen.z;
    const double view_y = (screen.y - half_h) / proj_scale * screen.z;
    const double view_z = screen.z;

    return camera.from_camera_space({view_x, view_y, view_z});
}
