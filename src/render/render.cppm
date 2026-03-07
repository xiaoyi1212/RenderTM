module;

#include "../prelude.hpp"

export module render;

export import math;
export import camera;
export import world;
export import noise;
export import terrain;
export import settings;
export import lighting;
export import framebuffer;
export import rasterizer;
export import post;

struct FrameState
{
    size_t width = 0;
    size_t height = 0;
    size_t samples = 0;
    double width_d = 0.0;
    double height_d = 0.0;
    float depth_max = 0.0f;
    uint32_t frame_index = 0;

    bool taa_on = false;
    float taa_factor = 0.0f;
    bool clamp_history = false;

    bool gi_on = false;
    int gi_bounces = 0;
    float gi_scale = 0.0f;
    bool gi_active = false;

    float jitter_x = 0.0f;
    float jitter_y = 0.0f;
    double jitter_x_d = 0.0;
    double jitter_y_d = 0.0;

    double proj_scale_x = 0.0;
    double proj_scale_y = 0.0;
    double proj_a = 0.0;
    double proj_b = 0.0;

    ViewRotation view_rot{};
    Vec3 camera_pos{};
    double yaw = 0.0;
    double pitch = 0.0;

    LinearColor sky_top{};
    LinearColor sky_bottom{};
    LinearColor ambient_top{};
    LinearColor ambient_bottom{};
    LinearColor hemi_ground{};
    float sky_scale = 0.0f;
    float stars = 0.0f;

    bool shadows_on = false;
    bool direct_on = false;
    bool ao_on = false;
    double ambient = 0.0;

    std::array<ShadingContext::DirectionalLightInfo, 2> lights{};
    std::array<Vec3, 2> lights_right_scaled{};
    std::array<Vec3, 2> lights_up_scaled{};
    std::array<BlueNoise::Shift, 2> shadow_shift_u{};
    std::array<BlueNoise::Shift, 2> shadow_shift_v{};
};

export struct RenderEngine
{
    Camera camera{};
    World world{};
    Terrain terrain{};
    RenderSettings settings{};
    uint32_t renderFrameIndex = 0;
    RenderBuffers buffers{};
    Rasterizer rasterizer{};
    LightingEngine lighting{};
    PostProcessor post{};

    auto update(uint32_t* framebuffer, size_t width, size_t height) -> void
    {
        FrameState frame = begin_frame(width, height);
        rasterize_scene(frame);
        lighting.resolve_direct(DirectFrame{
            .width = frame.width,
            .height = frame.height,
            .samples = frame.samples,
            .depth_max = frame.depth_max,
            .shadows_on = frame.shadows_on
        }, buffers, settings.shadow);
        if (frame.gi_active)
        {
            lighting.gi_pass(GiFrame{
                .width = frame.width,
                .height = frame.height,
                .width_d = frame.width_d,
                .height_d = frame.height_d,
                .depth_max = frame.depth_max,
                .frame_index = frame.frame_index,
                .jitter_x_d = frame.jitter_x_d,
                .jitter_y_d = frame.jitter_y_d,
                .proj_a = frame.proj_a,
                .proj_b = frame.proj_b,
                .gi_bounces = frame.gi_bounces,
                .gi_scale = frame.gi_scale,
                .sky_scale = frame.sky_scale,
                .sky_top = frame.ambient_top,
                .hemi_ground = frame.hemi_ground,
                .lights = frame.lights,
                .inv_current_vp = post.inverseCurrentVP
            }, buffers, terrain, settings.gi);
        }
        post_process(frame, framebuffer);
    }

private:
    static constexpr int kTaaJitterSalt = 37;

    auto begin_frame(const size_t width, const size_t height) -> FrameState
    {
        FrameState frame{};
        frame.width = width;
        frame.height = height;
        frame.samples = width * height;
        frame.width_d = static_cast<double>(width);
        frame.height_d = static_cast<double>(height);
        frame.depth_max = std::numeric_limits<float>::max();

        frame.taa_on = settings.get_taa_enabled();
        frame.taa_factor = static_cast<float>(std::clamp(settings.get_taa_blend(), 0.0, 1.0));
        frame.clamp_history = settings.get_taa_clamp_enabled();
        frame.gi_on = settings.get_gi_enabled();
        frame.gi_scale = static_cast<float>(std::max(0.0, settings.get_gi_strength()));

        if (buffers.resize(width, height, frame.depth_max))
        {
            post.resize_buffers(frame.samples);
        }

        const bool paused = settings.is_paused();
        frame.frame_index = renderFrameIndex++;
        post.previousVP = post.currentVP;

        world.update_orbits(paused);

        frame.camera_pos = camera.position;
        frame.yaw = camera.rotation.x;
        frame.pitch = camera.rotation.y;

        post.update_taa_state(frame.taa_on, width, height, frame.samples,
                              frame.camera_pos, frame.yaw, frame.pitch);

        const double view_yaw = -frame.yaw;
        const double view_pitch = -frame.pitch;
        const Mat4 view = camera.view_matrix();
        frame.proj_scale_y = frame.height_d * 0.8;
        frame.proj_scale_x = frame.proj_scale_y;
        const Mat4 proj = Camera::projection(frame.width_d, frame.height_d,
                                             frame.proj_scale_x, frame.proj_scale_y);
        post.currentVP = proj * view;
        if (const auto inv = post.currentVP.invert())
        {
            post.inverseCurrentVP = *inv;
        }
        else
        {
            post.inverseCurrentVP = Mat4::identity();
        }

        const double inv_dist = 1.0 / (Camera::far_plane - Camera::near_plane);
        frame.proj_a = Camera::far_plane * inv_dist;
        frame.proj_b = -Camera::near_plane * Camera::far_plane * inv_dist;

        if (frame.taa_on)
        {
            const float jitter_scale = 1.0f;
            const float u = BlueNoise::sample(0, 0, static_cast<int>(frame.frame_index), kTaaJitterSalt);
            const float v = BlueNoise::sample(1, 0, static_cast<int>(frame.frame_index), kTaaJitterSalt + 1);
            frame.jitter_x = (u - 0.5f) * jitter_scale;
            frame.jitter_y = (v - 0.5f) * jitter_scale;
        }
        frame.jitter_x_d = static_cast<double>(frame.jitter_x);
        frame.jitter_y_d = static_cast<double>(frame.jitter_y);
        frame.view_rot = ViewRotation::from_yaw_pitch(view_yaw, view_pitch);

        const bool sun_orbit = world.sun.orbit_enabled;
        Vec3 sun_dir = world.sun.direction.normalize();
        double sun_intensity = world.sun.intensity;
        if (sun_orbit)
        {
            sun_dir = world.sun.direction_at(world.sun.orbit_angle);
            const double visibility = world.sun.height_factor(sun_dir);
            sun_intensity = world.sun.intensity * visibility;
        }
        sun_intensity *= settings.lighting.sun_intensity_boost;

        Vec3 moon_dir{-sun_dir.x, -sun_dir.y, -sun_dir.z};
        if (world.moon.orbit_enabled)
        {
            moon_dir = world.moon.direction_at(world.moon.orbit_angle);
        }
        if (moon_dir.x == 0.0 && moon_dir.y == 0.0 && moon_dir.z == 0.0)
        {
            moon_dir = world.moon.direction.normalize();
        }
        else
        {
            moon_dir = moon_dir.normalize();
        }

        double moon_intensity = std::max(0.0, world.moon.intensity);
        const double moon_visibility = world.moon.height_factor(moon_dir);
        moon_intensity *= moon_visibility;

        const float sun_height_sky = sun_orbit
                                         ? static_cast<float>(world.sun.height_signed(sun_dir))
                                         : 1.0f;
        const float sun_height_moon = static_cast<float>(world.sun.height_signed(sun_dir));
        const auto sky = world.sky.state(sun_height_sky, static_cast<float>(moon_intensity));
        const auto ambient = world.sky.ambient_gradient(sun_height_sky);
        const float moon_weight = world.sky.moon_weight(sun_height_moon);
        moon_intensity *= static_cast<double>(moon_weight);
        frame.sky_top = sky.zenith;
        frame.sky_bottom = sky.horizon;
        frame.ambient_top = ambient.zenith;
        frame.ambient_bottom = ambient.horizon;
        frame.sky_scale = sky.intensity;
        frame.stars = world.sky.star_visibility(sun_height_moon);

        frame.shadows_on = settings.get_shadow_enabled();
        frame.lights = {
            ShadingContext::DirectionalLightInfo{sun_dir, sun_intensity, world.sun.color, world.sun.angular_radius},
            ShadingContext::DirectionalLightInfo{moon_dir, moon_intensity, world.moon.color, world.moon.angular_radius}
        };

        for (int i = 0; i < 2; ++i)
        {
            if (frame.lights[i].angular_radius > 0.0 && frame.lights[i].intensity > 0.0)
            {
                auto [right, up, forward] = Vec3::get_basis(frame.lights[i].dir);
                const double scale = std::tan(frame.lights[i].angular_radius);

                frame.lights_right_scaled[i] = {right.x * scale, right.y * scale, right.z * scale};
                frame.lights_up_scaled[i]    = {up.x * scale, up.y * scale, up.z * scale};
            }
            else
            {
                frame.lights_right_scaled[i] = {0.0, 0.0, 0.0};
                frame.lights_up_scaled[i]    = {0.0, 0.0, 0.0};
            }
        }

        const auto& shadow_settings = settings.shadow;
        frame.shadow_shift_u = {
            BlueNoise::shift(static_cast<int>(frame.frame_index), shadow_settings.sun_salt),
            BlueNoise::shift(static_cast<int>(frame.frame_index), shadow_settings.moon_salt)
        };
        frame.shadow_shift_v = {
            BlueNoise::shift(static_cast<int>(frame.frame_index), shadow_settings.sun_salt + 1),
            BlueNoise::shift(static_cast<int>(frame.frame_index), shadow_settings.moon_salt + 1)
        };

        frame.direct_on = (frame.lights[0].intensity > 0.0) || (frame.lights[1].intensity > 0.0);
        frame.ao_on = settings.get_ambient_occlusion_enabled();
        frame.hemi_ground = lighting.hemi_ground(frame.ambient_bottom, frame.lights, settings.lighting);
        frame.ambient = settings.get_ambient_light();

        const int raw_bounces = settings.get_gi_bounce_count();
        frame.gi_bounces = raw_bounces > 0 ? raw_bounces : 0;
        frame.gi_active = frame.gi_on && frame.gi_scale > 0.0f && frame.gi_bounces > 0;

        return frame;
    }

    auto rasterize_scene(const FrameState& frame) -> void
    {
        if (terrain.blocks.empty() || terrain.mesh.empty())
        {
            terrain.generate();
        }

        const Material base_material{0xFFFFFFFF, 0.25, 1.0, 0.15, 24.0};

        std::array<uint32_t, 8> ctx_colors{};
        std::array<ShadingContext, 8> ctx_cache{};
        size_t ctx_count = 0;

        auto get_ctx = [&](const uint32_t color) -> ShadingContext& {
            for (size_t i = 0; i < ctx_count; ++i)
            {
                if (ctx_colors[i] == color)
                {
                    return ctx_cache[i];
                }
            }
            Material material = base_material;
            material.color = color;
            const ColorSrgb albedo_srgb = ColorSrgb::from_hex(material.color);
            const LinearColor albedo = albedo_srgb.to_linear();
            ShadingContext ctx{
                albedo,
                frame.ambient_top,
                frame.ambient_bottom,
                frame.hemi_ground,
                frame.sky_scale,
                frame.camera_pos,
                frame.ambient,
                material,
                frame.direct_on,
                frame.ao_on,
                frame.shadows_on,
                frame.lights
            };
            size_t slot = 0;
            if (ctx_count < ctx_cache.size())
            {
                slot = ctx_count++;
            }
            ctx_cache[slot] = ctx;
            ctx_colors[slot] = color;
            return ctx_cache[slot];
        };

        const RasterTarget target{
            .zbuffer = buffers.zbuffer.data(),
            .sample_colors = buffers.sample_colors.data(),
            .sample_direct_sun = buffers.sample_direct_sun.data(),
            .sample_direct_moon = buffers.sample_direct_moon.data(),
            .shadow_mask_sun = buffers.shadow_mask_sun.data(),
            .shadow_mask_moon = buffers.shadow_mask_moon.data(),
            .sample_normals = buffers.sample_normals.data(),
            .sample_albedo = buffers.sample_albedo.data(),
            .sample_ao = buffers.sample_ao.data(),
            .world_positions = buffers.world_positions.data(),
            .world_stamp = buffers.world_stamp.data(),
            .frame_index = frame.frame_index,
            .width = frame.width,
            .height = frame.height
        };

        for (const auto& quad : terrain.mesh)
        {
            ShadingContext& ctx = get_ctx(quad.color);
            const RasterInputs inputs{
                .terrain = terrain,
                .lighting = lighting,
                .shadow_settings = settings.shadow,
                .ctx = ctx,
                .jitter_x = frame.jitter_x,
                .jitter_y = frame.jitter_y,
                .lights_right_scaled = frame.lights_right_scaled,
                .lights_up_scaled = frame.lights_up_scaled,
                .shadow_shift_u = frame.shadow_shift_u,
                .shadow_shift_v = frame.shadow_shift_v
            };
            const RasterQuadInput quad_input{
                .quad = quad,
                .proj_scale_x = frame.proj_scale_x,
                .proj_scale_y = frame.proj_scale_y,
                .camera_pos = frame.camera_pos,
                .view_rot = frame.view_rot,
                .near_plane = Camera::near_plane
            };
            rasterizer.render_quad(target, quad_input, inputs);
        }
    }

    auto post_process(const FrameState& frame, uint32_t* framebuffer) -> void
    {
        const float exposure = static_cast<float>(std::max(0.0, world.sky.exposure));
        post.resolve_frame(framebuffer, buffers,
                           frame.sky_top, frame.sky_bottom, world.sky,
                           frame.taa_on, frame.clamp_history, frame.taa_factor,
                           frame.gi_active, settings.gi, frame.frame_index,
                           frame.jitter_x, frame.jitter_y,
                           exposure, frame.stars);
    }
};
