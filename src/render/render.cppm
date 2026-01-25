module;

#include "../prelude.hpp"

export module render;

export import :math;
export import :noise;
export import :terrain;

export void render_update_array(uint32_t* framebuffer, size_t width, size_t height);
export void render_set_paused(bool paused);
export bool render_is_paused();
export void render_toggle_pause();
export void render_set_light_direction(Vec3 dir);
export Vec3 render_get_light_direction();
export void render_set_light_intensity(double intensity);
export double render_get_light_intensity();
export void render_set_sun_orbit_enabled(bool enabled);
export bool render_get_sun_orbit_enabled();
export void render_set_sun_orbit_angle(double angle);
export double render_get_sun_orbit_angle();
export void render_set_moon_direction(Vec3 dir);
export void render_set_moon_intensity(double intensity);
export void render_set_sky_top_color(uint32_t color);
export uint32_t render_get_sky_top_color();
export void render_set_sky_bottom_color(uint32_t color);
export uint32_t render_get_sky_bottom_color();
export void render_set_sky_light_intensity(double intensity);
export double render_get_sky_light_intensity();
export void render_set_exposure(double exposure);
export double render_get_exposure();
export void render_set_taa_enabled(bool enabled);
export bool render_get_taa_enabled();
export void render_set_taa_blend(double blend);
export double render_get_taa_blend();
export void render_set_taa_clamp_enabled(bool enabled);
export bool render_get_taa_clamp_enabled();
export void render_set_gi_enabled(bool enabled);
export bool render_get_gi_enabled();
export void render_set_gi_strength(double strength);
export double render_get_gi_strength();
export void render_set_gi_bounce_count(int count);
export int render_get_gi_bounce_count();
export void render_reset_taa_history();
export void render_set_ambient_occlusion_enabled(bool enabled);
export void render_set_shadow_enabled(bool enabled);
export bool render_get_shadow_factor_at_point(Vec3 world, Vec3 normal, float* out_factor);
export bool render_get_terrain_vertex_sky_visibility(int x, int y, int z, int face, int corner, float* out_visibility);
export void render_set_camera_position(Vec3 pos);
export Vec3 render_get_camera_position();
export void render_move_camera(Vec3 delta);
export void render_move_camera_local(Vec3 delta);
export void render_set_camera_rotation(Vec2 rot);
export Vec2 render_get_camera_rotation();
export void render_rotate_camera(Vec2 delta);
export Vec2 render_project_point(Vec3 world, size_t width, size_t height);
export Vec3 render_unproject_point(Vec3 screen, size_t width, size_t height);
export Vec2 render_reproject_point(Vec3 world, size_t width, size_t height);
export bool render_debug_depth_at_sample(Vec3 v0, Vec3 v1, Vec3 v2, Vec2 p, float* out_depth);
export double render_debug_eval_specular(double ndoth, double vdoth, double ndotl,
                                         double shininess, double f0);
export Vec3 render_debug_tonemap_reinhard(Vec3 color, double exposure);
export Vec3 render_debug_sample_history_bilinear(const Vec3* buffer, size_t width, size_t height, Vec2 screen_coord);
export void render_debug_set_sky_colors_raw(uint32_t top, uint32_t bottom);
export Mat4 render_debug_get_current_vp();
export Mat4 render_debug_get_previous_vp();
export Mat4 render_debug_get_inverse_current_vp();
export double render_debug_get_taa_sharpen_strength();
export double render_debug_get_taa_sharpen_percent();
export void render_debug_set_frame_index(uint32_t frame_index);
export uint32_t render_debug_get_frame_index();
export bool render_debug_shadow_factor_with_frame(Vec3 world, Vec3 normal, Vec3 light_dir,
                                                  int pixel_x, int pixel_y, int frame,
                                                  float* out_factor);
export float render_debug_shadow_filter(const float* mask, const float* depth, const Vec3* normals);
export size_t render_debug_get_terrain_block_count();
export size_t render_debug_get_terrain_visible_face_count();
export size_t render_debug_get_terrain_triangle_count();
export bool render_should_rasterize_triangle(Vec3 v0, Vec3 v1, Vec3 v2);
export double render_get_near_plane();
export size_t render_clip_triangle_to_near_plane(Vec3 v0, Vec3 v1, Vec3 v2, Vec3* out_vertices, size_t max_vertices);

static std::atomic<bool> rotationPaused{false};
static std::atomic<uint32_t> renderFrameIndex{0};
static std::atomic<double> lightDirectionX{0.0};
static std::atomic<double> lightDirectionY{0.0};
static std::atomic<double> lightDirectionZ{-1.0};
static std::atomic<uint64_t> lightDirectionSeq{0};
static std::atomic<double> lightIntensity{1.1};
static std::atomic<bool> sunOrbitEnabled{true};
static std::atomic<double> sunOrbitAngle{1.5707963267948966};
static std::atomic<double> sunOrbitSpeed{0.00075};
static std::atomic<double> moonDirectionX{0.0};
static std::atomic<double> moonDirectionY{1.0};
static std::atomic<double> moonDirectionZ{0.0};
static std::atomic<uint64_t> moonDirectionSeq{0};
static std::atomic<double> moonIntensity{0.2};
static std::atomic<uint32_t> skyTopColor{0xFF78C2FF};
static std::atomic<uint32_t> skyBottomColor{0xFF172433};
static std::atomic<double> skyLightIntensity{0.32};
static std::atomic<double> ambientLight{0.13};
static std::atomic<double> exposure{1.0};
static std::atomic<bool> taaEnabled{true};
static std::atomic<double> taaBlend{0.05};
static std::atomic<bool> taaClampEnabled{true};
static std::atomic<double> taaSharpenStrength{0.0};
static std::atomic<bool> giEnabled{false};
static std::atomic<double> giStrength{0.0};
static std::atomic<int> giBounceCount{2};
static std::atomic<uint64_t> renderStateVersion{1};
static std::atomic<double> camera_x{16.0};
static std::atomic<double> camera_y{-19.72};
static std::atomic<double> camera_z{-1.93};
static std::atomic<double> camera_yaw{-0.6911503837897546};
static std::atomic<double> camera_pitch{-0.6003932626860493};
static std::atomic<bool> ambientOcclusionEnabled{true};
static std::atomic<bool> shadowEnabled{true};
static Mat4 currentVP{{{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0}
}}};
static Mat4 previousVP{{{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0}
}}};
static Mat4 inverseCurrentVP{{{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0}
}}};
static Terrain terrain{};


static void mark_render_state_dirty()
{
    renderStateVersion.fetch_add(1u, std::memory_order_relaxed);
}

static Vec3 load_light_direction()
{
    for (;;)
    {
        const uint64_t seq0 = lightDirectionSeq.load(std::memory_order_acquire);
        if (seq0 & 1u)
        {
            continue;
        }
        const Vec3 value{
            lightDirectionX.load(std::memory_order_relaxed),
            lightDirectionY.load(std::memory_order_relaxed),
            lightDirectionZ.load(std::memory_order_relaxed)
        };
        std::atomic_signal_fence(std::memory_order_acquire);
        const uint64_t seq1 = lightDirectionSeq.load(std::memory_order_acquire);
        if (seq0 == seq1)
        {
            return value;
        }
    }
}

static void store_light_direction(const Vec3& dir)
{
    lightDirectionSeq.fetch_add(1u, std::memory_order_acq_rel);
    lightDirectionX.store(dir.x, std::memory_order_relaxed);
    lightDirectionY.store(dir.y, std::memory_order_relaxed);
    lightDirectionZ.store(dir.z, std::memory_order_relaxed);
    lightDirectionSeq.fetch_add(1u, std::memory_order_acq_rel);
}

static Vec3 load_moon_direction()
{
    for (;;)
    {
        const uint64_t seq0 = moonDirectionSeq.load(std::memory_order_acquire);
        if (seq0 & 1u)
        {
            continue;
        }
        const Vec3 value{
            moonDirectionX.load(std::memory_order_relaxed),
            moonDirectionY.load(std::memory_order_relaxed),
            moonDirectionZ.load(std::memory_order_relaxed)
        };
        std::atomic_signal_fence(std::memory_order_acquire);
        const uint64_t seq1 = moonDirectionSeq.load(std::memory_order_acquire);
        if (seq0 == seq1)
        {
            return value;
        }
    }
}

static void store_moon_direction(const Vec3& dir)
{
    moonDirectionSeq.fetch_add(1u, std::memory_order_acq_rel);
    moonDirectionX.store(dir.x, std::memory_order_relaxed);
    moonDirectionY.store(dir.y, std::memory_order_relaxed);
    moonDirectionZ.store(dir.z, std::memory_order_relaxed);
    moonDirectionSeq.fetch_add(1u, std::memory_order_acq_rel);
}

struct ScreenVertex
{
    float x, y, z;
};

struct ViewRotation
{
    double cy;
    double sy;
    double cp;
    double sp;
};

struct Material
{
    uint32_t color;
    double ambient;
    double diffuse;
    double specular;
    double shininess;
};

struct ShadingContext
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

struct TaaContext {
    std::array<std::vector<LinearColor>, 2> buffers;
    int write_index = 0;
    bool valid = false;
    size_t width = 0, height = 0;

    void ensure_size(size_t w, size_t h) {
        if (width != w || height != h) {
            width = w; height = h;
            buffers[0].assign(w * h, {0,0,0});
            buffers[1].assign(w * h, {0,0,0});
            valid = false;
            write_index = 0;
        }
    }

    const LinearColor* get_read_buffer() const {
        return buffers[(write_index + 1) % 2].data();
    }

    LinearColor* get_write_buffer() {
        return buffers[write_index].data();
    }

    void swap() {
        valid = true;
        write_index = (write_index + 1) % 2;
    }
};

constexpr double kShadowRayBias = 0.05;
constexpr double kGiRayBias = 0.04;
constexpr double kGiMaxDistance = 12.0;
constexpr int kGiNoiseSalt = 73;
constexpr float kGiClamp = 4.0f;
constexpr int kGiSampleCount = 1;
constexpr float kGiAoLift = 0.15f;
constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kSunLatitudeDeg = 30.0;
constexpr double kSunLatitudeRad = kPi * kSunLatitudeDeg / 180.0;
constexpr double kSunDiskRadius = 0.03;
constexpr int kSunShadowSalt = 17;
constexpr int kMoonShadowSalt = 19;
constexpr uint32_t kSkySunriseTopColor = 0xFFB55A1A;
constexpr uint32_t kSkySunriseBottomColor = 0xFF4A200A;
constexpr double kHemisphereBounceStrength = 0.35;
constexpr LinearColor kHemisphereBounceColorLinear{1.0f, 0.9046612f, 0.7758222f};
constexpr double kSkyLightHeightPower = 0.5;
constexpr LinearColor kSunLightColorLinear{1.0f, 0.94f, 0.88f};
constexpr LinearColor kMoonLightColorLinear{1.0f, 1.0f, 1.0f};
constexpr double kSunIntensityBoost = 1.2;
constexpr double kMoonSkyLightFloor = 0.22;
constexpr double kNearPlane = 0.05;
constexpr double kFarPlane = 1000.0;
constexpr int kTaaJitterSalt = 37;
constexpr double kTaaSharpenMax = 0.25;
constexpr double kTaaSharpenRotThreshold = 0.25;
constexpr double kTaaSharpenMoveThreshold = 0.5;
constexpr double kTaaSharpenMoveGain = 10.0;
constexpr double kTaaSharpenRotGain = 20.0;
constexpr float kTaaSharpenAttack = 0.5f;
constexpr float kTaaSharpenRelease = 0.2f;
constexpr bool kShadowFilterEnabled = true;
constexpr float kShadowFilterDepthThreshold = 1.0f;
constexpr float kShadowFilterNormalThreshold = 0.5f;
constexpr float kShadowFilterCenterWeight = 4.0f;
constexpr float kShadowFilterNeighborWeight = 1.0f;

static float compute_shadow_factor(const Vec3& light_dir, const Vec3& world, const Vec3& normal);
static bool triangle_in_front_of_near_plane(double z0, double z1, double z2);
static Vec3 jitter_shadow_direction(const Vec3& light_dir, 
                                    const Vec3& right_scaled,
                                    const Vec3& up_scaled,
                                    const int px, const int py,
                                    const BlueNoise::Shift& shift_u,
                                    const BlueNoise::Shift& shift_v);
static float shadow_filter_at(std::span<const float> mask,
                              std::span<const float> depth,
                              std::span<const Vec3> normals,
                              size_t width, size_t height, int x, int y, float depth_max);
static void filter_shadow_masks(std::span<const float> mask_a,
                                std::span<const float> mask_b,
                                std::span<float> out_a,
                                std::span<float> out_b,
                                std::span<const float> depth,
                                std::span<const Vec3> normals,
                                size_t width, size_t height, float depth_max);

static const std::array<Vec2, 64>& disk_sample_points()
{
    static const std::array<Vec2, 64> samples = [] {
        std::array<Vec2, 64> points{};
        constexpr double golden_angle = 2.39996322972865332;
        const double count_inv = 1.0 / static_cast<double>(points.size());
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

template <typename ColorT>
static ColorT lerp_color(const ColorT& a, const ColorT& b, float t)
{
    return {
        std::lerp(a.r, b.r, t),
        std::lerp(a.g, b.g, t),
        std::lerp(a.b, b.b, t)
    };
}

static Vec3 lerp_vec3(const Vec3& a, const Vec3& b, const double t)
{
    return {
        std::lerp(a.x, b.x, t),
        std::lerp(a.y, b.y, t),
        std::lerp(a.z, b.z, t)
    };
}

static LinearColor sample_bilinear_history(std::span<const LinearColor> buffer, const size_t width,
                                        const size_t height, const double screen_x, const double screen_y)
{
    if (buffer.empty() || width == 0 || height == 0)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    double x = screen_x - 0.5;
    double y = screen_y - 0.5;
    x = std::clamp(x, 0.0, static_cast<double>(width - 1));
    y = std::clamp(y, 0.0, static_cast<double>(height - 1));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, static_cast<int>(width) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(height) - 1);
    const float fx = static_cast<float>(x - static_cast<double>(x0));
    const float fy = static_cast<float>(y - static_cast<double>(y0));

    const size_t row0 = static_cast<size_t>(y0) * width;
    const size_t row1 = static_cast<size_t>(y1) * width;
    const LinearColor c00 = buffer[row0 + static_cast<size_t>(x0)];
    const LinearColor c10 = buffer[row0 + static_cast<size_t>(x1)];
    const LinearColor c01 = buffer[row1 + static_cast<size_t>(x0)];
    const LinearColor c11 = buffer[row1 + static_cast<size_t>(x1)];
    const LinearColor top = lerp_color(c00, c10, fx);
    const LinearColor bottom = lerp_color(c01, c11, fx);
    return lerp_color(top, bottom, fy);
}

static Vec3 sample_bilinear_history_vec3(std::span<const Vec3> buffer, const size_t width,
                                         const size_t height, const double screen_x, const double screen_y)
{
    if (buffer.empty() || width == 0 || height == 0)
    {
        return {0.0, 0.0, 0.0};
    }

    double x = screen_x - 0.5;
    double y = screen_y - 0.5;
    x = std::clamp(x, 0.0, static_cast<double>(width - 1));
    y = std::clamp(y, 0.0, static_cast<double>(height - 1));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, static_cast<int>(width) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(height) - 1);
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);

    const size_t row0 = static_cast<size_t>(y0) * width;
    const size_t row1 = static_cast<size_t>(y1) * width;
    const Vec3 c00 = buffer[row0 + static_cast<size_t>(x0)];
    const Vec3 c10 = buffer[row0 + static_cast<size_t>(x1)];
    const Vec3 c01 = buffer[row1 + static_cast<size_t>(x0)];
    const Vec3 c11 = buffer[row1 + static_cast<size_t>(x1)];
    const Vec3 top = lerp_vec3(c00, c10, fx);
    const Vec3 bottom = lerp_vec3(c01, c11, fx);
    return lerp_vec3(top, bottom, fy);
}

static LinearColor scale_color(const LinearColor& color, float scale)
{
    return {color.r * scale, color.g * scale, color.b * scale};
}

static LinearColor mul_color(const LinearColor& a, const LinearColor& b)
{
    return {a.r * b.r, a.g * b.g, a.b * b.b};
}

static LinearColor add_color(const LinearColor& a, const LinearColor& b)
{
    return {a.r + b.r, a.g + b.g, a.b + b.b};
}

static LinearColor compute_hemisphere_ground(const LinearColor& base_ground,
                                          const std::array<ShadingContext::DirectionalLightInfo, 2>& lights)
{
    double bounce_energy = 0.0;
    for (const auto& light : lights)
    {
        if (light.intensity <= 0.0)
        {
            continue;
        }
        const double height = std::clamp(-light.dir.y, 0.0, 1.0);
        bounce_energy += light.intensity * height;
    }
    const double bounce = std::clamp(bounce_energy * kHemisphereBounceStrength, 0.0, 1.0);
    if (bounce <= 0.0)
    {
        return base_ground;
    }
    return lerp_color(base_ground, kHemisphereBounceColorLinear, static_cast<float>(bounce));
}

static double pow5(const double value)
{
    const double v2 = value * value;
    return v2 * v2 * value;
}

static double schlick_fresnel(double vdoth, double f0)
{
    if (f0 <= 0.0)
    {
        return 0.0;
    }
    f0 = std::clamp(f0, 0.0, 1.0);
    vdoth = std::clamp(vdoth, 0.0, 1.0);
    const double one_minus = 1.0 - vdoth;
    return f0 + (1.0 - f0) * pow5(one_minus);
}

static double blinn_phong_normalization(const double shininess)
{
    return (shininess + 8.0) / (8.0 * kPi);
}

static inline void sincos_double(const double angle, double* out_sin, double* out_cos)
{
#if defined(__GNUC__)
    __builtin_sincos(angle, out_sin, out_cos);
#else
    *out_sin = std::sin(angle);
    *out_cos = std::cos(angle);
#endif
}

static double eval_specular_term(const double ndoth, const double vdoth, const double ndotl,
                                 const double shininess, const double f0)
{
    if (ndoth <= 0.0 || ndotl <= 0.0 || f0 <= 0.0)
    {
        return 0.0;
    }
    const double ndoth_clamped = std::clamp(ndoth, 0.0, 1.0);
    const double ndotl_clamped = std::clamp(ndotl, 0.0, 1.0);
    const double power = std::pow(ndoth_clamped, shininess);
    const double fresnel = schlick_fresnel(vdoth, f0);
    return blinn_phong_normalization(shininess) * power * fresnel * ndotl_clamped;
}

static bool triangle_in_front_of_near_plane(const double z0, const double z1, const double z2)
{
    return z0 >= kNearPlane || z1 >= kNearPlane || z2 >= kNearPlane;
}

struct ClipVertex
{
    Vec3 view;
    Vec3 world;
    Vec3 normal;
    float sky_visibility;
};

static ClipVertex clip_lerp(const ClipVertex& a, const ClipVertex& b, const double t)
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

static size_t clip_triangle_to_near_plane(std::span<const ClipVertex> input,
                                          std::span<ClipVertex> output)
{
    if (input.empty() || output.empty())
    {
        return 0;
    }
    size_t out_count = 0;
    ClipVertex prev = input[input.size() - 1];
    bool prev_inside = prev.view.z >= kNearPlane;

    for (size_t i = 0; i < input.size(); ++i)
    {
        const ClipVertex cur = input[i];
        const bool cur_inside = cur.view.z >= kNearPlane;

        if (cur_inside)
        {
            if (!prev_inside)
            {
                const double t = (kNearPlane - prev.view.z) / (cur.view.z - prev.view.z);
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
            const double t = (kNearPlane - prev.view.z) / (cur.view.z - prev.view.z);
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

static Vec3 compute_sun_orbit_direction(double angle)
{
    const double max_alt = kPi * 0.5 - kSunLatitudeRad;
    angle = std::fmod(angle, kPi);
    if (angle < 0.0)
    {
        angle += kPi;
    }
    const double alt = max_alt * std::sin(angle);
    const double az = angle - kPi * 0.5;
    const double cos_alt = std::cos(alt);
    const double sin_alt = std::sin(alt);
    const double x = cos_alt * std::sin(az);
    const double z = cos_alt * std::cos(az);
    const double y = -sin_alt;
    return Vec3{x, y, z}.normalize();
}

static double compute_sun_height(const Vec3& sun_dir)
{
    const double max_y = std::cos(kSunLatitudeRad);
    if (max_y <= 0.0)
    {
        return 0.0;
    }
    const double height = sun_dir.y < 0.0 ? (-sun_dir.y) / max_y : 0.0;
    return std::clamp(height, 0.0, 1.0);
}

static float tonemap_reinhard_channel(float value)
{
    if (value <= 0.0f)
    {
        return 0.0f;
    }
    return value / (1.0f + value);
}

static LinearColor tonemap_reinhard(const LinearColor& color, const float exposure_factor)
{
    if (exposure_factor <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }
    const float r = color.r * exposure_factor;
    const float g = color.g * exposure_factor;
    const float b = color.b * exposure_factor;
    return {
        tonemap_reinhard_channel(r),
        tonemap_reinhard_channel(g),
        tonemap_reinhard_channel(b)
    };
}

static uint32_t pack_color(const ColorSrgb& color)
{
    const auto clamp_channel = [](float value) {
        if (value < 0.0f) value = 0.0f;
        if (value > 255.0f) value = 255.0f;
        return static_cast<uint32_t>(std::lround(value));
    };
    const uint32_t r = clamp_channel(color.r);
    const uint32_t g = clamp_channel(color.g);
    const uint32_t b = clamp_channel(color.b);
    return 0xFF000000 | (r << 16) | (g << 8) | b;
}

static Vec3 rotate_yaw_pitch_cached(const Vec3& v, const double cy, const double sy,
                                    const double cp, const double sp)
{
    const double x1 = v.x * cy + v.z * sy;
    const double z1 = -v.x * sy + v.z * cy;
    const double y1 = v.y;

    const double y2 = y1 * cp - z1 * sp;
    const double z2 = y1 * sp + z1 * cp;

    return {x1, y2, z2};
}

static Vec3 rotate_yaw_pitch(const Vec3& v, const double yaw, const double pitch)
{
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);
    return rotate_yaw_pitch_cached(v, cy, sy, cp, sp);
}

static Vec3 rotate_pitch_yaw(const Vec3& v, const double yaw, const double pitch)
{
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);

    const double x1 = v.x;
    const double y1 = v.y * cp - v.z * sp;
    const double z1 = v.y * sp + v.z * cp;

    const double x2 = x1 * cy + z1 * sy;
    const double z2 = -x1 * sy + z1 * cy;

    return {x2, y1, z2};
}

static ViewRotation make_view_rotation(const double yaw, const double pitch)
{
    return {std::cos(yaw), std::sin(yaw), std::cos(pitch), std::sin(pitch)};
}

static auto make_view_matrix(const Vec3& pos, const double yaw, const double pitch) -> Mat4
{
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);

    Mat4 m = Mat4::identity();
    m.m[0][0] = cy;
    m.m[0][1] = 0.0;
    m.m[0][2] = sy;

    m.m[1][0] = sy * sp;
    m.m[1][1] = cp;
    m.m[1][2] = -cy * sp;

    m.m[2][0] = -sy * cp;
    m.m[2][1] = sp;
    m.m[2][2] = cy * cp;

    m.m[0][3] = -(m.m[0][0] * pos.x + m.m[0][1] * pos.y + m.m[0][2] * pos.z);
    m.m[1][3] = -(m.m[1][0] * pos.x + m.m[1][1] * pos.y + m.m[1][2] * pos.z);
    m.m[2][3] = -(m.m[2][0] * pos.x + m.m[2][1] * pos.y + m.m[2][2] * pos.z);
    return m;
}

static auto make_projection_matrix(const double width, const double height,
                                   const double fov_x, const double fov_y) -> Mat4
{
    if (width <= 0.0 || height <= 0.0 || kFarPlane <= kNearPlane)
    {
        return Mat4::identity();
    }
    const double sx = 2.0 * fov_x / width;
    const double sy = 2.0 * fov_y / height;
    const double inv_range = 1.0 / (kFarPlane - kNearPlane);
    const double a = kFarPlane * inv_range;
    const double b = -kNearPlane * kFarPlane * inv_range;

    Mat4 m{};
    m.m[0][0] = sx;
    m.m[1][1] = sy;
    m.m[2][2] = a;
    m.m[2][3] = b;
    m.m[3][2] = 1.0;
    return m;
}

static double clamp_pitch(const double pitch)
{
    const double limit = 1.4;
    if (pitch > limit) return limit;
    if (pitch < -limit) return -limit;
    return pitch;
}

static inline float edge_function(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c)
{
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

static void draw_shaded_triangle(float* zbuffer, LinearColor* sample_ambient,
                                 LinearColor* sample_direct_sun, LinearColor* sample_direct_moon,
                                 float* shadow_mask_sun, float* shadow_mask_moon,
                                 Vec3* sample_normals, LinearColor* sample_albedo, float* sample_ao,
                                 Vec3* world_positions, uint32_t* world_stamp, uint32_t frame_index,
                                 size_t width, size_t height,
                                 const ScreenVertex& v0, const ScreenVertex& v1, const ScreenVertex& v2,
                                 const Vec3& wp0, const Vec3& wp1, const Vec3& wp2,
                                 const Vec3& n0, const Vec3& n1, const Vec3& n2,
                                 const float vis0, const float vis1, const float vis2,
                                 const ShadingContext& ctx,
                                 const float jitter_x, const float jitter_y,
                                 const std::array<Vec3, 2>& lights_right_scaled,
                                 const std::array<Vec3, 2>& lights_up_scaled,
                                 const std::array<BlueNoise::Shift, 2>& shadow_shift_u,
                                 const std::array<BlueNoise::Shift, 2>& shadow_shift_v)
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
                            const LinearColor sky = lerp_color(hemi_ground, sky_top, sky_t);
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
                                    const Vec3 shadow_dir = jitter_shadow_direction(light.dir,
                                                            lights_right_scaled[light_idx],
                                                            lights_up_scaled[light_idx],
                                                            x, y,
                                                            shadow_shift_u[light_idx],
                                                            shadow_shift_v[light_idx]);
                                    out_shadow = compute_shadow_factor(shadow_dir, world, normal);
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

static Vec3 jitter_shadow_direction(const Vec3& light_dir, 
                                    const Vec3& right_scaled,
                                    const Vec3& up_scaled,
                                    const int px, const int py,
                                    const BlueNoise::Shift& shift_u,
                                    const BlueNoise::Shift& shift_v)
{
    if (right_scaled.x == 0.0 && right_scaled.y == 0.0 && right_scaled.z == 0.0)
    {
        return light_dir;
    }

    const float u1 = BlueNoise::sample(px, py, shift_u);
    const float u2 = BlueNoise::sample(px, py, shift_v);
    const int ix = static_cast<int>(u1 * 8.0f);
    const int iy = static_cast<int>(u2 * 8.0f);
    const size_t idx = static_cast<size_t>((iy & 7) * 8 + (ix & 7));
    const Vec2 sample = disk_sample_points()[idx];
    const double dx = sample.x;
    const double dy = sample.y;

    return Vec3{
        light_dir.x + right_scaled.x * dx + up_scaled.x * dy,
        light_dir.y + right_scaled.y * dx + up_scaled.y * dy,
        light_dir.z + right_scaled.z * dx + up_scaled.z * dy
    };
}

struct GiHit
{
    Vec3 position;
    Vec3 normal;
    LinearColor albedo;
    float sky_visibility;
};

static bool gi_raymarch_hit(const Vec3& world, const Vec3& normal, const Vec3& dir,
                            const double max_distance, GiHit* out_hit)
{
    if (!out_hit)
    {
        return false;
    }
    const int terrain_size = terrain.chunk_size;
    const int terrain_height = terrain.max_height();
    if (terrain_size <= 0 || terrain_height <= 0)
    {
        return false;
    }

    const double block_size = terrain.block_size();
    const double half = block_size * 0.5;
    const double start_x = terrain.start_x();
    const double start_z = terrain.start_z();
    const double base_y = terrain.base_y();
    const double inv_block = 1.0 / block_size;

    const Vec3 origin_world{
        world.x + normal.x * kGiRayBias,
        world.y + normal.y * kGiRayBias,
        world.z + normal.z * kGiRayBias
    };
    const Vec3 origin{
        (origin_world.x - start_x + half) * inv_block,
        (base_y - origin_world.y + half) * inv_block,
        (origin_world.z - start_z + half) * inv_block
    };
    const Vec3 dir_grid{
        dir.x * inv_block,
        -dir.y * inv_block,
        dir.z * inv_block
    };
    if (dir_grid.x == 0.0 && dir_grid.y == 0.0 && dir_grid.z == 0.0)
    {
        return false;
    }

    const double origin_x_floor = std::floor(origin.x);
    const double origin_y_floor = std::floor(origin.y);
    const double origin_z_floor = std::floor(origin.z);

    int x = static_cast<int>(origin_x_floor);
    int y = static_cast<int>(origin_y_floor);
    int z = static_cast<int>(origin_z_floor);

    if (x < 0 || x >= terrain_size || z < 0 || z >= terrain_size || y < 0 || y >= terrain_height)
    {
        return false;
    }

    const double inf = std::numeric_limits<double>::infinity();
    const int step_x = dir_grid.x > 0.0 ? 1 : (dir_grid.x < 0.0 ? -1 : 0);
    const int step_y = dir_grid.y > 0.0 ? 1 : (dir_grid.y < 0.0 ? -1 : 0);
    const int step_z = dir_grid.z > 0.0 ? 1 : (dir_grid.z < 0.0 ? -1 : 0);

    const double t_delta_x = step_x != 0 ? 1.0 / std::abs(dir_grid.x) : inf;
    const double t_delta_y = step_y != 0 ? 1.0 / std::abs(dir_grid.y) : inf;
    const double t_delta_z = step_z != 0 ? 1.0 / std::abs(dir_grid.z) : inf;

    const double next_x = step_x > 0 ? (origin_x_floor + 1.0) : origin_x_floor;
    const double next_y = step_y > 0 ? (origin_y_floor + 1.0) : origin_y_floor;
    const double next_z = step_z > 0 ? (origin_z_floor + 1.0) : origin_z_floor;

    double t_max_x = step_x != 0 ? (next_x - origin.x) / dir_grid.x : inf;
    double t_max_y = step_y != 0 ? (next_y - origin.y) / dir_grid.y : inf;
    double t_max_z = step_z != 0 ? (next_z - origin.z) / dir_grid.z : inf;

    const double max_t = max_distance * inv_block;
    const int max_steps = (terrain_size + terrain_size + terrain_height) * 4;
    bool skip_first = true;
    double traveled = 0.0;
    int last_axis = -1;

    for (int i = 0; i < max_steps; ++i)
    {
        if (!skip_first)
        {
            const VoxelBlock* block = terrain.block_at(x, y, z);
            if (block)
            {
                Vec3 hit_normal{0.0, 0.0, 0.0};
                int face = FaceTop;
                if (last_axis == 0)
                {
                    hit_normal = {-static_cast<double>(step_x), 0.0, 0.0};
                    face = (step_x > 0) ? FaceLeft : FaceRight;
                }
                else if (last_axis == 1)
                {
                    hit_normal = {0.0, static_cast<double>(step_y), 0.0};
                    face = (step_y > 0) ? FaceBottom : FaceTop;
                }
                else if (last_axis == 2)
                {
                    hit_normal = {0.0, 0.0, -static_cast<double>(step_z)};
                    face = (step_z > 0) ? FaceBack : FaceFront;
                }

                float visibility = 1.0f;
                if (face >= 0 && face < 6)
                {
                    float sum = 0.0f;
                    for (int c = 0; c < 4; ++c)
                    {
                        sum += block->face_sky_visibility[static_cast<size_t>(face)][static_cast<size_t>(c)];
                    }
                    visibility = sum * 0.25f;
                }

                const double hit_dist = traveled * block_size;
                out_hit->position = {
                    origin_world.x + dir.x * hit_dist,
                    origin_world.y + dir.y * hit_dist,
                    origin_world.z + dir.z * hit_dist
                };
                out_hit->normal = hit_normal;
                out_hit->albedo = block->albedo_linear;
                out_hit->sky_visibility = std::clamp(visibility, 0.0f, 1.0f);
                return true;
            }
        }
        skip_first = false;

        if (t_max_x < t_max_y)
        {
            if (t_max_x < t_max_z)
            {
                x += step_x;
                traveled = t_max_x;
                t_max_x += t_delta_x;
                last_axis = 0;
            }
            else
            {
                z += step_z;
                traveled = t_max_z;
                t_max_z += t_delta_z;
                last_axis = 2;
            }
        }
        else
        {
            if (t_max_y < t_max_z)
            {
                y += step_y;
                traveled = t_max_y;
                t_max_y += t_delta_y;
                last_axis = 1;
            }
            else
            {
                z += step_z;
                traveled = t_max_z;
                t_max_z += t_delta_z;
                last_axis = 2;
            }
        }

        if (traveled > max_t)
        {
            return false;
        }
        if (x < 0 || x >= terrain_size || z < 0 || z >= terrain_size || y < 0 || y >= terrain_height)
        {
            return false;
        }
    }
    return false;
}

static bool shadow_raymarch_hit(const Vec3& world, const Vec3& normal, const Vec3& light_dir)
{
    const int terrain_size = terrain.chunk_size;
    const int terrain_height = terrain.max_height();
    if (terrain_size <= 0 || terrain_height <= 0)
    {
        return false;
    }

    const double block_size = terrain.block_size();
    const double half = block_size * 0.5;
    const double start_x = terrain.start_x();
    const double start_z = terrain.start_z();
    const double base_y = terrain.base_y();
    const double inv_block = 1.0 / block_size;

    const Vec3 origin_world{
        world.x + normal.x * kShadowRayBias,
        world.y + normal.y * kShadowRayBias,
        world.z + normal.z * kShadowRayBias
    };
    const Vec3 origin{
        (origin_world.x - start_x + half) * inv_block,
        (base_y - origin_world.y + half) * inv_block,
        (origin_world.z - start_z + half) * inv_block
    };
    const Vec3 dir{
        light_dir.x * inv_block,
        -light_dir.y * inv_block,
        light_dir.z * inv_block
    };
    if (dir.x == 0.0 && dir.y == 0.0 && dir.z == 0.0)
    {
        return false;
    }

    int x = static_cast<int>(std::floor(origin.x));
    int y = static_cast<int>(std::floor(origin.y));
    int z = static_cast<int>(std::floor(origin.z));

    if (x < 0 || x >= terrain_size || z < 0 || z >= terrain_size || y < 0 || y >= terrain_height)
    {
        return false;
    }

    const double inf = std::numeric_limits<double>::infinity();
    const int step_x = dir.x > 0.0 ? 1 : (dir.x < 0.0 ? -1 : 0);
    const int step_y = dir.y > 0.0 ? 1 : (dir.y < 0.0 ? -1 : 0);
    const int step_z = dir.z > 0.0 ? 1 : (dir.z < 0.0 ? -1 : 0);

    const double t_delta_x = step_x != 0 ? 1.0 / std::abs(dir.x) : inf;
    const double t_delta_y = step_y != 0 ? 1.0 / std::abs(dir.y) : inf;
    const double t_delta_z = step_z != 0 ? 1.0 / std::abs(dir.z) : inf;

    const double next_x = step_x > 0 ? (std::floor(origin.x) + 1.0) : std::floor(origin.x);
    const double next_y = step_y > 0 ? (std::floor(origin.y) + 1.0) : std::floor(origin.y);
    const double next_z = step_z > 0 ? (std::floor(origin.z) + 1.0) : std::floor(origin.z);

    double t_max_x = step_x != 0 ? (next_x - origin.x) / dir.x : inf;
    double t_max_y = step_y != 0 ? (next_y - origin.y) / dir.y : inf;
    double t_max_z = step_z != 0 ? (next_z - origin.z) / dir.z : inf;

    const int max_steps = (terrain_size + terrain_size + terrain_height) * 4;
    bool skip_first = true;

    for (int i = 0; i < max_steps; ++i)
    {
        if (!skip_first && terrain.has_block(x, y, z))
        {
            return true;
        }
        skip_first = false;

        if (t_max_x < t_max_y)
        {
            if (t_max_x < t_max_z)
            {
                x += step_x;
                t_max_x += t_delta_x;
            }
            else
            {
                z += step_z;
                t_max_z += t_delta_z;
            }
        }
        else
        {
            if (t_max_y < t_max_z)
            {
                y += step_y;
                t_max_y += t_delta_y;
            }
            else
            {
                z += step_z;
                t_max_z += t_delta_z;
            }
        }

        if (x < 0 || x >= terrain_size || z < 0 || z >= terrain_size || y < 0 || y >= terrain_height)
        {
            return false;
        }
    }
    return false;
}

static float compute_shadow_factor(const Vec3& light_dir, const Vec3& world, const Vec3& normal)
{
    const double ndotl = std::max(0.0, normal.dot(light_dir));
    if (ndotl <= 0.0)
    {
        return 1.0f;
    }
    return shadow_raymarch_hit(world, normal, light_dir) ? 0.0f : 1.0f;
}

static float shadow_filter_at(std::span<const float> mask, std::span<const float> depth,
                              std::span<const Vec3> normals, const size_t width, const size_t height,
                              const int x, const int y, const float depth_max)
{
    const int ix = std::clamp(x, 0, static_cast<int>(width) - 1);
    const int iy = std::clamp(y, 0, static_cast<int>(height) - 1);
    const size_t center_idx = static_cast<size_t>(iy) * width + static_cast<size_t>(ix);
    const float center_depth = depth[center_idx];
    if (center_depth >= depth_max)
    {
        return mask[center_idx];
    }
    const Vec3 center_normal = normals[center_idx];
    const double normal_len_sq = center_normal.x * center_normal.x +
                                 center_normal.y * center_normal.y +
                                 center_normal.z * center_normal.z;
    if (normal_len_sq <= 1e-6)
    {
        return mask[center_idx];
    }

    float sum = mask[center_idx] * kShadowFilterCenterWeight;
    float weight_sum = kShadowFilterCenterWeight;

    auto try_neighbor = [&](const int nx, const int ny) {
        if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height))
        {
            return;
        }
        const size_t idx = static_cast<size_t>(ny) * width + static_cast<size_t>(nx);
        const float neighbor_depth = depth[idx];
        if (neighbor_depth >= depth_max)
        {
            return;
        }
        const Vec3 neighbor_normal = normals[idx];
        const double neighbor_len_sq = neighbor_normal.x * neighbor_normal.x +
                                       neighbor_normal.y * neighbor_normal.y +
                                       neighbor_normal.z * neighbor_normal.z;
        if (neighbor_len_sq <= 1e-6)
        {
            return;
        }
        const float depth_diff = std::fabs(neighbor_depth - center_depth);
        if (depth_diff > kShadowFilterDepthThreshold)
        {
            return;
        }
        const float dot = static_cast<float>(center_normal.x * neighbor_normal.x +
                                             center_normal.y * neighbor_normal.y +
                                             center_normal.z * neighbor_normal.z);
        const float clamped_dot = std::clamp(dot, -1.0f, 1.0f);
        if (clamped_dot < kShadowFilterNormalThreshold)
        {
            return;
        }
        sum += mask[idx] * kShadowFilterNeighborWeight;
        weight_sum += kShadowFilterNeighborWeight;
    };

    try_neighbor(ix - 1, iy);
    try_neighbor(ix + 1, iy);
    try_neighbor(ix, iy - 1);
    try_neighbor(ix, iy + 1);

    if (weight_sum <= 0.0f)
    {
        return mask[center_idx];
    }
    float filtered = sum / weight_sum;
    filtered = std::clamp(filtered, 0.0f, 1.0f);
    return filtered;
}

static void filter_shadow_masks(std::span<const float> mask_a, std::span<const float> mask_b,
                                std::span<float> out_a, std::span<float> out_b,
                                std::span<const float> depth, std::span<const Vec3> normals,
                                const size_t width, const size_t height, const float depth_max)
{
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            const size_t center_idx = y * width + x;
            const float center_depth = depth[center_idx];
            if (center_depth >= depth_max)
            {
                out_a[center_idx] = mask_a[center_idx];
                out_b[center_idx] = mask_b[center_idx];
                continue;
            }
            const Vec3 center_normal = normals[center_idx];
            const double normal_len_sq = center_normal.x * center_normal.x +
                                         center_normal.y * center_normal.y +
                                         center_normal.z * center_normal.z;
            if (normal_len_sq <= 1e-6)
            {
                out_a[center_idx] = mask_a[center_idx];
                out_b[center_idx] = mask_b[center_idx];
                continue;
            }

            float sum_a = mask_a[center_idx] * kShadowFilterCenterWeight;
            float sum_b = mask_b[center_idx] * kShadowFilterCenterWeight;
            float weight_sum = kShadowFilterCenterWeight;

            auto try_neighbor = [&](const size_t idx) {
                const float neighbor_depth = depth[idx];
                if (neighbor_depth >= depth_max)
                {
                    return;
                }
                const Vec3 neighbor_normal = normals[idx];
                const double neighbor_len_sq = neighbor_normal.x * neighbor_normal.x +
                                               neighbor_normal.y * neighbor_normal.y +
                                               neighbor_normal.z * neighbor_normal.z;
                if (neighbor_len_sq <= 1e-6)
                {
                    return;
                }
                const float depth_diff = std::fabs(neighbor_depth - center_depth);
                if (depth_diff > kShadowFilterDepthThreshold)
                {
                    return;
                }
                const float dot = static_cast<float>(center_normal.x * neighbor_normal.x +
                                                     center_normal.y * neighbor_normal.y +
                                                     center_normal.z * neighbor_normal.z);
                const float clamped_dot = std::clamp(dot, -1.0f, 1.0f);
                if (clamped_dot < kShadowFilterNormalThreshold)
                {
                    return;
                }
                sum_a += mask_a[idx] * kShadowFilterNeighborWeight;
                sum_b += mask_b[idx] * kShadowFilterNeighborWeight;
                weight_sum += kShadowFilterNeighborWeight;
            };

            if (x > 0)
            {
                try_neighbor(center_idx - 1);
            }
            if (x + 1 < width)
            {
                try_neighbor(center_idx + 1);
            }
            if (y > 0)
            {
                try_neighbor(center_idx - width);
            }
            if (y + 1 < height)
            {
                try_neighbor(center_idx + width);
            }

            if (weight_sum <= 0.0f)
            {
                out_a[center_idx] = mask_a[center_idx];
                out_b[center_idx] = mask_b[center_idx];
                continue;
            }

            float filtered_a = sum_a / weight_sum;
            float filtered_b = sum_b / weight_sum;
            filtered_a = std::clamp(filtered_a, 0.0f, 1.0f);
            filtered_b = std::clamp(filtered_b, 0.0f, 1.0f);
            out_a[center_idx] = filtered_a;
            out_b[center_idx] = filtered_b;
        }
    }
}

static void render_quad(float* zbuffer, LinearColor* sample_ambient,
                        LinearColor* sample_direct_sun, LinearColor* sample_direct_moon,
                        float* shadow_mask_sun, float* shadow_mask_moon, Vec3* sample_normals,
                        LinearColor* sample_albedo, float* sample_ao,
                        Vec3* world_positions, uint32_t* world_stamp, uint32_t frame_index,
                        size_t width, size_t height,
                        const RenderQuad& quad, const double fov_x, const double fov_y,
                        const Vec3& camera_pos, const ViewRotation& view_rot, const ShadingContext& ctx,
                        const float jitter_x, const float jitter_y,
                        const std::array<Vec3, 2>& lights_right_scaled,
                        const std::array<Vec3, 2>& lights_up_scaled,
                        const std::array<BlueNoise::Shift, 2>& shadow_shift_u,
                        const std::array<BlueNoise::Shift, 2>& shadow_shift_v)

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
            static_cast<float>(view.x * invZ * fov_x + width / 2.0),
            static_cast<float>(view.y * invZ * fov_y + height / 2.0),
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
        const size_t clipped_count = clip_triangle_to_near_plane(input, clipped);
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

static Vec3 unproject_fast(double screen_x, double screen_y, double depth, 
                           const Mat4& inv_vp, double width, double height,
                           double proj_a, double proj_b)
{
    const double ndc_x = (screen_x / width - 0.5) * 2.0;
    const double ndc_y = (screen_y / height - 0.5) * 2.0;
    
    const double view_z = depth;
    const double ndc_z = proj_a + proj_b / view_z;
    
    const double clip_w = view_z;
    const double clip_x = ndc_x * clip_w;
    const double clip_y = ndc_y * clip_w;
    const double clip_z = ndc_z * clip_w;

    double wx = inv_vp.m[0][0]*clip_x + inv_vp.m[0][1]*clip_y + inv_vp.m[0][2]*clip_z + inv_vp.m[0][3]*clip_w;
    double wy = inv_vp.m[1][0]*clip_x + inv_vp.m[1][1]*clip_y + inv_vp.m[1][2]*clip_z + inv_vp.m[1][3]*clip_w;
    double wz = inv_vp.m[2][0]*clip_x + inv_vp.m[2][1]*clip_y + inv_vp.m[2][2]*clip_z + inv_vp.m[2][3]*clip_w;
    double ww = inv_vp.m[3][0]*clip_x + inv_vp.m[3][1]*clip_y + inv_vp.m[3][2]*clip_z + inv_vp.m[3][3]*clip_w;

    if (std::abs(ww) > 1e-6) {
        const double inv_ww = 1.0 / ww;
        wx *= inv_ww; wy *= inv_ww; wz *= inv_ww;
    }

    return {wx, wy, wz};
}

void render_update_array(uint32_t* framebuffer, size_t width, size_t height)
{
    static size_t cached_width = 0;
    static size_t cached_height = 0;
    static std::vector<float> zbuffer;
    static std::vector<LinearColor> sample_colors;
    static std::vector<LinearColor> sample_direct;
    static std::vector<LinearColor> sample_direct_sun;
    static std::vector<LinearColor> sample_direct_moon;
    static std::vector<float> shadow_mask_sun;
    static std::vector<float> shadow_mask_moon;
    static std::vector<float> shadow_mask_filtered_sun;
    static std::vector<float> shadow_mask_filtered_moon;
    static std::vector<Vec3> sample_normals;
    static std::vector<LinearColor> sample_albedo;
    static std::vector<LinearColor> sample_indirect;
    static std::vector<float> sample_ao;
    static std::vector<Vec3> world_positions;
    static std::vector<uint32_t> world_stamp;
    static std::array<std::vector<LinearColor>, 2> taa_history;
    static std::vector<LinearColor> taa_resolved;
    static std::vector<LinearColor> current_linear_buffer;
    static std::vector<uint8_t> taa_history_mask;
    static int taa_ping_pong = 0;
    static size_t taa_width = 0;
    static size_t taa_height = 0;
    static bool taa_history_valid = false;
    static uint64_t taa_state_version = 0;
    static bool taa_was_enabled = false;
    static Vec3 last_camera_pos{0.0, 0.0, 0.0};
    static double last_camera_yaw = 0.0;
    static double last_camera_pitch = 0.0;
    static bool last_camera_valid = false;
    static float taa_motion_activity = 0.0f;
    static uint64_t taa_motion_state_version = 0;
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

    if (width != cached_width || height != cached_height)
    {
        cached_width = width;
        cached_height = height;
        zbuffer.assign(sample_count, depth_max);
        sample_colors.assign(sample_count, {0.0f, 0.0f, 0.0f});
        sample_direct.assign(sample_count, {0.0f, 0.0f, 0.0f});
        sample_direct_sun.assign(sample_count, {0.0f, 0.0f, 0.0f});
        sample_direct_moon.assign(sample_count, {0.0f, 0.0f, 0.0f});
        shadow_mask_sun.assign(sample_count, 1.0f);
        shadow_mask_moon.assign(sample_count, 1.0f);
        shadow_mask_filtered_sun.assign(sample_count, 1.0f);
        shadow_mask_filtered_moon.assign(sample_count, 1.0f);
        sample_normals.assign(sample_count, {0.0, 0.0, 0.0});
        sample_albedo.assign(sample_count, {0.0f, 0.0f, 0.0f});
        sample_indirect.assign(sample_count, {0.0f, 0.0f, 0.0f});
        sample_ao.assign(sample_count, 1.0f);
        world_positions.assign(sample_count, {0.0, 0.0, 0.0});
        world_stamp.assign(sample_count, 0);
        taa_resolved.assign(sample_count, {0.0f, 0.0f, 0.0f});
        current_linear_buffer.assign(sample_count, {0.0f, 0.0f, 0.0f});
        taa_history_mask.assign(sample_count, 0);
    }
    else
    {
        std::fill(zbuffer.begin(), zbuffer.end(), depth_max);
    }

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

    const bool paused = rotationPaused.load(std::memory_order_relaxed);
    const uint32_t frame_index = renderFrameIndex.fetch_add(1u, std::memory_order_relaxed);
    previousVP = currentVP;
    
    if (sunOrbitEnabled.load(std::memory_order_relaxed) && !paused)
    {
        double angle = sunOrbitAngle.load(std::memory_order_relaxed) +
                       sunOrbitSpeed.load(std::memory_order_relaxed);
        angle = std::fmod(angle, kPi);
        if (angle < 0.0)
        {
            angle += kPi;
        }
        sunOrbitAngle.store(angle, std::memory_order_relaxed);
    }

    const Vec3 camera_pos_snapshot{
        camera_x.load(std::memory_order_relaxed),
        camera_y.load(std::memory_order_relaxed),
        camera_z.load(std::memory_order_relaxed)
    };
    const double yaw_snapshot = camera_yaw.load(std::memory_order_relaxed);
    const double pitch_snapshot = camera_pitch.load(std::memory_order_relaxed);

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
            const double dx = camera_pos_snapshot.x - last_camera_pos.x;
            const double dy = camera_pos_snapshot.y - last_camera_pos.y;
            const double dz = camera_pos_snapshot.z - last_camera_pos.z;
            const double move_dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double dyaw = yaw_snapshot - last_camera_yaw;
            const double dpitch = pitch_snapshot - last_camera_pitch;
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
    taaSharpenStrength.store(static_cast<double>(taa_sharpen_strength), std::memory_order_relaxed);
    last_camera_pos = camera_pos_snapshot;
    last_camera_yaw = yaw_snapshot;
    last_camera_pitch = pitch_snapshot;
    last_camera_valid = true;

    const double fov_y = static_cast<double>(height) * 0.8;
    const double fov_x = fov_y;
    
    const double view_yaw = -yaw_snapshot;
    const double view_pitch = -pitch_snapshot;
    const Mat4 view = make_view_matrix(camera_pos_snapshot, view_yaw, view_pitch);
    const Mat4 proj = make_projection_matrix(static_cast<double>(width),
                                             static_cast<double>(height),
                                             fov_x, fov_y);
    currentVP = proj * view;
    if (const auto inv = currentVP.invert())
    {
        inverseCurrentVP = *inv;
    }
    else
    {
        inverseCurrentVP = Mat4::identity();
    }

    const double inv_dist = 1.0 / (kFarPlane - kNearPlane);
    const double proj_a = kFarPlane * inv_dist;
    const double proj_b = -kNearPlane * kFarPlane * inv_dist;

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

    const bool sun_orbit = sunOrbitEnabled.load(std::memory_order_relaxed);
    Vec3 sun_dir = load_light_direction().normalize();
    double sun_intensity = lightIntensity.load(std::memory_order_relaxed);
    if (sun_orbit)
    {
        const double angle = sunOrbitAngle.load(std::memory_order_relaxed);
        sun_dir = compute_sun_orbit_direction(angle);
        const double max_y = std::cos(kSunLatitudeRad);
        double visibility = sun_dir.y < 0.0 ? (-sun_dir.y) / max_y : 0.0;
        visibility = std::clamp(visibility, 0.0, 1.0);
        sun_intensity = lightIntensity.load(std::memory_order_relaxed) * visibility;
    }
    sun_intensity *= kSunIntensityBoost;

    ColorSrgb sky_top = ColorSrgb::from_hex(skyTopColor.load(std::memory_order_relaxed));
    ColorSrgb sky_bottom = ColorSrgb::from_hex(skyBottomColor.load(std::memory_order_relaxed));
    double sun_height = 1.0;
    if (sun_orbit)
    {
        sun_height = compute_sun_height(sun_dir);
        const float t = static_cast<float>(sun_height);
        const ColorSrgb sunrise_top = ColorSrgb::from_hex(kSkySunriseTopColor);
        const ColorSrgb sunrise_bottom = ColorSrgb::from_hex(kSkySunriseBottomColor);
        sky_top = lerp_color(sunrise_top, sky_top, t);
        sky_bottom = lerp_color(sunrise_bottom, sky_bottom, t);
    }

    const LinearColor sky_top_linear = sky_top.to_linear();
    const LinearColor sky_bottom_linear = sky_bottom.to_linear();
    const double moon_intensity = moonIntensity.load(std::memory_order_relaxed);
    double effective_sky_intensity = skyLightIntensity.load(std::memory_order_relaxed);
    if (sun_orbit)
    {
        const double sun_factor = std::pow(std::clamp(sun_height, 0.0, 1.0), kSkyLightHeightPower);
        effective_sky_intensity *= sun_factor;

        const double moon_factor = std::clamp(moon_intensity, 0.0, 1.0) * kMoonSkyLightFloor * (1.0 - sun_factor);
        effective_sky_intensity = std::min(1.0, effective_sky_intensity + moon_factor);
    }

    const Vec3 moon_dir = load_moon_direction().normalize();
    const bool shadows_on = shadowEnabled.load(std::memory_order_relaxed);
    const std::array<ShadingContext::DirectionalLightInfo, 2> lights = {
        ShadingContext::DirectionalLightInfo{sun_dir, sun_intensity, kSunLightColorLinear, kSunDiskRadius},
        ShadingContext::DirectionalLightInfo{moon_dir, moon_intensity, kMoonLightColorLinear, 0.0}
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
    const LinearColor hemi_ground = compute_hemisphere_ground(sky_bottom_linear, lights);
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
        render_quad(zbuffer.data(), sample_colors.data(),
                    sample_direct_sun.data(), sample_direct_moon.data(),
                    shadow_mask_sun.data(), shadow_mask_moon.data(),
                    sample_normals.data(), sample_albedo.data(), sample_ao.data(),
                    world_positions.data(), world_stamp.data(), frame_index,
                    width, height, quad, fov_x, fov_y,
                    camera_pos_snapshot, view_rot, ctx, jitter_x, jitter_y,
                    lights_right_scaled, lights_up_scaled,
                    shadow_shift_u, shadow_shift_v); 
    }

    if (shadows_on)
    {
        const float* shadow_sun = shadow_mask_sun.data();
        const float* shadow_moon = shadow_mask_moon.data();
        if (kShadowFilterEnabled)
        {
            const std::span<const float> shadow_sun_span(shadow_mask_sun.data(), sample_count);
            const std::span<const float> shadow_moon_span(shadow_mask_moon.data(), sample_count);
            const std::span<float> shadow_sun_out_span(shadow_mask_filtered_sun.data(), sample_count);
            const std::span<float> shadow_moon_out_span(shadow_mask_filtered_moon.data(), sample_count);
            const std::span<const float> depth_span(zbuffer.data(), sample_count);
            const std::span<const Vec3> normals_span(sample_normals.data(), sample_count);
            filter_shadow_masks(shadow_sun_span, shadow_moon_span,
                                shadow_sun_out_span, shadow_moon_out_span,
                                depth_span, normals_span,
                                width, height, depth_max);
            shadow_sun = shadow_mask_filtered_sun.data();
            shadow_moon = shadow_mask_filtered_moon.data();
        }
        for (size_t i = 0; i < sample_count; ++i)
        {
            const LinearColor sun = sample_direct_sun[i];
            const LinearColor moon = sample_direct_moon[i];
            const LinearColor sun_shadowed = scale_color(sun, shadow_sun[i]);
            const LinearColor moon_shadowed = scale_color(moon, shadow_moon[i]);
            sample_direct[i] = add_color(sun_shadowed, moon_shadowed);
        }
    }
    else
    {
        for (size_t i = 0; i < sample_count; ++i)
        {
            sample_direct[i] = add_color(sample_direct_sun[i], sample_direct_moon[i]);
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
                if (zbuffer[idx] >= depth_max)
                {
                    continue;
                }
                sample_indirect[idx] = {0.0f, 0.0f, 0.0f};
                Vec3 normal = sample_normals[idx];
                const double normal_len_sq = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
                if (normal_len_sq <= 1e-6)
                {
                    continue;
                }
                const LinearColor origin_albedo = sample_albedo[idx];

                const double screen_x = static_cast<double>(x) + 0.5 + jitter_x_d;
                const Vec3 world = unproject_fast(screen_x, screen_y, zbuffer[idx],
                                                  inverseCurrentVP, width_d, height_d,
                                                  proj_a, proj_b);
                world_positions[idx] = world;
                world_stamp[idx] = frame_index;

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
                        const LinearColor sky = lerp_color(hemi_ground, sky_top_linear, sky_t);
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
                    return gi_raymarch_hit(world_origin, surface_normal, dir, kGiMaxDistance, &out_hit);
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
                            LinearColor bounced = mul_color(incoming, hit.albedo);
                            bounced = mul_color(bounced, throughput);
                            bounced = scale_color(bounced, static_cast<float>(cos_theta));
                            gi_sample = add_color(gi_sample, bounced);
                            hit_any = true;
                        }

                        LinearColor next_throughput = mul_color(throughput, hit.albedo);
                        next_throughput = scale_color(next_throughput, static_cast<float>(cos_theta));
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
                    gi_sample = scale_color(gi_sample, gi_scale);
                    gi_sample.r = std::clamp(gi_sample.r, 0.0f, kGiClamp);
                    gi_sample.g = std::clamp(gi_sample.g, 0.0f, kGiClamp);
                    gi_sample.b = std::clamp(gi_sample.b, 0.0f, kGiClamp);
                    sample_indirect[idx] = gi_sample;
                }
            }
        }
    }

    static constexpr std::array<std::array<int, 4>, 4> bayer4 = {{
        {{0, 8, 2, 10}},
        {{12, 4, 14, 6}},
        {{3, 11, 1, 9}},
        {{15, 7, 13, 5}}
    }};
    const float dither_strength = 2.0f;
    const float dither_scale = dither_strength / 16.0f;
    const float exposure_factor = static_cast<float>(std::max(0.0, exposure.load(std::memory_order_relaxed)));
    const bool use_history = taa_on && taa_history_valid;

    const int read_idx = (taa_ping_pong + 1) % 2;
    const int write_idx = taa_ping_pong;
    const LinearColor* history_read_ptr = taa_history[read_idx].data();
    LinearColor* history_write_ptr = taa_history[write_idx].data();
    const std::span<const LinearColor> history_read_span(history_read_ptr, sample_count);

    for (size_t y = 0; y < height; ++y)
    {
        const float sky_t = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
        const LinearColor sky_row_linear = lerp_color(sky_top_linear, sky_bottom_linear, sky_t);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            if (zbuffer[idx] >= depth_max)
            {
                current_linear_buffer[idx] = sky_row_linear;
                continue;
            }
            LinearColor accum = sample_colors[idx];
            const LinearColor direct = sample_direct[idx];
            accum.r += direct.r;
            accum.g += direct.g;
            accum.b += direct.b;
            if (gi_active)
            {
                const LinearColor indirect = sample_indirect[idx];
                const float gi_ao = std::min(1.0f, sample_ao[idx] + kGiAoLift);
                const LinearColor indirect_scaled = scale_color(indirect, gi_ao);
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
            const float depth = zbuffer[pixel];
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
                        if (world_stamp[pixel] == frame_index)
                        {
                            world = world_positions[pixel];
                        }
                        else
                        {
                            world = unproject_fast(screen_x, screen_y, depth,
                                                   inverseCurrentVP, width_d, height_d,
                                                   proj_a, proj_b);
                        }

                        Vec2 prev_screen = render_reproject_point(world, width, height);

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
            const bool is_sky = zbuffer[pixel] >= depth_max;
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

double render_debug_eval_specular(const double ndoth, const double vdoth,
                                  const double ndotl, const double shininess, const double f0)
{
    return eval_specular_term(ndoth, vdoth, ndotl, shininess, f0);
}

Vec3 render_debug_tonemap_reinhard(const Vec3 color, const double exposure_value)
{
    const float exposure_factor = static_cast<float>(std::max(0.0, exposure_value));
    const LinearColor input{
        static_cast<float>(color.x),
        static_cast<float>(color.y),
        static_cast<float>(color.z)
    };
    const LinearColor mapped = tonemap_reinhard(input, exposure_factor);
    return {mapped.r, mapped.g, mapped.b};
}

void render_debug_set_sky_colors_raw(const uint32_t top, const uint32_t bottom)
{
    skyTopColor.store(top, std::memory_order_relaxed);
    skyBottomColor.store(bottom, std::memory_order_relaxed);
}

Vec3 render_debug_sample_history_bilinear(const Vec3* buffer, const size_t width, const size_t height,
                                          const Vec2 screen_coord)
{
    if (!buffer)
    {
        return {0.0, 0.0, 0.0};
    }
    const std::span<const Vec3> span(buffer, width * height);
    return sample_bilinear_history_vec3(span, width, height, screen_coord.x, screen_coord.y);
}

void render_debug_set_frame_index(const uint32_t frame_index)
{
    renderFrameIndex.store(frame_index, std::memory_order_relaxed);
}

uint32_t render_debug_get_frame_index()
{
    return renderFrameIndex.load(std::memory_order_relaxed);
}

Mat4 render_debug_get_current_vp()
{
    return currentVP;
}

Mat4 render_debug_get_previous_vp()
{
    return previousVP;
}

Mat4 render_debug_get_inverse_current_vp()
{
    return inverseCurrentVP;
}

double render_debug_get_taa_sharpen_strength()
{
    return taaSharpenStrength.load(std::memory_order_relaxed);
}

double render_debug_get_taa_sharpen_percent()
{
    const double max_strength = kTaaSharpenMax;
    if (max_strength <= 0.0)
    {
        return 0.0;
    }
    const double strength = render_debug_get_taa_sharpen_strength();
    const double ratio = std::clamp(strength / max_strength, 0.0, 1.0);
    return ratio * 100.0;
}

void render_set_paused(const bool paused)
{
    rotationPaused.store(paused, std::memory_order_relaxed);
}

bool render_is_paused()
{
    return rotationPaused.load(std::memory_order_relaxed);
}

void render_toggle_pause()
{
    rotationPaused.store(!rotationPaused.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

void render_set_light_direction(const Vec3 dir)
{
    store_light_direction(dir);
    mark_render_state_dirty();
}

Vec3 render_get_light_direction()
{
    return load_light_direction();
}

void render_set_light_intensity(const double intensity)
{
    lightIntensity.store(intensity, std::memory_order_relaxed);
    mark_render_state_dirty();
}

double render_get_light_intensity()
{
    return lightIntensity.load(std::memory_order_relaxed);
}

void render_set_sun_orbit_enabled(const bool enabled)
{
    sunOrbitEnabled.store(enabled, std::memory_order_relaxed);
    mark_render_state_dirty();
}

bool render_get_sun_orbit_enabled()
{
    return sunOrbitEnabled.load(std::memory_order_relaxed);
}

void render_set_sun_orbit_angle(const double angle)
{
    sunOrbitAngle.store(angle, std::memory_order_relaxed);
    mark_render_state_dirty();
}

double render_get_sun_orbit_angle()
{
    return sunOrbitAngle.load(std::memory_order_relaxed);
}

void render_set_moon_direction(const Vec3 dir)
{
    store_moon_direction(dir);
    mark_render_state_dirty();
}

void render_set_moon_intensity(const double intensity)
{
    moonIntensity.store(intensity, std::memory_order_relaxed);
    mark_render_state_dirty();
}

void render_set_sky_top_color(const uint32_t color)
{
    skyTopColor.store(color, std::memory_order_relaxed);
    mark_render_state_dirty();
}

uint32_t render_get_sky_top_color()
{
    return skyTopColor.load(std::memory_order_relaxed);
}

void render_set_sky_bottom_color(const uint32_t color)
{
    skyBottomColor.store(color, std::memory_order_relaxed);
    mark_render_state_dirty();
}

uint32_t render_get_sky_bottom_color()
{
    return skyBottomColor.load(std::memory_order_relaxed);
}

void render_set_sky_light_intensity(const double intensity)
{
    skyLightIntensity.store(intensity, std::memory_order_relaxed);
    mark_render_state_dirty();
}

double render_get_sky_light_intensity()
{
    return skyLightIntensity.load(std::memory_order_relaxed);
}

void render_set_exposure(const double value)
{
    exposure.store(std::max(0.0, value), std::memory_order_relaxed);
    mark_render_state_dirty();
}

double render_get_exposure()
{
    return exposure.load(std::memory_order_relaxed);
}

void render_set_taa_enabled(const bool enabled)
{
    taaEnabled.store(enabled, std::memory_order_relaxed);
    mark_render_state_dirty();
}

bool render_get_taa_enabled()
{
    return taaEnabled.load(std::memory_order_relaxed);
}

void render_set_taa_blend(const double blend)
{
    taaBlend.store(std::clamp(blend, 0.0, 1.0), std::memory_order_relaxed);
    mark_render_state_dirty();
}

double render_get_taa_blend()
{
    return taaBlend.load(std::memory_order_relaxed);
}

void render_set_taa_clamp_enabled(const bool enabled)
{
    taaClampEnabled.store(enabled, std::memory_order_relaxed);
    mark_render_state_dirty();
}

bool render_get_taa_clamp_enabled()
{
    return taaClampEnabled.load(std::memory_order_relaxed);
}

void render_set_gi_enabled(const bool enabled)
{
    giEnabled.store(enabled, std::memory_order_relaxed);
    mark_render_state_dirty();
}

bool render_get_gi_enabled()
{
    return giEnabled.load(std::memory_order_relaxed);
}

void render_set_gi_strength(const double strength)
{
    giStrength.store(std::max(0.0, strength), std::memory_order_relaxed);
    mark_render_state_dirty();
}

double render_get_gi_strength()
{
    return giStrength.load(std::memory_order_relaxed);
}

void render_set_gi_bounce_count(const int count)
{
    giBounceCount.store(count, std::memory_order_relaxed);
    mark_render_state_dirty();
}

int render_get_gi_bounce_count()
{
    return giBounceCount.load(std::memory_order_relaxed);
}

void render_reset_taa_history()
{
    mark_render_state_dirty();
}

void render_set_ambient_occlusion_enabled(const bool enabled)
{
    ambientOcclusionEnabled.store(enabled, std::memory_order_relaxed);
    mark_render_state_dirty();
}

void render_set_shadow_enabled(const bool enabled)
{
    shadowEnabled.store(enabled, std::memory_order_relaxed);
    mark_render_state_dirty();
}

bool render_get_terrain_vertex_sky_visibility(const int x, const int y, const int z,
                                              const int face, const int corner,
                                              float* out_visibility)
{
    if (!out_visibility)
    {
        return false;
    }
    terrain.generate();
    const auto visibility = terrain.get_sky_visibility_at(x, y, z, face, corner);
    if (!visibility)
    {
        return false;
    }
    *out_visibility = *visibility;
    return true;
}

bool render_get_shadow_factor_at_point(const Vec3 world, const Vec3 normal, float* out_factor)
{
    if (!out_factor)
    {
        return false;
    }
    terrain.generate();
    if (!shadowEnabled.load(std::memory_order_relaxed))
    {
        *out_factor = 1.0f;
        return true;
    }
    const bool sun_orbit = sunOrbitEnabled.load(std::memory_order_relaxed);
    const double base_intensity = lightIntensity.load(std::memory_order_relaxed);
    Vec3 light_dir = load_light_direction().normalize();
    double sun_intensity = base_intensity;
    if (sun_orbit)
    {
        const double angle = sunOrbitAngle.load(std::memory_order_relaxed);
        light_dir = compute_sun_orbit_direction(angle);
        const double max_y = std::cos(kSunLatitudeRad);
        double visibility = light_dir.y < 0.0 ? (-light_dir.y) / max_y : 0.0;
        visibility = std::clamp(visibility, 0.0, 1.0);
        sun_intensity = base_intensity * visibility;
    }
    sun_intensity *= kSunIntensityBoost;
    if (sun_intensity <= 0.0)
    {
        *out_factor = 1.0f;
        return true;
    }

    *out_factor = compute_shadow_factor(light_dir, world, normal.normalize());
    return true;
}

bool render_debug_shadow_factor_with_frame(const Vec3 world, const Vec3 normal, const Vec3 light_dir,
                                           const int pixel_x, const int pixel_y, const int frame,
                                           float* out_factor)
{
    if (!out_factor)
    {
        return false;
    }
    terrain.generate();
    if (!shadowEnabled.load(std::memory_order_relaxed))
    {
        *out_factor = 1.0f;
        return true;
    }

    const Vec3 dir = light_dir.normalize();

    auto [right, up, forward] = Vec3::get_basis(dir);

    const double scale = std::tan(kSunDiskRadius);
    const Vec3 right_scaled = {right.x * scale, right.y * scale, right.z * scale};
    const Vec3 up_scaled    = {up.x * scale, up.y * scale, up.z * scale};

    const BlueNoise::Shift shift_u = BlueNoise::shift(frame, kSunShadowSalt);
    const BlueNoise::Shift shift_v = BlueNoise::shift(frame, kSunShadowSalt + 1);
    const Vec3 shadow_dir = jitter_shadow_direction(dir, 
                                                    right_scaled, up_scaled,
                                                    pixel_x, pixel_y,
                                                    shift_u, shift_v);
                                                    
    *out_factor = compute_shadow_factor(shadow_dir, world, normal.normalize());
    return true;
}

float render_debug_shadow_filter(const float* mask, const float* depth, const Vec3* normals)
{
    if (!mask || !depth || !normals)
    {
        return 0.0f;
    }
    const float depth_max = std::numeric_limits<float>::max();
    constexpr size_t kDebugShadowFilterSize = 3 * 3;
    const std::span<const float> mask_span(mask, kDebugShadowFilterSize);
    const std::span<const float> depth_span(depth, kDebugShadowFilterSize);
    const std::span<const Vec3> normals_span(normals, kDebugShadowFilterSize);
    return shadow_filter_at(mask_span, depth_span, normals_span, 3, 3, 1, 1, depth_max);
}

void render_set_camera_position(const Vec3 pos)
{
    camera_x.store(pos.x, std::memory_order_relaxed);
    camera_y.store(pos.y, std::memory_order_relaxed);
    camera_z.store(pos.z, std::memory_order_relaxed);
}

Vec3 render_get_camera_position()
{
    return {
        camera_x.load(std::memory_order_relaxed),
        camera_y.load(std::memory_order_relaxed),
        camera_z.load(std::memory_order_relaxed)
    };
}

void render_move_camera(const Vec3 delta)
{
    terrain.generate();
    Vec3 pos = render_get_camera_position();
    if (delta.x != 0.0)
    {
        const Vec3 candidate{pos.x + delta.x, pos.y, pos.z};
        if (!terrain.intersects_block(candidate))
        {
            pos.x = candidate.x;
        }
    }
    if (delta.y != 0.0)
    {
        const Vec3 candidate{pos.x, pos.y + delta.y, pos.z};
        if (!terrain.intersects_block(candidate))
        {
            pos.y = candidate.y;
        }
    }
    if (delta.z != 0.0)
    {
        const Vec3 candidate{pos.x, pos.y, pos.z + delta.z};
        if (!terrain.intersects_block(candidate))
        {
            pos.z = candidate.z;
        }
    }
    render_set_camera_position(pos);
}

void render_move_camera_local(const Vec3 delta)
{
    terrain.generate();
    const double yaw = camera_yaw.load(std::memory_order_relaxed);
    const double pitch = camera_pitch.load(std::memory_order_relaxed);
    Vec3 rotated = rotate_yaw_pitch(delta, yaw, pitch);
    if (delta.z > 0.0)
    {
        if (pitch < 0.0 && rotated.y < 0.0)
        {
            rotated.y = 0.0;
        }
        else if (pitch > 0.0 && rotated.y > 0.0)
        {
            rotated.y = 0.0;
        }
    }
    if (rotated.y > 0.0)
    {
        const Vec3 pos = render_get_camera_position();
        const Vec3 candidate{pos.x + rotated.x, pos.y + rotated.y, pos.z + rotated.z};
        if (terrain.intersects_block(candidate))
        {
            rotated.y = 0.0;
        }
    }
    render_move_camera(rotated);
}

void render_set_camera_rotation(const Vec2 rot)
{
    camera_yaw.store(rot.x, std::memory_order_relaxed);
    camera_pitch.store(clamp_pitch(rot.y), std::memory_order_relaxed);
}

Vec2 render_get_camera_rotation()
{
    return {
        camera_yaw.load(std::memory_order_relaxed),
        camera_pitch.load(std::memory_order_relaxed)
    };
}

void render_rotate_camera(const Vec2 delta)
{
    const double yaw = camera_yaw.load(std::memory_order_relaxed) + delta.x;
    const double pitch = clamp_pitch(camera_pitch.load(std::memory_order_relaxed) + delta.y);
    camera_yaw.store(yaw, std::memory_order_relaxed);
    camera_pitch.store(pitch, std::memory_order_relaxed);
}

bool render_should_rasterize_triangle(const Vec3 v0, const Vec3 v1, const Vec3 v2)
{
    return triangle_in_front_of_near_plane(v0.z, v1.z, v2.z);
}

double render_get_near_plane()
{
    return kNearPlane;
}

size_t render_clip_triangle_to_near_plane(const Vec3 v0, const Vec3 v1, const Vec3 v2,
                                          Vec3* out_vertices, const size_t max_vertices)
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
    const size_t count = clip_triangle_to_near_plane(input, clipped);
    const size_t out_count = std::min(count, max_vertices);
    for (size_t i = 0; i < out_count; ++i)
    {
        out_vertices[i] = clipped[i].view;
    }
    return out_count;
}

Vec2 render_project_point(const Vec3 world, const size_t width, const size_t height)
{
    if (width == 0 || height == 0)
    {
        return {0.0, 0.0};
    }

    const Vec3 camera_pos{
        camera_x.load(std::memory_order_relaxed),
        camera_y.load(std::memory_order_relaxed),
        camera_z.load(std::memory_order_relaxed)
    };
    const double yaw = camera_yaw.load(std::memory_order_relaxed);
    const double pitch = camera_pitch.load(std::memory_order_relaxed);

    Vec3 view{
        world.x - camera_pos.x,
        world.y - camera_pos.y,
        world.z - camera_pos.z
    };
    view = rotate_yaw_pitch(view, -yaw, -pitch);

    if (view.z <= kNearPlane)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    const double inv_z = 1.0 / view.z;
    const double fov_y = static_cast<double>(height) * 0.8;
    const double fov_x = fov_y;
    const double proj_x = view.x * inv_z * fov_x;
    const double proj_y = view.y * inv_z * fov_y;

    return {proj_x + static_cast<double>(width) / 2.0, proj_y + static_cast<double>(height) / 2.0};
}

Vec3 render_unproject_point(const Vec3 screen, const size_t width, const size_t height)
{
    if (width == 0 || height == 0)
    {
        return {0.0, 0.0, 0.0};
    }

    const Vec3 camera_pos{
        camera_x.load(std::memory_order_relaxed),
        camera_y.load(std::memory_order_relaxed),
        camera_z.load(std::memory_order_relaxed)
    };
    const double yaw = camera_yaw.load(std::memory_order_relaxed);
    const double pitch = camera_pitch.load(std::memory_order_relaxed);

    const double fov_y = static_cast<double>(height) * 0.8;
    const double fov_x = fov_y;
    const double half_w = static_cast<double>(width) / 2.0;
    const double half_h = static_cast<double>(height) / 2.0;

    const double view_x = (screen.x - half_w) / fov_x * screen.z;
    const double view_y = (screen.y - half_h) / fov_y * screen.z;
    const double view_z = screen.z;

    Vec3 view{view_x, view_y, view_z};
    Vec3 world = rotate_pitch_yaw(view, yaw, pitch);
    world.x += camera_pos.x;
    world.y += camera_pos.y;
    world.z += camera_pos.z;
    return world;
}

Vec2 render_reproject_point(const Vec3 world, const size_t width, const size_t height)
{
    if (width == 0 || height == 0)
    {
        return {0.0, 0.0};
    }

    const Mat4& vp = previousVP;
    const double clip_x = vp.m[0][0] * world.x + vp.m[0][1] * world.y + vp.m[0][2] * world.z + vp.m[0][3];
    const double clip_y = vp.m[1][0] * world.x + vp.m[1][1] * world.y + vp.m[1][2] * world.z + vp.m[1][3];
    const double clip_w = vp.m[3][0] * world.x + vp.m[3][1] * world.y + vp.m[3][2] * world.z + vp.m[3][3];

    if (clip_w <= kNearPlane)
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

size_t render_debug_get_terrain_block_count()
{
    terrain.generate();
    return terrain.block_count();
}

size_t render_debug_get_terrain_visible_face_count()
{
    terrain.generate();
    return terrain.visible_face_count();
}

size_t render_debug_get_terrain_triangle_count()
{
    terrain.generate();
    return terrain.triangle_count();
}

bool render_debug_depth_at_sample(const Vec3 v0, const Vec3 v1, const Vec3 v2, const Vec2 p, float* out_depth)
{
    if (!out_depth)
    {
        return false;
    }
    const ScreenVertex sv0{static_cast<float>(v0.x), static_cast<float>(v0.y), static_cast<float>(v0.z)};
    const ScreenVertex sv1{static_cast<float>(v1.x), static_cast<float>(v1.y), static_cast<float>(v1.z)};
    const ScreenVertex sv2{static_cast<float>(v2.x), static_cast<float>(v2.y), static_cast<float>(v2.z)};

    const float area = edge_function(sv0, sv1, sv2);
    if (area == 0.0f)
    {
        return false;
    }
    const bool area_positive = area > 0.0f;
    const ScreenVertex sp{static_cast<float>(p.x), static_cast<float>(p.y), 0.0f};

    float w0 = edge_function(sv1, sv2, sp);
    float w1 = edge_function(sv2, sv0, sp);
    float w2 = edge_function(sv0, sv1, sp);

    if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f && area_positive) ||
        (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f && !area_positive))
    {
        w0 /= area;
        w1 /= area;
        w2 /= area;
        const float inv_z0 = 1.0f / sv0.z;
        const float inv_z1 = 1.0f / sv1.z;
        const float inv_z2 = 1.0f / sv2.z;
        const float inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2;
        if (inv_z <= 0.0f)
        {
            return false;
        }
        *out_depth = 1.0f / inv_z;
        return true;
    }
    return false;
}
