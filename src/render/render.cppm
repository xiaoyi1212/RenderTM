module;

#include "../prelude.hpp"

export module render;

export import :math;
export import :camera;
export import :world;
export import :noise;
export import :terrain;
export struct ClipVertex;
export struct GiHit;
export struct LightingEngine;
export struct RenderEngine;
export struct ScreenVertex;
export struct RenderQuad;
export struct ShadingContext;
export struct ViewRotation;

export struct RenderBuffers
{
    size_t width = 0;
    size_t height = 0;
    std::vector<float> zbuffer;
    std::vector<LinearColor> sample_colors;
    std::vector<LinearColor> sample_direct;
    std::vector<LinearColor> sample_direct_sun;
    std::vector<LinearColor> sample_direct_moon;
    std::vector<float> shadow_mask_sun;
    std::vector<float> shadow_mask_moon;
    std::vector<float> shadow_mask_filtered_sun;
    std::vector<float> shadow_mask_filtered_moon;
    std::vector<Vec3> sample_normals;
    std::vector<LinearColor> sample_albedo;
    std::vector<LinearColor> sample_indirect;
    std::vector<float> sample_ao;
    std::vector<Vec3> world_positions;
    std::vector<uint32_t> world_stamp;

    auto resize(const size_t new_width, const size_t new_height, const float depth_max) -> bool
    {
        const size_t sample_count = new_width * new_height;
        if (new_width != width || new_height != height)
        {
            width = new_width;
            height = new_height;
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
            return true;
        }

        std::fill(zbuffer.begin(), zbuffer.end(), depth_max);
        return false;
    }
};

export struct Rasterizer
{
    auto should_rasterize_triangle(const Vec3 v0, const Vec3 v1, const Vec3 v2,
                                   double near_plane) const -> bool;
    auto clip_triangle_to_near_plane(double near_plane,
                                     std::span<const ClipVertex> input,
                                     std::span<ClipVertex> output) const -> size_t;
    static auto edge_function(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c) -> float;
    auto render_quad(float* zbuffer, LinearColor* sample_ambient,
                     LinearColor* sample_direct_sun, LinearColor* sample_direct_moon,
                     float* shadow_mask_sun, float* shadow_mask_moon,
                     Vec3* sample_normals, LinearColor* sample_albedo, float* sample_ao,
                     Vec3* world_positions, uint32_t* world_stamp, uint32_t frame_index,
                     size_t width, size_t height,
                     const RenderQuad& quad, const double proj_scale_x, const double proj_scale_y,
                     const Vec3& camera_pos, const ViewRotation& view_rot,
                     double near_plane,
                     const Terrain& terrain,
                     const LightingEngine& lighting,
                     const ShadingContext& ctx,
                     const float jitter_x, const float jitter_y,
                     const std::array<Vec3, 2>& lights_right_scaled,
                     const std::array<Vec3, 2>& lights_up_scaled,
                     const std::array<BlueNoise::Shift, 2>& shadow_shift_u,
                      const std::array<BlueNoise::Shift, 2>& shadow_shift_v) const -> void;

private:
    static auto clip_lerp(const ClipVertex& a, const ClipVertex& b, const double t) -> ClipVertex;
    auto draw_shaded_triangle(float* zbuffer, LinearColor* sample_ambient,
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
                              const std::array<BlueNoise::Shift, 2>& shadow_shift_v) const -> void;
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
};

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

export struct LightingEngine
{
    auto compute_shadow_factor(const Terrain& terrain,
                               const Vec3& light_dir, const Vec3& world, const Vec3& normal) const -> float;
    auto jitter_shadow_direction(const Vec3& light_dir,
                                 const Vec3& right_scaled,
                                 const Vec3& up_scaled,
                                 const int px, const int py,
                                 const BlueNoise::Shift& shift_u,
                                 const BlueNoise::Shift& shift_v) const -> Vec3;
    auto compute_hemisphere_ground(const LinearColor& base_ground,
                                   const std::array<ShadingContext::DirectionalLightInfo, 2>& lights) const -> LinearColor;
    auto gi_raymarch_hit(const Terrain& terrain,
                         const Vec3& world, const Vec3& normal, const Vec3& dir,
                         double max_distance, GiHit* out_hit) const -> bool;
    auto shadow_filter_at(std::span<const float> mask,
                          std::span<const float> depth,
                          std::span<const Vec3> normals,
                          size_t width, size_t height, int x, int y, float depth_max) const -> float;
    auto filter_shadow_masks(std::span<const float> mask_a,
                             std::span<const float> mask_b,
                             std::span<float> out_a,
                             std::span<float> out_b,
                             std::span<const float> depth,
                             std::span<const Vec3> normals,
                             size_t width, size_t height, float depth_max) const -> void;

private:
    auto shadow_raymarch_hit(const Terrain& terrain,
                             const Vec3& world, const Vec3& normal, const Vec3& light_dir) const -> bool;
};

export struct PostProcessor
{
    std::array<std::vector<LinearColor>, 2> taa_history;
    std::vector<LinearColor> taa_resolved;
    std::vector<LinearColor> current_linear_buffer;
    std::vector<uint8_t> taa_history_mask;
    int taa_ping_pong = 0;
    size_t taa_width = 0;
    size_t taa_height = 0;
    bool taa_history_valid = false;
    uint64_t taa_state_version = 0;
    bool taa_was_enabled = false;
    Vec3 last_camera_pos{0.0, 0.0, 0.0};
    double last_camera_yaw = 0.0;
    double last_camera_pitch = 0.0;
    bool last_camera_valid = false;
    float taa_motion_activity = 0.0f;
    uint64_t taa_motion_state_version = 0;

    auto resize_buffers(const size_t sample_count) -> void
    {
        taa_resolved.assign(sample_count, {0.0f, 0.0f, 0.0f});
        current_linear_buffer.assign(sample_count, {0.0f, 0.0f, 0.0f});
        taa_history_mask.assign(sample_count, 0);
    }

    auto update_taa_state(bool taa_on, size_t width, size_t height, size_t sample_count,
                          uint64_t state_version,
                          const Vec3& camera_pos, double yaw, double pitch) -> float;
    auto resolve_frame(uint32_t* framebuffer, size_t width, size_t height, size_t sample_count,
                       float depth_max, const RenderBuffers& buffers,
                       const LinearColor& sky_top_linear, const LinearColor& sky_bottom_linear,
                       bool taa_on, bool clamp_history, float taa_factor, float taa_sharpen_strength,
                       bool gi_active, uint32_t frame_index,
                       float jitter_x, float jitter_y, double jitter_x_d, double jitter_y_d,
                       double width_d, double height_d, double proj_a, double proj_b,
                       const Mat4& inverse_current_vp, const Mat4& previous_vp, double camera_near_plane,
                       float exposure_factor) -> void;
};

export struct RenderEngine
{
    Camera camera{};
    World world{};
    Terrain terrain{};
    std::atomic<bool> rotationPaused{false};
    std::atomic<uint32_t> renderFrameIndex{0};
    std::atomic<double> ambientLight{0.13};
    std::atomic<bool> taaEnabled{true};
    std::atomic<double> taaBlend{0.05};
    std::atomic<bool> taaClampEnabled{true};
    std::atomic<double> taaSharpenStrength{0.0};
    std::atomic<bool> giEnabled{false};
    std::atomic<double> giStrength{0.0};
    std::atomic<int> giBounceCount{2};
    std::atomic<uint64_t> renderStateVersion{1};
    std::atomic<bool> ambientOcclusionEnabled{true};
    std::atomic<bool> shadowEnabled{true};
    Mat4 currentVP{{{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    }}};
    Mat4 previousVP{{{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    }}};
    Mat4 inverseCurrentVP{{{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    }}};
    RenderBuffers buffers{};
    Rasterizer rasterizer{};
    LightingEngine lighting{};
    PostProcessor post{};

    auto update(uint32_t* framebuffer, size_t width, size_t height) -> void;

    auto set_paused(bool paused) -> void;
    auto is_paused() const -> bool;
    auto toggle_pause() -> void;

    auto set_light_direction(Vec3 dir) -> void;
    auto get_light_direction() const -> Vec3;
    auto set_light_intensity(double intensity) -> void;
    auto get_light_intensity() const -> double;
    auto set_sun_orbit_enabled(bool enabled) -> void;
    auto get_sun_orbit_enabled() const -> bool;
    auto set_sun_orbit_angle(double angle) -> void;
    auto get_sun_orbit_angle() const -> double;
    auto set_moon_direction(Vec3 dir) -> void;
    auto set_moon_intensity(double intensity) -> void;

    auto set_sky_top_color(uint32_t color) -> void;
    auto get_sky_top_color() const -> uint32_t;
    auto set_sky_bottom_color(uint32_t color) -> void;
    auto get_sky_bottom_color() const -> uint32_t;
    auto set_sky_light_intensity(double intensity) -> void;
    auto get_sky_light_intensity() const -> double;
    auto set_exposure(double exposure) -> void;
    auto get_exposure() const -> double;

    auto set_taa_enabled(bool enabled) -> void;
    auto get_taa_enabled() const -> bool;
    auto set_taa_blend(double blend) -> void;
    auto get_taa_blend() const -> double;
    auto set_taa_clamp_enabled(bool enabled) -> void;
    auto get_taa_clamp_enabled() const -> bool;

    auto set_gi_enabled(bool enabled) -> void;
    auto get_gi_enabled() const -> bool;
    auto set_gi_strength(double strength) -> void;
    auto get_gi_strength() const -> double;
    auto set_gi_bounce_count(int count) -> void;
    auto get_gi_bounce_count() const -> int;

    auto reset_taa_history() -> void;
    auto set_ambient_occlusion_enabled(bool enabled) -> void;
    auto set_shadow_enabled(bool enabled) -> void;

    auto set_camera_position(Vec3 pos) -> void;
    auto get_camera_position() const -> Vec3;
    auto move_camera(Vec3 delta) -> void;
    auto move_camera_local(Vec3 delta) -> void;
    auto set_camera_rotation(Vec2 rot) -> void;
    auto get_camera_rotation() const -> Vec2;
    auto rotate_camera(Vec2 delta) -> void;

    auto project_point(Vec3 world, size_t width, size_t height) const -> Vec2;
    auto unproject_point(Vec3 screen, size_t width, size_t height) const -> Vec3;
    auto reproject_point(Vec3 world, size_t width, size_t height) const -> Vec2;

    auto should_rasterize_triangle(Vec3 v0, Vec3 v1, Vec3 v2) const -> bool;
    auto clip_triangle_to_near_plane(Vec3 v0, Vec3 v1, Vec3 v2,
                                     Vec3* out_vertices, size_t max_vertices) const -> size_t;

    auto get_near_plane() const -> double;

    auto taa_sharpen_strength() const -> double;
    auto taa_sharpen_percent() const -> double;

private:
    auto mark_state_dirty() -> void;
};

inline constexpr double kShadowRayBias = 0.05;
inline constexpr double kGiRayBias = 0.04;
inline constexpr double kGiMaxDistance = 12.0;
inline constexpr int kGiNoiseSalt = 73;
inline constexpr float kGiClamp = 4.0f;
inline constexpr int kGiSampleCount = 1;
inline constexpr float kGiAoLift = 0.15f;
inline constexpr double kPi = std::numbers::pi_v<double>;
inline constexpr int kSunShadowSalt = 17;
inline constexpr int kMoonShadowSalt = 19;
inline constexpr double kHemisphereBounceStrength = 0.35;
inline constexpr LinearColor kHemisphereBounceColorLinear{1.0f, 0.9046612f, 0.7758222f};
inline constexpr double kSunIntensityBoost = 1.2;
inline constexpr int kTaaJitterSalt = 37;
inline constexpr double kTaaSharpenMax = 0.25;
inline constexpr double kTaaSharpenRotThreshold = 0.25;
inline constexpr double kTaaSharpenMoveThreshold = 0.5;
inline constexpr double kTaaSharpenMoveGain = 10.0;
inline constexpr double kTaaSharpenRotGain = 20.0;
inline constexpr float kTaaSharpenAttack = 0.5f;
inline constexpr float kTaaSharpenRelease = 0.2f;
inline constexpr bool kShadowFilterEnabled = true;
inline constexpr float kShadowFilterDepthThreshold = 1.0f;
inline constexpr float kShadowFilterNormalThreshold = 0.5f;
inline constexpr float kShadowFilterCenterWeight = 4.0f;
inline constexpr float kShadowFilterNeighborWeight = 1.0f;

LinearColor sample_bilinear_history(std::span<const LinearColor> buffer, const size_t width,
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
    const LinearColor top = LinearColor::lerp(c00, c10, fx);
    const LinearColor bottom = LinearColor::lerp(c01, c11, fx);
    return LinearColor::lerp(top, bottom, fy);
}

export Vec3 sample_bilinear_history_vec3(std::span<const Vec3> buffer, const size_t width,
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
    const Vec3 top = Vec3::lerp(c00, c10, fx);
    const Vec3 bottom = Vec3::lerp(c01, c11, fx);
    return Vec3::lerp(top, bottom, fy);
}

static double pow5(const double value)
{
    const double v2 = value * value;
    return v2 * v2 * value;
}

double schlick_fresnel(double vdoth, double f0)
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

double blinn_phong_normalization(const double shininess)
{
    return (shininess + 8.0) / (8.0 * kPi);
}

inline void sincos_double(const double angle, double* out_sin, double* out_cos)
{
#if defined(__GNUC__)
    __builtin_sincos(angle, out_sin, out_cos);
#else
    *out_sin = std::sin(angle);
    *out_cos = std::cos(angle);
#endif
}

export double eval_specular_term(const double ndoth, const double vdoth, const double ndotl,
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

export struct ClipVertex
{
    Vec3 view;
    Vec3 world;
    Vec3 normal;
    float sky_visibility;
};

static float tonemap_reinhard_channel(float value)
{
    if (value <= 0.0f)
    {
        return 0.0f;
    }
    return value / (1.0f + value);
}

export LinearColor tonemap_reinhard(const LinearColor& color, const float exposure_factor)
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

uint32_t pack_color(const ColorSrgb& color)
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

Vec3 rotate_yaw_pitch_cached(const Vec3& v, const double cy, const double sy,
                                    const double cp, const double sp)
{
    const double x1 = v.x * cy + v.z * sy;
    const double z1 = -v.x * sy + v.z * cy;
    const double y1 = v.y;

    const double y2 = y1 * cp - z1 * sp;
    const double z2 = y1 * sp + z1 * cp;

    return {x1, y2, z2};
}

ViewRotation make_view_rotation(const double yaw, const double pitch)
{
    return {std::cos(yaw), std::sin(yaw), std::cos(pitch), std::sin(pitch)};
}

auto make_projection_matrix(const double width, const double height,
                            const double proj_scale_x, const double proj_scale_y,
                            const Camera& camera) -> Mat4
{
    if (width <= 0.0 || height <= 0.0 ||
        Camera::far_plane <= Camera::near_plane)
    {
        return Mat4::identity();
    }
    const double sx = 2.0 * proj_scale_x / width;
    const double sy = 2.0 * proj_scale_y / height;
    const double inv_range = 1.0 / (Camera::far_plane - Camera::near_plane);
    const double a = Camera::far_plane * inv_range;
    const double b = -Camera::near_plane * Camera::far_plane * inv_range;

    Mat4 m{};
    m.m[0][0] = sx;
    m.m[1][1] = sy;
    m.m[2][2] = a;
    m.m[2][3] = b;
    m.m[3][2] = 1.0;
    return m;
}

export struct GiHit
{
    Vec3 position;
    Vec3 normal;
    LinearColor albedo;
    float sky_visibility;
};

Vec3 unproject_fast(double screen_x, double screen_y, double depth, 
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
