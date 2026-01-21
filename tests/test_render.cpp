#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

#include "render.h"
#include "blue_noise.h"

static void reset_camera()
{
    render_set_camera_position({0.0, 0.0, -4.0});
    render_set_camera_rotation({0.0, 0.0});
    render_set_sky_top_color(0xFF78C2FF);
    render_set_sky_bottom_color(0xFF172433);
    render_set_sky_light_intensity(0.0);
    render_set_ambient_occlusion_enabled(true);
    render_set_sun_orbit_enabled(false);
    render_set_sun_orbit_angle(0.0);
    render_set_moon_direction({0.0, 1.0, 0.0});
    render_set_moon_intensity(0.0);
    render_set_shadow_enabled(true);
    render_set_exposure(1.0);
    render_set_taa_enabled(true);
    render_set_taa_blend(0.2);
}

static Mat4 mat4_multiply(const Mat4& a, const Mat4& b)
{
    Mat4 r{};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k)
            {
                sum += a.m[i][k] * b.m[k][j];
            }
            r.m[i][j] = sum;
        }
    }
    return r;
}

static bool mat4_near_equal(const Mat4& a, const Mat4& b, const double eps)
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            if (std::fabs(a.m[i][j] - b.m[i][j]) > eps)
            {
                return false;
            }
        }
    }
    return true;
}

static bool mat4_is_identity(const Mat4& m, const double eps)
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            const double expected = (i == j) ? 1.0 : 0.0;
            if (std::fabs(m.m[i][j] - expected) > eps)
            {
                return false;
            }
        }
    }
    return true;
}

static std::string mat4_to_string(const Mat4& m)
{
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            oss << m.m[i][j];
            if (j < 3)
            {
                oss << ", ";
            }
        }
        if (i < 3)
        {
            oss << '\n';
        }
    }
    return oss.str();
}

static Vec3 rotate_yaw_pitch(const Vec3& v, const double yaw, const double pitch)
{
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);
    const double x1 = v.x * cy + v.z * sy;
    const double z1 = -v.x * sy + v.z * cy;
    const double y1 = v.y;
    const double y2 = y1 * cp - z1 * sp;
    const double z2 = y1 * sp + z1 * cp;
    return {x1, y2, z2};
}

static float srgb_channel_to_linear(float channel);
static float linear_channel_to_srgb(float channel);

static uint32_t sky_color_for_row(size_t y, size_t height, uint32_t sky_top, uint32_t sky_bottom)
{
    const float t = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
    const uint32_t r0 = (sky_top >> 16) & 0xFF;
    const uint32_t g0 = (sky_top >> 8) & 0xFF;
    const uint32_t b0 = sky_top & 0xFF;
    const uint32_t r1 = (sky_bottom >> 16) & 0xFF;
    const uint32_t g1 = (sky_bottom >> 8) & 0xFF;
    const uint32_t b1 = sky_bottom & 0xFF;
    const float r0_lin = srgb_channel_to_linear(static_cast<float>(r0));
    const float g0_lin = srgb_channel_to_linear(static_cast<float>(g0));
    const float b0_lin = srgb_channel_to_linear(static_cast<float>(b0));
    const float r1_lin = srgb_channel_to_linear(static_cast<float>(r1));
    const float g1_lin = srgb_channel_to_linear(static_cast<float>(g1));
    const float b1_lin = srgb_channel_to_linear(static_cast<float>(b1));
    const float r_lin = r0_lin + (r1_lin - r0_lin) * t;
    const float g_lin = g0_lin + (g1_lin - g0_lin) * t;
    const float b_lin = b0_lin + (b1_lin - b0_lin) * t;
    const float exposure = static_cast<float>(std::max(0.0, render_get_exposure()));
    const float r_mapped = exposure <= 0.0f ? 0.0f : (r_lin * exposure) / (1.0f + r_lin * exposure);
    const float g_mapped = exposure <= 0.0f ? 0.0f : (g_lin * exposure) / (1.0f + g_lin * exposure);
    const float b_mapped = exposure <= 0.0f ? 0.0f : (b_lin * exposure) / (1.0f + b_lin * exposure);
    const uint32_t r = static_cast<uint32_t>(std::lround(linear_channel_to_srgb(r_mapped) * 255.0f));
    const uint32_t g = static_cast<uint32_t>(std::lround(linear_channel_to_srgb(g_mapped) * 255.0f));
    const uint32_t b = static_cast<uint32_t>(std::lround(linear_channel_to_srgb(b_mapped) * 255.0f));
    return 0xFF000000u | (r << 16) | (g << 8) | b;
}

static size_t count_geometry_pixels(const std::vector<uint32_t>& framebuffer, size_t width, size_t height,
                                    uint32_t sky_top, uint32_t sky_bottom)
{
    size_t count = 0;
    for (size_t y = 0; y < height; ++y)
    {
        const uint32_t sky = sky_color_for_row(y, height, sky_top, sky_bottom);
        for (size_t x = 0; x < width; ++x)
        {
            if (framebuffer[y * width + x] != sky)
            {
                count++;
            }
        }
    }
    return count;
}

static uint32_t pixel_luminance(uint32_t color)
{
    return ((color >> 16) & 0xFF) + ((color >> 8) & 0xFF) + (color & 0xFF);
}

static double average_luminance_delta_masked(const std::vector<uint32_t>& a,
                                             const std::vector<uint32_t>& b,
                                             size_t width, size_t height,
                                             uint32_t sky_top, uint32_t sky_bottom)
{
    double sum = 0.0;
    size_t count = 0;
    for (size_t y = 0; y < height; ++y)
    {
        const uint32_t sky = sky_color_for_row(y, height, sky_top, sky_bottom);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            if (a[idx] == sky && b[idx] == sky)
            {
                continue;
            }
            const uint32_t la = pixel_luminance(a[idx]);
            const uint32_t lb = pixel_luminance(b[idx]);
            sum += static_cast<double>(la > lb ? la - lb : lb - la);
            count++;
        }
    }
    if (count == 0)
    {
        return 0.0;
    }
    return sum / static_cast<double>(count);
}

static std::array<int, 512> make_permutation(const int seed)
{
    std::array<int, 256> p{};
    std::iota(p.begin(), p.end(), 0);
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::shuffle(p.begin(), p.end(), rng);
    std::array<int, 512> perm{};
    for (size_t i = 0; i < perm.size(); ++i)
    {
        perm[i] = p[i & 255];
    }
    return perm;
}

static double simplex_noise(const double xin, const double yin)
{
    static const std::array<int, 512> perm = make_permutation(1337);
    static constexpr int grad2[8][2] = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {-1, 1}, {1, -1}, {-1, -1}
    };

    static constexpr double f2 = 0.366025403784438646;
    static constexpr double g2 = 0.211324865405187117;

    const double s = (xin + yin) * f2;
    const int i = static_cast<int>(std::floor(xin + s));
    const int j = static_cast<int>(std::floor(yin + s));
    const double t = (i + j) * g2;
    const double x0 = xin - (static_cast<double>(i) - t);
    const double y0 = yin - (static_cast<double>(j) - t);

    const int i1 = (x0 > y0) ? 1 : 0;
    const int j1 = (x0 > y0) ? 0 : 1;

    const double x1 = x0 - static_cast<double>(i1) + g2;
    const double y1 = y0 - static_cast<double>(j1) + g2;
    const double x2 = x0 - 1.0 + 2.0 * g2;
    const double y2 = y0 - 1.0 + 2.0 * g2;

    const int ii = i & 255;
    const int jj = j & 255;

    auto grad_dot = [&](int hash, double x, double y) {
        const int* g = grad2[hash & 7];
        return static_cast<double>(g[0]) * x + static_cast<double>(g[1]) * y;
    };

    double n0 = 0.0;
    double t0 = 0.5 - x0 * x0 - y0 * y0;
    if (t0 > 0.0)
    {
        t0 *= t0;
        n0 = t0 * t0 * grad_dot(perm[ii + perm[jj]], x0, y0);
    }

    double n1 = 0.0;
    double t1 = 0.5 - x1 * x1 - y1 * y1;
    if (t1 > 0.0)
    {
        t1 *= t1;
        n1 = t1 * t1 * grad_dot(perm[ii + i1 + perm[jj + j1]], x1, y1);
    }

    double n2 = 0.0;
    double t2 = 0.5 - x2 * x2 - y2 * y2;
    if (t2 > 0.0)
    {
        t2 *= t2;
        n2 = t2 * t2 * grad_dot(perm[ii + 1 + perm[jj + 1]], x2, y2);
    }

    return 70.0 * (n0 + n1 + n2);
}

static void build_heightmap(std::vector<int>& heights, std::vector<uint32_t>& top_colors)
{
    const int chunk_size = 16;
    const int base_height = 4;
    const int height_variation = 6;
    const double height_freq = 0.12;
    const double surface_freq = 0.4;
    const uint32_t dirt_color = 0xFF8A4F22;
    const uint32_t grass_color = 0xFF3B8A38;
    const uint32_t water_color = 0xFF2B5FA8;

    heights.assign(static_cast<size_t>(chunk_size * chunk_size), 0);
    top_colors.assign(static_cast<size_t>(chunk_size * chunk_size), grass_color);

    auto index = [chunk_size](int x, int z) {
        return static_cast<size_t>(z * chunk_size + x);
    };

    for (int z = 0; z < chunk_size; ++z)
    {
        for (int x = 0; x < chunk_size; ++x)
        {
            const double h = simplex_noise(x * height_freq, z * height_freq);
            int height = base_height + static_cast<int>(std::lround((h + 1.0) * 0.5 * height_variation));
            if (height < 3)
            {
                height = 3;
            }

            const double surface = simplex_noise(x * surface_freq + 100.0, z * surface_freq - 100.0);
            uint32_t top_color = grass_color;
            if (surface > 0.55)
            {
                top_color = water_color;
            }
            else if (surface < -0.35)
            {
                top_color = dirt_color;
            }

            heights[index(x, z)] = height;
            top_colors[index(x, z)] = top_color;
        }
    }
}

static bool heightmap_has_block(const std::vector<int>& heights, const int gx, const int gy, const int gz)
{
    const int chunk_size = 16;
    if (gx < 0 || gx >= chunk_size || gz < 0 || gz >= chunk_size || gy < 0)
    {
        return false;
    }
    const size_t idx = static_cast<size_t>(gz * chunk_size + gx);
    if (idx >= heights.size())
    {
        return false;
    }
    return gy < heights[idx];
}

static int heightmap_max_height(const std::vector<int>& heights)
{
    int max_height = 0;
    for (int value : heights)
    {
        if (value > max_height)
        {
            max_height = value;
        }
    }
    return max_height;
}

static Vec3 world_to_grid_coords(const Vec3& pos)
{
    const int chunk_size = 16;
    const double block_size = 2.0;
    const double half = block_size * 0.5;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;
    return {
        (pos.x - start_x + half) / block_size,
        (base_y - pos.y + half) / block_size,
        (pos.z - start_z + half) / block_size
    };
}

static bool raymarch_shadow_hit(const std::vector<int>& heights, const Vec3& world,
                                const Vec3& light_dir, const Vec3& normal)
{
    const int chunk_size = 16;
    const int max_height = heightmap_max_height(heights);
    if (max_height <= 0)
    {
        return false;
    }

    const double step = 0.1;
    const int max_steps = 2000;
    const double bias = 0.05;
    Vec3 pos{
        world.x + normal.x * bias,
        world.y + normal.y * bias,
        world.z + normal.z * bias
    };

    for (int i = 0; i < max_steps; ++i)
    {
        pos.x += light_dir.x * step;
        pos.y += light_dir.y * step;
        pos.z += light_dir.z * step;

        const Vec3 grid = world_to_grid_coords(pos);
        if (grid.x < 0.0 || grid.x >= chunk_size ||
            grid.z < 0.0 || grid.z >= chunk_size ||
            grid.y < 0.0 || grid.y >= max_height)
        {
            return false;
        }

        const int gx = static_cast<int>(std::floor(grid.x));
        const int gy = static_cast<int>(std::floor(grid.y));
        const int gz = static_cast<int>(std::floor(grid.z));
        if (heightmap_has_block(heights, gx, gy, gz))
        {
            return true;
        }
    }
    return false;
}

static bool find_shadow_sample(const std::vector<int>& heights, const Vec3& light_dir,
                               const Vec3& normal, const bool want_shadow, Vec3* out_world)
{
    if (!out_world)
    {
        return false;
    }
    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    auto index = [chunk_size](int x, int z) {
        return static_cast<size_t>(z * chunk_size + x);
    };

    for (int z = 0; z < chunk_size; ++z)
    {
        for (int x = 0; x < chunk_size; ++x)
        {
            const int height = heights[index(x, z)];
            if (height <= 0)
            {
                continue;
            }
            const double center_y = base_y - (height - 1) * block_size;
            const double top_y = center_y - block_size * 0.5;
            const Vec3 world{
                start_x + x * block_size,
                top_y,
                start_z + z * block_size
            };
            const bool hit = raymarch_shadow_hit(heights, world, light_dir, normal);
            if (hit == want_shadow)
            {
                *out_world = world;
                return true;
            }
        }
    }
    return false;
}

static size_t count_blocks(const std::vector<int>& heights, const int chunk_size)
{
    size_t total = 0;
    for (int z = 0; z < chunk_size; ++z)
    {
        for (int x = 0; x < chunk_size; ++x)
        {
            total += static_cast<size_t>(heights[static_cast<size_t>(z * chunk_size + x)]);
        }
    }
    return total;
}

static bool terrain_has_block(const std::vector<int>& heights, const int chunk_size,
                              const int gx, const int gy, const int gz);

extern double render_debug_eval_specular(double ndoth, double vdoth, double ndotl,
                                         double shininess, double f0);

constexpr int kFaceTop = 0;
constexpr int kFaceBottom = 1;
constexpr int kFaceLeft = 2;
constexpr int kFaceRight = 3;
constexpr int kFaceBack = 4;
constexpr int kFaceFront = 5;

constexpr int kCubeFaceNormal[6][3] = {
    {0, 1, 0},
    {0, -1, 0},
    {-1, 0, 0},
    {1, 0, 0},
    {0, 0, -1},
    {0, 0, 1}
};

constexpr int kCubeFaceVertices[6][4] = {
    {0, 1, 5, 4},
    {3, 2, 6, 7},
    {0, 3, 7, 4},
    {1, 2, 6, 5},
    {0, 1, 2, 3},
    {4, 5, 6, 7}
};

constexpr int kCubeVertexGrid[8][3] = {
    {0, 1, 0},
    {1, 1, 0},
    {1, 0, 0},
    {0, 0, 0},
    {0, 1, 1},
    {1, 1, 1},
    {1, 0, 1},
    {0, 0, 1}
};

constexpr double kPi = 3.14159265358979323846;
constexpr size_t kSkyRayCount = 128;
constexpr double kSkyRayStep = 0.25;
constexpr double kSkyRayMaxDistance = 6.0;
constexpr double kSkyRayBias = 0.02;
constexpr double kSkyRayCenterBias = 0.02;

static double radical_inverse_vdc(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<double>(bits) * 2.3283064365386963e-10;
}

static const std::vector<Vec3>& sky_sample_dirs()
{
    static std::vector<Vec3> samples = [] {
        std::vector<Vec3> dirs;
        dirs.reserve(kSkyRayCount);
        for (size_t i = 0; i < kSkyRayCount; ++i)
        {
            const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(kSkyRayCount);
            const double v = radical_inverse_vdc(static_cast<uint32_t>(i));
            const double r = std::sqrt(u);
            const double theta = 2.0 * kPi * v;
            const double x = r * std::cos(theta);
            const double y = r * std::sin(theta);
            const double z = std::sqrt(std::max(0.0, 1.0 - u));
            dirs.push_back({x, y, z});
        }
        return dirs;
    }();
    return samples;
}

static Vec3 normalize_vec3(const Vec3& v)
{
    const double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len == 0.0) return {0.0, 0.0, 0.0};
    return {v.x / len, v.y / len, v.z / len};
}

static void build_basis(const Vec3& n, Vec3& t, Vec3& b)
{
    Vec3 up{0.0, 1.0, 0.0};
    if (std::abs(n.x * up.x + n.y * up.y + n.z * up.z) > 0.99)
    {
        up = {1.0, 0.0, 0.0};
    }
    t = normalize_vec3({up.y * n.z - up.z * n.y, up.z * n.x - up.x * n.z, up.x * n.y - up.y * n.x});
    b = {n.y * t.z - n.z * t.y, n.z * t.x - n.x * t.z, n.x * t.y - n.y * t.x};
}

static Vec3 face_corner_position_grid(const int gx, const int gy, const int gz,
                                      const int face, const int corner)
{
    const int vi = kCubeFaceVertices[face][corner];
    return {
        static_cast<double>(gx + kCubeVertexGrid[vi][0]),
        static_cast<double>(gy + kCubeVertexGrid[vi][1]),
        static_cast<double>(gz + kCubeVertexGrid[vi][2])
    };
}

static float raycast_vertex_sky_visibility(const std::vector<int>& heights, const int chunk_size,
                                           const int gx, const int gy, const int gz,
                                           const int face, const int corner)
{
    const Vec3 normal = normalize_vec3({
        static_cast<double>(kCubeFaceNormal[face][0]),
        static_cast<double>(kCubeFaceNormal[face][1]),
        static_cast<double>(kCubeFaceNormal[face][2])
    });
    Vec3 tangent{};
    Vec3 bitangent{};
    build_basis(normal, tangent, bitangent);

    const Vec3 vertex = face_corner_position_grid(gx, gy, gz, face, corner);
    const Vec3 center{
        static_cast<double>(gx) + 0.5,
        static_cast<double>(gy) + 0.5,
        static_cast<double>(gz) + 0.5
    };
    Vec3 origin = vertex;
    origin.x += normal.x * kSkyRayBias;
    origin.y += normal.y * kSkyRayBias;
    origin.z += normal.z * kSkyRayBias;
    origin.x += (center.x - vertex.x) * kSkyRayCenterBias;
    origin.y += (center.y - vertex.y) * kSkyRayCenterBias;
    origin.z += (center.z - vertex.z) * kSkyRayCenterBias;

    const auto& samples = sky_sample_dirs();
    size_t occluded = 0;
    for (const auto& sample : samples)
    {
        Vec3 dir{
            tangent.x * sample.x + bitangent.x * sample.y + normal.x * sample.z,
            tangent.y * sample.x + bitangent.y * sample.y + normal.y * sample.z,
            tangent.z * sample.x + bitangent.z * sample.y + normal.z * sample.z
        };

        bool hit = false;
        for (double t = kSkyRayStep; t <= kSkyRayMaxDistance; t += kSkyRayStep)
        {
            const Vec3 p{
                origin.x + dir.x * t,
                origin.y + dir.y * t,
                origin.z + dir.z * t
            };
            const int vx = static_cast<int>(std::floor(p.x));
            const int vy = static_cast<int>(std::floor(p.y));
            const int vz = static_cast<int>(std::floor(p.z));
            if (terrain_has_block(heights, chunk_size, vx, vy, vz))
            {
                hit = true;
                break;
            }
        }
        if (hit)
        {
            occluded++;
        }
    }

    const double visibility = 1.0 - static_cast<double>(occluded) / static_cast<double>(samples.size());
    return static_cast<float>(std::clamp(visibility, 0.0, 1.0));
}

static bool terrain_has_block(const std::vector<int>& heights, const int chunk_size,
                              const int gx, const int gy, const int gz)
{
    if (gx < 0 || gx >= chunk_size || gz < 0 || gz >= chunk_size)
    {
        return false;
    }
    if (gy < 0)
    {
        return false;
    }
    const size_t idx = static_cast<size_t>(gz * chunk_size + gx);
    return gy < heights[idx];
}

static size_t count_visible_faces(const std::vector<int>& heights, const int chunk_size)
{
    size_t faces = 0;
    for (int z = 0; z < chunk_size; ++z)
    {
        for (int x = 0; x < chunk_size; ++x)
        {
            const int height = heights[static_cast<size_t>(z * chunk_size + x)];
            for (int y = 0; y < height; ++y)
            {
                if (!terrain_has_block(heights, chunk_size, x + 1, y, z)) faces++;
                if (!terrain_has_block(heights, chunk_size, x - 1, y, z)) faces++;
                if (!terrain_has_block(heights, chunk_size, x, y + 1, z)) faces++;
                if (!terrain_has_block(heights, chunk_size, x, y - 1, z)) faces++;
                if (!terrain_has_block(heights, chunk_size, x, y, z + 1)) faces++;
                if (!terrain_has_block(heights, chunk_size, x, y, z - 1)) faces++;
            }
        }
    }
    return faces;
}

static bool find_sloped_grass_cell(int& out_x, int& out_z, int& out_height)
{
    const int chunk_size = 16;
    const uint32_t grass_color = 0xFF3B8A38;

    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);

    auto index = [chunk_size](int x, int z) {
        return static_cast<size_t>(z * chunk_size + x);
    };

    int best_slope = -1;
    int best_x = 0;
    int best_z = 0;
    int best_height = 0;

    for (int z = 1; z < chunk_size - 1; ++z)
    {
        for (int x = 1; x < chunk_size - 1; ++x)
        {
            if (top_colors[index(x, z)] != grass_color)
            {
                continue;
            }
            const int h_l = heights[index(x - 1, z)];
            const int h_r = heights[index(x + 1, z)];
            const int h_d = heights[index(x, z - 1)];
            const int h_u = heights[index(x, z + 1)];
            const int slope = std::abs(h_r - h_l) + std::abs(h_u - h_d);
            if (slope > best_slope)
            {
                best_slope = slope;
                best_x = x;
                best_z = z;
                best_height = heights[index(x, z)];
            }
        }
    }

    if (best_slope <= 0)
    {
        return false;
    }

    out_x = best_x;
    out_z = best_z;
    out_height = best_height;
    return true;
}

struct FlatCornerProbe
{
    int x;
    int z;
    int height;
    int sx;
    int sz;
};

static bool find_flat_shared_corner(FlatCornerProbe& out_probe)
{
    const int chunk_size = 16;
    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);

    auto index = [chunk_size](int x, int z) {
        return static_cast<size_t>(z * chunk_size + x);
    };

    constexpr int corners[4][2] = {
        {1, 1},
        {1, -1},
        {-1, 1},
        {-1, -1}
    };

    for (int z = 1; z < chunk_size - 1; ++z)
    {
        for (int x = 1; x < chunk_size - 1; ++x)
        {
            const int h = heights[index(x, z)];
            for (const auto& corner : corners)
            {
                const int sx = corner[0];
                const int sz = corner[1];
                const int nx = x + sx;
                const int nz = z + sz;
                if (nx < 0 || nx >= chunk_size || nz < 0 || nz >= chunk_size)
                {
                    continue;
                }
                if (heights[index(nx, z)] != h)
                {
                    continue;
                }
                if (heights[index(x, nz)] > h)
                {
                    continue;
                }
                if (heights[index(nx, nz)] > h)
                {
                    continue;
                }
                out_probe = {x, z, h, sx, sz};
                return true;
            }
        }
    }

    return false;
}

static int top_face_corner_from_offsets(const int sx, const int sz)
{
    if (sx > 0 && sz > 0) return 2;
    if (sx > 0 && sz < 0) return 1;
    if (sx < 0 && sz > 0) return 3;
    return 0;
}

struct SideFaceAoProbe
{
    int x;
    int z;
    int height;
};

static bool find_right_face_diagonal_occluder(SideFaceAoProbe& out_probe)
{
    const int chunk_size = 16;
    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);

    auto index = [chunk_size](int x, int z) {
        return static_cast<size_t>(z * chunk_size + x);
    };

    for (int z = 0; z < chunk_size - 1; ++z)
    {
        for (int x = 0; x < chunk_size - 1; ++x)
        {
            const int height = heights[index(x, z)];
            const int y = height - 1;
            if (y < 0)
            {
                continue;
            }
            const int right_height = heights[index(x + 1, z)];
            const int z_height = heights[index(x, z + 1)];
            const int diag_height = heights[index(x + 1, z + 1)];
            if (right_height != height - 1)
            {
                continue;
            }
            if (z_height > y)
            {
                continue;
            }
            if (diag_height <= y)
            {
                continue;
            }
            out_probe = {x, z, height};
            return true;
        }
    }

    return false;
}

static bool find_front_column(int& out_x, int& out_height)
{
    const int chunk_size = 16;
    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);

    auto index = [chunk_size](int x, int z) {
        return static_cast<size_t>(z * chunk_size + x);
    };

    int best_height = -1;
    int best_x = 0;
    const int z = 0;
    for (int x = 0; x < chunk_size; ++x)
    {
        const int height = heights[index(x, z)];
        if (height > best_height)
        {
            best_height = height;
            best_x = x;
        }
    }

    if (best_height <= 0)
    {
        return false;
    }

    out_x = best_x;
    out_height = best_height;
    return true;
}

static bool colors_similar(uint32_t a, uint32_t b, int tolerance)
{
    const int ar = static_cast<int>((a >> 16) & 0xFF);
    const int ag = static_cast<int>((a >> 8) & 0xFF);
    const int ab = static_cast<int>(a & 0xFF);
    const int br = static_cast<int>((b >> 16) & 0xFF);
    const int bg = static_cast<int>((b >> 8) & 0xFF);
    const int bb = static_cast<int>(b & 0xFF);
    return std::abs(ar - br) <= tolerance &&
           std::abs(ag - bg) <= tolerance &&
           std::abs(ab - bb) <= tolerance;
}

static bool sample_average_luminance(const std::vector<uint32_t>& frame, size_t width, size_t height,
                                     int px, int py, uint32_t sky_top, uint32_t sky_bottom,
                                     double& out_avg)
{
    const int radius = 4;
    double sum = 0.0;
    size_t count = 0;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int y = std::clamp(py + dy, 0, static_cast<int>(height) - 1);
        const uint32_t sky = sky_color_for_row(static_cast<size_t>(y), height, sky_top, sky_bottom);
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = std::clamp(px + dx, 0, static_cast<int>(width) - 1);
            const uint32_t pixel = frame[static_cast<size_t>(y) * width + static_cast<size_t>(x)];
            if (pixel == sky)
            {
                continue;
            }
            sum += static_cast<double>(pixel_luminance(pixel));
            count++;
        }
    }

    if (count == 0)
    {
        return false;
    }
    out_avg = sum / static_cast<double>(count);
    return true;
}

static bool sample_average_color(const std::vector<uint32_t>& frame, size_t width, size_t height,
                                 int px, int py, uint32_t sky_top, uint32_t sky_bottom,
                                 double& out_r, double& out_g, double& out_b)
{
    const int radius = 4;
    double sum_r = 0.0;
    double sum_g = 0.0;
    double sum_b = 0.0;
    size_t count = 0;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int y = std::clamp(py + dy, 0, static_cast<int>(height) - 1);
        const uint32_t sky = sky_color_for_row(static_cast<size_t>(y), height, sky_top, sky_bottom);
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = std::clamp(px + dx, 0, static_cast<int>(width) - 1);
            const uint32_t pixel = frame[static_cast<size_t>(y) * width + static_cast<size_t>(x)];
            if (pixel == sky)
            {
                continue;
            }
            sum_r += static_cast<double>((pixel >> 16) & 0xFF);
            sum_g += static_cast<double>((pixel >> 8) & 0xFF);
            sum_b += static_cast<double>(pixel & 0xFF);
            count++;
        }
    }

    if (count == 0)
    {
        return false;
    }
    out_r = sum_r / static_cast<double>(count);
    out_g = sum_g / static_cast<double>(count);
    out_b = sum_b / static_cast<double>(count);
    return true;
}

static float srgb_channel_to_linear(float channel)
{
    const float c = channel / 255.0f;
    if (c <= 0.04045f)
    {
        return c / 12.92f;
    }
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

static float linear_channel_to_srgb(float channel)
{
    channel = std::clamp(channel, 0.0f, 1.0f);
    if (channel <= 0.0031308f)
    {
        return channel * 12.92f;
    }
    return 1.055f * std::pow(channel, 1.0f / 2.4f) - 0.055f;
}

static uint32_t expected_gamma_luminance(uint32_t albedo, float intensity)
{
    const float r_lin = srgb_channel_to_linear(static_cast<float>((albedo >> 16) & 0xFF)) * intensity;
    const float g_lin = srgb_channel_to_linear(static_cast<float>((albedo >> 8) & 0xFF)) * intensity;
    const float b_lin = srgb_channel_to_linear(static_cast<float>(albedo & 0xFF)) * intensity;
    const float exposure = static_cast<float>(std::max(0.0, render_get_exposure()));
    const float r_mapped = exposure <= 0.0f ? 0.0f : (r_lin * exposure) / (1.0f + r_lin * exposure);
    const float g_mapped = exposure <= 0.0f ? 0.0f : (g_lin * exposure) / (1.0f + g_lin * exposure);
    const float b_mapped = exposure <= 0.0f ? 0.0f : (b_lin * exposure) / (1.0f + b_lin * exposure);
    const uint32_t r = static_cast<uint32_t>(std::lround(linear_channel_to_srgb(r_mapped) * 255.0f));
    const uint32_t g = static_cast<uint32_t>(std::lround(linear_channel_to_srgb(g_mapped) * 255.0f));
    const uint32_t b = static_cast<uint32_t>(std::lround(linear_channel_to_srgb(b_mapped) * 255.0f));
    return r + g + b;
}

TEST_CASE("blue noise sampling responds to salt and frame")
{
    const float base = sample_noise(5, 7, 0, 0);
    REQUIRE(base >= 0.0f);
    REQUIRE(base < 1.0f);

    int same_salt = 0;
    int same_frame = 0;
    int total = 0;
    for (int y = 0; y < 8; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            const float a = sample_noise(x, y, 0, 0);
            const float b = sample_noise(x, y, 0, 1);
            const float c = sample_noise(x, y, 1, 0);
            if (a == b) same_salt++;
            if (a == c) same_frame++;
            total++;
        }
    }

    REQUIRE(same_salt < total);
    REQUIRE(same_frame < total);
}

TEST_CASE("shadow spatial filter applies gaussian blur with bilateral rejection")
{
    const std::array<float, 9> mask = {1.0f, 1.0f, 1.0f,
                                       1.0f, 0.0f, 1.0f,
                                       1.0f, 1.0f, 1.0f};
    std::array<float, 9> depth_same{};
    depth_same.fill(5.0f);
    std::array<Vec3, 9> normals_same{};
    for (auto& n : normals_same)
    {
        n = {0.0, 1.0, 0.0};
    }

    const float baseline = render_debug_shadow_filter_3x3(mask.data(), depth_same.data(), normals_same.data());
    REQUIRE(baseline == Catch::Approx(0.75f).margin(0.02f));

    std::array<float, 9> depth_far = depth_same;
    for (size_t i = 0; i < depth_far.size(); ++i)
    {
        if (i != 4)
        {
            depth_far[i] = 50.0f;
        }
    }
    const float depth_filtered = render_debug_shadow_filter_3x3(mask.data(), depth_far.data(), normals_same.data());
    REQUIRE(depth_filtered < baseline);
    REQUIRE(depth_filtered < 0.2f);

    std::array<Vec3, 9> normals_flipped = normals_same;
    for (size_t i = 0; i < normals_flipped.size(); ++i)
    {
        if (i != 4)
        {
            normals_flipped[i] = {0.0, -1.0, 0.0};
        }
    }
    const float normal_filtered = render_debug_shadow_filter_3x3(mask.data(), depth_same.data(), normals_flipped.data());
    REQUIRE(normal_filtered < baseline);
    REQUIRE(normal_filtered < 0.2f);
}

TEST_CASE("render_update_array clears framebuffer and draws geometry")
{
    reset_camera();
    render_set_paused(false);
    render_set_light_direction({0.0, 0.0, -1.0});
    render_set_light_intensity(1.0);
    const size_t width = 120;
    const size_t height = 80;

    std::vector<uint32_t> framebuffer(width * height, 0xFFFFFFFF);
    render_update_array(framebuffer.data(), width, height);

    for (const uint32_t pixel : framebuffer)
    {
        REQUIRE((pixel & 0xFF000000u) == 0xFF000000u);
    }

    const size_t colored = count_geometry_pixels(framebuffer, width, height, 0xFF78C2FF, 0xFF172433);
    REQUIRE(colored > 0);
}

TEST_CASE("render_update_array handles tiny even-sized buffers")
{
    reset_camera();
    render_set_paused(false);
    render_set_light_direction({0.0, 0.0, 1.0});
    render_set_light_intensity(1.0);
    const size_t width = 2;
    const size_t height = 2;

    std::vector<uint32_t> framebuffer(width * height, 0xFFFFFFFF);
    render_update_array(framebuffer.data(), width, height);

    for (const uint32_t pixel : framebuffer)
    {
        REQUIRE((pixel & 0xFF000000u) == 0xFF000000u);
    }
}

TEST_CASE("pause stops sun orbit but keeps rendering active")
{
    reset_camera();
    const size_t width = 64;
    const size_t height = 64;

    render_set_light_direction({0.0, 0.0, 1.0});
    render_set_light_intensity(1.0);
    render_set_sun_orbit_enabled(true);
    render_set_sun_orbit_angle(0.5);

    std::vector<uint32_t> framebuffer(width * height, 0u);

    render_set_paused(false);
    for (int i = 0; i < 4; ++i)
    {
        render_update_array(framebuffer.data(), width, height);
    }
    const double moved_angle = render_get_sun_orbit_angle();
    REQUIRE(std::abs(moved_angle - 0.5) > 1e-4);

    render_set_paused(true);
    const double paused_angle = render_get_sun_orbit_angle();
    for (int i = 0; i < 4; ++i)
    {
        render_update_array(framebuffer.data(), width, height);
    }
    render_set_paused(false);

    REQUIRE(render_get_sun_orbit_angle() == Catch::Approx(paused_angle));
}

TEST_CASE("temporal accumulation reduces frame-to-frame noise")
{
    reset_camera();
    render_set_sun_orbit_enabled(false);
    render_set_light_intensity(1.0);
    render_set_moon_intensity(0.0);
    render_set_shadow_enabled(true);
    render_set_sky_light_intensity(0.0);
    render_set_ambient_occlusion_enabled(false);
    render_set_exposure(1.0);
    render_set_paused(false);

    const size_t width = 200;
    const size_t height = 160;
    const uint32_t sky_top = 0xFF78C2FF;
    const uint32_t sky_bottom = 0xFF172433;

    struct Config
    {
        Vec3 camera_pos;
        Vec2 camera_rot;
        Vec3 light_dir;
    };

    const std::array<Config, 3> configs = {{
        {{0.0, 0.0, -4.0}, {0.0, 0.0}, {0.6, -0.3, 0.8}},
        {{0.0, 10.0, -12.0}, {0.0, -0.5}, {0.9, -0.2, 0.3}},
        {{0.0, 22.0, -10.0}, {0.0, -0.8}, {0.4, -0.5, 0.8}}
    }};

    auto max_luminance_delta = [&](const std::vector<uint32_t>& a,
                                   const std::vector<uint32_t>& b) {
        double max_delta = 0.0;
        for (size_t y = 0; y < height; ++y)
        {
            const uint32_t sky = sky_color_for_row(y, height, sky_top, sky_bottom);
            for (size_t x = 0; x < width; ++x)
            {
                const size_t idx = y * width + x;
                if (a[idx] == sky || b[idx] == sky)
                {
                    continue;
                }
                const double delta = std::abs(static_cast<double>(pixel_luminance(a[idx])) -
                                              static_cast<double>(pixel_luminance(b[idx])));
                if (delta > max_delta)
                {
                    max_delta = delta;
                }
            }
        }
        return max_delta;
    };

    std::vector<uint32_t> frame0(width * height, 0u);
    std::vector<uint32_t> frame1(width * height, 0u);

    double delta_no_taa = 0.0;
    bool found = false;
    for (const auto& cfg : configs)
    {
        render_set_camera_position(cfg.camera_pos);
        render_set_camera_rotation(cfg.camera_rot);
        render_set_light_direction(cfg.light_dir);

        render_set_taa_enabled(false);
        render_update_array(frame0.data(), width, height);
        render_update_array(frame1.data(), width, height);

        delta_no_taa = max_luminance_delta(frame0, frame1);
        if (delta_no_taa > 0.0)
        {
            found = true;
            break;
        }
    }

    if (!found)
    {
        SUCCEED("No temporal noise detected in the current configurations.");
        return;
    }

    render_set_taa_enabled(true);
    render_set_taa_blend(0.2);

    std::vector<uint32_t> scratch(width * height, 0u);
    render_update_array(scratch.data(), width, height);
    render_update_array(scratch.data(), width, height);

    std::vector<uint32_t> frame2(width * height, 0u);
    std::vector<uint32_t> frame3(width * height, 0u);
    render_update_array(frame2.data(), width, height);
    render_update_array(frame3.data(), width, height);

    const double delta_taa = max_luminance_delta(frame2, frame3);
    REQUIRE(delta_taa < delta_no_taa);
}

TEST_CASE("render_update_array fills front face")
{
    reset_camera();
    const size_t width = 120;
    const size_t height = 80;

    std::vector<uint32_t> framebuffer(width * height, 0u);

    render_set_light_direction({0.0, 0.0, 1.0});
    render_set_light_intensity(1.0);
    render_set_paused(true);
    render_update_array(framebuffer.data(), width, height);
    render_set_paused(false);
    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const Vec3 probe{
        start_x + (chunk_size / 2) * block_size,
        base_y,
        start_z
    };
    const Vec2 projected = render_project_point(probe, width, height);
    const int px = std::clamp(static_cast<int>(std::lround(projected.x)), 0, static_cast<int>(width) - 1);
    const int py = std::clamp(static_cast<int>(std::lround(projected.y)), 0, static_cast<int>(height) - 1);
    const uint32_t sky = sky_color_for_row(static_cast<size_t>(py), height, 0xFF78C2FF, 0xFF172433);
    REQUIRE(framebuffer[static_cast<size_t>(py) * width + static_cast<size_t>(px)] != sky);
}

TEST_CASE("render_update_array shows multiple terrain materials")
{
    reset_camera();
    const size_t width = 160;
    const size_t height = 120;

    std::vector<uint32_t> framebuffer(width * height, 0u);

    render_set_paused(true);
    render_set_light_direction({0.0, 1.0, 1.0});
    render_set_light_intensity(0.0);
    render_set_sky_light_intensity(1.0);

    render_update_array(framebuffer.data(), width, height);
    render_set_paused(false);

    const uint32_t sky_top = 0xFF78C2FF;
    const uint32_t sky_bottom = 0xFF172433;
    std::unordered_set<uint32_t> colors;
    for (size_t y = 0; y < height; ++y)
    {
        const uint32_t sky = sky_color_for_row(y, height, sky_top, sky_bottom);
        for (size_t x = 0; x < width; ++x)
        {
            const uint32_t pixel = framebuffer[y * width + x];
            if (pixel != sky)
            {
                colors.insert(pixel);
            }
        }
    }

    REQUIRE(colors.size() >= 3);
}

TEST_CASE("render_update_array applies lighting as multiple shades")
{
    reset_camera();
    const size_t width = 160;
    const size_t height = 120;

    std::vector<uint32_t> framebuffer(width * height, 0u);

    render_set_paused(true);
    render_set_light_direction({0.5, -1.0, 0.7});
    render_set_light_intensity(1.0);
    render_set_sky_light_intensity(1.0);

    render_update_array(framebuffer.data(), width, height);

    const uint32_t sky_top = 0xFF78C2FF;
    const uint32_t sky_bottom = 0xFF172433;
    std::unordered_set<uint32_t> colors;
    for (size_t y = 0; y < height; ++y)
    {
        const uint32_t sky = sky_color_for_row(y, height, sky_top, sky_bottom);
        for (size_t x = 0; x < width; ++x)
        {
            const uint32_t pixel = framebuffer[y * width + x];
            if (pixel != sky)
            {
                colors.insert(pixel);
            }
        }
    }

    render_set_paused(false);

    REQUIRE(colors.size() >= 2);
}

TEST_CASE("render state setters and getters")
{
    reset_camera();

    render_set_light_direction({0.1, 0.2, -0.3});
    const Vec3 dir = render_get_light_direction();
    REQUIRE(dir.x == Catch::Approx(0.1));
    REQUIRE(dir.y == Catch::Approx(0.2));
    REQUIRE(dir.z == Catch::Approx(-0.3));

    render_set_light_intensity(0.42);
    REQUIRE(render_get_light_intensity() == Catch::Approx(0.42));

    render_set_sun_orbit_enabled(true);
    REQUIRE(render_get_sun_orbit_enabled());
    render_set_sun_orbit_enabled(false);
    REQUIRE_FALSE(render_get_sun_orbit_enabled());

    render_set_sun_orbit_angle(1.23);
    REQUIRE(render_get_sun_orbit_angle() == Catch::Approx(1.23));

    render_set_sky_top_color(0xFF112233);
    render_set_sky_bottom_color(0xFF445566);
    REQUIRE(render_get_sky_top_color() == 0xFF112233);
    REQUIRE(render_get_sky_bottom_color() == 0xFF445566);

    render_set_sky_light_intensity(0.77);
    REQUIRE(render_get_sky_light_intensity() == Catch::Approx(0.77));

    render_set_exposure(1.4);
    REQUIRE(render_get_exposure() == Catch::Approx(1.4));

    render_set_taa_enabled(true);
    REQUIRE(render_get_taa_enabled());
    render_set_taa_enabled(false);
    REQUIRE_FALSE(render_get_taa_enabled());

    render_set_taa_blend(0.25);
    REQUIRE(render_get_taa_blend() == Catch::Approx(0.25));

    render_set_paused(false);
    REQUIRE_FALSE(render_is_paused());
    render_toggle_pause();
    REQUIRE(render_is_paused());
    render_toggle_pause();
    REQUIRE_FALSE(render_is_paused());

    render_set_light_direction({0.0, 0.0, -1.0});
    render_set_light_intensity(1.0);
    render_set_sky_top_color(0xFF78C2FF);
    render_set_sky_bottom_color(0xFF172433);
    render_set_sky_light_intensity(0.0);
    render_set_exposure(1.0);
    render_set_taa_enabled(true);
    render_set_taa_blend(0.2);
    render_set_paused(false);
}

TEST_CASE("terrain mesh culls internal faces")
{
    reset_camera();
    const int chunk_size = 16;
    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);

    const size_t total_blocks = count_blocks(heights, chunk_size);
    const size_t expected_faces = count_visible_faces(heights, chunk_size);

    REQUIRE(render_debug_get_terrain_block_count() == total_blocks);
    REQUIRE(render_debug_get_terrain_visible_face_count() == expected_faces);
    REQUIRE(render_debug_get_terrain_triangle_count() < total_blocks * 12);
}

TEST_CASE("render_update_array renders sky gradient")
{
    reset_camera();
    render_set_camera_position({0.0, 25.0, -10.0});
    render_set_camera_rotation({0.0, -0.8});
    render_set_paused(true);
    render_set_light_direction({0.0, 0.0, 1.0});
    render_set_light_intensity(0.0);

    const uint32_t sky_top = 0xFFFF4D4D;
    const uint32_t sky_bottom = 0xFF2A1B70;
    render_set_sky_top_color(sky_top);
    render_set_sky_bottom_color(sky_bottom);
    render_set_sky_light_intensity(0.0);

    const size_t width = 120;
    const size_t height = 80;
    std::vector<uint32_t> framebuffer(width * height, 0u);
    render_update_array(framebuffer.data(), width, height);
    render_set_paused(false);

    const uint32_t top_color = sky_color_for_row(0, height, sky_top, sky_bottom);
    const size_t mid = height / 2;
    const uint32_t mid_color = sky_color_for_row(mid, height, sky_top, sky_bottom);

    bool found_top = false;
    for (size_t x = 0; x < width; ++x)
    {
        if (framebuffer[x] == top_color)
        {
            found_top = true;
            break;
        }
    }
    REQUIRE(found_top);

    bool found_mid = false;
    for (size_t x = 0; x < width; ++x)
    {
        if (framebuffer[mid * width + x] == mid_color)
        {
            found_mid = true;
            break;
        }
    }
    REQUIRE(found_mid);
}

TEST_CASE("sky gradient follows sun altitude when orbit enabled")
{
    reset_camera();
    render_set_camera_position({0.0, 25.0, -10.0});
    render_set_camera_rotation({0.0, -0.8});
    render_set_light_direction({0.0, 0.0, 1.0});
    render_set_light_intensity(1.0);
    render_set_sky_light_intensity(0.0);

    const uint32_t noon_top = 0xFF78C2FF;
    const uint32_t noon_bottom = 0xFF172433;
    render_set_sky_top_color(noon_top);
    render_set_sky_bottom_color(noon_bottom);

    const uint32_t sunrise_top = 0xFFB55A1A;
    const uint32_t sunrise_bottom = 0xFF4A200A;

    render_set_sun_orbit_enabled(true);
    render_set_paused(true);

    const size_t width = 120;
    const size_t height = 80;
    std::vector<uint32_t> sunrise_frame(width * height, 0u);
    std::vector<uint32_t> noon_frame(width * height, 0u);

    render_set_sun_orbit_angle(0.0);
    render_update_array(sunrise_frame.data(), width, height);

    render_set_sun_orbit_angle(1.5707963267948966);
    render_update_array(noon_frame.data(), width, height);

    render_set_paused(false);
    render_set_sun_orbit_enabled(false);

    const uint32_t sunrise_top_row = sky_color_for_row(0, height, sunrise_top, sunrise_bottom);
    const uint32_t noon_top_row = sky_color_for_row(0, height, noon_top, noon_bottom);

    bool found_sunrise = false;
    bool found_noon = false;
    for (size_t x = 0; x < width; ++x)
    {
        if (sunrise_frame[x] == sunrise_top_row)
        {
            found_sunrise = true;
        }
        if (noon_frame[x] == noon_top_row)
        {
            found_noon = true;
        }
    }
    REQUIRE(sunrise_top_row != noon_top_row);
    REQUIRE(found_sunrise);
    REQUIRE(found_noon);
}

TEST_CASE("sky light intensity follows sun altitude")
{
    reset_camera();
    render_set_camera_position({0.0, -12.0, -10.0});
    render_set_camera_rotation({0.0, -0.7});
    render_set_light_intensity(0.0);
    render_set_moon_intensity(0.0);
    render_set_shadow_enabled(false);

    const uint32_t sunrise_top = 0xFFB55A1A;
    const uint32_t sunrise_bottom = 0xFF4A200A;
    render_set_sky_top_color(sunrise_top);
    render_set_sky_bottom_color(sunrise_bottom);
    render_set_sky_light_intensity(1.0);

    render_set_sun_orbit_enabled(true);
    render_set_paused(true);

    const size_t width = 160;
    const size_t height = 120;
    std::vector<uint32_t> sunrise_frame(width * height, 0u);
    std::vector<uint32_t> noon_frame(width * height, 0u);

    render_set_sun_orbit_angle(0.0);
    render_update_array(sunrise_frame.data(), width, height);

    render_set_sun_orbit_angle(1.5707963267948966);
    render_update_array(noon_frame.data(), width, height);

    render_set_paused(false);
    render_set_sun_orbit_enabled(false);

    double sum_sunrise = 0.0;
    double sum_noon = 0.0;
    size_t count = 0;
    for (size_t y = 0; y < height; ++y)
    {
        const uint32_t sky = sky_color_for_row(y, height, sunrise_top, sunrise_bottom);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            if (sunrise_frame[idx] == sky || noon_frame[idx] == sky)
            {
                continue;
            }
            sum_sunrise += static_cast<double>(pixel_luminance(sunrise_frame[idx]));
            sum_noon += static_cast<double>(pixel_luminance(noon_frame[idx]));
            count++;
        }
    }

    REQUIRE(count > 0);
    REQUIRE(sum_noon > sum_sunrise + 50.0);
}

TEST_CASE("low sun altitude keeps ambient above black")
{
    reset_camera();
    render_set_paused(true);
    render_set_sun_orbit_enabled(true);
    render_set_sun_orbit_angle(0.08);
    render_set_light_intensity(0.0);
    render_set_moon_intensity(0.0);
    render_set_shadow_enabled(false);
    render_set_ambient_occlusion_enabled(false);

    const uint32_t sky_color = 0xFFFFFFFF;
    render_set_sky_top_color(sky_color);
    render_set_sky_bottom_color(sky_color);
    render_set_sky_light_intensity(1.0);

    int cell_x = 0;
    int cell_z = 0;
    int cell_height = 0;
    REQUIRE(find_sloped_grass_cell(cell_x, cell_z, cell_height));

    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const double center_x = start_x + cell_x * block_size;
    const double center_z = start_z + cell_z * block_size;
    const double center_y = base_y - (cell_height - 1) * block_size;

    render_set_camera_position({center_x, center_y - 20.0, center_z - 12.0});
    render_set_camera_rotation({0.0, -0.8});

    const size_t width = 200;
    const size_t height = 160;
    std::vector<uint32_t> framebuffer(width * height, 0u);
    render_update_array(framebuffer.data(), width, height);
    render_set_paused(false);
    render_set_sun_orbit_enabled(false);

    const Vec2 projected = render_project_point({center_x, center_y - 1.0, center_z}, width, height);
    const int px = std::clamp(static_cast<int>(std::lround(projected.x)), 0, static_cast<int>(width) - 1);
    const int py = std::clamp(static_cast<int>(std::lround(projected.y)), 0, static_cast<int>(height) - 1);

    double avg_lum = 0.0;
    REQUIRE(sample_average_luminance(framebuffer, width, height, px, py, sky_color, sky_color, avg_lum));
    REQUIRE(avg_lum > 30.0);
}

TEST_CASE("sun light is warmer than moon light")
{
    reset_camera();
    render_set_paused(true);
    render_set_sun_orbit_enabled(false);
    render_set_shadow_enabled(false);
    render_set_ambient_occlusion_enabled(false);
    render_set_sky_light_intensity(0.0);

    const uint32_t sky_color = 0xFF010203;
    render_set_sky_top_color(sky_color);
    render_set_sky_bottom_color(sky_color);

    int cell_x = 0;
    int cell_z = 0;
    int cell_height = 0;
    REQUIRE(find_sloped_grass_cell(cell_x, cell_z, cell_height));

    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const double center_x = start_x + cell_x * block_size;
    const double center_z = start_z + cell_z * block_size;
    const double center_y = base_y - (cell_height - 1) * block_size;

    render_set_camera_position({center_x, center_y - 20.0, center_z - 12.0});
    render_set_camera_rotation({0.0, -0.8});

    const Vec3 sample_point{center_x, center_y - 1.0, center_z};

    const size_t width = 200;
    const size_t height = 160;
    std::vector<uint32_t> sun_frame(width * height, 0u);
    std::vector<uint32_t> moon_frame(width * height, 0u);

    render_set_light_direction({0.0, -1.0, 0.0});
    render_set_light_intensity(1.0);
    render_set_moon_direction({0.0, -1.0, 0.0});
    render_set_moon_intensity(0.0);
    render_update_array(sun_frame.data(), width, height);

    render_set_light_intensity(0.0);
    render_set_moon_intensity(1.0);
    render_update_array(moon_frame.data(), width, height);
    render_set_paused(false);

    const Vec2 projected = render_project_point(sample_point, width, height);
    const int px = std::clamp(static_cast<int>(std::lround(projected.x)), 0, static_cast<int>(width) - 1);
    const int py = std::clamp(static_cast<int>(std::lround(projected.y)), 0, static_cast<int>(height) - 1);

    double r_sun = 0.0;
    double g_sun = 0.0;
    double b_sun = 0.0;
    double r_moon = 0.0;
    double g_moon = 0.0;
    double b_moon = 0.0;
    REQUIRE(sample_average_color(sun_frame, width, height, px, py, sky_color, sky_color, r_sun, g_sun, b_sun));
    REQUIRE(sample_average_color(moon_frame, width, height, px, py, sky_color, sky_color, r_moon, g_moon, b_moon));

    REQUIRE(b_sun > 1.0);
    REQUIRE(b_moon > 1.0);
    const double ratio_sun = r_sun / b_sun;
    const double ratio_moon = r_moon / b_moon;
    REQUIRE(ratio_sun > ratio_moon + 0.05);
}

TEST_CASE("moonlight adds ambient when sun is below horizon")
{
    reset_camera();
    render_set_paused(true);
    render_set_sun_orbit_enabled(true);
    render_set_sun_orbit_angle(0.0);
    render_set_light_intensity(0.0);
    render_set_shadow_enabled(false);
    render_set_ambient_occlusion_enabled(false);
    render_set_sky_light_intensity(1.0);
    render_set_sky_top_color(0xFFFFFFFF);
    render_set_sky_bottom_color(0xFFFFFFFF);
    render_set_moon_direction({0.0, 1.0, 0.0});

    int cell_x = 0;
    int cell_z = 0;
    int cell_height = 0;
    REQUIRE(find_sloped_grass_cell(cell_x, cell_z, cell_height));

    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const double center_x = start_x + cell_x * block_size;
    const double center_z = start_z + cell_z * block_size;
    const double center_y = base_y - (cell_height - 1) * block_size;

    render_set_camera_position({center_x, center_y - 20.0, center_z - 12.0});
    render_set_camera_rotation({0.0, -0.8});

    const size_t width = 200;
    const size_t height = 160;
    std::vector<uint32_t> moon_on(width * height, 0u);
    std::vector<uint32_t> moon_off(width * height, 0u);

    render_set_moon_intensity(0.6);
    render_update_array(moon_on.data(), width, height);
    render_set_moon_intensity(0.0);
    render_update_array(moon_off.data(), width, height);
    render_set_paused(false);
    render_set_sun_orbit_enabled(false);

    const Vec2 projected = render_project_point({center_x, center_y - 1.0, center_z}, width, height);
    const int px = std::clamp(static_cast<int>(std::lround(projected.x)), 0, static_cast<int>(width) - 1);
    const int py = std::clamp(static_cast<int>(std::lround(projected.y)), 0, static_cast<int>(height) - 1);

    double avg_on = 0.0;
    double avg_off = 0.0;
    REQUIRE(sample_average_luminance(moon_on, width, height, px, py, 0xFFFFFFFF, 0xFFFFFFFF, avg_on));
    REQUIRE(sample_average_luminance(moon_off, width, height, px, py, 0xFFFFFFFF, 0xFFFFFFFF, avg_off));
    REQUIRE(avg_on > avg_off + 8.0);
}

TEST_CASE("sky light increases ambient brightness")
{
    reset_camera();
    render_set_paused(true);
    render_set_light_direction({0.0, 0.0, 1.0});
    render_set_light_intensity(0.0);

    const uint32_t sky_top = 0xFF9AD4FF;
    const uint32_t sky_bottom = 0xFF101820;
    render_set_sky_top_color(sky_top);
    render_set_sky_bottom_color(sky_bottom);

    const size_t width = 120;
    const size_t height = 80;
    std::vector<uint32_t> no_sky_light(width * height, 0u);
    std::vector<uint32_t> with_sky_light(width * height, 0u);

    render_set_sky_light_intensity(0.0);
    render_update_array(no_sky_light.data(), width, height);

    render_set_sky_light_intensity(1.0);
    render_update_array(with_sky_light.data(), width, height);
    render_set_paused(false);

    double sum_no = 0.0;
    double sum_with = 0.0;
    size_t count = 0;
    for (size_t y = 0; y < height; ++y)
    {
        const uint32_t sky = sky_color_for_row(y, height, sky_top, sky_bottom);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            if (no_sky_light[idx] == sky)
            {
                continue;
            }
            sum_no += static_cast<double>(pixel_luminance(no_sky_light[idx]));
            sum_with += static_cast<double>(pixel_luminance(with_sky_light[idx]));
            count++;
        }
    }

    REQUIRE(count > 0);
    REQUIRE(sum_with > sum_no);
}

TEST_CASE("gamma correction applies to midtone ambient")
{
    reset_camera();
    render_set_paused(true);
    render_set_sun_orbit_enabled(false);
    render_set_light_intensity(0.0);
    render_set_moon_intensity(0.0);
    render_set_shadow_enabled(false);
    render_set_ambient_occlusion_enabled(false);

    const uint32_t sky_color = 0xFFFFFFFF;
    render_set_sky_top_color(sky_color);
    render_set_sky_bottom_color(sky_color);
    render_set_sky_light_intensity(0.5);

    int cell_x = 0;
    int cell_z = 0;
    int cell_height = 0;
    REQUIRE(find_sloped_grass_cell(cell_x, cell_z, cell_height));

    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const double center_x = start_x + cell_x * block_size;
    const double center_z = start_z + cell_z * block_size;
    const double center_y = base_y - (cell_height - 1) * block_size;

    render_set_camera_position({center_x, center_y - 20.0, center_z - 12.0});
    render_set_camera_rotation({0.0, -0.8});

    const size_t width = 200;
    const size_t height = 160;
    std::vector<uint32_t> framebuffer(width * height, 0u);
    render_update_array(framebuffer.data(), width, height);
    render_set_paused(false);

    const Vec2 projected = render_project_point({center_x, center_y - 1.0, center_z}, width, height);
    const int px = std::clamp(static_cast<int>(std::lround(projected.x)), 0, static_cast<int>(width) - 1);
    const int py = std::clamp(static_cast<int>(std::lround(projected.y)), 0, static_cast<int>(height) - 1);

    double avg_lum = 0.0;
    REQUIRE(sample_average_luminance(framebuffer, width, height, px, py, sky_color, sky_color, avg_lum));

    const uint32_t expected = expected_gamma_luminance(0xFF3B8A38, 0.5f);
    REQUIRE(avg_lum == Catch::Approx(static_cast<double>(expected)).margin(12.0));
}

TEST_CASE("Reinhard tone mapping applies exposure in linear space")
{
    const Vec3 color{2.0, 0.5, 0.0};
    const Vec3 mapped = render_debug_tonemap_reinhard(color, 1.0);
    REQUIRE(mapped.x == Catch::Approx(2.0 / 3.0).margin(1e-6));
    REQUIRE(mapped.y == Catch::Approx(0.5 / 1.5).margin(1e-6));
    REQUIRE(mapped.z == Catch::Approx(0.0).margin(1e-9));

    const Vec3 mapped_boost = render_debug_tonemap_reinhard(color, 2.0);
    REQUIRE(mapped_boost.x == Catch::Approx(4.0 / 5.0).margin(1e-6));
    REQUIRE(mapped_boost.y == Catch::Approx(1.0 / 2.0).margin(1e-6));
    REQUIRE(mapped_boost.x > mapped.x);
    REQUIRE(mapped_boost.y > mapped.y);

    const Vec3 mapped_zero = render_debug_tonemap_reinhard(color, -1.0);
    REQUIRE(mapped_zero.x == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(mapped_zero.y == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(mapped_zero.z == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("hemisphere lighting adds sun bounce to shadowed faces")
{
    reset_camera();
    render_set_paused(true);
    render_set_sun_orbit_enabled(false);
    render_set_light_direction({0.0, -1.0, 0.0});
    render_set_moon_intensity(0.0);
    render_set_shadow_enabled(false);
    render_set_ambient_occlusion_enabled(false);
    render_set_camera_position({0.0, -10.0, -12.0});
    render_set_camera_rotation({0.0, -0.6});

    const uint32_t sky_top = 0xFFFFFFFF;
    const uint32_t sky_bottom = 0xFF202020;
    render_set_sky_top_color(sky_top);
    render_set_sky_bottom_color(sky_bottom);
    render_set_sky_light_intensity(1.0);

    int column_x = 0;
    int column_height = 0;
    REQUIRE(find_front_column(column_x, column_height));

    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const double center_x = start_x + column_x * block_size;
    const double center_y = base_y - (column_height - 1) * block_size;
    const double front_z = start_z - 1.0;
    const Vec3 probe{center_x, center_y, front_z};

    const size_t width = 200;
    const size_t height = 160;
    std::vector<uint32_t> sun_on(width * height, 0u);
    std::vector<uint32_t> sun_off(width * height, 0u);

    render_set_light_intensity(1.0);
    render_update_array(sun_on.data(), width, height);
    render_set_light_intensity(0.0);
    render_update_array(sun_off.data(), width, height);
    render_set_paused(false);

    const Vec2 projected = render_project_point(probe, width, height);
    const int px = std::clamp(static_cast<int>(std::lround(projected.x)), 0, static_cast<int>(width) - 1);
    const int py = std::clamp(static_cast<int>(std::lround(projected.y)), 0, static_cast<int>(height) - 1);

    double avg_on = 0.0;
    double avg_off = 0.0;
    REQUIRE(sample_average_luminance(sun_on, width, height, px, py, sky_top, sky_bottom, avg_on));
    REQUIRE(sample_average_luminance(sun_off, width, height, px, py, sky_top, sky_bottom, avg_off));
    REQUIRE(avg_on > avg_off + 3.0);
}

TEST_CASE("directional shadowing darkens terrain")
{
    reset_camera();
    render_set_paused(true);
    render_set_light_direction({0.6, -0.3, 0.8});
    render_set_light_intensity(1.0);
    render_set_sky_light_intensity(0.0);
    render_set_ambient_occlusion_enabled(false);
    render_set_shadow_enabled(true);

    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);

    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    auto index = [chunk_size](int x, int z) {
        return static_cast<size_t>(z * chunk_size + x);
    };

    bool found_shadow = false;
    for (int z = 0; z < chunk_size && !found_shadow; ++z)
    {
        for (int x = 0; x < chunk_size && !found_shadow; ++x)
        {
            const int height = heights[index(x, z)];
            const double center_y = base_y - (height - 1) * block_size;
            const double top_y = center_y - block_size * 0.5;
            const Vec3 world{
                start_x + x * block_size,
                top_y,
                start_z + z * block_size
            };
            float factor = 1.0f;
            if (render_get_shadow_factor_at_point(world, {0.0, -1.0, 0.0}, &factor) && factor < 0.98f)
            {
                found_shadow = true;
            }
        }
    }

    render_set_paused(false);

    REQUIRE(found_shadow);
}

TEST_CASE("light direction reads stay coherent under concurrent updates")
{
    reset_camera();
    render_set_sun_orbit_enabled(false);

    const Vec3 a{1.0, 2.0, 3.0};
    const Vec3 b{-4.0, -5.0, -6.0};

    std::atomic<bool> stop{false};
    std::atomic<bool> saw_mixed{false};
    std::atomic<bool> ready{false};

    render_set_light_direction(a);

    std::thread writer([&] {
        for (int i = 0; i < 2000000 && !saw_mixed.load(std::memory_order_relaxed); ++i)
        {
            render_set_light_direction(a);
            render_set_light_direction(b);
            if (i == 0)
            {
                ready.store(true, std::memory_order_release);
            }
        }
        stop.store(true, std::memory_order_relaxed);
    });

    std::thread reader([&] {
        while (!ready.load(std::memory_order_acquire))
        {
        }
        while (!stop.load(std::memory_order_relaxed))
        {
            const Vec3 v = render_get_light_direction();
            const bool is_a = (v.x == a.x && v.y == a.y && v.z == a.z);
            const bool is_b = (v.x == b.x && v.y == b.y && v.z == b.z);
            if (!is_a && !is_b)
            {
                saw_mixed.store(true, std::memory_order_relaxed);
                break;
            }
        }
    });

    writer.join();
    stop.store(true, std::memory_order_relaxed);
    reader.join();

    REQUIRE(!saw_mixed.load(std::memory_order_relaxed));
}

TEST_CASE("stochastic DDA varies across frames near shadow boundaries")
{
    reset_camera();
    render_set_paused(true);
    render_set_sun_orbit_enabled(false);
    render_set_light_direction({0.6, -0.2, 0.7});
    render_set_light_intensity(1.0);
    render_set_moon_intensity(0.0);
    render_set_shadow_enabled(true);

    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);

    const Vec3 normal{0.0, -1.0, 0.0};
    const Vec3 light_dir = normalize_vec3({0.6, -0.2, 0.7});

    bool saw_variation = false;
    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    auto index = [chunk_size](int x, int z) {
        return static_cast<size_t>(z * chunk_size + x);
    };

    const std::array<double, 3> offsets = {-0.35, 0.0, 0.35};
    for (int z = 0; z < chunk_size && !saw_variation; z += 2)
    {
        for (int x = 0; x < chunk_size && !saw_variation; x += 2)
        {
            const int height = heights[index(x, z)];
            if (height <= 0)
            {
                continue;
            }
            const double center_y = base_y - (height - 1) * block_size;
            const double top_y = center_y - block_size * 0.5;
            for (double ox : offsets)
            {
                for (double oz : offsets)
                {
                    const Vec3 world{
                        start_x + x * block_size + ox * block_size,
                        top_y,
                        start_z + z * block_size + oz * block_size
                    };
                    float min_factor = 1.0f;
                    float max_factor = 0.0f;
                    for (int frame = 0; frame < 24; ++frame)
                    {
                        float sample = 1.0f;
                        if (!render_debug_shadow_factor_with_frame(world, normal, light_dir,
                                                                   x, z, frame, &sample))
                        {
                            continue;
                        }
                        min_factor = std::min(min_factor, sample);
                        max_factor = std::max(max_factor, sample);
                        if (min_factor < 0.05f && max_factor > 0.95f)
                        {
                            saw_variation = true;
                            break;
                        }
                    }
                    if (saw_variation) break;
                }
                if (saw_variation) break;
            }
        }
    }

    render_set_paused(false);

    REQUIRE(saw_variation);
}

TEST_CASE("moon light contributes when sun is disabled")
{
    reset_camera();
    render_set_paused(true);
    render_set_light_direction({0.0, 1.0, 0.0});
    render_set_light_intensity(0.0);
    render_set_moon_direction({-0.4, 1.0, 0.3});
    render_set_moon_intensity(0.8);
    render_set_sky_light_intensity(0.0);
    render_set_shadow_enabled(false);

    const size_t width = 160;
    const size_t height = 120;
    std::vector<uint32_t> moon_on(width * height, 0u);
    std::vector<uint32_t> moon_off(width * height, 0u);

    render_update_array(moon_on.data(), width, height);

    render_set_moon_intensity(0.0);
    render_update_array(moon_off.data(), width, height);
    render_set_paused(false);

    double sum_on = 0.0;
    double sum_off = 0.0;
    size_t count = 0;
    const uint32_t sky_top = render_get_sky_top_color();
    const uint32_t sky_bottom = render_get_sky_bottom_color();
    for (size_t y = 0; y < height; ++y)
    {
        const uint32_t sky = sky_color_for_row(y, height, sky_top, sky_bottom);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            if (moon_on[idx] == sky || moon_off[idx] == sky)
            {
                continue;
            }
            sum_on += static_cast<double>(pixel_luminance(moon_on[idx]));
            sum_off += static_cast<double>(pixel_luminance(moon_off[idx]));
            count++;
        }
    }

    REQUIRE(count > 0);
    REQUIRE(sum_on > sum_off);
}

TEST_CASE("phong shading varies within a terrain top face")
{
    reset_camera();
    render_set_paused(true);
    render_set_light_intensity(0.0);
    render_set_sky_light_intensity(1.0);

    const uint32_t sky_top = 0xFFFFFFFF;
    const uint32_t sky_bottom = 0xFF000000;
    render_set_sky_top_color(sky_top);
    render_set_sky_bottom_color(sky_bottom);

    int cell_x = 0;
    int cell_z = 0;
    int cell_height = 0;
    REQUIRE(find_sloped_grass_cell(cell_x, cell_z, cell_height));

    const int chunk_size = 16;
    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const double center_x = start_x + cell_x * block_size;
    const double center_z = start_z + cell_z * block_size;
    const double center_y = base_y - (cell_height - 1) * block_size;

    render_set_camera_position({center_x, center_y - 20.0, center_z - 12.0});
    render_set_camera_rotation({0.0, -0.8});

    const size_t width = 200;
    const size_t height = 160;
    std::vector<uint32_t> framebuffer(width * height, 0u);
    render_update_array(framebuffer.data(), width, height);
    render_set_paused(false);

    const Vec2 projected = render_project_point({center_x, center_y - 1.0, center_z}, width, height);
    const int px = std::clamp(static_cast<int>(std::lround(projected.x)), 0, static_cast<int>(width) - 1);
    const int py = std::clamp(static_cast<int>(std::lround(projected.y)), 0, static_cast<int>(height) - 1);

    const int radius = 3;
    uint32_t base_color = 0;
    bool found_base = false;
    for (int dy = -radius; dy <= radius && !found_base; ++dy)
    {
        const int y = std::clamp(py + dy, 0, static_cast<int>(height) - 1);
        const uint32_t sky = sky_color_for_row(static_cast<size_t>(y), height, sky_top, sky_bottom);
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = std::clamp(px + dx, 0, static_cast<int>(width) - 1);
            const uint32_t pixel = framebuffer[static_cast<size_t>(y) * width + static_cast<size_t>(x)];
            if (pixel != sky)
            {
                base_color = pixel;
                found_base = true;
                break;
            }
        }
    }
    REQUIRE(found_base);

    std::vector<uint32_t> samples;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int y = std::clamp(py + dy, 0, static_cast<int>(height) - 1);
        const uint32_t sky = sky_color_for_row(static_cast<size_t>(y), height, sky_top, sky_bottom);
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = std::clamp(px + dx, 0, static_cast<int>(width) - 1);
            const uint32_t pixel = framebuffer[static_cast<size_t>(y) * width + static_cast<size_t>(x)];
            if (pixel == sky)
            {
                continue;
            }
            if (colors_similar(pixel, base_color, 30))
            {
                samples.push_back(pixel_luminance(pixel));
            }
        }
    }

    REQUIRE(samples.size() >= 6);
    const auto minmax = std::minmax_element(samples.begin(), samples.end());
    REQUIRE((*minmax.second - *minmax.first) >= 8);
}

TEST_CASE("sky lighting is consistent across a terrain side face")
{
    reset_camera();
    render_set_paused(true);
    render_set_light_intensity(0.0);
    render_set_shadow_enabled(false);
    render_set_ambient_occlusion_enabled(false);
    render_set_sky_light_intensity(1.0);

    const uint32_t sky_top = 0xFFFFFFFF;
    const uint32_t sky_bottom = 0xFF000000;
    render_set_sky_top_color(sky_top);
    render_set_sky_bottom_color(sky_bottom);

    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);
    constexpr int chunk_size = 16;
    const int cell_x = 0;
    const int cell_z = 0;
    const int cell_height = heights[static_cast<size_t>(cell_z * chunk_size + cell_x)];
    REQUIRE(cell_height > 0);

    const double block_size = 2.0;
    const double half = block_size * 0.5;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const double center_x = start_x + cell_x * block_size;
    const double center_z = start_z + cell_z * block_size;
    const double center_y = base_y - (cell_height - 1) * block_size;
    const double face_x = center_x - half;

    constexpr double half_pi = 1.5707963267948966;
    render_set_camera_position({face_x - 8.0, center_y, center_z});
    render_set_camera_rotation({half_pi, 0.0});

    const size_t width = 200;
    const size_t height = 160;
    std::vector<uint32_t> framebuffer(width * height, 0u);
    render_update_array(framebuffer.data(), width, height);
    render_set_paused(false);

    const double inset = half * 0.7;
    const Vec2 top_proj = render_project_point({face_x, center_y - inset, center_z}, width, height);
    const Vec2 bottom_proj = render_project_point({face_x, center_y + inset, center_z}, width, height);

    REQUIRE(std::isfinite(top_proj.x));
    REQUIRE(std::isfinite(top_proj.y));
    REQUIRE(std::isfinite(bottom_proj.x));
    REQUIRE(std::isfinite(bottom_proj.y));

    const int top_x = std::clamp(static_cast<int>(std::lround(top_proj.x)), 0, static_cast<int>(width) - 1);
    const int top_y = std::clamp(static_cast<int>(std::lround(top_proj.y)), 0, static_cast<int>(height) - 1);
    const int bottom_x = std::clamp(static_cast<int>(std::lround(bottom_proj.x)), 0, static_cast<int>(width) - 1);
    const int bottom_y = std::clamp(static_cast<int>(std::lround(bottom_proj.y)), 0, static_cast<int>(height) - 1);

    double top_lum = 0.0;
    double bottom_lum = 0.0;
    REQUIRE(sample_average_luminance(framebuffer, width, height, top_x, top_y, sky_top, sky_bottom, top_lum));
    REQUIRE(sample_average_luminance(framebuffer, width, height, bottom_x, bottom_y, sky_top, sky_bottom, bottom_lum));
    REQUIRE(std::abs(top_lum - bottom_lum) <= 6.0);
}

TEST_CASE("normalized Blinn-Phong specular uses Schlick Fresnel")
{
    const double shininess = 24.0;
    const double ndoth = 0.8;
    const double ndotl = 0.75;
    const double f0 = 0.2;

    const double spec_normal = render_debug_eval_specular(ndoth, 1.0, ndotl, shininess, f0);
    const double expected = ((shininess + 8.0) / (8.0 * kPi)) * std::pow(ndoth, shininess) * f0 * ndotl;
    REQUIRE(spec_normal == Catch::Approx(expected).margin(1e-6));

    const double spec_grazing = render_debug_eval_specular(ndoth, 0.2, ndotl, shininess, f0);
    REQUIRE(spec_grazing > spec_normal);

    const double spec_low_ndotl = render_debug_eval_specular(ndoth, 1.0, 0.2, shininess, f0);
    REQUIRE(spec_low_ndotl < spec_normal);

    REQUIRE(render_debug_eval_specular(ndoth, 0.2, ndotl, shininess, 0.0) == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("sky visibility darkens terrain when enabled")
{
    reset_camera();
    render_set_paused(true);
    render_set_light_intensity(0.0);
    render_set_sky_light_intensity(1.0);
    render_set_sky_top_color(0xFFBBD7FF);
    render_set_sky_bottom_color(0xFF1A2430);
    render_set_camera_position({0.0, -10.0, -12.0});
    render_set_camera_rotation({0.0, -0.6});

    const size_t width = 160;
    const size_t height = 120;
    std::vector<uint32_t> no_ao(width * height, 0u);
    std::vector<uint32_t> with_ao(width * height, 0u);

    render_set_ambient_occlusion_enabled(false);
    render_update_array(no_ao.data(), width, height);
    render_set_ambient_occlusion_enabled(true);
    render_update_array(with_ao.data(), width, height);
    render_set_paused(false);

    double sum_no = 0.0;
    double sum_with = 0.0;
    size_t count = 0;
    for (size_t y = 0; y < height; ++y)
    {
        const uint32_t sky = sky_color_for_row(y, height, 0xFFBBD7FF, 0xFF1A2430);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            if (no_ao[idx] == sky)
            {
                continue;
            }
            sum_no += static_cast<double>(pixel_luminance(no_ao[idx]));
            sum_with += static_cast<double>(pixel_luminance(with_ao[idx]));
            count++;
        }
    }

    REQUIRE(count > 0);
    const double avg_no = sum_no / static_cast<double>(count);
    const double avg_with = sum_with / static_cast<double>(count);
    REQUIRE(avg_with < avg_no);
    REQUIRE(avg_no - avg_with > 0.3);
}

TEST_CASE("side face sky visibility responds to diagonal neighbor blocks")
{
    reset_camera();
    render_set_paused(true);

    SideFaceAoProbe probe{};
    REQUIRE(find_right_face_diagonal_occluder(probe));

    constexpr int kFaceRight = 3;
    constexpr int kCornerTopFront = 3;
    float visibility = 1.0f;
    REQUIRE(render_get_terrain_vertex_sky_visibility(probe.x, probe.height - 1, probe.z,
                                                     kFaceRight, kCornerTopFront, &visibility));

    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);
    const float expected = raycast_vertex_sky_visibility(heights, 16, probe.x, probe.height - 1, probe.z,
                                                         kFaceRight, kCornerTopFront);

    REQUIRE(visibility == Catch::Approx(expected).margin(0.02f));
    render_set_paused(false);
}

TEST_CASE("side face sky visibility captures off-axis occluders")
{
    reset_camera();
    render_set_paused(true);

    SideFaceAoProbe probe{};
    REQUIRE(find_right_face_diagonal_occluder(probe));

    constexpr int kFaceRight = 3;
    constexpr int kCornerTopFront = 3;
    float visibility = 1.0f;
    REQUIRE(render_get_terrain_vertex_sky_visibility(probe.x, probe.height - 1, probe.z,
                                                     kFaceRight, kCornerTopFront, &visibility));
    REQUIRE(visibility < 0.98f);

    render_set_paused(false);
}

TEST_CASE("flat terrain top faces match sky visibility raycast")
{
    reset_camera();

    FlatCornerProbe probe{};
    REQUIRE(find_flat_shared_corner(probe));

    const int y = probe.height - 1;
    float visibility = 1.0f;
    const int corner = top_face_corner_from_offsets(probe.sx, probe.sz);
    REQUIRE(render_get_terrain_vertex_sky_visibility(probe.x, y, probe.z, kFaceTop, corner, &visibility));

    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);
    const float expected = raycast_vertex_sky_visibility(heights, 16, probe.x, y, probe.z,
                                                         kFaceTop, corner);

    REQUIRE(visibility == Catch::Approx(expected).margin(0.02f));
}

TEST_CASE("vertex sky visibility varies within some top faces")
{
    reset_camera();
    render_set_paused(true);

    constexpr int chunk_size = 16;
    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);

    bool found = false;
    float visibility = 1.0f;
    for (int z = 0; z < chunk_size && !found; ++z)
    {
        for (int x = 0; x < chunk_size && !found; ++x)
        {
            const int height = heights[static_cast<size_t>(z * chunk_size + x)];
            const int y = height - 1;
            if (y < 0)
            {
                continue;
            }
            float min_v = 1.0f;
            float max_v = 0.0f;
            for (int corner = 0; corner < 4; ++corner)
            {
                if (!render_get_terrain_vertex_sky_visibility(x, y, z, kFaceTop, corner, &visibility))
                {
                    min_v = 1.0f;
                    max_v = 0.0f;
                    break;
                }
                min_v = std::min(min_v, visibility);
                max_v = std::max(max_v, visibility);
            }
            if (max_v - min_v > 0.02f)
            {
                found = true;
            }
        }
    }

    render_set_paused(false);

    REQUIRE(found);
}

TEST_CASE("top face sky visibility varies beyond four discrete levels")
{
    reset_camera();
    render_set_paused(true);

    constexpr int chunk_size = 16;
    std::unordered_set<int> buckets;
    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);
    float visibility = 1.0f;

    for (int z = 0; z < chunk_size; ++z)
    {
        for (int x = 0; x < chunk_size; ++x)
        {
            const int height = heights[static_cast<size_t>(z * chunk_size + x)];
            const int y = height - 1;
            for (int corner = 0; corner < 4; ++corner)
            {
                if (!render_get_terrain_vertex_sky_visibility(x, y, z, kFaceTop, corner, &visibility))
                {
                    continue;
                }
                const int bucket = static_cast<int>(std::lround(visibility * 1000.0f));
                buckets.insert(bucket);
            }
        }
    }

    render_set_paused(false);

    REQUIRE(buckets.size() > 4);
}

TEST_CASE("side face sky visibility varies beyond four discrete levels")
{
    reset_camera();
    render_set_paused(true);

    constexpr int chunk_size = 16;
    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);
    const int max_height = *std::max_element(heights.begin(), heights.end());

    std::unordered_set<int> buckets;
    float visibility = 1.0f;
    constexpr int kFaceRight = 3;
    constexpr int kFaceFront = 5;

    for (int z = 0; z < chunk_size; ++z)
    {
        for (int x = 0; x < chunk_size; ++x)
        {
            for (int y = 0; y < max_height; ++y)
            {
                if (!render_get_terrain_vertex_sky_visibility(x, y, z, kFaceRight, 0, &visibility))
                {
                    break;
                }
                const int bucket_right0 = static_cast<int>(std::lround(visibility * 1000.0f));
                buckets.insert(bucket_right0);
                for (int corner = 1; corner < 4; ++corner)
                {
                    if (render_get_terrain_vertex_sky_visibility(x, y, z, kFaceRight, corner, &visibility))
                    {
                        const int bucket_right = static_cast<int>(std::lround(visibility * 1000.0f));
                        buckets.insert(bucket_right);
                    }
                }
                if (!render_get_terrain_vertex_sky_visibility(x, y, z, kFaceFront, 0, &visibility))
                {
                    break;
                }
                const int bucket_front0 = static_cast<int>(std::lround(visibility * 1000.0f));
                buckets.insert(bucket_front0);
                for (int corner = 1; corner < 4; ++corner)
                {
                    if (render_get_terrain_vertex_sky_visibility(x, y, z, kFaceFront, corner, &visibility))
                    {
                        const int bucket_front = static_cast<int>(std::lround(visibility * 1000.0f));
                        buckets.insert(bucket_front);
                    }
                }
            }
        }
    }

    render_set_paused(false);

    REQUIRE(buckets.size() > 4);
}

TEST_CASE("camera state setters and getters")
{
    reset_camera();
    render_set_camera_position({1.0, -2.0, 3.5});
    const Vec3 pos = render_get_camera_position();
    REQUIRE(pos.x == Catch::Approx(1.0));
    REQUIRE(pos.y == Catch::Approx(-2.0));
    REQUIRE(pos.z == Catch::Approx(3.5));

    render_set_camera_rotation({0.25, -0.5});
    const Vec2 rot = render_get_camera_rotation();
    REQUIRE(rot.x == Catch::Approx(0.25));
    REQUIRE(rot.y == Catch::Approx(-0.5));
}

TEST_CASE("camera movement in world and local space")
{
    reset_camera();
    render_set_camera_position({0.0, -20.0, -20.0});
    render_set_camera_rotation({0.0, 0.0});
    render_move_camera({1.0, -2.0, 3.0});
    Vec3 pos = render_get_camera_position();
    REQUIRE(pos.x == Catch::Approx(1.0));
    REQUIRE(pos.y == Catch::Approx(-22.0));
    REQUIRE(pos.z == Catch::Approx(-17.0));

    render_move_camera_local({0.0, 0.0, 1.0});
    pos = render_get_camera_position();
    REQUIRE(pos.x == Catch::Approx(1.0));
    REQUIRE(pos.y == Catch::Approx(-22.0));
    REQUIRE(pos.z == Catch::Approx(-16.0));

    constexpr double half_pi = 1.5707963267948966;
    render_set_camera_position({0.0, -20.0, -20.0});
    render_set_camera_rotation({half_pi, 0.0});
    render_move_camera_local({0.0, 0.0, 1.0});
    pos = render_get_camera_position();
    REQUIRE(pos.x == Catch::Approx(1.0));
    REQUIRE(pos.z == Catch::Approx(-20.0).margin(1e-6));
}

TEST_CASE("reprojection matrices stay stable without camera motion")
{
    reset_camera();
    render_set_taa_enabled(true);
    const size_t width = 96;
    const size_t height = 72;
    std::vector<uint32_t> framebuffer(width * height);

    render_update_array(framebuffer.data(), width, height);
    render_update_array(framebuffer.data(), width, height);

    const Mat4 prev = render_debug_get_previous_vp();
    const Mat4 curr = render_debug_get_current_vp();
    const Mat4 inv = render_debug_get_inverse_current_vp();
    const Mat4 product = mat4_multiply(curr, inv);

    INFO("PreviousVP:\n" << mat4_to_string(prev));
    INFO("CurrentVP:\n" << mat4_to_string(curr));
    INFO("CurrentVP * Inverse(CurrentVP):\n" << mat4_to_string(product));

    REQUIRE(mat4_near_equal(prev, curr, 1e-12));
    REQUIRE(mat4_is_identity(product, 1e-7));
}

TEST_CASE("reprojection matrices change when camera moves")
{
    reset_camera();
    render_set_taa_enabled(true);
    const size_t width = 96;
    const size_t height = 72;
    std::vector<uint32_t> framebuffer(width * height);

    render_update_array(framebuffer.data(), width, height);
    Vec3 pos = render_get_camera_position();
    render_set_camera_position({pos.x + 1.0, pos.y, pos.z});
    render_update_array(framebuffer.data(), width, height);

    const Mat4 prev = render_debug_get_previous_vp();
    const Mat4 curr = render_debug_get_current_vp();
    INFO("PreviousVP:\n" << mat4_to_string(prev));
    INFO("CurrentVP:\n" << mat4_to_string(curr));
    REQUIRE_FALSE(mat4_near_equal(prev, curr, 1e-9));
}

TEST_CASE("camera movement blocks entry into terrain")
{
    reset_camera();
    render_set_paused(true);

    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);
    const int cell_x = 0;
    const int cell_z = 0;
    const int cell_height = heights[static_cast<size_t>(cell_z * 16 + cell_x)];
    REQUIRE(cell_height > 0);

    const double block_size = 2.0;
    const double half = block_size * 0.5;
    const double start_x = -(16 - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const double block_x = start_x + cell_x * block_size;
    const double block_z = start_z + cell_z * block_size;
    const double block_y = base_y - (cell_height - 1) * block_size;

    const Vec3 start{block_x - half - 0.2, block_y, block_z};
    render_set_camera_position(start);
    render_move_camera({0.3, 0.0, 0.0});
    Vec3 pos = render_get_camera_position();

    const bool inside = std::abs(pos.x - block_x) < half &&
                        std::abs(pos.y - block_y) < half &&
                        std::abs(pos.z - block_z) < half;
    REQUIRE_FALSE(inside);

    render_set_camera_position(start);
    render_move_camera({-0.3, 0.0, 0.0});
    pos = render_get_camera_position();
    REQUIRE(pos.x == Catch::Approx(start.x - 0.3));
    render_set_paused(false);
}

TEST_CASE("camera forward movement does not climb when blocked by top face")
{
    reset_camera();
    render_set_paused(true);

    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    build_heightmap(heights, top_colors);

    const int cell_x = 0;
    const int cell_z = 0;
    const int cell_height = heights[static_cast<size_t>(cell_z * 16 + cell_x)];
    REQUIRE(cell_height > 0);

    const double block_size = 2.0;
    const double half = block_size * 0.5;
    const double start_x = -(16 - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const double block_x = start_x + cell_x * block_size;
    const double block_z = start_z + cell_z * block_size;
    const double block_y = base_y - (cell_height - 1) * block_size;
    const double top_face = block_y - half;

    render_set_camera_position({block_x, top_face - 0.2, block_z - 1.5});
    render_set_camera_rotation({0.0, -1.2});

    Vec3 pos = render_get_camera_position();
    double last_y = pos.y;
    for (int i = 0; i < 20; ++i)
    {
        render_move_camera_local({0.0, 0.0, 0.3});
        pos = render_get_camera_position();
        REQUIRE(pos.y + 1e-6 >= last_y);
        last_y = pos.y;
    }

    render_set_paused(false);
}

TEST_CASE("render_unproject_point reconstructs world position")
{
    reset_camera();
    const size_t width = 160;
    const size_t height = 120;
    const Vec3 camera_pos{2.0, -1.5, -3.0};
    const Vec2 camera_rot{0.35, -0.2};
    render_set_camera_position(camera_pos);
    render_set_camera_rotation(camera_rot);

    const Vec3 target{10.0, 0.0, 5.0};
    const Vec2 projected = render_project_point(target, width, height);
    REQUIRE_FALSE(std::isnan(projected.x));
    REQUIRE_FALSE(std::isnan(projected.y));

    const Vec3 view = rotate_yaw_pitch({target.x - camera_pos.x,
                                        target.y - camera_pos.y,
                                        target.z - camera_pos.z},
                                       -camera_rot.x, -camera_rot.y);
    REQUIRE(view.z > 0.0);

    const Vec3 reconstructed = render_unproject_point({projected.x, projected.y, view.z},
                                                      width, height);
    REQUIRE(reconstructed.x == Catch::Approx(target.x).margin(1e-6));
    REQUIRE(reconstructed.y == Catch::Approx(target.y).margin(1e-6));
    REQUIRE(reconstructed.z == Catch::Approx(target.z).margin(1e-6));
}

TEST_CASE("render_reproject_point maps world position into previous frame")
{
    reset_camera();
    render_set_camera_rotation({0.0, 0.0});
    const size_t width = 160;
    const size_t height = 120;
    std::vector<uint32_t> framebuffer(width * height);

    render_set_camera_position({0.0, 0.0, 0.0});
    render_update_array(framebuffer.data(), width, height);

    render_set_camera_position({-10.0, 0.0, 0.0});
    render_update_array(framebuffer.data(), width, height);

    const Vec3 point{0.0, 0.0, 100.0};
    const Vec2 current = render_project_point(point, width, height);
    REQUIRE(current.x > width / 2.0);

    const Vec3 camera_pos = render_get_camera_position();
    const double view_z = point.z - camera_pos.z;
    REQUIRE(view_z > 0.0);

    const Vec3 reconstructed = render_unproject_point({current.x, current.y, view_z}, width, height);
    REQUIRE(reconstructed.x == Catch::Approx(point.x).margin(1e-6));
    REQUIRE(reconstructed.y == Catch::Approx(point.y).margin(1e-6));
    REQUIRE(reconstructed.z == Catch::Approx(point.z).margin(1e-6));

    const Vec2 prev = render_reproject_point(reconstructed, width, height);
    REQUIRE(prev.x == Catch::Approx(width / 2.0).margin(1e-6));
    REQUIRE(prev.y == Catch::Approx(height / 2.0).margin(1e-6));

    const double fov_x = static_cast<double>(height) * 0.8;
    const double expected_delta = (point.x - camera_pos.x) / view_z * fov_x;
    REQUIRE((current.x - prev.x) == Catch::Approx(expected_delta).margin(1e-6));
}

TEST_CASE("history bilinear sampling blends across top row midpoint")
{
    const size_t width = 2;
    const size_t height = 2;
    const std::array<Vec3, 4> history = {{
        {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}
    }};
    const Vec3 sample = render_debug_sample_history_bilinear(history.data(), width, height, {1.0, 0.5});
    REQUIRE(sample.x == Catch::Approx(0.5));
    REQUIRE(sample.y == Catch::Approx(0.5));
    REQUIRE(sample.z == Catch::Approx(0.5));
}

TEST_CASE("history bilinear sampling uses fractional weights")
{
    const size_t width = 2;
    const size_t height = 2;
    const std::array<Vec3, 4> history = {{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}, {1.0, 1.0, 0.0}
    }};
    const Vec3 sample = render_debug_sample_history_bilinear(history.data(), width, height, {1.25, 0.75});
    REQUIRE(sample.x == Catch::Approx(0.375));
    REQUIRE(sample.y == Catch::Approx(0.75));
    REQUIRE(sample.z == Catch::Approx(0.0625));
}

TEST_CASE("render_project_point centers consistently")
{
    reset_camera();
    const size_t width = 100;
    const size_t height = 80;

    const Vec2 center = render_project_point({0.0, 0.0, 0.0}, width, height);
    REQUIRE(center.x == Catch::Approx(width / 2.0));
    REQUIRE(center.y == Catch::Approx(height / 2.0));

    const Vec2 square = render_project_point({1.0, 0.0, 0.0}, 80, 80);
    const Vec2 wide = render_project_point({1.0, 0.0, 0.0}, 200, 80);
    const double square_offset = square.x - 40.0;
    const double wide_offset = wide.x - 100.0;
    REQUIRE(wide_offset == Catch::Approx(square_offset));
}

TEST_CASE("render_project_point returns NaN for points behind camera")
{
    reset_camera();
    render_set_camera_position({0.0, 0.0, 0.0});
    render_set_camera_rotation({0.0, 0.0});
    const size_t width = 120;
    const size_t height = 90;

    const Vec2 projected = render_project_point({0.0, 0.0, -1.0}, width, height);
    REQUIRE(std::isnan(projected.x));
    REQUIRE(std::isnan(projected.y));
}

TEST_CASE("render_debug_depth_at_sample uses perspective-correct interpolation")
{
    const Vec3 v0{10.0, 10.0, 1.0};
    const Vec3 v1{30.0, 10.0, 4.0};
    const Vec3 v2{10.0, 30.0, 8.0};
    const Vec2 p{15.0, 15.0};

    float depth = 0.0f;
    REQUIRE(render_debug_depth_at_sample(v0, v1, v2, p, &depth));

    auto edge = [](const Vec3& a, const Vec3& b, const Vec2& c) {
        return static_cast<float>((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x));
    };

    const float area = edge(v0, v1, {v2.x, v2.y});
    REQUIRE(area != 0.0f);
    float w0 = edge(v1, v2, p) / area;
    float w1 = edge(v2, v0, p) / area;
    float w2 = edge(v0, v1, p) / area;

    const float inv_z = w0 * (1.0f / static_cast<float>(v0.z)) +
                        w1 * (1.0f / static_cast<float>(v1.z)) +
                        w2 * (1.0f / static_cast<float>(v2.z));
    const float expected = 1.0f / inv_z;

    REQUIRE(depth == Catch::Approx(expected).margin(1e-4f));
}

TEST_CASE("render_should_rasterize_triangle allows near-plane clipping")
{
    const Vec3 in_front{0.0, 0.0, 0.2};
    const Vec3 behind{0.0, 0.0, 0.01};

    REQUIRE(render_should_rasterize_triangle(in_front, in_front, in_front));
    REQUIRE(render_should_rasterize_triangle(behind, in_front, in_front));
    REQUIRE_FALSE(render_should_rasterize_triangle(behind, behind, behind));
}

TEST_CASE("render_clip_triangle_to_near_plane clips partially occluded triangles")
{
    const double near_plane = render_get_near_plane();
    const Vec3 in_front{0.0, 0.0, near_plane + 0.1};
    const Vec3 behind{0.0, 0.0, near_plane * 0.5};

    Vec3 clipped[4]{};
    const size_t count = render_clip_triangle_to_near_plane(behind, in_front, in_front, clipped, 4);
    REQUIRE(count == 4);
    for (size_t i = 0; i < count; ++i)
    {
        REQUIRE(clipped[i].z >= Catch::Approx(near_plane).margin(1e-6));
    }

    const size_t count_inside = render_clip_triangle_to_near_plane(in_front, in_front, in_front, clipped, 4);
    REQUIRE(count_inside == 3);

    const size_t count_outside = render_clip_triangle_to_near_plane(behind, behind, behind, clipped, 4);
    REQUIRE(count_outside == 0);
}

TEST_CASE("render_project_point responds to yaw rotation")
{
    reset_camera();
    const size_t width = 120;
    const size_t height = 90;
    render_set_camera_position({0.0, 0.0, 0.0});

    const Vec3 point{10.0, 0.0, 10.0};

    render_set_camera_rotation({0.0, 0.0});
    const Vec2 no_yaw = render_project_point(point, width, height);
    REQUIRE(no_yaw.x > width / 2.0);

    const double quarter_pi = 0.7853981633974483;
    render_set_camera_rotation({quarter_pi, 0.0});
    const Vec2 rotated = render_project_point(point, width, height);
    REQUIRE(rotated.x == Catch::Approx(width / 2.0).margin(1e-6));
}
