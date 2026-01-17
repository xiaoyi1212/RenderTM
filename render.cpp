#include "render.h"
#include "cmath"
#include "cstring"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

const Vec3 cubeVertices[8] = {
    {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
    {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}
};

const int cubeTriangles[12][3] = {
    {4,5,6}, {4,6,7}, // front (+z)
    {0,2,1}, {0,3,2}, // back (-z)
    {0,7,3}, {0,4,7}, // left (-x)
    {1,2,6}, {1,6,5}, // right (+x)
    {3,6,2}, {3,7,6}, // top (+y)
    {0,1,5}, {0,5,4}  // bottom (-y)
};

static float rotationAngle = 0.0f;
static std::atomic<bool> rotationPaused{false};
static RenderScene activeScene = RenderScene::CubeOnly;
static Vec3 lightDirection{0.0, 0.0, -1.0};
static double lightIntensity = 1.0;
static std::atomic<bool> sunOrbitEnabled{true};
static std::atomic<double> sunOrbitAngle{1.5707963267948966};
static std::atomic<double> sunOrbitSpeed{0.0015};
static Vec3 moonDirection{0.0, 1.0, 0.0};
static double moonIntensity = 0.0;
static uint32_t skyTopColor = 0xFF78C2FF;
static uint32_t skyBottomColor = 0xFF172433;
static double skyLightIntensity = 0.35;
static double ambientLight = 0.15;
static std::atomic<double> camera_x{0.0};
static std::atomic<double> camera_y{-10.0};
static std::atomic<double> camera_z{-12.0};
static std::atomic<double> camera_yaw{0.0};
static std::atomic<double> camera_pitch{-0.6};
static std::atomic<bool> ambientOcclusionEnabled{true};
static std::atomic<bool> shadowEnabled{true};

struct ScreenVertex
{
    float x, y, z;
};

struct Mesh
{
    const Vec3* vertices;
    size_t vertexCount;
    const int (*triangles)[3];
    size_t triangleCount;
    Vec3 position;
    bool rotate;
};

struct Material
{
    uint32_t color;
    double ambient;
    double diffuse;
    double specular;
    double shininess;
};

struct ColorRGB
{
    float r;
    float g;
    float b;
};

struct TopFaceLighting
{
    std::array<Vec3, 4> normals;
    std::array<float, 4> ao;
};

struct ShadowMap
{
    size_t resolution = 0;
    std::vector<float> depth;
    float min_x = 0.0f;
    float max_x = 0.0f;
    float min_y = 0.0f;
    float max_y = 0.0f;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    float depth_min = 0.0f;
    Vec3 right{1.0, 0.0, 0.0};
    Vec3 up{0.0, 1.0, 0.0};
    Vec3 forward{0.0, 0.0, 1.0};
    bool valid = false;
};

struct ShadingContext
{
    ColorRGB albedo;
    float albedo_r;
    float albedo_g;
    float albedo_b;
    ColorRGB sky_top;
    ColorRGB sky_bottom;
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
        const struct ShadowMap* shadow;
    };
    std::array<DirectionalLightInfo, 2> lights;
};

constexpr int kMsaaSamples = 2;
constexpr size_t kShadowMapResolution = 256;
constexpr double kPi = 3.14159265358979323846;
constexpr double kSunLatitudeDeg = 30.0;
constexpr double kSunLatitudeRad = kPi * kSunLatitudeDeg / 180.0;

static Vec3 normalize_vec(const Vec3& v);
static float compute_shadow_factor(const ShadowMap* shadow, const Vec3& light_dir,
                                   const Vec3& world, const Vec3& normal);

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

    static constexpr double f2 = 0.366025403784438646; // (sqrt(3)-1)/2
    static constexpr double g2 = 0.211324865405187117; // (3-sqrt(3))/6

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

struct VoxelBlock
{
    Vec3 position;
    uint32_t color;
    bool isTop;
    TopFaceLighting topFace;
};

static std::vector<VoxelBlock> terrainBlocks;
static bool terrainReady = false;
static int terrainSize = 0;
static std::vector<int> terrainHeights;
static std::vector<uint32_t> terrainTopColors;
static std::vector<TopFaceLighting> terrainTopFaces;

static void generate_terrain_chunk()
{
    if (terrainReady)
    {
        return;
    }
    terrainReady = true;

    const int chunk_size = 16;
    const int base_height = 4;
    const int dirt_thickness = 2;
    const int height_variation = 6;
    const double height_freq = 0.12;
    const double surface_freq = 0.4;

    const double block_size = 2.0;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = 4.0;
    const double base_y = 2.0;

    const uint32_t stone_color = 0xFF7A7A7A;
    const uint32_t dirt_color = 0xFFB36A2E;
    const uint32_t grass_color = 0xFF3AAA35;
    const uint32_t water_color = 0xFF3B7BFF;

    terrainSize = chunk_size;
    terrainHeights.assign(static_cast<size_t>(chunk_size * chunk_size), 0);
    terrainTopColors.assign(static_cast<size_t>(chunk_size * chunk_size), grass_color);
    terrainTopFaces.assign(static_cast<size_t>(chunk_size * chunk_size), {});

    auto index = [chunk_size](int x, int z) {
        return static_cast<size_t>(z * chunk_size + x);
    };
    auto height_at_clamped = [&](int x, int z) {
        x = std::clamp(x, 0, chunk_size - 1);
        z = std::clamp(z, 0, chunk_size - 1);
        return terrainHeights[index(x, z)];
    };
    auto height_at_or_zero = [&](int x, int z) {
        if (x < 0 || x >= chunk_size || z < 0 || z >= chunk_size)
        {
            return 0;
        }
        return terrainHeights[index(x, z)];
    };

    terrainBlocks.reserve(static_cast<size_t>(chunk_size * chunk_size * (base_height + height_variation + 3)));

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

            terrainHeights[index(x, z)] = height;
            terrainTopColors[index(x, z)] = top_color;
        }
    }

    const int vertex_size = chunk_size + 1;
    std::vector<double> vertexHeights(static_cast<size_t>(vertex_size * vertex_size), 0.0);
    std::vector<Vec3> vertexNormals(static_cast<size_t>(vertex_size * vertex_size), {0.0, -1.0, 0.0});

    auto vertex_index = [vertex_size](int x, int z) {
        return static_cast<size_t>(z * vertex_size + x);
    };

    for (int z = 0; z < vertex_size; ++z)
    {
        for (int x = 0; x < vertex_size; ++x)
        {
            double sum = 0.0;
            int count = 0;
            for (int dz = -1; dz <= 0; ++dz)
            {
                for (int dx = -1; dx <= 0; ++dx)
                {
                    const int cx = x + dx;
                    const int cz = z + dz;
                    if (cx < 0 || cx >= chunk_size || cz < 0 || cz >= chunk_size)
                    {
                        continue;
                    }
                    sum += static_cast<double>(terrainHeights[index(cx, cz)]);
                    count++;
                }
            }
            vertexHeights[vertex_index(x, z)] = count > 0 ? sum / static_cast<double>(count) : static_cast<double>(base_height);
        }
    }

    auto vertex_height = [&](int x, int z) {
        x = std::clamp(x, 0, vertex_size - 1);
        z = std::clamp(z, 0, vertex_size - 1);
        return vertexHeights[vertex_index(x, z)];
    };

    for (int z = 0; z < vertex_size; ++z)
    {
        for (int x = 0; x < vertex_size; ++x)
        {
            const double h_l = vertex_height(x - 1, z);
            const double h_r = vertex_height(x + 1, z);
            const double h_d = vertex_height(x, z - 1);
            const double h_u = vertex_height(x, z + 1);
            const double dy_dx = -(h_r - h_l) * 0.5;
            const double dy_dz = -(h_u - h_d) * 0.5;
            vertexNormals[vertex_index(x, z)] = normalize_vec({dy_dx, -1.0, dy_dz});
        }
    }

    for (int z = 0; z < chunk_size; ++z)
    {
        for (int x = 0; x < chunk_size; ++x)
        {
            const int height = terrainHeights[index(x, z)];
            const uint32_t top_color = terrainTopColors[index(x, z)];
            const int top_y = height - 1;
            TopFaceLighting top_face{};
            top_face.normals = {
                vertexNormals[vertex_index(x, z)],
                vertexNormals[vertex_index(x + 1, z)],
                vertexNormals[vertex_index(x + 1, z + 1)],
                vertexNormals[vertex_index(x, z + 1)]
            };

            const float ao_strength = 0.35f;
            auto corner_ao = [&](int sx, int sz) {
                const int side1 = height_at_or_zero(x + sx, z) > height ? 1 : 0;
                const int side2 = height_at_or_zero(x, z + sz) > height ? 1 : 0;
                const int corner = height_at_or_zero(x + sx, z + sz) > height ? 1 : 0;
                int occlusion = side1 + side2 + corner;
                if (side1 && side2)
                {
                    occlusion = 3;
                }
                float ao = 1.0f - ao_strength * (static_cast<float>(occlusion) / 3.0f);
                return std::clamp(ao, 0.0f, 1.0f);
            };
            top_face.ao = {
                corner_ao(-1, -1),
                corner_ao(1, -1),
                corner_ao(1, 1),
                corner_ao(-1, 1)
            };
            terrainTopFaces[index(x, z)] = top_face;

            const TopFaceLighting empty_top{
                {Vec3{0.0, -1.0, 0.0}, Vec3{0.0, -1.0, 0.0}, Vec3{0.0, -1.0, 0.0}, Vec3{0.0, -1.0, 0.0}},
                {1.0f, 1.0f, 1.0f, 1.0f}
            };

            for (int y = 0; y < height; ++y)
            {
                uint32_t color = stone_color;
                if (y >= height - 1)
                {
                    color = top_color;
                }
                else if (y >= height - 1 - dirt_thickness)
                {
                    color = dirt_color;
                }

                const bool is_top = (y == height - 1);
                terrainBlocks.push_back({
                    {start_x + x * block_size, base_y - y * block_size, start_z + z * block_size},
                    color,
                    is_top,
                    is_top ? top_face : empty_top
                });
            }
        }
    }
}

static ColorRGB lerp_color(const ColorRGB& a, const ColorRGB& b, float t)
{
    return {
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t
    };
}

static Vec3 normalize_vec(const Vec3& v)
{
    const double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len == 0.0) return {0.0, 0.0, 0.0};
    return {v.x / len, v.y / len, v.z / len};
}

static Vec3 add_vec(const Vec3& a, const Vec3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static Vec3 scale_vec(const Vec3& v, const double s)
{
    return {v.x * s, v.y * s, v.z * s};
}

static double dot_vec(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 cross_vec(const Vec3& a, const Vec3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static Vec3 rotate_around_axis(const Vec3& v, const Vec3& axis, const double angle)
{
    const Vec3 a = normalize_vec(axis);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const Vec3 term1 = scale_vec(v, c);
    const Vec3 term2 = scale_vec(cross_vec(a, v), s);
    const Vec3 term3 = scale_vec(a, dot_vec(a, v) * (1.0 - c));
    return add_vec(add_vec(term1, term2), term3);
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
    return normalize_vec({x, y, z});
}

static ColorRGB unpack_color(uint32_t color)
{
    return {
        static_cast<float>((color >> 16) & 0xFF),
        static_cast<float>((color >> 8) & 0xFF),
        static_cast<float>(color & 0xFF)
    };
}

static uint32_t pack_color(const ColorRGB& color)
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

static uint32_t apply_specular(uint32_t color, double specular)
{
    specular = std::clamp(specular, 0.0, 1.0);
    const uint32_t r = (color >> 16) & 0xFF;
    const uint32_t g = (color >> 8) & 0xFF;
    const uint32_t b = color & 0xFF;
    const uint32_t nr = static_cast<uint32_t>(std::round(r + (255.0 - r) * specular));
    const uint32_t ng = static_cast<uint32_t>(std::round(g + (255.0 - g) * specular));
    const uint32_t nb = static_cast<uint32_t>(std::round(b + (255.0 - b) * specular));
    return (color & 0xFF000000) | (nr << 16) | (ng << 8) | nb;
}

static Vec3 rotate_vertex(const Vec3& v, const double c, const double s)
{
    const double tempX = v.x;
    const double tempY = v.y * c - v.z * s;
    const double tempZ = v.y * s + v.z * c;

    return {
        tempX * c - tempZ * s,
        tempY,
        tempX * s + tempZ * c
    };
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

static void draw_filled_triangle(uint32_t* framebuffer, float* zbuffer, size_t width, size_t height,
                                 const ScreenVertex& v0, const ScreenVertex& v1, const ScreenVertex& v2, uint32_t color)
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

    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            ScreenVertex p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, 0.0f};
            float w0 = edge_function(v1, v2, p);
            float w1 = edge_function(v2, v0, p);
            float w2 = edge_function(v0, v1, p);

            if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f && area_positive) ||
                (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f && !area_positive))
            {
                w0 /= area;
                w1 /= area;
                w2 /= area;
                const float depth = w0 * v0.z + w1 * v1.z + w2 * v2.z;
                const size_t idx = static_cast<size_t>(y) * width + static_cast<size_t>(x);
                if (depth < zbuffer[idx])
                {
                    zbuffer[idx] = depth;
                    framebuffer[idx] = color;
                }
            }
        }
    }
}

static void draw_shaded_triangle(float* zbuffer, ColorRGB* sample_ambient, ColorRGB* sample_direct, uint8_t* sample_mask,
                                 size_t width, size_t height,
                                 const ScreenVertex& v0, const ScreenVertex& v1, const ScreenVertex& v2,
                                 const Vec3& wp0, const Vec3& wp1, const Vec3& wp2,
                                 const Vec3& n0, const Vec3& n1, const Vec3& n2,
                                 float ao0, float ao1, float ao2,
                                 const ShadingContext& ctx)
{
    static constexpr float sample_offsets[kMsaaSamples][2] = {
        {0.25f, 0.75f},
        {0.75f, 0.25f}
    };

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

    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            for (int s = 0; s < kMsaaSamples; ++s)
            {
                ScreenVertex p{
                    static_cast<float>(x) + sample_offsets[s][0],
                    static_cast<float>(y) + sample_offsets[s][1],
                    0.0f
                };
                float w0 = edge_function(v1, v2, p);
                float w1 = edge_function(v2, v0, p);
                float w2 = edge_function(v0, v1, p);

                if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f && area_positive) ||
                    (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f && !area_positive))
                {
                    w0 /= area;
                    w1 /= area;
                    w2 /= area;
                    const float depth = w0 * v0.z + w1 * v1.z + w2 * v2.z;
                    const size_t base = (static_cast<size_t>(y) * width + static_cast<size_t>(x)) * kMsaaSamples;
                    const size_t idx = base + static_cast<size_t>(s);
                    if (depth < zbuffer[idx])
                    {
                        zbuffer[idx] = depth;
                        Vec3 normal{
                            w0 * n0.x + w1 * n1.x + w2 * n2.x,
                            w0 * n0.y + w1 * n1.y + w2 * n2.y,
                            w0 * n0.z + w1 * n1.z + w2 * n2.z
                        };
                        normal = normalize_vec(normal);

                        Vec3 world{
                            w0 * wp0.x + w1 * wp1.x + w2 * wp2.x,
                            w0 * wp0.y + w1 * wp1.y + w2 * wp2.y,
                            w0 * wp0.z + w1 * wp1.z + w2 * wp2.z
                        };

                        float ao = w0 * ao0 + w1 * ao1 + w2 * ao2;
                        if (!ctx.ambient_occlusion_enabled)
                        {
                            ao = 1.0f;
                        }
                        ao = std::clamp(ao, 0.0f, 1.0f);

                        const double ambient = ctx.direct_lighting_enabled ? ctx.ambient_light * ctx.material.ambient : 0.0;
                        ColorRGB ambient_color{
                            static_cast<float>(ctx.albedo.r * ambient),
                            static_cast<float>(ctx.albedo.g * ambient),
                            static_cast<float>(ctx.albedo.b * ambient)
                        };
                        if (ctx.sky_scale > 0.0f)
                        {
                            float sky_t = static_cast<float>((-normal.y) * 0.5 + 0.5);
                            sky_t = std::clamp(sky_t, 0.0f, 1.0f);
                            const ColorRGB sky = lerp_color(ctx.sky_bottom, ctx.sky_top, sky_t);
                            ambient_color.r = sky.r * ctx.sky_scale * ctx.albedo_r;
                            ambient_color.g = sky.g * ctx.sky_scale * ctx.albedo_g;
                            ambient_color.b = sky.b * ctx.sky_scale * ctx.albedo_b;
                        }
                        ambient_color.r *= ao;
                        ambient_color.g *= ao;
                        ambient_color.b *= ao;

                        ColorRGB direct_color{0.0f, 0.0f, 0.0f};
                        if (ctx.direct_lighting_enabled)
                        {
                            const Vec3 view_vec{
                                ctx.camera_pos.x - world.x,
                                ctx.camera_pos.y - world.y,
                                ctx.camera_pos.z - world.z
                            };
                            const Vec3 view_dir = normalize_vec(view_vec);

                            auto add_light = [&](const ShadingContext::DirectionalLightInfo& light) {
                                if (light.intensity <= 0.0)
                                {
                                    return;
                                }
                                const double ndotl = std::max(0.0, dot_vec(normal, light.dir));
                                if (ndotl <= 0.0)
                                {
                                    return;
                                }
                                float shadow_factor = 1.0f;
                                if (ctx.shadows_enabled)
                                {
                                    shadow_factor = compute_shadow_factor(light.shadow, light.dir, world, normal);
                                }

                                const double diffuse = ndotl * light.intensity * ctx.material.diffuse * shadow_factor;
                                ColorRGB light_color{
                                    static_cast<float>(ctx.albedo.r * diffuse),
                                    static_cast<float>(ctx.albedo.g * diffuse),
                                    static_cast<float>(ctx.albedo.b * diffuse)
                                };
                                if (ctx.material.specular > 0.0)
                                {
                                    const Vec3 half_vec = normalize_vec(add_vec(light.dir, view_dir));
                                    const double spec_dot = std::max(0.0, dot_vec(normal, half_vec));
                                    double spec = std::pow(spec_dot, ctx.material.shininess) * ctx.material.specular;
                                    spec *= light.intensity * shadow_factor;
                                    const uint32_t packed = apply_specular(pack_color(light_color), spec);
                                    light_color = unpack_color(packed);
                                }
                                direct_color.r += light_color.r;
                                direct_color.g += light_color.g;
                                direct_color.b += light_color.b;
                            };

                            add_light(ctx.lights[0]);
                            add_light(ctx.lights[1]);
                        }

                        sample_ambient[idx] = ambient_color;
                        sample_direct[idx] = direct_color;
                        if (sample_mask)
                        {
                            sample_mask[idx] = 1;
                        }
                    }
                }
            }
        }
    }
}

static void build_light_basis(const Vec3& light_dir, Vec3& right, Vec3& up, Vec3& forward)
{
    forward = normalize_vec(light_dir);
    Vec3 up_guess{0.0, 1.0, 0.0};
    if (std::abs(dot_vec(forward, up_guess)) > 0.99)
    {
        up_guess = {0.0, 0.0, 1.0};
    }
    right = normalize_vec(cross_vec(up_guess, forward));
    up = cross_vec(forward, right);
}

static void rasterize_shadow_triangle(std::vector<float>& depth, size_t resolution,
                                      const ScreenVertex& v0, const ScreenVertex& v1, const ScreenVertex& v2)
{
    float min_x = std::min({v0.x, v1.x, v2.x});
    float max_x = std::max({v0.x, v1.x, v2.x});
    float min_y = std::min({v0.y, v1.y, v2.y});
    float max_y = std::max({v0.y, v1.y, v2.y});

    const int x0 = std::max(0, static_cast<int>(std::floor(min_x)));
    const int x1 = std::min(static_cast<int>(resolution) - 1, static_cast<int>(std::ceil(max_x)));
    const int y0 = std::max(0, static_cast<int>(std::floor(min_y)));
    const int y1 = std::min(static_cast<int>(resolution) - 1, static_cast<int>(std::ceil(max_y)));

    const float area = edge_function(v0, v1, v2);
    if (area == 0.0f) return;

    const bool area_positive = area > 0.0f;

    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            ScreenVertex p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, 0.0f};
            float w0 = edge_function(v1, v2, p);
            float w1 = edge_function(v2, v0, p);
            float w2 = edge_function(v0, v1, p);

            if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f && area_positive) ||
                (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f && !area_positive))
            {
                w0 /= area;
                w1 /= area;
                w2 /= area;
                const float depth_value = w0 * v0.z + w1 * v1.z + w2 * v2.z;
                const size_t idx = static_cast<size_t>(y) * resolution + static_cast<size_t>(x);
                if (depth_value > depth[idx])
                {
                    depth[idx] = depth_value;
                }
            }
        }
    }
}

static float compute_shadow_factor(const ShadowMap* shadow, const Vec3& light_dir,
                                   const Vec3& world, const Vec3& normal)
{
    if (!shadow || !shadow->valid)
    {
        return 1.0f;
    }
    const double ndotl = std::max(0.0, dot_vec(normal, light_dir));
    if (ndotl <= 0.0)
    {
        return 1.0f;
    }
    const double slope = 1.0 - ndotl;
    const float bias = static_cast<float>(0.02 + 0.08 * slope);
    const float lx = static_cast<float>(dot_vec(world, shadow->right));
    const float ly = static_cast<float>(dot_vec(world, shadow->up));
    const float lz = static_cast<float>(dot_vec(world, shadow->forward));

    if (lx < shadow->min_x || lx > shadow->max_x || ly < shadow->min_y || ly > shadow->max_y)
    {
        return 1.0f;
    }

    const float u = (lx - shadow->min_x) * shadow->scale_x;
    const float v = (ly - shadow->min_y) * shadow->scale_y;
    const int res = static_cast<int>(shadow->resolution);
    const int cx = std::clamp(static_cast<int>(std::lround(u)), 0, res - 1);
    const int cy = std::clamp(static_cast<int>(std::lround(v)), 0, res - 1);
    int lit = 0;
    int samples = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        const int sy = std::clamp(cy + dy, 0, res - 1);
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int sx = std::clamp(cx + dx, 0, res - 1);
            const float depth = shadow->depth[static_cast<size_t>(sy) * shadow->resolution + static_cast<size_t>(sx)];
            if (depth <= shadow->depth_min)
            {
                continue;
            }
            samples++;
            if (lz + bias >= depth)
            {
                lit++;
            }
        }
    }
    if (samples <= 0)
    {
        return 1.0f;
    }
    return static_cast<float>(lit) / static_cast<float>(samples);
}

static bool build_shadow_map(ShadowMap& shadow, const Vec3& light_dir,
                             const std::vector<VoxelBlock>& blocks,
                             const Mesh& base_mesh, const double c, const double s)
{
    const Vec3 forward = normalize_vec(light_dir);
    if (forward.x == 0.0 && forward.y == 0.0 && forward.z == 0.0)
    {
        shadow.valid = false;
        return false;
    }

    Vec3 right;
    Vec3 up;
    build_light_basis(forward, right, up, shadow.forward);
    shadow.right = right;
    shadow.up = up;

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    bool has_bounds = false;

    std::vector<Vec3> world_vertices(base_mesh.vertexCount);
    for (const auto& block : blocks)
    {
        for (size_t i = 0; i < base_mesh.vertexCount; ++i)
        {
            Vec3 v = base_mesh.vertices[i];
            if (base_mesh.rotate)
            {
                v = rotate_vertex(v, c, s);
            }
            v.x += block.position.x;
            v.y += block.position.y;
            v.z += block.position.z;
            world_vertices[i] = v;
            const float lx = static_cast<float>(dot_vec(v, right));
            const float ly = static_cast<float>(dot_vec(v, up));
            min_x = std::min(min_x, lx);
            max_x = std::max(max_x, lx);
            min_y = std::min(min_y, ly);
            max_y = std::max(max_y, ly);
            has_bounds = true;
        }
    }

    if (!has_bounds)
    {
        shadow.valid = false;
        return false;
    }

    const float padding = 1.0f;
    min_x -= padding;
    max_x += padding;
    min_y -= padding;
    max_y += padding;
    if (std::abs(max_x - min_x) < 1e-3f)
    {
        min_x -= 1.0f;
        max_x += 1.0f;
    }
    if (std::abs(max_y - min_y) < 1e-3f)
    {
        min_y -= 1.0f;
        max_y += 1.0f;
    }

    shadow.min_x = min_x;
    shadow.max_x = max_x;
    shadow.min_y = min_y;
    shadow.max_y = max_y;
    shadow.scale_x = static_cast<float>(kShadowMapResolution - 1) / (max_x - min_x);
    shadow.scale_y = static_cast<float>(kShadowMapResolution - 1) / (max_y - min_y);
    shadow.resolution = kShadowMapResolution;
    shadow.depth_min = std::numeric_limits<float>::lowest();
    shadow.valid = true;

    const size_t map_size = kShadowMapResolution * kShadowMapResolution;
    if (shadow.depth.size() != map_size)
    {
        shadow.depth.assign(map_size, shadow.depth_min);
    }
    else
    {
        std::fill(shadow.depth.begin(), shadow.depth.end(), shadow.depth_min);
    }

    for (const auto& block : blocks)
    {
        for (size_t i = 0; i < base_mesh.vertexCount; ++i)
        {
            Vec3 v = base_mesh.vertices[i];
            if (base_mesh.rotate)
            {
                v = rotate_vertex(v, c, s);
            }
            v.x += block.position.x;
            v.y += block.position.y;
            v.z += block.position.z;
            world_vertices[i] = v;
        }

        for (size_t t = 0; t < base_mesh.triangleCount; ++t)
        {
            const int i0 = base_mesh.triangles[t][0];
            const int i1 = base_mesh.triangles[t][1];
            const int i2 = base_mesh.triangles[t][2];

            const Vec3& a = world_vertices[static_cast<size_t>(i0)];
            const Vec3& b = world_vertices[static_cast<size_t>(i1)];
            const Vec3& cpos = world_vertices[static_cast<size_t>(i2)];

            const auto to_shadow = [&](const Vec3& v) {
                const float lx = static_cast<float>(dot_vec(v, right));
                const float ly = static_cast<float>(dot_vec(v, up));
                const float lz = static_cast<float>(dot_vec(v, forward));
                return ScreenVertex{
                    (lx - min_x) * shadow.scale_x,
                    (ly - min_y) * shadow.scale_y,
                    lz
                };
            };

            const ScreenVertex sv0 = to_shadow(a);
            const ScreenVertex sv1 = to_shadow(b);
            const ScreenVertex sv2 = to_shadow(cpos);

            rasterize_shadow_triangle(shadow.depth, shadow.resolution, sv0, sv1, sv2);
        }
    }

    return true;
}

static void render_mesh(float* zbuffer, ColorRGB* sample_ambient, ColorRGB* sample_direct, uint8_t* sample_mask,
                        size_t width, size_t height,
                        const Mesh& mesh, const double c, const double s,
                        const double fov, const Vec3& camera_pos,
                        const double yaw, const double pitch, const Material& material,
                        const ColorRGB& sky_top, const ColorRGB& sky_bottom,
                        const double sky_light_intensity,
                        const std::array<ShadingContext::DirectionalLightInfo, 2>& lights,
                        bool shadows_enabled,
                        const TopFaceLighting* top_face)
{
    std::vector<Vec3> world(mesh.vertexCount);
    std::vector<ScreenVertex> projected(mesh.vertexCount);

    for (size_t i = 0; i < mesh.vertexCount; ++i)
    {
        Vec3 v = mesh.vertices[i];
        if (mesh.rotate)
        {
            v = rotate_vertex(v, c, s);
        }

        v.x += mesh.position.x;
        v.y += mesh.position.y;
        v.z += mesh.position.z;
        world[i] = v;

        Vec3 view = {v.x - camera_pos.x, v.y - camera_pos.y, v.z - camera_pos.z};
        view = rotate_yaw_pitch(view, -yaw, -pitch);

        const double z = view.z <= 0.01 ? 0.01 : view.z;
        const double invZ = 1.0 / z;
        const double projX = view.x * invZ * fov;
        const double projY = view.y * invZ * fov;
        projected[i].x = static_cast<float>(projX + width / 2.0);
        projected[i].y = static_cast<float>(projY + height / 2.0);
        projected[i].z = static_cast<float>(z);
    }

    const ColorRGB albedo = unpack_color(material.color);
    const float albedo_r = albedo.r / 255.0f;
    const float albedo_g = albedo.g / 255.0f;
    const float albedo_b = albedo.b / 255.0f;
    const float sky_scale = static_cast<float>(std::clamp(sky_light_intensity, 0.0, 1.0));
    const bool direct_lighting_enabled = (lights[0].intensity > 0.0) || (lights[1].intensity > 0.0);
    const bool ao_enabled = ambientOcclusionEnabled.load(std::memory_order_relaxed);
    const ShadingContext ctx{
        albedo,
        albedo_r,
        albedo_g,
        albedo_b,
        sky_top,
        sky_bottom,
        sky_scale,
        camera_pos,
        ambientLight,
        material,
        direct_lighting_enabled,
        ao_enabled,
        shadows_enabled,
        lights
    };

    auto top_corner_index = [](int vertex_index) -> int {
        switch (vertex_index)
        {
            case 0: return 0;
            case 1: return 1;
            case 5: return 2;
            case 4: return 3;
            default: return -1;
        }
    };

    for (size_t t = 0; t < mesh.triangleCount; ++t)
    {
        const int i0 = mesh.triangles[t][0];
        const int i1 = mesh.triangles[t][1];
        const int i2 = mesh.triangles[t][2];

        const Vec3& a = world[i0];
        const Vec3& b = world[i1];
        const Vec3& cpos = world[i2];

        const Vec3 ab{b.x - a.x, b.y - a.y, b.z - a.z};
        const Vec3 ac{cpos.x - a.x, cpos.y - a.y, cpos.z - a.z};
        const Vec3 normal_raw{
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x
        };

        const Vec3 center{
            (a.x + b.x + cpos.x) / 3.0,
            (a.y + b.y + cpos.y) / 3.0,
            (a.z + b.z + cpos.z) / 3.0
        };
        const Vec3 view_vec{
            camera_pos.x - center.x,
            camera_pos.y - center.y,
            camera_pos.z - center.z
        };
        const double facing = normal_raw.x * view_vec.x + normal_raw.y * view_vec.y + normal_raw.z * view_vec.z;
        if (facing <= 0.0) continue;

        const Vec3 face_normal = normalize_vec(normal_raw);
        Vec3 n0 = face_normal;
        Vec3 n1 = face_normal;
        Vec3 n2 = face_normal;
        float ao0 = 1.0f;
        float ao1 = 1.0f;
        float ao2 = 1.0f;

        if (top_face && face_normal.y < -0.5)
        {
            const int c0 = top_corner_index(i0);
            const int c1 = top_corner_index(i1);
            const int c2 = top_corner_index(i2);
            if (c0 >= 0)
            {
                n0 = top_face->normals[static_cast<size_t>(c0)];
                ao0 = top_face->ao[static_cast<size_t>(c0)];
            }
            if (c1 >= 0)
            {
                n1 = top_face->normals[static_cast<size_t>(c1)];
                ao1 = top_face->ao[static_cast<size_t>(c1)];
            }
            if (c2 >= 0)
            {
                n2 = top_face->normals[static_cast<size_t>(c2)];
                ao2 = top_face->ao[static_cast<size_t>(c2)];
            }
        }

        draw_shaded_triangle(zbuffer, sample_ambient, sample_direct, sample_mask, width, height,
                             projected[i0], projected[i1], projected[i2],
                             world[i0], world[i1], world[i2],
                             n0, n1, n2,
                             ao0, ao1, ao2,
                             ctx);
    }
}

static void draw_shadow_triangle(ColorRGB* sample_ambient, const uint8_t* plane_mask, size_t width, size_t height,
                                 const ScreenVertex& v0, const ScreenVertex& v1, const ScreenVertex& v2,
                                 float shadow_strength)
{
    static constexpr float sample_offsets[kMsaaSamples][2] = {
        {0.25f, 0.75f},
        {0.75f, 0.25f}
    };

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

    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            for (int s = 0; s < kMsaaSamples; ++s)
            {
                ScreenVertex p{
                    static_cast<float>(x) + sample_offsets[s][0],
                    static_cast<float>(y) + sample_offsets[s][1],
                    0.0f
                };
                float w0 = edge_function(v1, v2, p);
                float w1 = edge_function(v2, v0, p);
                float w2 = edge_function(v0, v1, p);

                if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f && area_positive) ||
                    (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f && !area_positive))
                {
                    const size_t base = (static_cast<size_t>(y) * width + static_cast<size_t>(x)) * kMsaaSamples;
                    const size_t idx = base + static_cast<size_t>(s);
                    if (plane_mask[idx])
                    {
                        sample_ambient[idx].r *= shadow_strength;
                        sample_ambient[idx].g *= shadow_strength;
                        sample_ambient[idx].b *= shadow_strength;
                    }
                }
            }
        }
    }
}

static void render_shadow_on_plane(ColorRGB* sample_ambient, size_t width, size_t height,
                                   const Mesh& mesh, const double c, const double s, const double fov,
                                   const Vec3& camera_pos, const double yaw, const double pitch,
                                   const Vec3& light_dir, const double plane_z, const uint8_t* plane_mask,
                                   float shadow_strength)
{
    const Vec3 ray_dir{-light_dir.x, -light_dir.y, -light_dir.z};
    if (std::abs(ray_dir.z) < 1e-6) return;

    std::vector<Vec3> world(mesh.vertexCount);
    std::vector<ScreenVertex> projected(mesh.vertexCount);
    std::vector<bool> valid(mesh.vertexCount, false);

    for (size_t i = 0; i < mesh.vertexCount; ++i)
    {
        Vec3 v = mesh.vertices[i];
        if (mesh.rotate)
        {
            v = rotate_vertex(v, c, s);
        }
        v.x += mesh.position.x;
        v.y += mesh.position.y;
        v.z += mesh.position.z;
        world[i] = v;

        const double t = (plane_z - v.z) / ray_dir.z;
        if (t <= 0.0) continue;

        Vec3 shadow_v{
            v.x + ray_dir.x * t,
            v.y + ray_dir.y * t,
            plane_z - 0.02
        };

        Vec3 view = {shadow_v.x - camera_pos.x, shadow_v.y - camera_pos.y, shadow_v.z - camera_pos.z};
        view = rotate_yaw_pitch(view, -yaw, -pitch);

        const double z = view.z <= 0.01 ? 0.01 : view.z;
        const double inv_z = 1.0 / z;
        const double proj_x = view.x * inv_z * fov;
        const double proj_y = view.y * inv_z * fov;
        projected[i].x = static_cast<float>(proj_x + width / 2.0);
        projected[i].y = static_cast<float>(proj_y + height / 2.0);
        projected[i].z = static_cast<float>(z);
        valid[i] = true;
    }

    for (size_t t = 0; t < mesh.triangleCount; ++t)
    {
        const int i0 = mesh.triangles[t][0];
        const int i1 = mesh.triangles[t][1];
        const int i2 = mesh.triangles[t][2];
        if (!valid[i0] || !valid[i1] || !valid[i2]) continue;

        const Vec3& a = world[i0];
        const Vec3& b = world[i1];
        const Vec3& cpos = world[i2];

        const Vec3 ab{b.x - a.x, b.y - a.y, b.z - a.z};
        const Vec3 ac{cpos.x - a.x, cpos.y - a.y, cpos.z - a.z};
        const Vec3 normal_raw{
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x
        };

        const double light_facing = dot_vec(normal_raw, light_dir);
        if (light_facing <= 0.0) continue;

        draw_shadow_triangle(sample_ambient, plane_mask, width, height,
                             projected[i0], projected[i1], projected[i2],
                             shadow_strength);
    }
}

void render_update_array(uint32_t* framebuffer, size_t width, size_t height)
{
    std::memset(framebuffer, 0, width * height * sizeof(uint32_t));

    static size_t cached_width = 0;
    static size_t cached_height = 0;
    static std::vector<float> zbuffer;
    static std::vector<ColorRGB> sample_colors;
    static std::vector<ColorRGB> sample_direct;
    static std::vector<uint8_t> pixel_coverage;
    const size_t pixel_count = width * height;
    const size_t sample_count = width * height * kMsaaSamples;
    const float depth_max = std::numeric_limits<float>::max();

    if (width != cached_width || height != cached_height)
    {
        cached_width = width;
        cached_height = height;
        zbuffer.assign(sample_count, depth_max);
        sample_colors.assign(sample_count, {0.0f, 0.0f, 0.0f});
        sample_direct.assign(sample_count, {0.0f, 0.0f, 0.0f});
        pixel_coverage.assign(pixel_count, 0);
    }
    else
    {
        std::fill(zbuffer.begin(), zbuffer.end(), depth_max);
        std::fill(sample_colors.begin(), sample_colors.end(), ColorRGB{0.0f, 0.0f, 0.0f});
        std::fill(sample_direct.begin(), sample_direct.end(), ColorRGB{0.0f, 0.0f, 0.0f});
        std::fill(pixel_coverage.begin(), pixel_coverage.end(), 0);
    }

    if (!rotationPaused.load(std::memory_order_relaxed))
    {
        rotationAngle += 0.002f;
    }
    if (sunOrbitEnabled.load(std::memory_order_relaxed) && !rotationPaused.load(std::memory_order_relaxed))
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
    const double c = cos(rotationAngle);
    const double s = sin(rotationAngle);

    const double fov = static_cast<double>(height) * 0.8;
    const Vec3 camera_pos{
        camera_x.load(std::memory_order_relaxed),
        camera_y.load(std::memory_order_relaxed),
        camera_z.load(std::memory_order_relaxed)
    };
    const double yaw = camera_yaw.load(std::memory_order_relaxed);
    const double pitch = camera_pitch.load(std::memory_order_relaxed);
    const ColorRGB sky_top = unpack_color(skyTopColor);
    const ColorRGB sky_bottom = unpack_color(skyBottomColor);

    const double mat_ambient = 0.25;
    const double mat_diffuse = 1.0;
    const Material cube_material{0xFFFFFFFF, mat_ambient, mat_diffuse, 0.15, 24.0};

    const Mesh cubeMesh{
        cubeVertices,
        8,
        cubeTriangles,
        12,
        {0.0, 0.0, 0.0},
        false
    };

    generate_terrain_chunk();

    const bool sun_orbit = sunOrbitEnabled.load(std::memory_order_relaxed);
    Vec3 sun_dir = normalize_vec(lightDirection);
    double sun_intensity = lightIntensity;
    if (sun_orbit)
    {
        const double angle = sunOrbitAngle.load(std::memory_order_relaxed);
        sun_dir = compute_sun_orbit_direction(angle);
        const double max_y = std::cos(kSunLatitudeRad);
        double visibility = sun_dir.y < 0.0 ? (-sun_dir.y) / max_y : 0.0;
        visibility = std::clamp(visibility, 0.0, 1.0);
        sun_intensity = lightIntensity * visibility;
    }

    const Vec3 moon_dir = normalize_vec(moonDirection);
    const bool shadows_on = shadowEnabled.load(std::memory_order_relaxed);
    static ShadowMap sun_shadow;
    static ShadowMap moon_shadow;
    const bool sun_shadow_valid = shadows_on && sun_intensity > 0.0 &&
                                  build_shadow_map(sun_shadow, sun_dir, terrainBlocks, cubeMesh, c, s);
    const bool moon_shadow_valid = shadows_on && moonIntensity > 0.0 &&
                                   build_shadow_map(moon_shadow, moon_dir, terrainBlocks, cubeMesh, c, s);

    const std::array<ShadingContext::DirectionalLightInfo, 2> lights = {
        ShadingContext::DirectionalLightInfo{sun_dir, sun_intensity, sun_shadow_valid ? &sun_shadow : nullptr},
        ShadingContext::DirectionalLightInfo{moon_dir, moonIntensity, moon_shadow_valid ? &moon_shadow : nullptr}
    };

    for (const auto& block : terrainBlocks)
    {
        Material material = cube_material;
        material.color = block.color;
        Mesh mesh = cubeMesh;
        mesh.position = block.position;
        const TopFaceLighting* top_face = block.isTop ? &block.topFace : nullptr;
        render_mesh(zbuffer.data(), sample_colors.data(), sample_direct.data(), nullptr, width, height, mesh, c, s, fov, camera_pos,
                    yaw, pitch, material, sky_top, sky_bottom, skyLightIntensity, lights, shadows_on, top_face);
    }

    for (size_t i = 0; i < sample_count; ++i)
    {
        const size_t pixel = i / kMsaaSamples;
        const float depth = zbuffer[i];
        if (depth < depth_max)
        {
            pixel_coverage[pixel] += 1;
        }
    }

    static constexpr int bayer4[4][4] = {
        {0, 8, 2, 10},
        {12, 4, 14, 6},
        {3, 11, 1, 9},
        {15, 7, 13, 5}
    };
    const float dither_strength = 2.0f;
    const float dither_scale = dither_strength / 16.0f;

    for (size_t y = 0; y < height; ++y)
    {
        const float sky_t = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
        const ColorRGB sky_row_color = lerp_color(sky_top, sky_bottom, sky_t);
        const uint32_t sky_row = pack_color(sky_row_color);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t pixel = y * width + x;
            const int covered = pixel_coverage[pixel];
            if (covered == 0)
            {
                framebuffer[y * width + x] = sky_row;
                continue;
            }

            const size_t base = (y * width + x) * kMsaaSamples;

            ColorRGB accum{0.0f, 0.0f, 0.0f};
            float weight_sum = 0.0f;
            for (int s = 0; s < kMsaaSamples; ++s)
            {
                const float depth = zbuffer[base + static_cast<size_t>(s)];
                if (depth >= depth_max)
                {
                    accum.r += sky_row_color.r;
                    accum.g += sky_row_color.g;
                    accum.b += sky_row_color.b;
                    weight_sum += 1.0f;
                    continue;
                }
                const float weight = 1.0f;
                ColorRGB c = sample_colors[base + static_cast<size_t>(s)];
                const ColorRGB direct = sample_direct[base + static_cast<size_t>(s)];
                c.r += direct.r;
                c.g += direct.g;
                c.b += direct.b;

                accum.r += c.r * weight;
                accum.g += c.g * weight;
                accum.b += c.b * weight;
                weight_sum += weight;
            }

            if (weight_sum <= 0.0f)
            {
                framebuffer[y * width + x] = sky_row;
                continue;
            }

            const float inv_weight = 1.0f / weight_sum;
            accum.r *= inv_weight;
            accum.g *= inv_weight;
            accum.b *= inv_weight;

            const float dither = (static_cast<float>(bayer4[y & 3][x & 3]) - 7.5f) * dither_scale;
            accum.r += dither;
            accum.g += dither;
            accum.b += dither;

            framebuffer[y * width + x] = pack_color(accum);
        }
    }
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

void render_set_rotation(const float angle)
{
    rotationAngle = angle;
}

float render_get_rotation()
{
    return rotationAngle;
}

void render_set_scene(const RenderScene scene)
{
    activeScene = scene;
}

RenderScene render_get_scene()
{
    return activeScene;
}

void render_set_light_direction(const Vec3 dir)
{
    lightDirection = dir;
}

Vec3 render_get_light_direction()
{
    return lightDirection;
}

void render_set_light_intensity(const double intensity)
{
    lightIntensity = intensity;
}

double render_get_light_intensity()
{
    return lightIntensity;
}

void render_set_sun_orbit_enabled(const bool enabled)
{
    sunOrbitEnabled.store(enabled, std::memory_order_relaxed);
}

bool render_get_sun_orbit_enabled()
{
    return sunOrbitEnabled.load(std::memory_order_relaxed);
}

void render_set_sun_orbit_angle(const double angle)
{
    sunOrbitAngle.store(angle, std::memory_order_relaxed);
}

double render_get_sun_orbit_angle()
{
    return sunOrbitAngle.load(std::memory_order_relaxed);
}

void render_set_moon_direction(const Vec3 dir)
{
    moonDirection = dir;
}

Vec3 render_get_moon_direction()
{
    return moonDirection;
}

void render_set_moon_intensity(const double intensity)
{
    moonIntensity = intensity;
}

double render_get_moon_intensity()
{
    return moonIntensity;
}

void render_set_sky_top_color(const uint32_t color)
{
    skyTopColor = color;
}

uint32_t render_get_sky_top_color()
{
    return skyTopColor;
}

void render_set_sky_bottom_color(const uint32_t color)
{
    skyBottomColor = color;
}

uint32_t render_get_sky_bottom_color()
{
    return skyBottomColor;
}

void render_set_sky_light_intensity(const double intensity)
{
    skyLightIntensity = intensity;
}

double render_get_sky_light_intensity()
{
    return skyLightIntensity;
}

void render_set_ambient_occlusion_enabled(const bool enabled)
{
    ambientOcclusionEnabled.store(enabled, std::memory_order_relaxed);
}

bool render_get_ambient_occlusion_enabled()
{
    return ambientOcclusionEnabled.load(std::memory_order_relaxed);
}

void render_set_shadow_enabled(const bool enabled)
{
    shadowEnabled.store(enabled, std::memory_order_relaxed);
}

bool render_get_shadow_enabled()
{
    return shadowEnabled.load(std::memory_order_relaxed);
}

bool render_get_terrain_top_ao(const int x, const int z, float out_ao[4])
{
    if (!out_ao)
    {
        return false;
    }
    generate_terrain_chunk();
    if (x < 0 || z < 0 || x >= terrainSize || z >= terrainSize)
    {
        return false;
    }
    const size_t idx = static_cast<size_t>(z * terrainSize + x);
    if (idx >= terrainTopFaces.size())
    {
        return false;
    }
    for (size_t i = 0; i < 4; ++i)
    {
        out_ao[i] = terrainTopFaces[idx].ao[i];
    }
    return true;
}

bool render_get_shadow_factor_at_point(const Vec3 world, const Vec3 normal, float* out_factor)
{
    if (!out_factor)
    {
        return false;
    }
    generate_terrain_chunk();
    if (!shadowEnabled.load(std::memory_order_relaxed))
    {
        *out_factor = 1.0f;
        return true;
    }
    const Vec3 light_dir = normalize_vec(lightDirection);
    if (lightIntensity <= 0.0)
    {
        *out_factor = 1.0f;
        return true;
    }

    static ShadowMap shadow;
    const Mesh cubeMesh{
        cubeVertices,
        8,
        cubeTriangles,
        12,
        {0.0, 0.0, 0.0},
        false
    };
    const double c = std::cos(rotationAngle);
    const double s = std::sin(rotationAngle);
    if (!build_shadow_map(shadow, light_dir, terrainBlocks, cubeMesh, c, s))
    {
        *out_factor = 1.0f;
        return false;
    }
    *out_factor = compute_shadow_factor(&shadow, light_dir, world, normalize_vec(normal));
    return true;
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
    camera_x.store(camera_x.load(std::memory_order_relaxed) + delta.x, std::memory_order_relaxed);
    camera_y.store(camera_y.load(std::memory_order_relaxed) + delta.y, std::memory_order_relaxed);
    camera_z.store(camera_z.load(std::memory_order_relaxed) + delta.z, std::memory_order_relaxed);
}

void render_move_camera_local(const Vec3 delta)
{
    const double yaw = camera_yaw.load(std::memory_order_relaxed);
    const double pitch = camera_pitch.load(std::memory_order_relaxed);
    const Vec3 rotated = rotate_yaw_pitch(delta, yaw, pitch);
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

    const double z = view.z <= 0.01 ? 0.01 : view.z;
    const double inv_z = 1.0 / z;
    const double fov = static_cast<double>(height) * 0.8;
    const double proj_x = view.x * inv_z * fov;
    const double proj_y = view.y * inv_z * fov;

    return {proj_x + static_cast<double>(width) / 2.0, proj_y + static_cast<double>(height) / 2.0};
}
