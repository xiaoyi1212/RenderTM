#include "render.h"
#include "blue_noise.h"
#include "cmath"
#include "cstring"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

const Vec3 cubeVertices[8] = {
    {-1, -1, -1}, {1, -1, -1}, {1,  1, -1}, {-1,  1, -1},
    {-1, -1,  1}, {1, -1,  1}, {1,  1,  1}, {-1,  1,  1}
};

enum FaceIndex
{
    FaceTop = 0,    // normal -Y
    FaceBottom = 1, // normal +Y
    FaceLeft = 2,   // normal -X
    FaceRight = 3,  // normal +X
    FaceBack = 4,   // normal -Z
    FaceFront = 5   // normal +Z
};

const int cubeFaceVertices[6][4] = {
    {0, 1, 5, 4}, // top (-y)
    {3, 2, 6, 7}, // bottom (+y)
    {0, 3, 7, 4}, // left (-x)
    {1, 2, 6, 5}, // right (+x)
    {0, 1, 2, 3}, // back (-z)
    {4, 5, 6, 7}  // front (+z)
};

const int cubeFaceQuadOrder[6][4] = {
    {0, 1, 5, 4}, // top (-y)
    {3, 7, 6, 2}, // bottom (+y)
    {0, 4, 7, 3}, // left (-x)
    {1, 2, 6, 5}, // right (+x)
    {0, 3, 2, 1}, // back (-z)
    {4, 5, 6, 7}  // front (+z)
};

const int cubeFaceNormal[6][3] = {
    {0, 1, 0},   // top (-y world, +y grid)
    {0, -1, 0},  // bottom (+y world, -y grid)
    {-1, 0, 0},  // left (-x)
    {1, 0, 0},   // right (+x)
    {0, 0, -1},  // back (-z)
    {0, 0, 1}    // front (+z)
};

const Vec3 cubeVerticesGrid[8] = {
    {0.0, 1.0, 0.0},
    {1.0, 1.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0},
    {0.0, 1.0, 1.0},
    {1.0, 1.0, 1.0},
    {1.0, 0.0, 1.0},
    {0.0, 0.0, 1.0}
};

static Vec3 face_normal_world(const int face)
{
    const double x = static_cast<double>(cubeFaceNormal[face][0]);
    const double y = static_cast<double>(-cubeFaceNormal[face][1]);
    const double z = static_cast<double>(cubeFaceNormal[face][2]);
    return {x, y, z};
}

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
static std::atomic<double> taaBlend{0.1};
static std::atomic<bool> taaClampEnabled{true};
static std::atomic<uint64_t> renderStateVersion{1};
static std::atomic<double> camera_x{16.0};
static std::atomic<double> camera_y{-19.72};
static std::atomic<double> camera_z{-1.93};
static std::atomic<double> camera_yaw{-0.6911503837897546};
static std::atomic<double> camera_pitch{-0.6003932626860493};
static std::atomic<bool> ambientOcclusionEnabled{true};
static std::atomic<bool> shadowEnabled{true};
static Mat4 currentVP{{{1.0, 0.0, 0.0, 0.0},
                       {0.0, 1.0, 0.0, 0.0},
                       {0.0, 0.0, 1.0, 0.0},
                       {0.0, 0.0, 0.0, 1.0}}};
static Mat4 previousVP{{{1.0, 0.0, 0.0, 0.0},
                        {0.0, 1.0, 0.0, 0.0},
                        {0.0, 0.0, 1.0, 0.0},
                        {0.0, 0.0, 0.0, 1.0}}};
static Mat4 inverseCurrentVP{{{1.0, 0.0, 0.0, 0.0},
                              {0.0, 1.0, 0.0, 0.0},
                              {0.0, 0.0, 1.0, 0.0},
                              {0.0, 0.0, 0.0, 1.0}}};

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

struct ColorRGB
{
    float r;
    float g;
    float b;
};

struct TopFaceLighting
{
    std::array<Vec3, 4> normals;
};

struct ShadingContext
{
    ColorRGB albedo;
    ColorRGB sky_top;
    ColorRGB sky_bottom;
    ColorRGB hemi_ground;
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
        ColorRGB color;
        double angular_radius;
    };
    std::array<DirectionalLightInfo, 2> lights;
};

constexpr double kShadowRayBias = 0.05;
constexpr double kPi = 3.14159265358979323846;
constexpr double kSunLatitudeDeg = 30.0;
constexpr double kSunLatitudeRad = kPi * kSunLatitudeDeg / 180.0;
constexpr double kSunDiskRadius = 0.03;
constexpr int kSunShadowSalt = 17;
constexpr int kMoonShadowSalt = 19;
constexpr uint32_t kSkySunriseTopColor = 0xFFB55A1A;
constexpr uint32_t kSkySunriseBottomColor = 0xFF4A200A;
constexpr double kHemisphereBounceStrength = 0.35;
constexpr ColorRGB kHemisphereBounceColorLinear{1.0f, 0.9046612f, 0.7758222f};
constexpr double kSkyLightHeightPower = 0.5;
constexpr ColorRGB kSunLightColorLinear{1.0f, 0.94f, 0.88f};
constexpr ColorRGB kMoonLightColorLinear{1.0f, 1.0f, 1.0f};
constexpr double kSunIntensityBoost = 1.2;
constexpr double kMoonSkyLightFloor = 0.22;
constexpr double kNearPlane = 0.05;
constexpr double kFarPlane = 1000.0;
constexpr int kTerrainChunkSize = 16;
constexpr double kTerrainBlockSize = 2.0;
constexpr double kTerrainStartZ = 4.0;
constexpr double kTerrainBaseY = 2.0;
constexpr size_t kSkyRayCount = 128;
constexpr double kSkyRayStep = 0.25;
constexpr double kSkyRayMaxDistance = 6.0;
constexpr double kSkyRayBias = 0.02;
constexpr double kSkyRayCenterBias = 0.02;
constexpr int kTaaJitterSalt = 37;
constexpr float kShadowFilterDepthSigma = 0.6f;
constexpr float kShadowFilterNormalSigma = 0.35f;
constexpr float kShadowGaussianWeights[9] = {
    1.0f, 2.0f, 1.0f,
    2.0f, 4.0f, 2.0f,
    1.0f, 2.0f, 1.0f
};

static Vec3 normalize_vec(const Vec3& v);
static float compute_shadow_factor(const Vec3& light_dir, const Vec3& world, const Vec3& normal);
static bool triangle_in_front_of_near_plane(double z0, double z1, double z2);
static void build_terrain_mesh();
static void build_light_basis(const Vec3& light_dir, Vec3& right, Vec3& up, Vec3& forward);
static Vec3 jitter_shadow_direction(const Vec3& light_dir, 
                                    const Vec3& right_scaled,
                                    const Vec3& up_scaled,
                                    const int px, const int py,
                                    const uint32_t frame, const int salt);
static float shadow_filter_3x3_at(const float* mask, const float* depth, const Vec3* normals,
                                  size_t width, size_t height, int x, int y, float depth_max);
static void filter_shadow_mask_3x3(const float* mask, float* out_mask, const float* depth,
                                   const Vec3* normals, size_t width, size_t height, float depth_max);
static Vec3 add_vec(const Vec3& a, const Vec3& b);
static double dot_vec(const Vec3& a, const Vec3& b);
static Vec3 cross_vec(const Vec3& a, const Vec3& b);

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
    TopFaceLighting topFace;
    std::array<std::array<Vec3, 4>, 6> face_normals;
    std::array<std::array<float, 4>, 6> face_sky_visibility;
};

struct RenderQuad
{
    Vec3 v[4];
    Vec3 n[4];
    std::array<float, 4> sky_visibility;
    uint32_t color;
};

static std::vector<VoxelBlock> terrainBlocks;
static int terrainSize = 0;
static int terrainMaxHeight = 0;
static std::vector<int> terrainHeights;
static std::vector<uint32_t> terrainTopColors;
static std::vector<int> terrainBlockIndex;
static std::vector<RenderQuad> terrainQuads;
static bool terrainMeshReady = false;
static size_t terrainVisibleFaces = 0;
static size_t terrainMeshTriangles = 0;

static bool terrain_has_block(const int gx, const int gy, const int gz)
{
    if (gx < 0 || gx >= terrainSize || gz < 0 || gz >= terrainSize)
    {
        return false;
    }
    if (gy < 0)
    {
        return false;
    }
    const size_t idx = static_cast<size_t>(gz * terrainSize + gx);
    return gy < terrainHeights[idx];
}

static size_t terrain_block_slot(const int gx, const int gy, const int gz)
{
    return (static_cast<size_t>(gz) * static_cast<size_t>(terrainMaxHeight) +
            static_cast<size_t>(gy)) * static_cast<size_t>(terrainSize) +
           static_cast<size_t>(gx);
}

static const VoxelBlock* terrain_block_at(const int gx, const int gy, const int gz)
{
    if (gx < 0 || gx >= terrainSize || gz < 0 || gz >= terrainSize)
    {
        return nullptr;
    }
    if (gy < 0 || gy >= terrainMaxHeight)
    {
        return nullptr;
    }
    const size_t slot = terrain_block_slot(gx, gy, gz);
    if (slot >= terrainBlockIndex.size())
    {
        return nullptr;
    }
    const int index = terrainBlockIndex[slot];
    if (index < 0 || static_cast<size_t>(index) >= terrainBlocks.size())
    {
        return nullptr;
    }
    return &terrainBlocks[static_cast<size_t>(index)];
}

static double radical_inverse_vdc(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<double>(bits) * 2.3283064365386963e-10;
}

static const std::array<Vec3, kSkyRayCount>& sky_sample_dirs()
{
    static const std::array<Vec3, kSkyRayCount> dirs = [] {
        std::array<Vec3, kSkyRayCount> samples{};
        for (size_t i = 0; i < kSkyRayCount; ++i)
        {
            const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(kSkyRayCount);
            const double v = radical_inverse_vdc(static_cast<uint32_t>(i));
            const double r = std::sqrt(u);
            const double theta = 2.0 * kPi * v;
            const double x = r * std::cos(theta);
            const double y = r * std::sin(theta);
            const double z = std::sqrt(std::max(0.0, 1.0 - u));
            samples[i] = {x, y, z};
        }
        return samples;
    }();
    return dirs;
}

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

static float compute_vertex_sky_visibility(const int gx, const int gy, const int gz,
                                           const int face, const int corner)
{
    Vec3 normal{
        static_cast<double>(cubeFaceNormal[face][0]),
        static_cast<double>(cubeFaceNormal[face][1]),
        static_cast<double>(cubeFaceNormal[face][2])
    };
    Vec3 tangent;
    Vec3 bitangent;
    Vec3 forward;
    build_light_basis(normal, tangent, bitangent, forward);
    normal = forward;

    const int vi = cubeFaceVertices[face][corner];
    const Vec3 offset = cubeVerticesGrid[vi];
    const Vec3 vertex{
        static_cast<double>(gx) + offset.x,
        static_cast<double>(gy) + offset.y,
        static_cast<double>(gz) + offset.z
    };
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
            if (terrain_has_block(vx, vy, vz))
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

    const double visibility = 1.0 - static_cast<double>(occluded) /
                                      static_cast<double>(samples.size());
    return static_cast<float>(std::clamp(visibility, 0.0, 1.0));
}

static void build_terrain_chunk()
{
    const int chunk_size = kTerrainChunkSize;
    const int base_height = 4;
    const int dirt_thickness = 2;
    const int height_variation = 6;
    const double height_freq = 0.12;
    const double surface_freq = 0.4;

    const double block_size = kTerrainBlockSize;
    const double start_x = -(chunk_size - 1) * block_size * 0.5;
    const double start_z = kTerrainStartZ;
    const double base_y = kTerrainBaseY;

    const uint32_t stone_color = 0xFF7A7A7A;
    const uint32_t dirt_color = 0xFF7D4714;
    const uint32_t grass_color = 0xFF3B8A38;
    const uint32_t water_color = 0xFF2B5FA8;

    terrainSize = chunk_size;
    terrainHeights.assign(static_cast<size_t>(chunk_size * chunk_size), 0);
    terrainTopColors.assign(static_cast<size_t>(chunk_size * chunk_size), grass_color);

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
            int height = base_height + static_cast<int>(((h + 1.0) * 0.5 * height_variation) + 0.5);
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

    terrainBlocks.clear();
    terrainMaxHeight = 0;
    for (int value : terrainHeights)
    {
        if (value > terrainMaxHeight)
        {
            terrainMaxHeight = value;
        }
    }
    if (terrainMaxHeight < 0)
    {
        terrainMaxHeight = 0;
    }
    terrainBlockIndex.assign(static_cast<size_t>(terrainSize * std::max(terrainMaxHeight, 1) * terrainSize), -1);
    terrainMeshReady = false;
    const Vec3 top_normal = face_normal_world(FaceTop);
    const TopFaceLighting empty_top{
        {top_normal, top_normal, top_normal, top_normal}
    };

    for (int z = 0; z < chunk_size; ++z)
    {
        for (int x = 0; x < chunk_size; ++x)
        {
            const int height = terrainHeights[index(x, z)];
            const uint32_t top_color = terrainTopColors[index(x, z)];
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

                std::array<std::array<Vec3, 4>, 6> face_normals{};
                for (int face = 0; face < 6; ++face)
                {
                    const Vec3 base = face_normal_world(face);
                    for (int corner = 0; corner < 4; ++corner)
                    {
                        face_normals[face][corner] = base;
                    }
                }
                std::array<std::array<float, 4>, 6> face_sky_visibility{};
                for (auto& face_visibility : face_sky_visibility)
                {
                    face_visibility.fill(0.0f);
                }
                for (int face = 0; face < 6; ++face)
                {
                    const int nx = x + cubeFaceNormal[face][0];
                    const int ny = y + cubeFaceNormal[face][1];
                    const int nz = z + cubeFaceNormal[face][2];
                    if (!terrain_has_block(nx, ny, nz))
                    {
                        for (int corner = 0; corner < 4; ++corner)
                        {
                            face_sky_visibility[face][corner] = compute_vertex_sky_visibility(x, y, z, face, corner);
                        }
                    }
                }
                terrainBlocks.push_back({
                    {start_x + x * block_size, base_y - y * block_size, start_z + z * block_size},
                    color,
                    empty_top,
                    face_normals,
                    face_sky_visibility
                });
                const size_t block_index = terrainBlocks.size() - 1;
                const size_t slot = terrain_block_slot(x, y, z);
                if (slot < terrainBlockIndex.size())
                {
                    terrainBlockIndex[slot] = static_cast<int>(block_index);
                }
            }
        }
    }

    build_terrain_mesh();
}

static void generate_terrain_chunk()
{
    static std::once_flag terrain_init_flag;
    std::call_once(terrain_init_flag, build_terrain_chunk);
}

static void emit_block_face_quad(const VoxelBlock& block, const int face)
{
    RenderQuad quad{};
    quad.color = block.color;
    auto corner_index = [&](const int vertex_index) -> int {
        for (int i = 0; i < 4; ++i)
        {
            if (cubeFaceVertices[face][i] == vertex_index)
            {
                return i;
            }
        }
        return -1;
    };
    for (int corner = 0; corner < 4; ++corner)
    {
        const int vi = cubeFaceQuadOrder[face][corner];
        quad.v[corner] = add_vec(block.position, cubeVertices[vi]);
        const int face_corner = corner_index(vi);
        const int attr_corner = face_corner < 0 ? corner : face_corner;
        quad.sky_visibility[corner] = block.face_sky_visibility[static_cast<size_t>(face)][static_cast<size_t>(attr_corner)];
        if (face == FaceTop)
        {
            quad.n[corner] = block.topFace.normals[static_cast<size_t>(attr_corner)];
        }
        else
        {
            quad.n[corner] = block.face_normals[static_cast<size_t>(face)][static_cast<size_t>(attr_corner)];
        }
    }
    terrainQuads.push_back(quad);
}

static void build_terrain_mesh()
{
    if (terrainMeshReady)
    {
        return;
    }
    terrainMeshReady = true;
    terrainQuads.clear();
    terrainVisibleFaces = 0;
    terrainMeshTriangles = 0;

    if (terrainSize <= 0 || terrainMaxHeight <= 0)
    {
        return;
    }

    static constexpr int face_order[6] = {
        FaceFront,
        FaceBack,
        FaceLeft,
        FaceRight,
        FaceBottom,
        FaceTop
    };
    for (int z = 0; z < terrainSize; ++z)
    {
        for (int x = 0; x < terrainSize; ++x)
        {
            const int height = terrainHeights[static_cast<size_t>(z * terrainSize + x)];
            for (int y = 0; y < height; ++y)
            {
                const VoxelBlock* block = terrain_block_at(x, y, z);
                if (!block)
                {
                    continue;
                }
                for (int i = 0; i < 6; ++i)
                {
                    const int face = face_order[i];
                    const int nx = x + cubeFaceNormal[face][0];
                    const int ny = y + cubeFaceNormal[face][1];
                    const int nz = z + cubeFaceNormal[face][2];
                    if (!terrain_has_block(nx, ny, nz))
                    {
                        terrainVisibleFaces++;
                        emit_block_face_quad(*block, face);
                    }
                }
            }
        }
    }

    terrainMeshTriangles = terrainQuads.size() * 2;
}

static ColorRGB lerp_color(const ColorRGB& a, const ColorRGB& b, float t)
{
    return {
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t
    };
}

static Vec3 lerp_vec3(const Vec3& a, const Vec3& b, const double t)
{
    return {
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    };
}

static ColorRGB sample_bilinear_history(const ColorRGB* buffer, const size_t width, const size_t height,
                                        const double screen_x, const double screen_y)
{
    if (!buffer || width == 0 || height == 0)
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
    const ColorRGB c00 = buffer[row0 + static_cast<size_t>(x0)];
    const ColorRGB c10 = buffer[row0 + static_cast<size_t>(x1)];
    const ColorRGB c01 = buffer[row1 + static_cast<size_t>(x0)];
    const ColorRGB c11 = buffer[row1 + static_cast<size_t>(x1)];
    const ColorRGB top = lerp_color(c00, c10, fx);
    const ColorRGB bottom = lerp_color(c01, c11, fx);
    return lerp_color(top, bottom, fy);
}

static Vec3 sample_bilinear_history_vec3(const Vec3* buffer, const size_t width, const size_t height,
                                         const double screen_x, const double screen_y)
{
    if (!buffer || width == 0 || height == 0)
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

static ColorRGB scale_color(const ColorRGB& color, float scale)
{
    return {color.r * scale, color.g * scale, color.b * scale};
}

static ColorRGB add_color(const ColorRGB& a, const ColorRGB& b)
{
    return {a.r + b.r, a.g + b.g, a.b + b.b};
}

static ColorRGB compute_hemisphere_ground(const ColorRGB& base_ground,
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

static Vec3 normalize_vec(const Vec3& v)
{
    const double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len == 0.0) return {0.0, 0.0, 0.0};
    return {v.x / len, v.y / len, v.z / len};
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
        a.view.x + (b.view.x - a.view.x) * t,
        a.view.y + (b.view.y - a.view.y) * t,
        a.view.z + (b.view.z - a.view.z) * t
    };
    const Vec3 world{
        a.world.x + (b.world.x - a.world.x) * t,
        a.world.y + (b.world.y - a.world.y) * t,
        a.world.z + (b.world.z - a.world.z) * t
    };
    Vec3 normal{
        a.normal.x + (b.normal.x - a.normal.x) * t,
        a.normal.y + (b.normal.y - a.normal.y) * t,
        a.normal.z + (b.normal.z - a.normal.z) * t
    };
    normal = normalize_vec(normal);
    const float visibility = a.sky_visibility + (b.sky_visibility - a.sky_visibility) * static_cast<float>(t);
    return {view, world, normal, visibility};
}

static size_t clip_triangle_to_near_plane(const ClipVertex* input, const size_t input_count,
                                          ClipVertex* output, const size_t max_output)
{
    if (!input || !output || input_count == 0 || max_output == 0)
    {
        return 0;
    }
    size_t out_count = 0;
    ClipVertex prev = input[input_count - 1];
    bool prev_inside = prev.view.z >= kNearPlane;

    for (size_t i = 0; i < input_count; ++i)
    {
        const ClipVertex cur = input[i];
        const bool cur_inside = cur.view.z >= kNearPlane;

        if (cur_inside)
        {
            if (!prev_inside)
            {
                const double t = (kNearPlane - prev.view.z) / (cur.view.z - prev.view.z);
                if (out_count < max_output)
                {
                    output[out_count++] = clip_lerp(prev, cur, t);
                }
            }
            if (out_count < max_output)
            {
                output[out_count++] = cur;
            }
        }
        else if (prev_inside)
        {
            const double t = (kNearPlane - prev.view.z) / (cur.view.z - prev.view.z);
            if (out_count < max_output)
            {
                output[out_count++] = clip_lerp(prev, cur, t);
            }
        }

        prev = cur;
        prev_inside = cur_inside;
    }

    return out_count;
}

static Vec3 add_vec(const Vec3& a, const Vec3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
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

static ColorRGB unpack_color(uint32_t color)
{
    return {
        static_cast<float>((color >> 16) & 0xFF),
        static_cast<float>((color >> 8) & 0xFF),
        static_cast<float>(color & 0xFF)
    };
}

static float srgb_channel_to_linear(const float channel)
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

static ColorRGB srgb_to_linear(const ColorRGB& color)
{
    return {
        srgb_channel_to_linear(color.r),
        srgb_channel_to_linear(color.g),
        srgb_channel_to_linear(color.b)
    };
}

static ColorRGB linear_to_srgb(const ColorRGB& color)
{
    return {
        linear_channel_to_srgb(color.r) * 255.0f,
        linear_channel_to_srgb(color.g) * 255.0f,
        linear_channel_to_srgb(color.b) * 255.0f
    };
}

static float tonemap_reinhard_channel(float value)
{
    if (value <= 0.0f)
    {
        return 0.0f;
    }
    return value / (1.0f + value);
}

static ColorRGB tonemap_reinhard(const ColorRGB& color, const float exposure_factor)
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

static Mat4 mat4_identity()
{
    Mat4 m{};
    m.m[0][0] = 1.0;
    m.m[1][1] = 1.0;
    m.m[2][2] = 1.0;
    m.m[3][3] = 1.0;
    return m;
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

static bool mat4_invert(const Mat4& m, Mat4* out)
{
    if (!out)
    {
        return false;
    }
    double aug[4][8]{};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            aug[i][j] = m.m[i][j];
        }
        for (int j = 0; j < 4; ++j)
        {
            aug[i][j + 4] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < 4; ++col)
    {
        int pivot = col;
        double max_abs = std::fabs(aug[col][col]);
        for (int row = col + 1; row < 4; ++row)
        {
            const double value = std::fabs(aug[row][col]);
            if (value > max_abs)
            {
                max_abs = value;
                pivot = row;
            }
        }
        if (max_abs < 1e-12)
        {
            return false;
        }
        if (pivot != col)
        {
            for (int j = 0; j < 8; ++j)
            {
                std::swap(aug[col][j], aug[pivot][j]);
            }
        }

        const double inv_pivot = 1.0 / aug[col][col];
        for (int j = 0; j < 8; ++j)
        {
            aug[col][j] *= inv_pivot;
        }

        for (int row = 0; row < 4; ++row)
        {
            if (row == col)
            {
                continue;
            }
            const double factor = aug[row][col];
            if (factor == 0.0)
            {
                continue;
            }
            for (int j = 0; j < 8; ++j)
            {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            out->m[i][j] = aug[i][j + 4];
        }
    }
    return true;
}

static Mat4 make_view_matrix(const Vec3& pos, const double yaw, const double pitch)
{
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);

    Mat4 m = mat4_identity();
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

static Mat4 make_projection_matrix(const double width, const double height,
                                   const double fov_x, const double fov_y)
{
    if (width <= 0.0 || height <= 0.0 || kFarPlane <= kNearPlane)
    {
        return mat4_identity();
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

static void draw_shaded_triangle(float* zbuffer, ColorRGB* sample_ambient,
                                 ColorRGB* sample_direct_sun, ColorRGB* sample_direct_moon,
                                 float* shadow_mask_sun, float* shadow_mask_moon,
                                 Vec3* sample_normals, size_t width, size_t height,
                                 const ScreenVertex& v0, const ScreenVertex& v1, const ScreenVertex& v2,
                                 const Vec3& wp0, const Vec3& wp1, const Vec3& wp2,
                                 const Vec3& n0, const Vec3& n1, const Vec3& n2,
                                 const float vis0, const float vis1, const float vis2,
                                 const ShadingContext& ctx, const uint32_t frame_index,
                                 const float jitter_x, const float jitter_y,
                                 const std::array<Vec3, 2>& lights_right_scaled,
                                 const std::array<Vec3, 2>& lights_up_scaled)
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
    const float inv_z0 = 1.0f / v0.z;
    const float inv_z1 = 1.0f / v1.z;
    const float inv_z2 = 1.0f / v2.z;

    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            ScreenVertex p{
                static_cast<float>(x) + 0.5f + jitter_x,
                static_cast<float>(y) + 0.5f + jitter_y,
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
                const float inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2;
                if (inv_z <= 0.0f)
                {
                    continue;
                }
                const float depth = 1.0f / inv_z;
                const size_t idx = static_cast<size_t>(y) * width + static_cast<size_t>(x);
                if (depth < zbuffer[idx])
                {
                    zbuffer[idx] = depth;
                    const float w0p = w0 * inv_z0 / inv_z;
                    const float w1p = w1 * inv_z1 / inv_z;
                    const float w2p = w2 * inv_z2 / inv_z;
                    const float visibility = ctx.ambient_occlusion_enabled
                                                 ? std::clamp(w0p * vis0 + w1p * vis1 + w2p * vis2, 0.0f, 1.0f)
                                                 : 1.0f;
                    Vec3 normal{
                        w0p * n0.x + w1p * n1.x + w2p * n2.x,
                        w0p * n0.y + w1p * n1.y + w2p * n2.y,
                        w0p * n0.z + w1p * n1.z + w2p * n2.z
                    };
                    normal = normalize_vec(normal);

                    Vec3 world{
                        w0p * wp0.x + w1p * wp1.x + w2p * wp2.x,
                        w0p * wp0.y + w1p * wp1.y + w2p * wp2.y,
                        w0p * wp0.z + w1p * wp1.z + w2p * wp2.z
                    };

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
                        const ColorRGB sky = lerp_color(ctx.hemi_ground, ctx.sky_top, sky_t);
                        ambient_color.r = sky.r * ctx.sky_scale * ctx.albedo.r;
                        ambient_color.g = sky.g * ctx.sky_scale * ctx.albedo.g;
                        ambient_color.b = sky.b * ctx.sky_scale * ctx.albedo.b;
                    }
                    ambient_color.r *= visibility;
                    ambient_color.g *= visibility;
                    ambient_color.b *= visibility;

                    ColorRGB direct_sun{0.0f, 0.0f, 0.0f};
                    ColorRGB direct_moon{0.0f, 0.0f, 0.0f};
                    float shadow_sun = 1.0f;
                    float shadow_moon = 1.0f;
                    if (ctx.direct_lighting_enabled)
                    {
                        const Vec3 view_vec{
                            ctx.camera_pos.x - world.x,
                            ctx.camera_pos.y - world.y,
                            ctx.camera_pos.z - world.z
                        };
                        const Vec3 view_dir = normalize_vec(view_vec);

                        auto eval_light = [&](const ShadingContext::DirectionalLightInfo& light,
                                              const int light_idx, const int shadow_salt,
                                              ColorRGB& out_direct, float& out_shadow) {
                            out_direct = {0.0f, 0.0f, 0.0f};
                            out_shadow = 1.0f;
                            if (light.intensity <= 0.0)
                            {
                                return;
                            }
                            const double ndotl = std::max(0.0, dot_vec(normal, light.dir));
                            if (ndotl <= 0.0)
                            {
                                return;
                            }
                            if (ctx.shadows_enabled)
                            {
                                const Vec3 shadow_dir = jitter_shadow_direction(light.dir,
                                                        lights_right_scaled[light_idx],
                                                        lights_up_scaled[light_idx],
                                                        x, y, frame_index,
                                                        shadow_salt);
                                out_shadow = compute_shadow_factor(shadow_dir, world, normal);
                            }

                            const Vec3 half_vec = normalize_vec(add_vec(light.dir, view_dir));
                            const double f0 = std::clamp(ctx.material.specular, 0.0, 1.0);
                            const double vdoth = std::max(0.0, dot_vec(view_dir, half_vec));
                            const double fresnel = schlick_fresnel(vdoth, f0);
                            const double diffuse_scale = std::clamp(1.0 - fresnel, 0.0, 1.0);
                            const double diffuse = ndotl * light.intensity * ctx.material.diffuse * diffuse_scale;
                            ColorRGB light_color{
                                static_cast<float>(ctx.albedo.r * diffuse),
                                static_cast<float>(ctx.albedo.g * diffuse),
                                static_cast<float>(ctx.albedo.b * diffuse)
                            };
                            light_color.r *= light.color.r;
                            light_color.g *= light.color.g;
                            light_color.b *= light.color.b;
                            if (f0 > 0.0)
                            {
                                const double spec_dot = std::max(0.0, dot_vec(normal, half_vec));
                                double spec = eval_specular_term(spec_dot, vdoth, ndotl,
                                                                 ctx.material.shininess, f0);
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

static Vec3 jitter_shadow_direction(const Vec3& light_dir, 
                                    const Vec3& right_scaled,
                                    const Vec3& up_scaled,
                                    const int px, const int py,
                                    const uint32_t frame, const int salt)
{
    if (right_scaled.x == 0.0 && right_scaled.y == 0.0 && right_scaled.z == 0.0)
    {
        return light_dir;
    }

    const float u1 = sample_noise(px, py, static_cast<int>(frame), salt);
    const float u2 = sample_noise(px, py, static_cast<int>(frame), salt + 1);
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

static bool shadow_raymarch_hit(const Vec3& world, const Vec3& normal, const Vec3& light_dir)
{
    if (terrainSize <= 0 || terrainMaxHeight <= 0)
    {
        return false;
    }

    const double block_size = kTerrainBlockSize;
    const double half = block_size * 0.5;
    const double start_x = -(terrainSize - 1) * block_size * 0.5;
    const double start_z = kTerrainStartZ;
    const double base_y = kTerrainBaseY;
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

    if (x < 0 || x >= terrainSize || z < 0 || z >= terrainSize || y < 0 || y >= terrainMaxHeight)
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

    const int max_steps = (terrainSize + terrainSize + terrainMaxHeight) * 4;
    bool skip_first = true;

    for (int i = 0; i < max_steps; ++i)
    {
        if (!skip_first && terrain_has_block(x, y, z))
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

        if (x < 0 || x >= terrainSize || z < 0 || z >= terrainSize || y < 0 || y >= terrainMaxHeight)
        {
            return false;
        }
    }
    return false;
}

static float compute_shadow_factor(const Vec3& light_dir, const Vec3& world, const Vec3& normal)
{
    const double ndotl = std::max(0.0, dot_vec(normal, light_dir));
    if (ndotl <= 0.0)
    {
        return 1.0f;
    }
    return shadow_raymarch_hit(world, normal, light_dir) ? 0.0f : 1.0f;
}

static float shadow_filter_3x3_at(const float* mask, const float* depth, const Vec3* normals,
                                  const size_t width, const size_t height,
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

    const float inv_depth_sigma2 = 1.0f / (2.0f * kShadowFilterDepthSigma * kShadowFilterDepthSigma);
    const float inv_normal_sigma2 = 1.0f / (2.0f * kShadowFilterNormalSigma * kShadowFilterNormalSigma);
    float sum = 0.0f;
    float weight_sum = 0.0f;
    int k = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        const int sy = std::clamp(iy + dy, 0, static_cast<int>(height) - 1);
        for (int dx = -1; dx <= 1; ++dx, ++k)
        {
            const int sx = std::clamp(ix + dx, 0, static_cast<int>(width) - 1);
            const size_t idx = static_cast<size_t>(sy) * width + static_cast<size_t>(sx);
            const float neighbor_depth = depth[idx];
            if (neighbor_depth >= depth_max)
            {
                continue;
            }
            const Vec3 neighbor_normal = normals[idx];
            const double neighbor_len_sq = neighbor_normal.x * neighbor_normal.x +
                                           neighbor_normal.y * neighbor_normal.y +
                                           neighbor_normal.z * neighbor_normal.z;
            if (neighbor_len_sq <= 1e-6)
            {
                continue;
            }

            float weight = kShadowGaussianWeights[k];
            const float depth_diff = neighbor_depth - center_depth;
            const float depth_w = std::exp(-(depth_diff * depth_diff) * inv_depth_sigma2);
            const float dot = static_cast<float>(center_normal.x * neighbor_normal.x +
                                                 center_normal.y * neighbor_normal.y +
                                                 center_normal.z * neighbor_normal.z);
            const float clamped_dot = std::clamp(dot, -1.0f, 1.0f);
            const float normal_diff = 1.0f - clamped_dot;
            const float normal_w = std::exp(-(normal_diff * normal_diff) * inv_normal_sigma2);
            weight *= depth_w * normal_w;
            sum += mask[idx] * weight;
            weight_sum += weight;
        }
    }

    if (weight_sum <= 0.0f)
    {
        return mask[center_idx];
    }
    float filtered = sum / weight_sum;
    filtered = std::clamp(filtered, 0.0f, 1.0f);
    return filtered;
}

static void filter_shadow_mask_3x3(const float* mask, float* out_mask, const float* depth,
                                   const Vec3* normals, const size_t width, const size_t height,
                                   const float depth_max)
{
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            out_mask[idx] = shadow_filter_3x3_at(mask, depth, normals,
                                                 width, height,
                                                 static_cast<int>(x),
                                                 static_cast<int>(y),
                                                 depth_max);
        }
    }
}

static void render_quad(float* zbuffer, ColorRGB* sample_ambient,
                        ColorRGB* sample_direct_sun, ColorRGB* sample_direct_moon,
                        float* shadow_mask_sun, float* shadow_mask_moon, Vec3* sample_normals,
                        size_t width, size_t height,
                        const RenderQuad& quad, const double fov_x, const double fov_y,
                        const Vec3& camera_pos, const ViewRotation& view_rot, const ShadingContext& ctx,
                        const uint32_t frame_index, const float jitter_x, const float jitter_y,
                        const std::array<Vec3, 2>& lights_right_scaled,
                        const std::array<Vec3, 2>& lights_up_scaled)

{
    Vec3 view_space[4];
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
                             shadow_mask_sun, shadow_mask_moon, sample_normals,
                             width, height,
                             sv0, sv1, sv2,
                             a.world, b.world, c.world,
                             a.normal, b.normal, c.normal,
                             a.sky_visibility, b.sky_visibility, c.sky_visibility,
                             ctx, frame_index,
                             jitter_x, jitter_y,
                             lights_right_scaled, lights_up_scaled);
    };

    auto draw_clipped = [&](int i0, int i1, int i2) {
        ClipVertex input[3]{
            {view_space[i0], quad.v[i0], quad.n[i0], quad.sky_visibility[i0]},
            {view_space[i1], quad.v[i1], quad.n[i1], quad.sky_visibility[i1]},
            {view_space[i2], quad.v[i2], quad.n[i2], quad.sky_visibility[i2]}
        };
        ClipVertex clipped[4]{};
        const size_t clipped_count = clip_triangle_to_near_plane(input, 3, clipped, 4);
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

void render_update_array(uint32_t* framebuffer, size_t width, size_t height)
{
    static size_t cached_width = 0;
    static size_t cached_height = 0;
    static std::vector<float> zbuffer;
    static std::vector<ColorRGB> sample_colors;
    static std::vector<ColorRGB> sample_direct;
    static std::vector<ColorRGB> sample_direct_sun;
    static std::vector<ColorRGB> sample_direct_moon;
    static std::vector<float> shadow_mask_sun;
    static std::vector<float> shadow_mask_moon;
    static std::vector<float> shadow_mask_filtered_sun;
    static std::vector<float> shadow_mask_filtered_moon;
    static std::vector<Vec3> sample_normals;
    static std::vector<ColorRGB> taa_history;
    static size_t taa_width = 0;
    static size_t taa_height = 0;
    static bool taa_history_valid = false;
    static uint64_t taa_state_version = 0;
    static bool taa_was_enabled = false;
    const size_t sample_count = width * height;
    const float depth_max = std::numeric_limits<float>::max();

    const bool taa_on = taaEnabled.load(std::memory_order_relaxed);
    const float base_blend = static_cast<float>(std::clamp(taaBlend.load(std::memory_order_relaxed), 0.0, 1.0));
    const float taa_factor = base_blend;
    const bool clamp_history = taaClampEnabled.load(std::memory_order_relaxed);
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
    }
    else
    {
        std::fill(zbuffer.begin(), zbuffer.end(), depth_max);
        std::fill(sample_colors.begin(), sample_colors.end(), ColorRGB{0.0f, 0.0f, 0.0f});
        std::fill(sample_direct.begin(), sample_direct.end(), ColorRGB{0.0f, 0.0f, 0.0f});
        std::fill(sample_direct_sun.begin(), sample_direct_sun.end(), ColorRGB{0.0f, 0.0f, 0.0f});
        std::fill(sample_direct_moon.begin(), sample_direct_moon.end(), ColorRGB{0.0f, 0.0f, 0.0f});
        std::fill(shadow_mask_sun.begin(), shadow_mask_sun.end(), 1.0f);
        std::fill(shadow_mask_moon.begin(), shadow_mask_moon.end(), 1.0f);
        std::fill(shadow_mask_filtered_sun.begin(), shadow_mask_filtered_sun.end(), 1.0f);
        std::fill(shadow_mask_filtered_moon.begin(), shadow_mask_filtered_moon.end(), 1.0f);
        std::fill(sample_normals.begin(), sample_normals.end(), Vec3{0.0, 0.0, 0.0});
    }

    if (taa_on)
    {
        if (width != taa_width || height != taa_height || !taa_was_enabled || taa_state_version != state_version)
        {
            taa_width = width;
            taa_height = height;
            taa_history.assign(sample_count, {0.0f, 0.0f, 0.0f});
            taa_history_valid = false;
            taa_state_version = state_version;
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

    const double fov_y = static_cast<double>(height) * 0.8;
    const double fov_x = fov_y;
    const Vec3 camera_pos{
        camera_x.load(std::memory_order_relaxed),
        camera_y.load(std::memory_order_relaxed),
        camera_z.load(std::memory_order_relaxed)
    };
    const double yaw = camera_yaw.load(std::memory_order_relaxed);
    const double pitch = camera_pitch.load(std::memory_order_relaxed);
    const double view_yaw = -yaw;
    const double view_pitch = -pitch;
    const Mat4 view = make_view_matrix(camera_pos, view_yaw, view_pitch);
    const Mat4 proj = make_projection_matrix(static_cast<double>(width),
                                             static_cast<double>(height),
                                             fov_x, fov_y);
    currentVP = mat4_multiply(proj, view);
    if (!mat4_invert(currentVP, &inverseCurrentVP))
    {
        inverseCurrentVP = mat4_identity();
    }

    float jitter_x = 0.0f;
    float jitter_y = 0.0f;
    if (taa_on)
    {
        const float jitter_scale = 1.0f;
        const float u = sample_noise(0, 0, static_cast<int>(frame_index), kTaaJitterSalt);
        const float v = sample_noise(1, 0, static_cast<int>(frame_index), kTaaJitterSalt + 1);
        jitter_x = (u - 0.5f) * jitter_scale;
        jitter_y = (v - 0.5f) * jitter_scale;
    }
    const ViewRotation view_rot = make_view_rotation(view_yaw, view_pitch);

    const double mat_ambient = 0.25;
    const double mat_diffuse = 1.0;
    const Material cube_material{0xFFFFFFFF, mat_ambient, mat_diffuse, 0.15, 24.0};

    generate_terrain_chunk();

    const bool sun_orbit = sunOrbitEnabled.load(std::memory_order_relaxed);
    Vec3 sun_dir = normalize_vec(load_light_direction());
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

    ColorRGB sky_top = unpack_color(skyTopColor.load(std::memory_order_relaxed));
    ColorRGB sky_bottom = unpack_color(skyBottomColor.load(std::memory_order_relaxed));
    double sun_height = 1.0;
    if (sun_orbit)
    {
        sun_height = compute_sun_height(sun_dir);
        const float t = static_cast<float>(sun_height);
        const ColorRGB sunrise_top = unpack_color(kSkySunriseTopColor);
        const ColorRGB sunrise_bottom = unpack_color(kSkySunriseBottomColor);
        sky_top = lerp_color(sunrise_top, sky_top, t);
        sky_bottom = lerp_color(sunrise_bottom, sky_bottom, t);
    }

    const ColorRGB sky_top_linear = srgb_to_linear(sky_top);
    const ColorRGB sky_bottom_linear = srgb_to_linear(sky_bottom);
    const double moon_intensity = moonIntensity.load(std::memory_order_relaxed);
    double effective_sky_intensity = skyLightIntensity.load(std::memory_order_relaxed);
    if (sun_orbit)
    {
        const double sun_factor = std::pow(std::clamp(sun_height, 0.0, 1.0), kSkyLightHeightPower);
        effective_sky_intensity *= sun_factor;

        const double moon_factor = std::clamp(moon_intensity, 0.0, 1.0) * kMoonSkyLightFloor * (1.0 - sun_factor);
        effective_sky_intensity = std::min(1.0, effective_sky_intensity + moon_factor);
    }

    const Vec3 moon_dir = normalize_vec(load_moon_direction());
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
            Vec3 right, up, forward;
            build_light_basis(lights[i].dir, right, up, forward);
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

    const bool direct_lighting_enabled = (lights[0].intensity > 0.0) || (lights[1].intensity > 0.0);
    const bool ao_enabled = ambientOcclusionEnabled.load(std::memory_order_relaxed);
    const float sky_scale = static_cast<float>(std::clamp(effective_sky_intensity, 0.0, 1.0));
    const ColorRGB hemi_ground = compute_hemisphere_ground(sky_bottom_linear, lights);
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
        const ColorRGB albedo_srgb = unpack_color(material.color);
        const ColorRGB albedo = srgb_to_linear(albedo_srgb);
        CachedContext entry{};
        entry.color = color;
        entry.ctx = {
            albedo,
            sky_top_linear,
            sky_bottom_linear,
            hemi_ground,
            sky_scale,
            camera_pos,
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

    for (const auto& quad : terrainQuads)
    {
        ShadingContext& ctx = get_ctx(quad.color);
        render_quad(zbuffer.data(), sample_colors.data(),
                    sample_direct_sun.data(), sample_direct_moon.data(),
                    shadow_mask_sun.data(), shadow_mask_moon.data(),
                    sample_normals.data(),
                    width, height, quad, fov_x, fov_y,
                    camera_pos, view_rot, ctx, frame_index, jitter_x, jitter_y,
                    lights_right_scaled, lights_up_scaled); 
    }

    if (shadows_on)
    {
        filter_shadow_mask_3x3(shadow_mask_sun.data(), shadow_mask_filtered_sun.data(),
                               zbuffer.data(), sample_normals.data(),
                               width, height, depth_max);
        filter_shadow_mask_3x3(shadow_mask_moon.data(), shadow_mask_filtered_moon.data(),
                               zbuffer.data(), sample_normals.data(),
                               width, height, depth_max);
        for (size_t i = 0; i < sample_count; ++i)
        {
            const ColorRGB sun = sample_direct_sun[i];
            const ColorRGB moon = sample_direct_moon[i];
            const ColorRGB sun_shadowed = scale_color(sun, shadow_mask_filtered_sun[i]);
            const ColorRGB moon_shadowed = scale_color(moon, shadow_mask_filtered_moon[i]);
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

    static constexpr int bayer4[4][4] = {
        {0, 8, 2, 10},
        {12, 4, 14, 6},
        {3, 11, 1, 9},
        {15, 7, 13, 5}
    };
    const float dither_strength = 2.0f;
    const float dither_scale = dither_strength / 16.0f;

    const float exposure_factor = static_cast<float>(std::max(0.0, exposure.load(std::memory_order_relaxed)));
    const bool use_history = taa_on && taa_history_valid;

    auto current_linear_at = [&](int ix, int iy) -> ColorRGB {
        ix = std::clamp(ix, 0, static_cast<int>(width) - 1);
        iy = std::clamp(iy, 0, static_cast<int>(height) - 1);
        const size_t idx = static_cast<size_t>(iy) * width + static_cast<size_t>(ix);
        if (zbuffer[idx] >= depth_max)
        {
            const float t = height > 1 ? static_cast<float>(iy) / static_cast<float>(height - 1) : 0.0f;
            return lerp_color(sky_top_linear, sky_bottom_linear, t);
        }
        ColorRGB accum = sample_colors[idx];
        const ColorRGB direct = sample_direct[idx];
        accum.r += direct.r;
        accum.g += direct.g;
        accum.b += direct.b;
        return accum;
    };

    for (size_t y = 0; y < height; ++y)
    {
        const float sky_t = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
        const ColorRGB sky_row_linear = lerp_color(sky_top_linear, sky_bottom_linear, sky_t);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t pixel = y * width + x;
            const float depth = zbuffer[pixel];
            const bool is_sky = depth >= depth_max;
            ColorRGB current_linear = sky_row_linear;
            if (!is_sky)
            {
                ColorRGB accum = sample_colors[pixel];
                const ColorRGB direct = sample_direct[pixel];
                accum.r += direct.r;
                accum.g += direct.g;
                accum.b += direct.b;
                current_linear = accum;
            }

            ColorRGB blended = current_linear;
            if (taa_on)
            {
                bool history_valid = use_history;
                ColorRGB prev = current_linear;
                if (history_valid)
                {
                    prev = taa_history[pixel];
                    if (!is_sky)
                    {
                        const double screen_x = static_cast<double>(x) + 0.5;
                        const double screen_y = static_cast<double>(y) + 0.5;
                        const Vec3 world = render_unproject_point({screen_x, screen_y, depth}, width, height);
                        const Vec2 prev_screen = render_reproject_point(world, width, height);
                        if (std::isfinite(prev_screen.x) && std::isfinite(prev_screen.y) &&
                            prev_screen.x >= 0.0 && prev_screen.x <= static_cast<double>(width) &&
                            prev_screen.y >= 0.0 && prev_screen.y <= static_cast<double>(height))
                        {
                            prev = sample_bilinear_history(taa_history.data(), width, height,
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
                    ColorRGB minc = current_linear;
                    ColorRGB maxc = current_linear;
                    for (int ny = -1; ny <= 1; ++ny)
                    {
                        for (int nx = -1; nx <= 1; ++nx)
                        {
                            const ColorRGB neighbor = current_linear_at(static_cast<int>(x) + nx,
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
                }
                else
                {
                    blended = current_linear;
                }

                taa_history[pixel] = blended;
            }

            const ColorRGB mapped = tonemap_reinhard(blended, exposure_factor);
            ColorRGB srgb = linear_to_srgb(mapped);
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
    const ColorRGB input{
        static_cast<float>(color.x),
        static_cast<float>(color.y),
        static_cast<float>(color.z)
    };
    const ColorRGB mapped = tonemap_reinhard(input, exposure_factor);
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
    return sample_bilinear_history_vec3(buffer, width, height, screen_coord.x, screen_coord.y);
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
    generate_terrain_chunk();
    if (face < 0 || face >= 6 || corner < 0 || corner >= 4)
    {
        return false;
    }
    if (x < 0 || z < 0 || x >= terrainSize || z >= terrainSize)
    {
        return false;
    }
    const size_t idx = static_cast<size_t>(z * terrainSize + x);
    if (idx >= terrainHeights.size())
    {
        return false;
    }
    const int height = terrainHeights[idx];
    if (y < 0 || y >= height)
    {
        return false;
    }
    const VoxelBlock* block = terrain_block_at(x, y, z);
    if (!block)
    {
        return false;
    }
    *out_visibility = block->face_sky_visibility[static_cast<size_t>(face)][static_cast<size_t>(corner)];
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
    const bool sun_orbit = sunOrbitEnabled.load(std::memory_order_relaxed);
    const double base_intensity = lightIntensity.load(std::memory_order_relaxed);
    Vec3 light_dir = normalize_vec(load_light_direction());
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

    *out_factor = compute_shadow_factor(light_dir, world, normalize_vec(normal));
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
    generate_terrain_chunk();
    if (!shadowEnabled.load(std::memory_order_relaxed))
    {
        *out_factor = 1.0f;
        return true;
    }

    const Vec3 dir = normalize_vec(light_dir);

    Vec3 right, up, forward;
    build_light_basis(dir, right, up, forward);

    const double scale = std::tan(kSunDiskRadius);
    const Vec3 right_scaled = {right.x * scale, right.y * scale, right.z * scale};
    const Vec3 up_scaled    = {up.x * scale, up.y * scale, up.z * scale};

    const Vec3 shadow_dir = jitter_shadow_direction(dir, 
                                                    right_scaled, up_scaled,
                                                    pixel_x, pixel_y, 
                                                    static_cast<uint32_t>(frame),
                                                    kSunShadowSalt);
                                                    
    *out_factor = compute_shadow_factor(shadow_dir, world, normalize_vec(normal));
    return true;
}

float render_debug_shadow_filter_3x3(const float* mask, const float* depth, const Vec3* normals)
{
    if (!mask || !depth || !normals)
    {
        return 0.0f;
    }
    const float depth_max = std::numeric_limits<float>::max();
    return shadow_filter_3x3_at(mask, depth, normals, 3, 3, 1, 1, depth_max);
}

static bool camera_intersects_block(const Vec3& pos)
{
    generate_terrain_chunk();
    const double block_size = kTerrainBlockSize;
    const double half = block_size * 0.5;
    const double start_x = -(kTerrainChunkSize - 1) * block_size * 0.5;
    const double start_z = kTerrainStartZ;
    const double base_y = kTerrainBaseY;

    const double gx_f = (pos.x - start_x + half) / block_size;
    const double gz_f = (pos.z - start_z + half) / block_size;
    const double gy_f = (base_y - pos.y + half) / block_size;

    const int gx = static_cast<int>(std::floor(gx_f));
    const int gz = static_cast<int>(std::floor(gz_f));
    const int gy = static_cast<int>(std::floor(gy_f));

    if (!terrain_has_block(gx, gy, gz))
    {
        return false;
    }

    const double center_x = start_x + gx * block_size;
    const double center_z = start_z + gz * block_size;
    const double center_y = base_y - gy * block_size;

    return std::abs(pos.x - center_x) < half &&
           std::abs(pos.y - center_y) < half &&
           std::abs(pos.z - center_z) < half;
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
    Vec3 pos = render_get_camera_position();
    if (delta.x != 0.0)
    {
        const Vec3 candidate{pos.x + delta.x, pos.y, pos.z};
        if (!camera_intersects_block(candidate))
        {
            pos.x = candidate.x;
        }
    }
    if (delta.y != 0.0)
    {
        const Vec3 candidate{pos.x, pos.y + delta.y, pos.z};
        if (!camera_intersects_block(candidate))
        {
            pos.y = candidate.y;
        }
    }
    if (delta.z != 0.0)
    {
        const Vec3 candidate{pos.x, pos.y, pos.z + delta.z};
        if (!camera_intersects_block(candidate))
        {
            pos.z = candidate.z;
        }
    }
    render_set_camera_position(pos);
}

void render_move_camera_local(const Vec3 delta)
{
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
        if (camera_intersects_block(candidate))
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
    ClipVertex input[3]{
        {v0, v0, {0.0, 0.0, 0.0}, 0.0f},
        {v1, v1, {0.0, 0.0, 0.0}, 0.0f},
        {v2, v2, {0.0, 0.0, 0.0}, 0.0f}
    };
    ClipVertex clipped[4]{};
    const size_t count = clip_triangle_to_near_plane(input, 3, clipped, 4);
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
    generate_terrain_chunk();
    return terrainBlocks.size();
}

size_t render_debug_get_terrain_visible_face_count()
{
    generate_terrain_chunk();
    return terrainVisibleFaces;
}

size_t render_debug_get_terrain_triangle_count()
{
    generate_terrain_chunk();
    return terrainMeshTriangles;
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
