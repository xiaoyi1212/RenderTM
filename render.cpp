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
    {3, 7, 6, 2}, // bottom (+y), matches cubeTriangles winding
    {0, 4, 7, 3}, // left (-x), matches cubeTriangles winding
    {1, 2, 6, 5}, // right (+x)
    {0, 3, 2, 1}, // back (-z), matches cubeTriangles winding
    {4, 5, 6, 7}  // front (+z)
};

const int cubeFaceAxis1[6] = {
    0, // top: X
    0, // bottom: X
    2, // left: Z
    2, // right: Z
    0, // back: X
    0  // front: X
};

const int cubeFaceAxis2[6] = {
    2, // top: Z
    2, // bottom: Z
    1, // left: Y
    1, // right: Y
    1, // back: Y
    1  // front: Y
};

const int cubeFaceNormal[6][3] = {
    {0, 1, 0},   // top (-y world, +y grid)
    {0, -1, 0},  // bottom (+y world, -y grid)
    {-1, 0, 0},  // left (-x)
    {1, 0, 0},   // right (+x)
    {0, 0, -1},  // back (-z)
    {0, 0, 1}    // front (+z)
};

static int top_corner_index_for_vertex(const int vertex_index)
{
    switch (vertex_index)
    {
        case 0: return 0;
        case 1: return 1;
        case 5: return 2;
        case 4: return 3;
        default: return -1;
    }
}

static Vec3 face_normal_world(const int face)
{
    const double x = static_cast<double>(cubeFaceNormal[face][0]);
    const double y = static_cast<double>(-cubeFaceNormal[face][1]);
    const double z = static_cast<double>(cubeFaceNormal[face][2]);
    return {x, y, z};
}

static std::atomic<float> rotationAngle{0.0f};
static std::atomic<bool> rotationPaused{false};
static std::atomic<RenderScene> activeScene{RenderScene::CubeOnly};
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
static std::atomic<double> camera_x{16.0};
static std::atomic<double> camera_y{-19.72};
static std::atomic<double> camera_z{-1.93};
static std::atomic<double> camera_yaw{-0.6911503837897546};
static std::atomic<double> camera_pitch{-0.6003932626860493};
static std::atomic<bool> ambientOcclusionEnabled{true};
static std::atomic<bool> shadowEnabled{true};
static std::atomic<bool> greedyMeshingEnabled{false};
static std::atomic<uint64_t> sunShadowBuilds{0};

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

struct ShadowCacheState
{
    Vec3 dir{0.0, 0.0, 0.0};
    double c = 0.0;
    double s = 0.0;
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
        const struct ShadowMap* shadow;
    };
    std::array<DirectionalLightInfo, 2> lights;
};

constexpr int kMsaaSamples = 2;
constexpr size_t kShadowMapResolution = 256;
constexpr int kShadowPcfRadius = 2;
constexpr int kShadowPcfKernel = kShadowPcfRadius * 2 + 1;
constexpr double kPi = 3.14159265358979323846;
constexpr double kSunLatitudeDeg = 30.0;
constexpr double kSunLatitudeRad = kPi * kSunLatitudeDeg / 180.0;
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
constexpr int kTerrainChunkSize = 16;
constexpr double kTerrainBlockSize = 2.0;
constexpr double kTerrainStartZ = 4.0;
constexpr double kTerrainBaseY = 2.0;
constexpr double kSideNormalBlend = 0.65;
constexpr int kAoSampleRadius = 6;
constexpr float kAoSlopeScale = 3.0f;
constexpr float kAoStrength = 0.35f;

static Vec3 normalize_vec(const Vec3& v);
static float compute_shadow_factor(const ShadowMap* shadow, const Vec3& light_dir,
                                   const Vec3& world, const Vec3& normal);
static bool triangle_in_front_of_near_plane(double z0, double z1, double z2);
static void build_terrain_mesh();
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
    bool isTop;
    TopFaceLighting topFace;
    std::array<std::array<float, 4>, 6> face_ao;
    std::array<std::array<Vec3, 4>, 6> face_normals;
};

struct RenderQuad
{
    Vec3 v[4];
    Vec3 n[4];
    float ao[4];
    uint32_t color;
};

static std::vector<VoxelBlock> terrainBlocks;
static bool terrainReady = false;
static int terrainSize = 0;
static int terrainMaxHeight = 0;
static std::vector<int> terrainHeights;
static std::vector<uint32_t> terrainTopColors;
static std::vector<TopFaceLighting> terrainTopFaces;
static std::vector<int> terrainBlockIndex;
static std::vector<RenderQuad> terrainQuads;
static bool terrainMeshReady = false;
static size_t terrainVisibleFaces = 0;
static size_t terrainMergedQuads = 0;
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

static void compute_face_ao_for_block(const int gx, const int gy, const int gz,
                                      std::array<std::array<float, 4>, 6>& out)
{
    for (int face = 0; face < 6; ++face)
    {
        const int axis1 = cubeFaceAxis1[face];
        const int axis2 = cubeFaceAxis2[face];
        const int nx = cubeFaceNormal[face][0];
        const int ny = cubeFaceNormal[face][1];
        const int nz = cubeFaceNormal[face][2];

        for (int corner = 0; corner < 4; ++corner)
        {
            const int vi = cubeFaceVertices[face][corner];
            const Vec3& v = cubeVertices[vi];
            const int sx = v.x >= 0.0 ? 1 : -1;
            const int sy_world = v.y >= 0.0 ? 1 : -1;
            const int sz = v.z >= 0.0 ? 1 : -1;
            const int sy = -sy_world;

            const int sign1 = (axis1 == 0) ? sx : (axis1 == 1 ? sy : sz);
            const int sign2 = (axis2 == 0) ? sx : (axis2 == 1 ? sy : sz);

            int d1x = 0, d1y = 0, d1z = 0;
            int d2x = 0, d2y = 0, d2z = 0;
            if (axis1 == 0) d1x = sign1;
            else if (axis1 == 1) d1y = sign1;
            else d1z = sign1;

            if (axis2 == 0) d2x = sign2;
            else if (axis2 == 1) d2y = sign2;
            else d2z = sign2;

            const int base_x = gx + nx;
            const int base_y = gy + ny;
            const int base_z = gz + nz;

            const int side1 = terrain_has_block(base_x + d1x, base_y + d1y, base_z + d1z) ? 1 : 0;
            const int side2 = terrain_has_block(base_x + d2x, base_y + d2y, base_z + d2z) ? 1 : 0;
            const int corner_block = terrain_has_block(base_x + d1x + d2x, base_y + d1y + d2y, base_z + d1z + d2z) ? 1 : 0;

            int occlusion = side1 + side2 + corner_block;
            if (side1 && side2)
            {
                occlusion = 3;
            }
            float ao = 1.0f - kAoStrength * (static_cast<float>(occlusion) / 3.0f);
            out[face][corner] = std::clamp(ao, 0.0f, 1.0f);
        }
    }
}

static void generate_terrain_chunk()
{
    if (terrainReady)
    {
        return;
    }
    terrainReady = true;

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

            auto corner_ao = [&](int sx, int sz) {
                const double base_height = static_cast<double>(height);

                auto direction_occlusion = [&](int dx, int dz) -> float {
                    if (dx == 0 && dz == 0)
                    {
                        return 0.0f;
                    }
                    const int nx = x + dx;
                    const int nz = z + dz;
                    if (nx < 0 || nx >= chunk_size || nz < 0 || nz >= chunk_size)
                    {
                        return 0.0f;
                    }
                    if (height_at_or_zero(nx, nz) <= height)
                    {
                        return 0.0f;
                    }
                    const float dist_unit = (dx != 0 && dz != 0) ? 1.41421356f : 1.0f;
                    float max_slope = 0.0f;
                    for (int step = 1; step <= kAoSampleRadius; ++step)
                    {
                        const int sxp = x + dx * step;
                        const int szp = z + dz * step;
                        if (sxp < 0 || sxp >= chunk_size || szp < 0 || szp >= chunk_size)
                        {
                            break;
                        }
                        const double h = static_cast<double>(height_at_or_zero(sxp, szp));
                        const double diff = h - base_height;
                        if (diff <= 0.0)
                        {
                            continue;
                        }
                        const double slope = diff / (dist_unit * static_cast<double>(step));
                        if (slope > max_slope)
                        {
                            max_slope = static_cast<float>(slope);
                        }
                    }
                    return std::clamp(max_slope / kAoSlopeScale, 0.0f, 1.0f);
                };

                const float occlusion = (direction_occlusion(sx, 0) +
                                         direction_occlusion(0, sz) +
                                         direction_occlusion(sx, sz)) / 3.0f;
                const float ao = 1.0f - kAoStrength * occlusion;
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
                std::array<std::array<float, 4>, 6> face_ao{};
                for (auto& face : face_ao)
                {
                    face.fill(1.0f);
                }
                compute_face_ao_for_block(x, y, z, face_ao);
                std::array<std::array<Vec3, 4>, 6> face_normals{};
                for (int face = 0; face < 6; ++face)
                {
                    const Vec3 base = face_normal_world(face);
                    for (int corner = 0; corner < 4; ++corner)
                    {
                        face_normals[face][corner] = base;
                    }
                }
                if (is_top)
                {
                    for (int face = FaceLeft; face <= FaceFront; ++face)
                    {
                        const Vec3 base = face_normal_world(face);
                        for (int corner = 0; corner < 4; ++corner)
                        {
                            const int vertex_index = cubeFaceVertices[face][corner];
                            const int top_corner = top_corner_index_for_vertex(vertex_index);
                            if (top_corner < 0)
                            {
                                continue;
                            }
                            const Vec3 slope = top_face.normals[static_cast<size_t>(top_corner)];
                            Vec3 blended{
                                base.x * (1.0 - kSideNormalBlend) + slope.x * kSideNormalBlend,
                                base.y * (1.0 - kSideNormalBlend) + slope.y * kSideNormalBlend,
                                base.z * (1.0 - kSideNormalBlend) + slope.z * kSideNormalBlend
                            };
                            blended = normalize_vec(blended);
                            face_normals[face][corner] = blended;
                        }
                    }
                }
                terrainBlocks.push_back({
                    {start_x + x * block_size, base_y - y * block_size, start_z + z * block_size},
                    color,
                    is_top,
                    is_top ? top_face : empty_top,
                    face_ao,
                    face_normals
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

struct FaceKey
{
    uint32_t color = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int ao = 0;

    bool operator==(const FaceKey& other) const
    {
        return color == other.color &&
               nx == other.nx &&
               ny == other.ny &&
               nz == other.nz &&
               ao == other.ao;
    }
};

static int quantize_key(const double value, const double scale)
{
    return static_cast<int>(std::lround(value * scale));
}

static bool approx_equal(const double a, const double b, const double eps)
{
    return std::abs(a - b) <= eps;
}

static bool approx_equal_vec(const Vec3& a, const Vec3& b, const double eps)
{
    return approx_equal(a.x, b.x, eps) &&
           approx_equal(a.y, b.y, eps) &&
           approx_equal(a.z, b.z, eps);
}

static bool face_uniform(const VoxelBlock& block, const int face, Vec3& out_normal, float& out_ao)
{
    const std::array<Vec3, 4>* normals = nullptr;
    const std::array<float, 4>* ao = nullptr;
    if (face == FaceTop)
    {
        normals = &block.topFace.normals;
        ao = &block.topFace.ao;
    }
    else
    {
        normals = &block.face_normals[static_cast<size_t>(face)];
        ao = &block.face_ao[static_cast<size_t>(face)];
    }
    const Vec3 n0 = (*normals)[0];
    const Vec3 n1 = (*normals)[1];
    const Vec3 n2 = (*normals)[2];
    const Vec3 n3 = (*normals)[3];
    const float a0 = (*ao)[0];
    const float a1 = (*ao)[1];
    const float a2 = (*ao)[2];
    const float a3 = (*ao)[3];
    if (!approx_equal_vec(n0, n1, 1e-6) ||
        !approx_equal_vec(n0, n2, 1e-6) ||
        !approx_equal_vec(n0, n3, 1e-6))
    {
        return false;
    }
    if (!approx_equal(a0, a1, 1e-5) ||
        !approx_equal(a0, a2, 1e-5) ||
        !approx_equal(a0, a3, 1e-5))
    {
        return false;
    }
    out_normal = n0;
    out_ao = a0;
    return true;
}

static FaceKey make_face_key(const uint32_t color, const Vec3& normal, const float ao)
{
    static constexpr double kNormalScale = 10000.0;
    static constexpr double kAoScale = 10000.0;
    return {
        color,
        quantize_key(normal.x, kNormalScale),
        quantize_key(normal.y, kNormalScale),
        quantize_key(normal.z, kNormalScale),
        quantize_key(ao, kAoScale)
    };
}

static void ensure_quad_winding(RenderQuad& quad, const Vec3& desired_normal)
{
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
    const Vec3 normal = cross_vec(ab, ac);
    if (dot_vec(normal, desired_normal) < 0.0)
    {
        std::swap(quad.v[1], quad.v[3]);
        std::swap(quad.n[1], quad.n[3]);
        std::swap(quad.ao[1], quad.ao[3]);
    }
}

static double terrain_axis_center(const int axis, const int index)
{
    const double block_size = kTerrainBlockSize;
    if (axis == 0)
    {
        const double start_x = -(terrainSize - 1) * block_size * 0.5;
        return start_x + static_cast<double>(index) * block_size;
    }
    if (axis == 1)
    {
        const double base_y = kTerrainBaseY;
        return base_y - static_cast<double>(index) * block_size;
    }
    const double start_z = kTerrainStartZ;
    return start_z + static_cast<double>(index) * block_size;
}

static double terrain_axis_min_edge(const int axis, const int index)
{
    const double half = kTerrainBlockSize * 0.5;
    const double center = terrain_axis_center(axis, index);
    return std::min(center - half, center + half);
}

static double terrain_axis_max_edge(const int axis, const int index)
{
    const double half = kTerrainBlockSize * 0.5;
    const double center = terrain_axis_center(axis, index);
    return std::max(center - half, center + half);
}

static void make_quad_vertices(const int normal_axis, const double plane,
                               const int u_axis, const double u_min, const double u_max,
                               const int v_axis, const double v_min, const double v_max,
                               Vec3 out[4])
{
    auto make_vertex = [&](const double u, const double v) {
        Vec3 p{0.0, 0.0, 0.0};
        if (normal_axis == 0) p.x = plane;
        else if (normal_axis == 1) p.y = plane;
        else p.z = plane;

        if (u_axis == 0) p.x = u;
        else if (u_axis == 1) p.y = u;
        else p.z = u;

        if (v_axis == 0) p.x = v;
        else if (v_axis == 1) p.y = v;
        else p.z = v;
        return p;
    };

    out[0] = make_vertex(u_min, v_max);
    out[1] = make_vertex(u_max, v_max);
    out[2] = make_vertex(u_max, v_min);
    out[3] = make_vertex(u_min, v_min);
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
        if (face == FaceTop)
        {
            quad.n[corner] = block.topFace.normals[static_cast<size_t>(attr_corner)];
            quad.ao[corner] = block.topFace.ao[static_cast<size_t>(attr_corner)];
        }
        else
        {
            quad.n[corner] = block.face_normals[static_cast<size_t>(face)][static_cast<size_t>(attr_corner)];
            quad.ao[corner] = block.face_ao[static_cast<size_t>(face)][static_cast<size_t>(attr_corner)];
        }
    }
    terrainQuads.push_back(quad);
    terrainMergedQuads++;
}

static void emit_merged_quad(const int face, const int normal_axis, const int normal_sign,
                             const int u_axis, const int v_axis,
                             const int slice, const int u0, const int v0,
                             const int u1, const int v1, const uint32_t color,
                             const Vec3& normal, const float ao)
{
    double u_min = 0.0;
    double u_max = 0.0;
    double v_min = 0.0;
    double v_max = 0.0;
    const double u_min0 = terrain_axis_min_edge(u_axis, u0);
    const double u_max0 = terrain_axis_max_edge(u_axis, u0);
    const double u_min1 = terrain_axis_min_edge(u_axis, u1);
    const double u_max1 = terrain_axis_max_edge(u_axis, u1);
    u_min = std::min(u_min0, u_min1);
    u_max = std::max(u_max0, u_max1);
    const double v_min0 = terrain_axis_min_edge(v_axis, v0);
    const double v_max0 = terrain_axis_max_edge(v_axis, v0);
    const double v_min1 = terrain_axis_min_edge(v_axis, v1);
    const double v_max1 = terrain_axis_max_edge(v_axis, v1);
    v_min = std::min(v_min0, v_min1);
    v_max = std::max(v_max0, v_max1);

    const double half = kTerrainBlockSize * 0.5;
    const double plane = terrain_axis_center(normal_axis, slice) + static_cast<double>(normal_sign) * half;

    RenderQuad quad{};
    quad.color = color;
    for (int i = 0; i < 4; ++i)
    {
        quad.n[i] = normal;
        quad.ao[i] = ao;
    }
    make_quad_vertices(normal_axis, plane, u_axis, u_min, u_max, v_axis, v_min, v_max, quad.v);
    ensure_quad_winding(quad, face_normal_world(face));
    terrainQuads.push_back(quad);
    terrainMergedQuads++;
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
    terrainMergedQuads = 0;
    terrainMeshTriangles = 0;

    if (terrainSize <= 0 || terrainMaxHeight <= 0)
    {
        return;
    }

    const bool greedy_enabled = greedyMeshingEnabled.load(std::memory_order_relaxed);
    if (!greedy_enabled)
    {
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
        terrainMeshTriangles = terrainMergedQuads * 2;
        return;
    }

    const int axis_sizes[3] = {terrainSize, terrainMaxHeight, terrainSize};

    struct FaceCell
    {
        bool present = false;
        bool mergeable = false;
        FaceKey key{};
        int block_index = -1;
    };

    auto process_face = [&](const int face, const int normal_axis, const int normal_sign,
                            const int u_axis, const int v_axis) {
        const int slice_count = axis_sizes[normal_axis];
        const int u_dim = axis_sizes[u_axis];
        const int v_dim = axis_sizes[v_axis];
        std::vector<FaceCell> cells(static_cast<size_t>(u_dim * v_dim));
        std::vector<uint8_t> used(static_cast<size_t>(u_dim * v_dim), 0);

        for (int slice = 0; slice < slice_count; ++slice)
        {
            for (int v = 0; v < v_dim; ++v)
            {
                for (int u = 0; u < u_dim; ++u)
                {
                    FaceCell& cell = cells[static_cast<size_t>(v * u_dim + u)];
                    cell.present = false;
                    cell.mergeable = false;
                    cell.block_index = -1;

                    int coord[3]{0, 0, 0};
                    coord[normal_axis] = slice;
                    coord[u_axis] = u;
                    coord[v_axis] = v;
                    const int gx = coord[0];
                    const int gy = coord[1];
                    const int gz = coord[2];
                    if (!terrain_has_block(gx, gy, gz))
                    {
                        continue;
                    }
                    int ncoord[3]{
                        gx + cubeFaceNormal[face][0],
                        gy + cubeFaceNormal[face][1],
                        gz + cubeFaceNormal[face][2]
                    };
                    if (terrain_has_block(ncoord[0], ncoord[1], ncoord[2]))
                    {
                        continue;
                    }
                    cell.present = true;
                    terrainVisibleFaces++;
                    const VoxelBlock* block = terrain_block_at(gx, gy, gz);
                    if (!block)
                    {
                        continue;
                    }
                    const size_t slot = terrain_block_slot(gx, gy, gz);
                    if (slot < terrainBlockIndex.size())
                    {
                        cell.block_index = terrainBlockIndex[slot];
                    }
                    Vec3 uniform_normal{};
                    float uniform_ao = 1.0f;
                    if (face_uniform(*block, face, uniform_normal, uniform_ao))
                    {
                        cell.mergeable = true;
                        cell.key = make_face_key(block->color, uniform_normal, uniform_ao);
                    }
                }
            }

            std::fill(used.begin(), used.end(), 0);

            for (int v = 0; v < v_dim; ++v)
            {
                for (int u = 0; u < u_dim; ++u)
                {
                    const size_t idx = static_cast<size_t>(v * u_dim + u);
                    FaceCell& cell = cells[idx];
                    if (!cell.present || used[idx])
                    {
                        continue;
                    }
                    if (!cell.mergeable)
                    {
                        if (cell.block_index >= 0 && static_cast<size_t>(cell.block_index) < terrainBlocks.size())
                        {
                            emit_block_face_quad(terrainBlocks[static_cast<size_t>(cell.block_index)], face);
                        }
                        used[idx] = 1;
                        continue;
                    }

                    int width = 1;
                    while (u + width < u_dim)
                    {
                        const size_t next_idx = static_cast<size_t>(v * u_dim + (u + width));
                        const FaceCell& next = cells[next_idx];
                        if (!next.present || used[next_idx] || !next.mergeable || !(next.key == cell.key))
                        {
                            break;
                        }
                        width++;
                    }

                    int height = 1;
                    bool can_expand = true;
                    while (v + height < v_dim && can_expand)
                    {
                        for (int uu = 0; uu < width; ++uu)
                        {
                            const size_t next_idx = static_cast<size_t>((v + height) * u_dim + (u + uu));
                            const FaceCell& next = cells[next_idx];
                            if (!next.present || used[next_idx] || !next.mergeable || !(next.key == cell.key))
                            {
                                can_expand = false;
                                break;
                            }
                        }
                        if (can_expand)
                        {
                            height++;
                        }
                    }

                    for (int dv = 0; dv < height; ++dv)
                    {
                        for (int du = 0; du < width; ++du)
                        {
                            const size_t mark_idx = static_cast<size_t>((v + dv) * u_dim + (u + du));
                            used[mark_idx] = 1;
                        }
                    }

                    const VoxelBlock* block = nullptr;
                    if (cell.block_index >= 0 && static_cast<size_t>(cell.block_index) < terrainBlocks.size())
                    {
                        block = &terrainBlocks[static_cast<size_t>(cell.block_index)];
                    }
                    Vec3 uniform_normal{};
                    float uniform_ao = 1.0f;
                    if (block && face_uniform(*block, face, uniform_normal, uniform_ao))
                    {
                        emit_merged_quad(face, normal_axis, normal_sign, u_axis, v_axis,
                                         slice, u, v, u + width - 1, v + height - 1,
                                         block->color, uniform_normal, uniform_ao);
                    }
                    else
                    {
                        if (block)
                        {
                            emit_block_face_quad(*block, face);
                        }
                    }
                }
            }
        }
    };

    process_face(FaceRight, 0, 1, 2, 1);
    process_face(FaceLeft, 0, -1, 2, 1);
    process_face(FaceFront, 2, 1, 0, 1);
    process_face(FaceBack, 2, -1, 0, 1);
    process_face(FaceBottom, 1, 1, 0, 2);
    process_face(FaceTop, 1, -1, 0, 2);

    terrainMeshTriangles = terrainMergedQuads * 2;
}

static ColorRGB lerp_color(const ColorRGB& a, const ColorRGB& b, float t)
{
    return {
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t
    };
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
    float ao;
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
    const float ao = a.ao + (b.ao - a.ao) * static_cast<float>(t);
    return {view, world, normal, ao};
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

static bool shadow_cache_matches(const ShadowCacheState& cache, const Vec3& dir, const double c, const double s)
{
    if (!cache.valid)
    {
        return false;
    }
    const double dx = cache.dir.x - dir.x;
    const double dy = cache.dir.y - dir.y;
    const double dz = cache.dir.z - dir.z;
    if ((dx * dx + dy * dy + dz * dz) > 1e-10)
    {
        return false;
    }
    return (std::abs(cache.c - c) <= 1e-10) && (std::abs(cache.s - s) <= 1e-10);
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

static ViewRotation make_view_rotation(const double yaw, const double pitch)
{
    return {std::cos(yaw), std::sin(yaw), std::cos(pitch), std::sin(pitch)};
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

static inline float smoothstep(const float edge0, const float edge1, const float x)
{
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
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
    const float inv_z0 = 1.0f / v0.z;
    const float inv_z1 = 1.0f / v1.z;
    const float inv_z2 = 1.0f / v2.z;

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
                    const float inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2;
                    if (inv_z <= 0.0f)
                    {
                        continue;
                    }
                    const float depth = 1.0f / inv_z;
                    const size_t base = (static_cast<size_t>(y) * width + static_cast<size_t>(x)) * kMsaaSamples;
                    const size_t idx = base + static_cast<size_t>(s);
                    if (depth < zbuffer[idx])
                    {
                        zbuffer[idx] = depth;
                        const float w0p = w0 * inv_z0 / inv_z;
                        const float w1p = w1 * inv_z1 / inv_z;
                        const float w2p = w2 * inv_z2 / inv_z;
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

                        float ao = w0p * ao0 + w1p * ao1 + w2p * ao2;
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
                            const ColorRGB sky = lerp_color(ctx.hemi_ground, ctx.sky_top, sky_t);
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
                                light_color.r *= light.color.r;
                                light_color.g *= light.color.g;
                                light_color.b *= light.color.b;
                                if (ctx.material.specular > 0.0)
                                {
                                    const Vec3 half_vec = normalize_vec(add_vec(light.dir, view_dir));
                                    const double spec_dot = std::max(0.0, dot_vec(normal, half_vec));
                                    double spec = std::pow(spec_dot, ctx.material.shininess) * ctx.material.specular;
                                    spec *= light.intensity * shadow_factor;
                                    spec = std::clamp(spec, 0.0, 1.0);
                                    light_color.r += light.color.r * static_cast<float>(spec);
                                    light_color.g += light.color.g * static_cast<float>(spec);
                                    light_color.b += light.color.b * static_cast<float>(spec);
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
    float lit = 0.0f;
    int samples = 0;
    const float softness = static_cast<float>(0.05 + 0.08 * slope);
    for (int dy = -kShadowPcfRadius; dy <= kShadowPcfRadius; ++dy)
    {
        const int sy = std::clamp(cy + dy, 0, res - 1);
        for (int dx = -kShadowPcfRadius; dx <= kShadowPcfRadius; ++dx)
        {
            const int sx = std::clamp(cx + dx, 0, res - 1);
            const float depth = shadow->depth[static_cast<size_t>(sy) * shadow->resolution + static_cast<size_t>(sx)];
            if (depth <= shadow->depth_min)
            {
                continue;
            }
            samples++;
            const float delta = static_cast<float>((lz + bias) - depth);
            lit += smoothstep(-softness, softness, delta);
        }
    }
    if (samples <= 0)
    {
        return 1.0f;
    }
    return lit / static_cast<float>(samples);
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

    const size_t block_count = blocks.size();
    const size_t vertex_count = base_mesh.vertexCount;
    thread_local std::vector<ScreenVertex> light_vertices;
    if (light_vertices.size() != block_count * vertex_count)
    {
        light_vertices.resize(block_count * vertex_count);
    }

    for (size_t b = 0; b < block_count; ++b)
    {
        const auto& block = blocks[b];
        const size_t base = b * vertex_count;
        for (size_t i = 0; i < vertex_count; ++i)
        {
            Vec3 v = base_mesh.vertices[i];
            if (base_mesh.rotate)
            {
                v = rotate_vertex(v, c, s);
            }
            v.x += block.position.x;
            v.y += block.position.y;
            v.z += block.position.z;
            const float lx = static_cast<float>(dot_vec(v, right));
            const float ly = static_cast<float>(dot_vec(v, up));
            const float lz = static_cast<float>(dot_vec(v, forward));
            light_vertices[base + i] = {lx, ly, lz};
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

    for (size_t b = 0; b < block_count; ++b)
    {
        const size_t base = b * vertex_count;
        for (size_t t = 0; t < base_mesh.triangleCount; ++t)
        {
            const int i0 = base_mesh.triangles[t][0];
            const int i1 = base_mesh.triangles[t][1];
            const int i2 = base_mesh.triangles[t][2];

            const ScreenVertex& a = light_vertices[base + static_cast<size_t>(i0)];
            const ScreenVertex& bvert = light_vertices[base + static_cast<size_t>(i1)];
            const ScreenVertex& cpos = light_vertices[base + static_cast<size_t>(i2)];

            const ScreenVertex sv0{
                (a.x - min_x) * shadow.scale_x,
                (a.y - min_y) * shadow.scale_y,
                a.z
            };
            const ScreenVertex sv1{
                (bvert.x - min_x) * shadow.scale_x,
                (bvert.y - min_y) * shadow.scale_y,
                bvert.z
            };
            const ScreenVertex sv2{
                (cpos.x - min_x) * shadow.scale_x,
                (cpos.y - min_y) * shadow.scale_y,
                cpos.z
            };

            rasterize_shadow_triangle(shadow.depth, shadow.resolution, sv0, sv1, sv2);
        }
    }

    return true;
}

static void render_quad(float* zbuffer, ColorRGB* sample_ambient, ColorRGB* sample_direct, uint8_t* sample_mask,
                        size_t width, size_t height,
                        const RenderQuad& quad, const double fov_x, const double fov_y,
                        const Vec3& camera_pos, const ViewRotation& view_rot, const ShadingContext& ctx)
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
        draw_shaded_triangle(zbuffer, sample_ambient, sample_direct, sample_mask, width, height,
                             sv0, sv1, sv2,
                             a.world, b.world, c.world,
                             a.normal, b.normal, c.normal,
                             a.ao, b.ao, c.ao,
                             ctx);
    };

    auto draw_clipped = [&](int i0, int i1, int i2) {
        ClipVertex input[3]{
            {view_space[i0], quad.v[i0], quad.n[i0], quad.ao[i0]},
            {view_space[i1], quad.v[i1], quad.n[i1], quad.ao[i1]},
            {view_space[i2], quad.v[i2], quad.n[i2], quad.ao[i2]}
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

static void render_mesh(float* zbuffer, ColorRGB* sample_ambient, ColorRGB* sample_direct, uint8_t* sample_mask,
                        size_t width, size_t height,
                        const Mesh& mesh, const double c, const double s,
                        const double fov_x, const double fov_y, const Vec3& camera_pos,
                        const ViewRotation& view_rot, const Material& material,
                        const ColorRGB& sky_top, const ColorRGB& sky_bottom,
                        const double sky_light_intensity,
                        const std::array<ShadingContext::DirectionalLightInfo, 2>& lights,
                        bool shadows_enabled,
                        const TopFaceLighting* top_face,
                        const std::array<std::array<float, 4>, 6>* face_ao,
                        const std::array<std::array<Vec3, 4>, 6>* face_normals)
{
    thread_local std::vector<Vec3> world;
    thread_local std::vector<Vec3> view_space;
    if (world.size() != mesh.vertexCount)
    {
        world.resize(mesh.vertexCount);
    }
    if (view_space.size() != mesh.vertexCount)
    {
        view_space.resize(mesh.vertexCount);
    }

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
        view = rotate_yaw_pitch_cached(view, view_rot.cy, view_rot.sy, view_rot.cp, view_rot.sp);
        view_space[i] = view;
    }

    const ColorRGB albedo_srgb = unpack_color(material.color);
    const ColorRGB albedo = srgb_to_linear(albedo_srgb);
    const float albedo_r = albedo.r;
    const float albedo_g = albedo.g;
    const float albedo_b = albedo.b;
    const float sky_scale = static_cast<float>(std::clamp(sky_light_intensity, 0.0, 1.0));
    const ColorRGB hemi_ground = compute_hemisphere_ground(sky_bottom, lights);
    const bool direct_lighting_enabled = (lights[0].intensity > 0.0) || (lights[1].intensity > 0.0);
    const bool ao_enabled = ambientOcclusionEnabled.load(std::memory_order_relaxed);
    const ShadingContext ctx{
        albedo,
        albedo_r,
        albedo_g,
        albedo_b,
        sky_top,
        sky_bottom,
        hemi_ground,
        sky_scale,
        camera_pos,
        ambientLight.load(std::memory_order_relaxed),
        material,
        direct_lighting_enabled,
        ao_enabled,
        shadows_enabled,
        lights
    };

    auto face_corner_index = [](int face, int vertex_index) -> int {
        for (int i = 0; i < 4; ++i)
        {
            if (cubeFaceVertices[face][i] == vertex_index)
            {
                return i;
            }
        }
        return -1;
    };

    auto face_from_normal = [](const Vec3& normal) -> int {
        if (std::abs(normal.x) > 0.5)
        {
            return normal.x > 0.0 ? FaceRight : FaceLeft;
        }
        if (std::abs(normal.y) > 0.5)
        {
            return normal.y > 0.0 ? FaceBottom : FaceTop;
        }
        if (std::abs(normal.z) > 0.5)
        {
            return normal.z > 0.0 ? FaceFront : FaceBack;
        }
        return -1;
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

        const int face_index = face_from_normal(face_normal);
        if (top_face && face_normal.y < -0.5)
        {
            const int c0 = top_corner_index_for_vertex(i0);
            const int c1 = top_corner_index_for_vertex(i1);
            const int c2 = top_corner_index_for_vertex(i2);
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
        else
        {
            if (face_normals && face_index >= 0)
            {
                const auto& face_values = (*face_normals)[static_cast<size_t>(face_index)];
                const int c0 = face_corner_index(face_index, i0);
                const int c1 = face_corner_index(face_index, i1);
                const int c2 = face_corner_index(face_index, i2);
                if (c0 >= 0)
                {
                    n0 = face_values[static_cast<size_t>(c0)];
                }
                if (c1 >= 0)
                {
                    n1 = face_values[static_cast<size_t>(c1)];
                }
                if (c2 >= 0)
                {
                    n2 = face_values[static_cast<size_t>(c2)];
                }
            }
            if (face_ao && face_index >= 0)
            {
                const auto& face_values = (*face_ao)[static_cast<size_t>(face_index)];
                const int c0 = face_corner_index(face_index, i0);
                const int c1 = face_corner_index(face_index, i1);
                const int c2 = face_corner_index(face_index, i2);
                if (c0 >= 0)
                {
                    ao0 = face_values[static_cast<size_t>(c0)];
                }
                if (c1 >= 0)
                {
                    ao1 = face_values[static_cast<size_t>(c1)];
                }
                if (c2 >= 0)
                {
                    ao2 = face_values[static_cast<size_t>(c2)];
                }
            }
        }

        ClipVertex input[3]{
            {view_space[i0], world[i0], n0, ao0},
            {view_space[i1], world[i1], n1, ao1},
            {view_space[i2], world[i2], n2, ao2}
        };
        ClipVertex clipped[4]{};
        const size_t clipped_count = clip_triangle_to_near_plane(input, 3, clipped, 4);
        if (clipped_count < 3)
        {
            continue;
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
            draw_shaded_triangle(zbuffer, sample_ambient, sample_direct, sample_mask, width, height,
                                 sv0, sv1, sv2,
                                 a.world, b.world, c.world,
                                 a.normal, b.normal, c.normal,
                                 a.ao, b.ao, c.ao,
                                 ctx);
        };

        if (clipped_count == 3)
        {
            draw_triangle(clipped[0], clipped[1], clipped[2]);
        }
        else if (clipped_count == 4)
        {
            draw_triangle(clipped[0], clipped[1], clipped[2]);
            draw_triangle(clipped[0], clipped[2], clipped[3]);
        }
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
        rotationAngle.fetch_add(0.002f, std::memory_order_relaxed);
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
    const double rotation = rotationAngle.load(std::memory_order_relaxed);
    const double c = cos(rotation);
    const double s = sin(rotation);

    const double fov_y = static_cast<double>(height) * 0.8;
    const double fov_x = fov_y;
    const Vec3 camera_pos{
        camera_x.load(std::memory_order_relaxed),
        camera_y.load(std::memory_order_relaxed),
        camera_z.load(std::memory_order_relaxed)
    };
    const double yaw = camera_yaw.load(std::memory_order_relaxed);
    const double pitch = camera_pitch.load(std::memory_order_relaxed);
    const ViewRotation view_rot = make_view_rotation(-yaw, -pitch);

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
    static ShadowMap sun_shadow;
    static ShadowMap moon_shadow;
    static ShadowCacheState sun_shadow_cache;
    static ShadowCacheState moon_shadow_cache;
    bool sun_shadow_valid = false;
    bool moon_shadow_valid = false;
    if (shadows_on && sun_intensity > 0.0)
    {
        const double cache_c = cubeMesh.rotate ? c : 1.0;
        const double cache_s = cubeMesh.rotate ? s : 0.0;
        const bool needs_rebuild = !sun_shadow.valid || !shadow_cache_matches(sun_shadow_cache, sun_dir, cache_c, cache_s);
        if (needs_rebuild)
        {
            sunShadowBuilds.fetch_add(1u, std::memory_order_relaxed);
            sun_shadow_valid = build_shadow_map(sun_shadow, sun_dir, terrainBlocks, cubeMesh, c, s);
            sun_shadow_cache.valid = sun_shadow_valid;
            if (sun_shadow_valid)
            {
                sun_shadow_cache.dir = sun_dir;
                sun_shadow_cache.c = cache_c;
                sun_shadow_cache.s = cache_s;
            }
        }
        else
        {
            sun_shadow_valid = sun_shadow.valid;
        }
    }
    if (shadows_on && moon_intensity > 0.0)
    {
        const double cache_c = cubeMesh.rotate ? c : 1.0;
        const double cache_s = cubeMesh.rotate ? s : 0.0;
        const bool needs_rebuild = !moon_shadow.valid || !shadow_cache_matches(moon_shadow_cache, moon_dir, cache_c, cache_s);
        if (needs_rebuild)
        {
            moon_shadow_valid = build_shadow_map(moon_shadow, moon_dir, terrainBlocks, cubeMesh, c, s);
            moon_shadow_cache.valid = moon_shadow_valid;
            if (moon_shadow_valid)
            {
                moon_shadow_cache.dir = moon_dir;
                moon_shadow_cache.c = cache_c;
                moon_shadow_cache.s = cache_s;
            }
        }
        else
        {
            moon_shadow_valid = moon_shadow.valid;
        }
    }

    const std::array<ShadingContext::DirectionalLightInfo, 2> lights = {
        ShadingContext::DirectionalLightInfo{sun_dir, sun_intensity, kSunLightColorLinear, sun_shadow_valid ? &sun_shadow : nullptr},
        ShadingContext::DirectionalLightInfo{moon_dir, moon_intensity, kMoonLightColorLinear, moon_shadow_valid ? &moon_shadow : nullptr}
    };

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
            albedo.r,
            albedo.g,
            albedo.b,
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
        render_quad(zbuffer.data(), sample_colors.data(), sample_direct.data(), nullptr, width, height, quad, fov_x, fov_y,
                    camera_pos, view_rot, ctx);
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
        const ColorRGB sky_row_srgb = lerp_color(sky_top, sky_bottom, sky_t);
        const ColorRGB sky_row_linear = srgb_to_linear(sky_row_srgb);
        const uint32_t sky_row = pack_color(sky_row_srgb);
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
                    accum.r += sky_row_linear.r;
                    accum.g += sky_row_linear.g;
                    accum.b += sky_row_linear.b;
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

            ColorRGB srgb = linear_to_srgb(accum);
            const float dither = (static_cast<float>(bayer4[y & 3][x & 3]) - 7.5f) * dither_scale;
            srgb.r += dither;
            srgb.g += dither;
            srgb.b += dither;

            framebuffer[y * width + x] = pack_color(srgb);
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
    rotationAngle.store(angle, std::memory_order_relaxed);
}

float render_get_rotation()
{
    return rotationAngle.load(std::memory_order_relaxed);
}

void render_set_scene(const RenderScene scene)
{
    activeScene.store(scene, std::memory_order_relaxed);
}

RenderScene render_get_scene()
{
    return activeScene.load(std::memory_order_relaxed);
}

void render_set_light_direction(const Vec3 dir)
{
    store_light_direction(dir);
}

Vec3 render_get_light_direction()
{
    return load_light_direction();
}

void render_set_light_intensity(const double intensity)
{
    lightIntensity.store(intensity, std::memory_order_relaxed);
}

double render_get_light_intensity()
{
    return lightIntensity.load(std::memory_order_relaxed);
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
    store_moon_direction(dir);
}

Vec3 render_get_moon_direction()
{
    return load_moon_direction();
}

void render_set_moon_intensity(const double intensity)
{
    moonIntensity.store(intensity, std::memory_order_relaxed);
}

double render_get_moon_intensity()
{
    return moonIntensity.load(std::memory_order_relaxed);
}

void render_set_sky_top_color(const uint32_t color)
{
    skyTopColor.store(color, std::memory_order_relaxed);
}

uint32_t render_get_sky_top_color()
{
    return skyTopColor.load(std::memory_order_relaxed);
}

void render_set_sky_bottom_color(const uint32_t color)
{
    skyBottomColor.store(color, std::memory_order_relaxed);
}

uint32_t render_get_sky_bottom_color()
{
    return skyBottomColor.load(std::memory_order_relaxed);
}

void render_set_sky_light_intensity(const double intensity)
{
    skyLightIntensity.store(intensity, std::memory_order_relaxed);
}

double render_get_sky_light_intensity()
{
    return skyLightIntensity.load(std::memory_order_relaxed);
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

void render_set_greedy_meshing_enabled(const bool enabled)
{
    greedyMeshingEnabled.store(enabled, std::memory_order_relaxed);
    terrainMeshReady = false;
}

bool render_get_greedy_meshing_enabled()
{
    return greedyMeshingEnabled.load(std::memory_order_relaxed);
}

size_t render_get_shadow_map_resolution()
{
    return kShadowMapResolution;
}

int render_get_shadow_pcf_kernel()
{
    return kShadowPcfKernel;
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

bool render_get_terrain_face_ao(const int x, const int y, const int z, const int face, float out_ao[4])
{
    if (!out_ao)
    {
        return false;
    }
    generate_terrain_chunk();
    if (face < 0 || face >= 6)
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
    std::array<std::array<float, 4>, 6> face_ao{};
    for (auto& entry : face_ao)
    {
        entry.fill(1.0f);
    }
    compute_face_ao_for_block(x, y, z, face_ao);
    for (size_t i = 0; i < 4; ++i)
    {
        out_ao[i] = face_ao[static_cast<size_t>(face)][i];
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

    static ShadowMap shadow;
    static ShadowCacheState shadow_cache;
    const Mesh cubeMesh{
        cubeVertices,
        8,
        cubeTriangles,
        12,
        {0.0, 0.0, 0.0},
        false
    };
    const double rotation = rotationAngle.load(std::memory_order_relaxed);
    const double c = std::cos(rotation);
    const double s = std::sin(rotation);
    const double cache_c = cubeMesh.rotate ? c : 1.0;
    const double cache_s = cubeMesh.rotate ? s : 0.0;
    const bool needs_rebuild = !shadow.valid || !shadow_cache_matches(shadow_cache, light_dir, cache_c, cache_s);
    if (needs_rebuild)
    {
        if (!build_shadow_map(shadow, light_dir, terrainBlocks, cubeMesh, c, s))
        {
            shadow_cache.valid = false;
            *out_factor = 1.0f;
            return false;
        }
        shadow_cache.valid = true;
        shadow_cache.dir = light_dir;
        shadow_cache.c = cache_c;
        shadow_cache.s = cache_s;
    }
    *out_factor = compute_shadow_factor(&shadow, light_dir, world, normalize_vec(normal));
    return true;
}

static bool camera_intersects_block(const Vec3& pos)
{
    generate_terrain_chunk();
    const double half = kTerrainBlockSize * 0.5;
    for (const auto& block : terrainBlocks)
    {
        if (std::abs(pos.x - block.position.x) < half &&
            std::abs(pos.y - block.position.y) < half &&
            std::abs(pos.z - block.position.z) < half)
        {
            return true;
        }
    }
    return false;
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
        {v0, v0, {0.0, 0.0, 0.0}, 1.0f},
        {v1, v1, {0.0, 0.0, 0.0}, 1.0f},
        {v2, v2, {0.0, 0.0, 0.0}, 1.0f}
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

void render_debug_reset_shadow_build_counts()
{
    sunShadowBuilds.store(0u, std::memory_order_relaxed);
}

uint64_t render_debug_get_sun_shadow_build_count()
{
    return sunShadowBuilds.load(std::memory_order_relaxed);
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

size_t render_debug_get_terrain_quad_count()
{
    generate_terrain_chunk();
    return terrainMergedQuads;
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
