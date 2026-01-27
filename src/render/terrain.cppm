module;

#include "../prelude.hpp"

export module render:terrain;

import :math;
import :noise;

export enum FaceIndex
{
    FaceTop = 0,    // normal -Y
    FaceBottom = 1, // normal +Y
    FaceLeft = 2,   // normal -X
    FaceRight = 3,  // normal +X
    FaceBack = 4,   // normal -Z
    FaceFront = 5   // normal +Z
};

constexpr std::array<Vec3, 8> cubeVertices = {{
    {-1, -1, -1}, {1, -1, -1}, {1,  1, -1}, {-1,  1, -1},
    {-1, -1,  1}, {1, -1,  1}, {1,  1,  1}, {-1,  1,  1}
}};

constexpr std::array<std::array<int, 4>, 6> cubeFaceVertices = {{
    {{0, 1, 5, 4}}, // top (-y)
    {{3, 2, 6, 7}}, // bottom (+y)
    {{0, 3, 7, 4}}, // left (-x)
    {{1, 2, 6, 5}}, // right (+x)
    {{0, 1, 2, 3}}, // back (-z)
    {{4, 5, 6, 7}}  // front (+z)
}};

constexpr std::array<std::array<int, 4>, 6> cubeFaceQuadOrder = {{
    {{0, 1, 5, 4}}, // top (-y)
    {{3, 7, 6, 2}}, // bottom (+y)
    {{0, 4, 7, 3}}, // left (-x)
    {{1, 2, 6, 5}}, // right (+x)
    {{0, 3, 2, 1}}, // back (-z)
    {{4, 5, 6, 7}}  // front (+z)
}};

constexpr std::array<std::array<int, 3>, 6> cubeFaceNormal = {{
    {{0, 1, 0}},   // top (-y world, +y grid)
    {{0, -1, 0}},  // bottom (+y world, -y grid)
    {{-1, 0, 0}},  // left (-x)
    {{1, 0, 0}},   // right (+x)
    {{0, 0, -1}},  // back (-z)
    {{0, 0, 1}}    // front (+z)
}};

constexpr std::array<Vec3, 8> cubeVerticesGrid = {{
    {0.0, 1.0, 0.0},
    {1.0, 1.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0},
    {0.0, 1.0, 1.0},
    {1.0, 1.0, 1.0},
    {1.0, 0.0, 1.0},
    {0.0, 0.0, 1.0}
}};

constexpr int kTerrainChunkSize = 16;
constexpr double kTerrainBlockSize = 2.0;
constexpr double kTerrainStartZ = 4.0;
constexpr double kTerrainBaseY = 2.0;
constexpr size_t kSkyRayCount = 128;
constexpr double kSkyRayStep = 0.25;
constexpr double kSkyRayMaxDistance = 6.0;
constexpr double kSkyRayBias = 0.02;
constexpr double kSkyRayCenterBias = 0.02;

static Vec3 face_normal_world(const int face)
{
    const double x = static_cast<double>(cubeFaceNormal[face][0]);
    const double y = static_cast<double>(-cubeFaceNormal[face][1]);
    const double z = static_cast<double>(cubeFaceNormal[face][2]);
    return {x, y, z};
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

static auto sky_sample_dirs() -> const std::array<Vec3, kSkyRayCount>&
{
    static const std::array<Vec3, kSkyRayCount> dirs = [] {
        std::array<Vec3, kSkyRayCount> samples{};
        for (size_t i = 0; i < kSkyRayCount; ++i)
        {
            const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(kSkyRayCount);
            const double v = radical_inverse_vdc(static_cast<uint32_t>(i));
            const double r = std::sqrt(u);
            const double theta = 2.0 * std::numbers::pi_v<double> * v;
            const double x = r * std::cos(theta);
            const double y = r * std::sin(theta);
            const double z = std::sqrt(std::max(0.0, 1.0 - u));
            samples[i] = {x, y, z};
        }
        return samples;
    }();
    return dirs;
}

export struct VoxelBlock
{
    Vec3 position;
    uint32_t color;
    LinearColor albedo_linear;
    std::array<std::array<Vec3, 4>, 6> face_normals;
    std::array<std::array<float, 4>, 6> face_sky_visibility;
};

export struct RenderQuad
{
    std::array<Vec3, 4> v;
    std::array<Vec3, 4> n;
    std::array<float, 4> sky_visibility;
    uint32_t color;
};

export struct Terrain
{
    std::vector<VoxelBlock> blocks;
    std::vector<RenderQuad> mesh;
    int chunk_size = kTerrainChunkSize;

    auto generate() -> void
    {
        std::call_once(init_flag, [&]() { build_chunk(); });
    }

    auto triangle_count() const -> size_t
    {
        return mesh.size() * 2;
    }

    auto block_count() const -> size_t
    {
        return blocks.size();
    }

    auto visible_face_count() const -> size_t
    {
        return visible_faces;
    }

    auto max_height() const -> int
    {
        return max_height_value;
    }

    auto block_size() const -> double
    {
        return kTerrainBlockSize;
    }

    auto start_z() const -> double
    {
        return kTerrainStartZ;
    }

    auto base_y() const -> double
    {
        return kTerrainBaseY;
    }

    auto start_x() const -> double
    {
        return -(static_cast<double>(chunk_size) - 1.0) * block_size() * 0.5;
    }

    auto has_block(const int gx, const int gy, const int gz) const -> bool
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
        if (idx >= heights.size())
        {
            return false;
        }
        return gy < heights[idx];
    }

    auto block_at(const int gx, const int gy, const int gz) const -> const VoxelBlock*
    {
        if (gx < 0 || gx >= chunk_size || gz < 0 || gz >= chunk_size)
        {
            return nullptr;
        }
        if (gy < 0 || gy >= max_height_value)
        {
            return nullptr;
        }
        const size_t slot = block_slot(gx, gy, gz);
        if (slot >= block_index.size())
        {
            return nullptr;
        }
        const int index = block_index[slot];
        if (index < 0 || static_cast<size_t>(index) >= blocks.size())
        {
            return nullptr;
        }
        return &blocks[static_cast<size_t>(index)];
    }

    auto get_sky_visibility_at(const int x, const int y, const int z,
                               const int face, const int corner) const -> std::optional<float>
    {
        if (face < 0 || face >= 6 || corner < 0 || corner >= 4)
        {
            return std::nullopt;
        }
        if (x < 0 || z < 0 || x >= chunk_size || z >= chunk_size)
        {
            return std::nullopt;
        }
        const size_t idx = static_cast<size_t>(z * chunk_size + x);
        if (idx >= heights.size())
        {
            return std::nullopt;
        }
        const int height = heights[idx];
        if (y < 0 || y >= height)
        {
            return std::nullopt;
        }
        const VoxelBlock* block = block_at(x, y, z);
        if (!block)
        {
            return std::nullopt;
        }
        return block->face_sky_visibility[static_cast<size_t>(face)][static_cast<size_t>(corner)];
    }

    auto intersects_block(const Vec3& pos) const -> bool
    {
        const double block = block_size();
        const double half = block * 0.5;
        const double gx_f = (pos.x - start_x() + half) / block;
        const double gz_f = (pos.z - start_z() + half) / block;
        const double gy_f = (base_y() - pos.y + half) / block;

        const int gx = static_cast<int>(std::floor(gx_f));
        const int gz = static_cast<int>(std::floor(gz_f));
        const int gy = static_cast<int>(std::floor(gy_f));

        if (!has_block(gx, gy, gz))
        {
            return false;
        }

        const double center_x = start_x() + gx * block;
        const double center_z = start_z() + gz * block;
        const double center_y = base_y() - gy * block;

        return std::abs(pos.x - center_x) < half &&
               std::abs(pos.y - center_y) < half &&
               std::abs(pos.z - center_z) < half;
    }

private:
    std::vector<int> heights;
    std::vector<uint32_t> top_colors;
    std::vector<int> block_index;
    int max_height_value = 0;
    bool mesh_ready = false;
    size_t visible_faces = 0;
    std::once_flag init_flag;

    auto block_slot(const int gx, const int gy, const int gz) const -> size_t
    {
        return (static_cast<size_t>(gz) * static_cast<size_t>(max_height_value) +
                static_cast<size_t>(gy)) * static_cast<size_t>(chunk_size) +
               static_cast<size_t>(gx);
    }

    auto compute_vertex_sky_visibility(const int gx, const int gy, const int gz,
                                       const int face, const int corner) const -> float
    {
        Vec3 normal{
            static_cast<double>(cubeFaceNormal[face][0]),
            static_cast<double>(cubeFaceNormal[face][1]),
            static_cast<double>(cubeFaceNormal[face][2])
        };
        auto [tangent, bitangent, forward] = Vec3::get_basis(normal);
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
                if (has_block(vx, vy, vz))
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

    auto emit_block_face_quad(const VoxelBlock& block, const int face) -> void
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
            quad.v[corner] = block.position + cubeVertices[vi];
            const int face_corner = corner_index(vi);
            const int attr_corner = face_corner < 0 ? corner : face_corner;
            quad.sky_visibility[corner] = block.face_sky_visibility[static_cast<size_t>(face)][static_cast<size_t>(attr_corner)];
            quad.n[corner] = block.face_normals[static_cast<size_t>(face)][static_cast<size_t>(attr_corner)];
        }
        mesh.push_back(quad);
    }

    auto build_mesh() -> void
    {
        if (mesh_ready)
        {
            return;
        }
        mesh_ready = true;
        mesh.clear();
        visible_faces = 0;

        if (chunk_size <= 0 || max_height_value <= 0)
        {
            return;
        }

        static constexpr std::array<int, 6> face_order = {{
            FaceFront,
            FaceBack,
            FaceLeft,
            FaceRight,
            FaceBottom,
            FaceTop
        }};
        for (int z = 0; z < chunk_size; ++z)
        {
            for (int x = 0; x < chunk_size; ++x)
            {
                const int height = heights[static_cast<size_t>(z * chunk_size + x)];
                for (int y = 0; y < height; ++y)
                {
                    const VoxelBlock* block = block_at(x, y, z);
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
                        if (!has_block(nx, ny, nz))
                        {
                            visible_faces++;
                            emit_block_face_quad(*block, face);
                        }
                    }
                }
            }
        }
    }

    auto build_chunk() -> void
    {
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

        heights.assign(static_cast<size_t>(chunk_size * chunk_size), 0);
        top_colors.assign(static_cast<size_t>(chunk_size * chunk_size), grass_color);

        auto index = [this](int x, int z) {
            return static_cast<size_t>(z * chunk_size + x);
        };
        blocks.reserve(static_cast<size_t>(chunk_size * chunk_size * (base_height + height_variation + 3)));

        for (int z = 0; z < chunk_size; ++z)
        {
            for (int x = 0; x < chunk_size; ++x)
            {
                const double h = SimplexNoise::sample(x * height_freq, z * height_freq);
                int height = base_height + static_cast<int>(((h + 1.0) * 0.5 * height_variation) + 0.5);
                if (height < 3)
                {
                    height = 3;
                }

                const double surface = SimplexNoise::sample(x * surface_freq + 100.0, z * surface_freq - 100.0);
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

        blocks.clear();
        max_height_value = 0;
        for (int value : heights)
        {
            if (value > max_height_value)
            {
                max_height_value = value;
            }
        }
        if (max_height_value < 0)
        {
            max_height_value = 0;
        }
        block_index.assign(static_cast<size_t>(chunk_size * std::max(max_height_value, 1) * chunk_size), -1);
        mesh_ready = false;

        for (int z = 0; z < chunk_size; ++z)
        {
            for (int x = 0; x < chunk_size; ++x)
            {
                const int height = heights[index(x, z)];
                const uint32_t top_color = top_colors[index(x, z)];
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
                        if (!has_block(nx, ny, nz))
                        {
                            for (int corner = 0; corner < 4; ++corner)
                            {
                                face_sky_visibility[face][corner] = compute_vertex_sky_visibility(x, y, z, face, corner);
                            }
                        }
                    }
                    const LinearColor albedo_linear = ColorSrgb::from_hex(color).to_linear();
                    blocks.push_back({
                        {start_x + x * block_size, base_y - y * block_size, start_z + z * block_size},
                        color,
                        albedo_linear,
                        face_normals,
                        face_sky_visibility
                    });
                    const size_t block_idx = blocks.size() - 1;
                    const size_t slot = block_slot(x, y, z);
                    if (slot < block_index.size())
                    {
                        block_index[slot] = static_cast<int>(block_idx);
                    }
                }
            }
        }

        build_mesh();
    }
};
