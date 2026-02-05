module;

#include "../prelude.hpp"

export module terrain;

import math;
import noise;

export struct BlockGeometry
{
    static constexpr int FaceTop = 0;
    static constexpr int FaceBottom = 1;
    static constexpr int FaceLeft = 2;
    static constexpr int FaceRight = 3;
    static constexpr int FaceBack = 4;
    static constexpr int FaceFront = 5;

    static constexpr std::array<Vec3, 8> vertices = {{
        {-1, -1, -1}, {1, -1, -1}, {1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, {1, -1,  1}, {1,  1,  1}, {-1,  1,  1}
    }};

    static constexpr std::array<std::array<int, 4>, 6> face_vertices = {{
        {{3, 2, 6, 7}}, // top (+y)
        {{0, 1, 5, 4}}, // bottom (-y)
        {{0, 3, 7, 4}}, // left (-x)
        {{1, 2, 6, 5}}, // right (+x)
        {{0, 1, 2, 3}}, // back (-z)
        {{4, 5, 6, 7}}  // front (+z)
    }};

    static constexpr std::array<std::array<int, 4>, 6> quad_order = {{
        {{3, 7, 6, 2}}, // top (+y)
        {{0, 1, 5, 4}}, // bottom (-y)
        {{0, 4, 7, 3}}, // left (-x)
        {{1, 2, 6, 5}}, // right (+x)
        {{0, 3, 2, 1}}, // back (-z)
        {{4, 5, 6, 7}}  // front (+z)
    }};

    static constexpr std::array<std::array<int, 3>, 6> face_normals = {{
        {{0, 1, 0}},   // top (+y)
        {{0, -1, 0}},  // bottom (-y)
        {{-1, 0, 0}},  // left (-x)
        {{1, 0, 0}},   // right (+x)
        {{0, 0, -1}},  // back (-z)
        {{0, 0, 1}}    // front (+z)
    }};

    static constexpr std::array<Vec3, 8> vertices_grid = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0},
        {0.0, 1.0, 1.0}
    }};

    [[nodiscard]]
    static constexpr auto face_normal_world(const int face) -> Vec3
    {
        const double x = static_cast<double>(face_normals[face][0]);
        const double y = static_cast<double>(face_normals[face][1]);
        const double z = static_cast<double>(face_normals[face][2]);
        return {x, y, z};
    }
};

export struct TerrainConfig
{
    int chunk_size = 16;
    double block_size = 2.0;
    double start_z = 4.0;
    double base_y = 2.0;
    int base_height = 4;
    int dirt_thickness = 2;
    int height_variation = 6;
    double height_freq = 0.12;
    double surface_freq = 0.4;

    uint32_t stone_color = 0xFF7A7A7A;
    uint32_t dirt_color  = 0xFF7D4714;
    uint32_t grass_color = 0xFF3B8A38;
    uint32_t water_color = 0xFF2B5FA8;
};

export struct VoxelBlock
{
    Vec3 position;
    uint32_t color;
    LinearColor albedo_linear;
    std::array<std::array<Vec3, 4>, 6> face_normals;
    std::array<std::array<float, 4>, 6> sky_visibility;
};

export struct RenderQuad
{
    std::array<Vec3, 4> v;
    std::array<Vec3, 4> n;
    std::array<float, 4> sky_visibility;
    uint32_t color;
};

struct BlockTopology
{
    int chunk_size = 0;
    int max_height = 0;
    std::vector<int> heights;
    std::vector<int> block_index;

    [[nodiscard]]
    constexpr auto index(const int gx, const int gz) const -> size_t
    {
        return static_cast<size_t>(gz * chunk_size + gx);
    }

    [[nodiscard]]
    constexpr auto block_slot(const int gx, const int gy, const int gz) const -> size_t
    {
        const size_t x = static_cast<size_t>(gx);
        const size_t y = static_cast<size_t>(gy);
        const size_t z = static_cast<size_t>(gz);

        const size_t width  = static_cast<size_t>(chunk_size);
        const size_t height = static_cast<size_t>(max_height);

        const size_t z_stride = height * width;
        const size_t y_stride = width;
        return (z * z_stride) + (y * y_stride) + x;
    }

    [[nodiscard]]
    auto has_block(const int gx, const int gy, const int gz) const -> bool
    {
        if (gx < 0 || gx >= chunk_size || gz < 0 || gz >= chunk_size)
        {
            return false;
        }
        if (gy < 0) return false;

        const size_t idx = index(gx, gz);
        if (idx >= heights.size())
        {
            return false;
        }

        return gy < heights[idx];
    }

    [[nodiscard]]
    auto block_at(std::span<const VoxelBlock> blocks,
                  const int gx, const int gy, const int gz) const
        -> std::optional<std::reference_wrapper<const VoxelBlock>>
    {
        if (gx < 0 || gx >= chunk_size || gz < 0 || gz >= chunk_size)
        {
            return std::nullopt;
        }
        if (gy < 0 || gy >= max_height)
        {
            return std::nullopt;
        }
        const size_t slot = block_slot(gx, gy, gz);
        if (slot >= block_index.size())
        {
            return std::nullopt;
        }
        const int index = block_index[slot];
        if (index < 0 || static_cast<size_t>(index) >= blocks.size())
        {
            return std::nullopt;
        }
        return std::cref(blocks[static_cast<size_t>(index)]);
    }
};

struct Occlusion
{
    Occlusion() = delete;

    [[nodiscard]]
    static auto sample(const BlockTopology& topology, 
                       const int gx, const int gy, const int gz,
                       const int face, const int corner) -> float
    {
        Vec3 normal{
            static_cast<double>(BlockGeometry::face_normals[face][0]),
            static_cast<double>(BlockGeometry::face_normals[face][1]),
            static_cast<double>(BlockGeometry::face_normals[face][2])
        };
        auto [tangent, bitangent, forward] = Vec3::get_basis(normal);
        normal = forward;

        const Vec3 grid_pos{
            static_cast<double>(gx),
            static_cast<double>(gy),
            static_cast<double>(gz)
        };
        const int vi = BlockGeometry::face_vertices[face][corner];
        const Vec3 vertex = grid_pos + BlockGeometry::vertices_grid[vi];
        const Vec3 center = grid_pos + Vec3{0.5, 0.5, 0.5};

        const Vec3 origin = vertex + 
                            normal * kRayBias + 
                            (center - vertex) * kRayCenterBias;

        const auto& samples = sample_dirs();
        size_t occluded = 0;
        for (const auto& sample : samples)
        {
            const Vec3 dir = tangent * sample.x +
                             bitangent * sample.y +
                             normal * sample.z;

            bool hit = false;
            for (double t = kRayStep; t <= kRayMaxDistance; t += kRayStep)
            {
                const Vec3 p = origin + dir * t;
                const int vx = static_cast<int>(std::floor(p.x));
                const int vy = static_cast<int>(std::floor(p.y));
                const int vz = static_cast<int>(std::floor(p.z));
                if (topology.has_block(vx, vy, vz))
                {
                    hit = true;
                    break;
                }
            }
            if (hit) occluded++;
        }

        const double total_samples = static_cast<double>(samples.size());
        const double occlusion_ratio = static_cast<double>(occluded) / total_samples;
        const double raw_visibility = 1.0 - occlusion_ratio;

        return static_cast<float>(std::clamp(raw_visibility, 0.0, 1.0));
    }

private:
    static constexpr size_t kRayCount = 128;
    static constexpr double kRayStep = 0.25;
    static constexpr double kRayMaxDistance = 6.0;
    static constexpr double kRayBias = 0.02;
    static constexpr double kRayCenterBias = 0.02;

    [[nodiscard]]
    static auto sample_dirs() -> const std::array<Vec3, kRayCount>&
    {
        static const auto dirs = [] {
            std::array<Vec3, kRayCount> samples{};
            constexpr double total_rays = static_cast<double>(kRayCount);

            for (size_t i = 0; i < kRayCount; ++i)
            {
                const double index = static_cast<double>(i);
                const double u = (index + 0.5) / total_rays;
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

    static double radical_inverse_vdc(uint32_t bits)
    {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return static_cast<double>(bits) * 2.3283064365386963e-10;
    }
};

struct TerrainGrid
{
    double block_size = 0.0;
    double start_x = 0.0;
    double start_z = 0.0;
    double base_y = 0.0;
};

struct TerrainMeshBuilder
{
    TerrainMeshBuilder() = delete;
    using FaceOrder = std::array<int, 6>;
    
    static auto build_mesh(const BlockTopology& topology,
                           std::span<const VoxelBlock> blocks,
                           std::vector<RenderQuad>& mesh,
                           size_t& visible_faces) -> void
    {
        const int chunk_size = topology.chunk_size;
        mesh.clear();
        visible_faces = 0;

        if (chunk_size <= 0 || topology.max_height <= 0) return;

        static constexpr std::array<int, 6> kFaceOrder{{
            BlockGeometry::FaceFront, BlockGeometry::FaceBack,
            BlockGeometry::FaceLeft, BlockGeometry::FaceRight,
            BlockGeometry::FaceBottom, BlockGeometry::FaceTop
        }};

        const auto append_quad = [&](const VoxelBlock& block, int face) 
        {
            RenderQuad quad{};
            quad.color = block.color;
            
            auto face_corner_index = [&](int v_idx) {
                for (int i = 0; i < 4; ++i)
                    if (BlockGeometry::face_vertices[face][i] == v_idx)
                        return i;
                return -1;
            };

            for (int corner = 0; corner < 4; ++corner)
            {
                const int vi = BlockGeometry::quad_order[face][corner];
                quad.v[corner] = block.position + BlockGeometry::vertices[vi];
                
                const int face_corner_raw = face_corner_index(vi);
                const int attr_corner = (face_corner_raw < 0) ? corner : face_corner_raw;
                
                quad.sky_visibility[corner] = block.sky_visibility[face][attr_corner];
                quad.n[corner] = block.face_normals[face][attr_corner];
            }
            mesh.push_back(quad);
        };

        const auto process_voxel = [&](int x, int y, int z) 
        {
            const size_t slot = topology.block_slot(x, y, z);
            const int index = topology.block_index[slot];
            const VoxelBlock& block = blocks[static_cast<size_t>(index)];

            for (const int face : kFaceOrder)
            {
                const int nx = x + BlockGeometry::face_normals[face][0];
                const int ny = y + BlockGeometry::face_normals[face][1];
                const int nz = z + BlockGeometry::face_normals[face][2];

                if (!topology.has_block(nx, ny, nz))
                {
                    append_quad(block, face);
                    visible_faces++;
                }
            }
        };

        const size_t total_columns = static_cast<size_t>(chunk_size * chunk_size);
        for (size_t i = 0; i < total_columns; ++i)
        {
            const int height = topology.heights[i];
            if (height <= 0) continue;

            const int z = static_cast<int>(i / chunk_size);
            const int x = static_cast<int>(i % chunk_size);

            for (int y = 0; y < height; ++y)
            {
                process_voxel(x, y, z);
            }
        }
    }
};

struct TerrainGenerator
{
    TerrainGenerator() = delete;

    static auto build_chunk(const TerrainConfig& config,
                            BlockTopology& topology,
                            std::vector<VoxelBlock>& blocks) -> void
    {
        const int chunk_size = config.chunk_size;
        const auto grid = grid_for(config, config.chunk_size);
        const auto& normals = normals_table();
        std::vector<uint32_t> top_colors;

        init_storage(config, topology, blocks, top_colors);
        fill_heights(config, topology, top_colors);
        finalize_topology(config, topology);

        const auto add_voxel = [&](const int x, const int y, const int z, 
                                   const int height, const uint32_t top_color) 
        {
            const uint32_t color = block_color(config, y, height, top_color);
            const auto sky = face_sky(topology, x, y, z);

            const Vec3 pos{
                grid.start_x + x * grid.block_size,
                grid.base_y + y * grid.block_size,
                grid.start_z + z * grid.block_size
            };

            blocks.push_back({
                pos,
                color,
                ColorSrgb::from_hex(color).to_linear(),
                normals,
                sky
            });

            const size_t slot = topology.block_slot(x, y, z);
            topology.block_index[slot] = static_cast<int>(blocks.size() - 1);
        };

        const auto process_column = [&](const size_t col_index) 
        {
            const int height = topology.heights[col_index];
            if (height <= 0) return;

            const int z = static_cast<int>(col_index / chunk_size);
            const int x = static_cast<int>(col_index % chunk_size);
            const uint32_t top_color = top_colors[col_index];

            for (int y = 0; y < height; ++y)
            {
                add_voxel(x, y, z, height, top_color);
            }
        };

        for (size_t i = 0; i < chunk_size * chunk_size; ++i) process_column(i);
    }

private:
    using FaceNormals = std::array<std::array<Vec3, 4>, 6>;
    using FaceSky = std::array<std::array<float, 4>, 6>;

    static auto grid_for(const TerrainConfig& cfg, const int chunk_size) -> TerrainGrid
    {
        const double max_index = static_cast<double>(chunk_size) - 1.0;
        return {
            .block_size = cfg.block_size,
            .start_x = -max_index * cfg.block_size * 0.5,
            .start_z = cfg.start_z,
            .base_y = cfg.base_y
        };
    }

    static auto normals_table() -> const FaceNormals&
    {
        const auto build_normals = [] {
            FaceNormals out{};
            for (int face = 0; face < 6; ++face)
            {
                const Vec3 base = BlockGeometry::face_normal_world(face);
                for (int corner = 0; corner < 4; ++corner)
                {
                    out[face][corner] = base;
                }
            }
            return out;
        };
        static const FaceNormals normals = build_normals();
        return normals;
    }

    static auto init_storage(const TerrainConfig& config,
                             BlockTopology& topology,
                             std::vector<VoxelBlock>& blocks,
                             std::vector<uint32_t>& top_colors) -> void
    {
        const int chunk_size = config.chunk_size;
        topology.chunk_size = chunk_size;

        const auto grid_cells = chunk_size * chunk_size;
        topology.heights.assign(grid_cells, 0);
        top_colors.assign(grid_cells, config.grass_color);
        blocks.clear();

        const auto reserve_rows = config.base_height + config.height_variation + 3;
        blocks.reserve(grid_cells * static_cast<size_t>(reserve_rows));
    }

    static auto fill_heights(const TerrainConfig& config,
                             BlockTopology& topology,
                             std::vector<uint32_t>& top_colors) -> void
    {
        const int chunk_size = config.chunk_size;
        for (int z = 0; z < chunk_size; ++z)
        {
            for (int x = 0; x < chunk_size; ++x)
            {
                const size_t idx = topology.index(x, z);
                topology.heights[idx] = height_at(config, x, z);
                top_colors[idx] = top_color_at(config, x, z);
            }
        }
    }

    static auto finalize_topology(const TerrainConfig& config,
                                  BlockTopology& topology) -> void
    {
        topology.max_height = 0;
        for (int value : topology.heights)
        {
            topology.max_height = std::max(topology.max_height, value);
        }
        if (topology.max_height < 0)
        {
            topology.max_height = 0;
        }
        const size_t stride = static_cast<size_t>(config.chunk_size);
        const size_t height = static_cast<size_t>(std::max(topology.max_height, 1));
        const size_t slots = stride * height * stride;
        topology.block_index.assign(slots, -1);
    }

    [[nodiscard]]
    static auto height_at(const TerrainConfig& cfg, const int x, const int z) -> int
    {
        const double h = SimplexNoise::sample(x * cfg.height_freq, z * cfg.height_freq);
        const double scaled = (h + 1.0) * 0.5 * static_cast<double>(cfg.height_variation);
        return std::max(cfg.base_height + static_cast<int>(scaled + 0.5), 3);
    }

    [[nodiscard]]
    static auto top_color_at(const TerrainConfig& cfg, const int x, const int z) -> uint32_t
    {
        const double surface = SimplexNoise::sample(x * cfg.surface_freq + 100.0,
                                                    z * cfg.surface_freq - 100.0);
        if (surface > 0.55)
        {
            return cfg.water_color;
        }
        if (surface < -0.35)
        {
            return cfg.dirt_color;
        }
        return cfg.grass_color;
    }

    [[nodiscard]]
    static auto block_color(const TerrainConfig& cfg, const int y, const int height,
                            const uint32_t top_color) -> uint32_t
    {
        if (y >= height - 1)
        {
            return top_color;
        }
        if (y >= height - 1 - cfg.dirt_thickness)
        {
            return cfg.dirt_color;
        }
        return cfg.stone_color;
    }

    [[nodiscard]]
    static auto face_sky(const BlockTopology& topology,
                         const int x, const int y, const int z) -> FaceSky
    {
        auto sky = FaceSky{};
        for (auto& face_visibility : sky)
        {
            face_visibility.fill(0.0f);
        }

        auto fill_corners = [&](const int face) {
            for (int corner = 0; corner < 4; ++corner)
            {
                sky[face][corner] = Occlusion::sample(topology, x, y, z, face, corner);
            }
        };

        auto fill_face = [&](const int face) {
            const int nx = x + BlockGeometry::face_normals[face][0];
            const int ny = y + BlockGeometry::face_normals[face][1];
            const int nz = z + BlockGeometry::face_normals[face][2];
            if (topology.has_block(nx, ny, nz))
            {
                return;
            }
            fill_corners(face);
        };

        for (int face = 0; face < 6; ++face) fill_face(face);

        return sky;
    }
};

export struct Terrain
{
    int chunk_size = 16;
    size_t visible_faces = 0;
    TerrainConfig config{};
    BlockTopology topology;
    std::vector<VoxelBlock> blocks;
    std::vector<RenderQuad> mesh;

    auto generate() -> void
    {
        chunk_size = config.chunk_size;
        TerrainGenerator::build_chunk(config, topology, blocks);
        TerrainMeshBuilder::build_mesh(topology, blocks, mesh, visible_faces);
    }
};
