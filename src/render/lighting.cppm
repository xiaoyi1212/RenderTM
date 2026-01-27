module;

#include "../prelude.hpp"

module render;

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

auto LightingEngine::compute_hemisphere_ground(const LinearColor& base_ground,
                                               const std::array<ShadingContext::DirectionalLightInfo, 2>& lights) const -> LinearColor
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
    return LinearColor::lerp(base_ground, kHemisphereBounceColorLinear, static_cast<float>(bounce));
}

auto LightingEngine::jitter_shadow_direction(const Vec3& light_dir,
                                             const Vec3& right_scaled,
                                             const Vec3& up_scaled,
                                             const int px, const int py,
                                             const BlueNoise::Shift& shift_u,
                                             const BlueNoise::Shift& shift_v) const -> Vec3
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

auto LightingEngine::gi_raymarch_hit(const Terrain& terrain,
                                     const Vec3& world, const Vec3& normal, const Vec3& dir,
                                     const double max_distance, GiHit* out_hit) const -> bool
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

auto LightingEngine::shadow_raymarch_hit(const Terrain& terrain,
                                         const Vec3& world, const Vec3& normal,
                                         const Vec3& light_dir) const -> bool
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

auto LightingEngine::compute_shadow_factor(const Terrain& terrain,
                                           const Vec3& light_dir, const Vec3& world, const Vec3& normal) const -> float
{
    const double ndotl = std::max(0.0, normal.dot(light_dir));
    if (ndotl <= 0.0)
    {
        return 1.0f;
    }
    return shadow_raymarch_hit(terrain, world, normal, light_dir) ? 0.0f : 1.0f;
}

auto LightingEngine::shadow_filter_at(std::span<const float> mask, std::span<const float> depth,
                                      std::span<const Vec3> normals, const size_t width, const size_t height,
                                      const int x, const int y, const float depth_max) const -> float
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

auto LightingEngine::filter_shadow_masks(std::span<const float> mask_a, std::span<const float> mask_b,
                                         std::span<float> out_a, std::span<float> out_b,
                                         std::span<const float> depth, std::span<const Vec3> normals,
                                         const size_t width, const size_t height, const float depth_max) const -> void
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
