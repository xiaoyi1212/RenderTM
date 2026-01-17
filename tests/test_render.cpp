#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "render.h"

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
}

static uint32_t lerp_channel(uint32_t top, uint32_t bottom, float t)
{
    const float value = static_cast<float>(top) + (static_cast<float>(bottom) - static_cast<float>(top)) * t;
    const long rounded = std::lround(value);
    if (rounded < 0) return 0;
    if (rounded > 255) return 255;
    return static_cast<uint32_t>(rounded);
}

static uint32_t sky_color_for_row(size_t y, size_t height, uint32_t sky_top, uint32_t sky_bottom)
{
    if (height <= 1)
    {
        return sky_top;
    }
    const float t = static_cast<float>(y) / static_cast<float>(height - 1);
    const uint32_t r0 = (sky_top >> 16) & 0xFF;
    const uint32_t g0 = (sky_top >> 8) & 0xFF;
    const uint32_t b0 = sky_top & 0xFF;
    const uint32_t r1 = (sky_bottom >> 16) & 0xFF;
    const uint32_t g1 = (sky_bottom >> 8) & 0xFF;
    const uint32_t b1 = sky_bottom & 0xFF;
    const uint32_t r = lerp_channel(r0, r1, t);
    const uint32_t g = lerp_channel(g0, g1, t);
    const uint32_t b = lerp_channel(b0, b1, t);
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
    const uint32_t dirt_color = 0xFFB36A2E;
    const uint32_t grass_color = 0xFF3AAA35;
    const uint32_t water_color = 0xFF3B7BFF;

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

static bool find_sloped_grass_cell(int& out_x, int& out_z, int& out_height)
{
    const int chunk_size = 16;
    const uint32_t grass_color = 0xFF3AAA35;

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

TEST_CASE("render_update_array clears framebuffer and draws geometry")
{
    reset_camera();
    render_set_scene(RenderScene::CubeOnly);
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
    render_set_scene(RenderScene::CubeOnly);
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

TEST_CASE("render_update_array is stable when paused")
{
    reset_camera();
    const size_t width = 64;
    const size_t height = 64;

    std::vector<uint32_t> framebuffer_a(width * height, 0u);
    std::vector<uint32_t> framebuffer_b(width * height, 0u);

    render_set_scene(RenderScene::CubeOnly);
    render_set_light_direction({0.0, 0.0, 1.0});
    render_set_light_intensity(1.0);
    render_set_paused(true);
    render_update_array(framebuffer_a.data(), width, height);
    render_update_array(framebuffer_b.data(), width, height);
    render_set_paused(false);

    REQUIRE(framebuffer_a == framebuffer_b);
}

TEST_CASE("render_update_array fills front face at zero rotation")
{
    reset_camera();
    const size_t width = 120;
    const size_t height = 80;

    std::vector<uint32_t> framebuffer(width * height, 0u);

    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.0f);
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

    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.0f);
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

    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.6f);
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
    render_set_scene(RenderScene::CubeOnly);
    REQUIRE(render_get_scene() == RenderScene::CubeOnly);

    render_set_rotation(1.234f);
    REQUIRE(render_get_rotation() == Catch::Approx(1.234f));

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

    render_set_paused(false);
    REQUIRE_FALSE(render_is_paused());
    render_toggle_pause();
    REQUIRE(render_is_paused());
    render_toggle_pause();
    REQUIRE_FALSE(render_is_paused());

    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.0f);
    render_set_light_direction({0.0, 0.0, -1.0});
    render_set_light_intensity(1.0);
    render_set_sky_top_color(0xFF78C2FF);
    render_set_sky_bottom_color(0xFF172433);
    render_set_sky_light_intensity(0.0);
    render_set_paused(false);
}

TEST_CASE("render_update_array renders sky gradient")
{
    reset_camera();
    render_set_camera_position({0.0, 25.0, -10.0});
    render_set_camera_rotation({0.0, -0.8});
    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.0f);
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

TEST_CASE("sky color affects anti-aliased edges")
{
    reset_camera();
    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.0f);
    render_set_paused(true);
    render_set_light_direction({0.0, 0.0, 1.0});
    render_set_light_intensity(1.0);
    render_set_sky_light_intensity(0.0);

    const size_t width = 120;
    const size_t height = 80;
    std::vector<uint32_t> frame_a(width * height, 0u);
    std::vector<uint32_t> frame_b(width * height, 0u);

    const uint32_t top_a = 0xFFFF00FF;
    const uint32_t bottom_a = 0xFF00FFFF;
    const uint32_t top_b = 0xFF112244;
    const uint32_t bottom_b = 0xFFCC6600;

    render_set_sky_top_color(top_a);
    render_set_sky_bottom_color(bottom_a);
    render_update_array(frame_a.data(), width, height);

    render_set_sky_top_color(top_b);
    render_set_sky_bottom_color(bottom_b);
    render_update_array(frame_b.data(), width, height);
    render_set_paused(false);

    bool found = false;
    for (size_t y = 0; y < height && !found; ++y)
    {
        const uint32_t sky_a = sky_color_for_row(y, height, top_a, bottom_a);
        const uint32_t sky_b = sky_color_for_row(y, height, top_b, bottom_b);
        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            const uint32_t pa = frame_a[idx];
            const uint32_t pb = frame_b[idx];
            if (pa != sky_a && pb != sky_b && pa != pb)
            {
                found = true;
                break;
            }
        }
    }

    REQUIRE(found);
}

TEST_CASE("sky light increases ambient brightness")
{
    reset_camera();
    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.0f);
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

TEST_CASE("directional shadow mapping darkens terrain")
{
    reset_camera();
    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.2f);
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
            const Vec3 world{
                start_x + x * block_size,
                base_y - (height - 1) * block_size,
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

TEST_CASE("moon light contributes when sun is disabled")
{
    reset_camera();
    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.1f);
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
    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.0f);
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

TEST_CASE("vertex ambient occlusion darkens terrain when enabled")
{
    reset_camera();
    render_set_scene(RenderScene::CubeOnly);
    render_set_rotation(0.0f);
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
    REQUIRE(avg_no - avg_with > 0.5);
}

TEST_CASE("flat terrain corners do not self-occlude with ambient occlusion")
{
    reset_camera();

    FlatCornerProbe probe{};
    REQUIRE(find_flat_shared_corner(probe));

    float ao[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    REQUIRE(render_get_terrain_top_ao(probe.x, probe.z, ao));

    int corner_index = 0;
    if (probe.sx == 1 && probe.sz == -1) corner_index = 1;
    else if (probe.sx == 1 && probe.sz == 1) corner_index = 2;
    else if (probe.sx == -1 && probe.sz == 1) corner_index = 3;

    REQUIRE(ao[corner_index] == Catch::Approx(1.0f).margin(1e-4f));
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
    render_set_camera_position({0.0, 0.0, 0.0});
    render_set_camera_rotation({0.0, 0.0});
    render_move_camera({1.0, -2.0, 3.0});
    Vec3 pos = render_get_camera_position();
    REQUIRE(pos.x == Catch::Approx(1.0));
    REQUIRE(pos.y == Catch::Approx(-2.0));
    REQUIRE(pos.z == Catch::Approx(3.0));

    render_move_camera_local({0.0, 0.0, 1.0});
    pos = render_get_camera_position();
    REQUIRE(pos.x == Catch::Approx(1.0));
    REQUIRE(pos.y == Catch::Approx(-2.0));
    REQUIRE(pos.z == Catch::Approx(4.0));

    constexpr double half_pi = 1.5707963267948966;
    render_set_camera_position({0.0, 0.0, 0.0});
    render_set_camera_rotation({half_pi, 0.0});
    render_move_camera_local({0.0, 0.0, 1.0});
    pos = render_get_camera_position();
    REQUIRE(pos.x == Catch::Approx(1.0));
    REQUIRE(pos.z == Catch::Approx(0.0).margin(1e-6));
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
