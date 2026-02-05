module;

#include "../prelude.hpp"

export module world;

import math;
import noise;

export struct Celestial
{
    Vec3 direction{0.0, 0.0, 0.0};
    LinearColor color{1.0f, 1.0f, 1.0f};
    double intensity = 0.0;
    double angular_radius = 0.0;
    bool orbit_enabled = false;
    double orbit_angle = 0.0;
    double orbit_speed = 0.0;
    double orbit_latitude_deg = 0.0;
    double night_length_ratio = 0.25;

    static constexpr double kPi = std::numbers::pi_v<double>;
    static constexpr double kTau = std::numbers::pi_v<double> * 2.0;

    auto update_orbit(const bool paused) -> void
    {
        if (!orbit_enabled)
        {
            return;
        }
        if (!paused)
        {
            orbit_angle += orbit_speed;
            orbit_angle = std::fmod(orbit_angle, kTau);
            if (orbit_angle < 0.0)
            {
                orbit_angle += kTau;
            }
        }
        direction = direction_at(orbit_angle);
    }

    [[nodiscard]]
    auto height_factor(const Vec3& dir) const -> double
    {
        const double latitude_rad = orbit_latitude_deg * kPi / 180.0;
        const double max_y = std::cos(latitude_rad);
        if (max_y <= 0.0)
        {
            return 0.0;
        }
        const double height = dir.y > 0.0 ? dir.y / max_y : 0.0;
        return std::clamp(height, 0.0, 1.0);
    }

    [[nodiscard]]
    auto height_signed(const Vec3& dir) const -> double
    {
        const double latitude_rad = orbit_latitude_deg * kPi / 180.0;
        const double max_y = std::cos(latitude_rad);
        if (max_y <= 0.0)
        {
            return 0.0;
        }
        const double height = dir.y / max_y;
        return std::clamp(height, -1.0, 1.0);
    }

    [[nodiscard]]
    auto direction_at(const double angle) const -> Vec3
    {
        const double latitude_rad = orbit_latitude_deg * kPi / 180.0;
        const double max_alt = kPi * 0.5 - latitude_rad;
        const double phase = orbit_phase(angle);
        const double alt = max_alt * std::sin(phase);
        const double az = phase - kPi * 0.5;
        const double cos_alt = std::cos(alt);
        const double sin_alt = std::sin(alt);
        const double x = cos_alt * std::sin(az);
        const double z = cos_alt * std::cos(az);
        const double y = sin_alt;
        return Vec3{x, y, z}.normalize();
    }

private:
    [[nodiscard]]
    auto orbit_phase(const double angle) const -> double
    {
        double wrapped = std::fmod(angle, kTau);
        if (wrapped < 0.0)
        {
            wrapped += kTau;
        }
        const double night_ratio = std::max(0.0, night_length_ratio);
        const double day_ratio = 1.0;
        const double total = day_ratio + night_ratio;
        if (total <= std::numeric_limits<double>::epsilon())
        {
            return wrapped;
        }
        const double day_fraction = day_ratio / total;
        const double night_fraction = night_ratio / total;
        const double t = wrapped / kTau;
        if (night_ratio <= std::numeric_limits<double>::epsilon())
        {
            return t * kPi;
        }
        if (t < day_fraction)
        {
            return (t / day_fraction) * kPi;
        }
        const double nt = (t - day_fraction) / night_fraction;
        return kPi + nt * kPi;
    }
};

export struct Skybox
{
    struct Gradient
    {
        LinearColor zenith;
        LinearColor horizon;
    };

    struct State
    {
        LinearColor zenith;
        LinearColor horizon;
        float intensity = 0.0f;
        float sun_height = 1.0f;
    };

    double sky_light_scale = 0.55;
    double sun_weight = 0.5;
    double moon_ambient_floor = 0.0;
    double exposure = 1.0;

    LinearColor day_zenith = ColorSrgb::from_hex(0xFF6FB7FF).to_linear();
    LinearColor day_horizon = ColorSrgb::from_hex(0xFFBFDFFF).to_linear();
    LinearColor golden_zenith = ColorSrgb::from_hex(0xFFF2E0C8).to_linear();
    LinearColor golden_horizon = ColorSrgb::from_hex(0xFFE2C299).to_linear();
    LinearColor dawn_zenith = ColorSrgb::from_hex(0xFFE09555).to_linear();
    LinearColor dawn_horizon = ColorSrgb::from_hex(0xFFB85C2E).to_linear();
    LinearColor blue_zenith = ColorSrgb::from_hex(0xFF4B3F7A).to_linear();
    LinearColor blue_horizon = ColorSrgb::from_hex(0xFF262B52).to_linear();
    LinearColor night_zenith = ColorSrgb::from_hex(0xFF090C17).to_linear();
    LinearColor night_horizon = ColorSrgb::from_hex(0xFF04060C).to_linear();
    LinearColor ambient_night_zenith = ColorSrgb::from_hex(0xFF1A2236).to_linear();
    LinearColor ambient_night_horizon = ColorSrgb::from_hex(0xFF0E1424).to_linear();

    double golden_height = 0.7;
    double golden_end = 0.25;
    double blue_height = -0.25;
    double night_height = -0.6;

    double dusk_light_ratio = 0.75;
    double blue_hour_light_ratio = 0.60;
    double night_light_ratio = 0.55;
    double midnight_light_ratio = 0.40;
    double star_small_threshold = 0.96;
    double star_large_threshold = 0.98;
    double star_glow_scale = 1.4;
    double star_fine_scale = 180.0;
    double star_big_scale = 45.0;
    double star_big_offset_u = 17.0;
    double star_big_offset_v = 29.0;
    double star_tint_r = 0.95;
    double star_tint_g = 0.98;
    double star_tint_b = 1.05;
    int star_noise_salt = 911;

    [[nodiscard]]
    auto sample(const float sun_height) const -> std::pair<LinearColor, LinearColor>
    {
        const float h = std::clamp(sun_height, -1.0f, 1.0f);
        const float golden = static_cast<float>(golden_height);
        const float golden_floor = static_cast<float>(golden_end);
        const float golden_hi = std::clamp(std::max(golden, golden_floor), 0.0f, 1.0f);
        const float golden_lo = std::clamp(std::min(golden, golden_floor), 0.0f, golden_hi);
        const float blue = static_cast<float>(blue_height);
        const float night = static_cast<float>(night_height);

        if (h >= golden_hi)
        {
            return {day_zenith, day_horizon};
        }
        if (h >= golden_lo)
        {
            const float t = Scalar::smoothstep(golden_lo, golden_hi, h);
            const auto zenith = LinearColor::lerp(golden_zenith, day_zenith, t);
            const auto horizon = LinearColor::lerp(golden_horizon, day_horizon, t);
            return {zenith, horizon};
        }
        if (h >= 0.0f)
        {
            const float t = Scalar::smoothstep(0.0f, golden_lo, h);
            const auto zenith = LinearColor::lerp(dawn_zenith, golden_zenith, t);
            const auto horizon = LinearColor::lerp(dawn_horizon, golden_horizon, t);
            return {zenith, horizon};
        }
        if (h >= blue)
        {
            const float t = Scalar::smoothstep(blue, 0.0f, h);
            const auto zenith = LinearColor::lerp(blue_zenith, dawn_zenith, t);
            const auto horizon = LinearColor::lerp(blue_horizon, dawn_horizon, t);
            return {zenith, horizon};
        }
        if (h >= night)
        {
            const float t = Scalar::smoothstep(night, blue, h);
            const auto zenith = LinearColor::lerp(night_zenith, blue_zenith, t);
            const auto horizon = LinearColor::lerp(night_horizon, blue_horizon, t);
            return {zenith, horizon};
        }
        return {night_zenith, night_horizon};
    }

    [[nodiscard]]
    auto intensity(const float sun_height, const float moon_intensity) const -> float
    {
        const float h = std::clamp(sun_height, -1.0f, 1.0f);
        const float golden = static_cast<float>(golden_height);
        const float blue = static_cast<float>(blue_height);
        const float night = static_cast<float>(night_height);
        const float dusk_base = static_cast<float>(dusk_light_ratio);
        const float blue_base = static_cast<float>(blue_hour_light_ratio);
        const float night_base = static_cast<float>(night_light_ratio);
        const float midnight_base = static_cast<float>(midnight_light_ratio);

        float base = 1.0f;
        if (h >= golden)
        {
            base = 1.0f;
        }
        else if (h >= 0.0f)
        {
            float t = Scalar::smoothstep(0.0f, golden, h);
            const double power = sun_weight <= 0.0 ? 1.0 : sun_weight;
            t = static_cast<float>(std::pow(t, power));
            base = std::lerp(dusk_base, 1.0f, t);
        }
        else if (h >= blue)
        {
            const float t = Scalar::smoothstep(blue, 0.0f, h);
            base = std::lerp(blue_base, dusk_base, t);
        }
        else if (h >= night)
        {
            const float t = Scalar::smoothstep(night, blue, h);
            base = std::lerp(night_base, blue_base, t);
        }
        else
        {
            const float t = Scalar::smoothstep(-1.0f, night, h);
            base = std::lerp(midnight_base, night_base, t);
        }

        float scale = static_cast<float>(sky_light_scale) * base;
        if (moon_intensity > 0.0f)
        {
            const float moon = std::clamp(moon_intensity, 0.0f, 1.0f);
            scale += moon * static_cast<float>(moon_ambient_floor) * moon_weight(h);
        }
        return std::clamp(scale, 0.0f, 1.0f);
    }

    [[nodiscard]]
    auto moon_weight(const float sun_height) const -> float
    {
        const float h = std::clamp(sun_height, -1.0f, 1.0f);
        const float golden = static_cast<float>(golden_height);
        if (golden <= 0.0f)
        {
            return h > 0.0f ? 0.0f : 1.0f;
        }
        const float t = Scalar::smoothstep(0.0f, golden, std::max(h, 0.0f));
        return 1.0f - t;
    }

    [[nodiscard]]
    auto star_visibility(const float sun_height) const -> float
    {
        const float h = std::clamp(sun_height, -1.0f, 1.0f);
        const float blue = static_cast<float>(blue_height);
        const float night = static_cast<float>(night_height);
        if (h >= blue)
        {
            return 0.0f;
        }
        if (h <= night)
        {
            return 1.0f;
        }
        return Scalar::smoothstep(blue, night, h);
    }

    [[nodiscard]]
    auto ambient_gradient(const float sun_height) const -> Gradient
    {
        const auto [zenith, horizon] = sample(sun_height);
        const float t = star_visibility(sun_height);
        const auto ambient_zenith = LinearColor::lerp(zenith, ambient_night_zenith, t);
        const auto ambient_horizon = LinearColor::lerp(horizon, ambient_night_horizon, t);
        return {ambient_zenith, ambient_horizon};
    }

    [[nodiscard]]
    auto apply_stars(const LinearColor& sky, const size_t x, const size_t y,
                     const size_t width, const size_t height,
                     const float visibility) const -> LinearColor
    {
        if (visibility <= 0.0f || width == 0 || height == 0)
        {
            return sky;
        }
        const float star = star_intensity(x, y, width, height);
        if (star <= 0.0f)
        {
            return sky;
        }
        const float glow = star * visibility * static_cast<float>(star_glow_scale);
        return {
            sky.r + glow * static_cast<float>(star_tint_r),
            sky.g + glow * static_cast<float>(star_tint_g),
            sky.b + glow * static_cast<float>(star_tint_b)
        };
    }

    [[nodiscard]]
    auto gradient(const float sun_height) const -> Gradient
    {
        const auto [zenith, horizon] = sample(sun_height);
        return {zenith, horizon};
    }

    [[nodiscard]]
    auto state(const float sun_height, const float moon_intensity) const -> State
    {
        const auto [zenith, horizon] = sample(sun_height);
        const float scale = intensity(sun_height, moon_intensity);
        return {zenith, horizon, scale, sun_height};
    }

private:
    [[nodiscard]]
    auto star_intensity(const size_t x, const size_t y,
                        const size_t width, const size_t height) const -> float
    {
        const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        const double n_fine = SimplexNoise::sample(u * star_fine_scale, v * star_fine_scale);
        const double n_big = SimplexNoise::sample(u * star_big_scale + star_big_offset_u,
                                                  v * star_big_scale + star_big_offset_v);
        const float fine = static_cast<float>(n_fine * 0.5 + 0.5);
        const float big = static_cast<float>(n_big * 0.5 + 0.5);
        const float sparkle = BlueNoise::sample(static_cast<int>(x), static_cast<int>(y),
                                                0, star_noise_salt);
        const float small_star = Scalar::smoothstep(static_cast<float>(star_small_threshold),
                                                    1.0f, fine) * (0.6f + 0.4f * sparkle);
        const float large_star = Scalar::smoothstep(static_cast<float>(star_large_threshold),
                                                    1.0f, big);
        return std::max(small_star, large_star);
    }

};

export struct World
{
    Celestial sun{
        .direction = {0.0, 0.0, -1.0},
        .color = {1.0f, 0.94f, 0.88f},
        .intensity = 1.1,
        .angular_radius = 0.03,
        .orbit_enabled = true,
        .orbit_angle = Celestial::kPi * 0.5,
        .orbit_speed = 0.00075,
        .orbit_latitude_deg = 30.0
    };

    Celestial moon{
        .direction = {0.0, -1.0, 0.0},
        .color = {1.0f, 1.0f, 1.0f},
        .intensity = 0.006,
        .angular_radius = 0.0,
        .orbit_enabled = false,
        .orbit_angle = 0.0,
        .orbit_speed = 0.0,
        .orbit_latitude_deg = 0.0
    };

    Skybox sky{};

    auto update_orbits(const bool paused) -> void
    {
        sun.update_orbit(paused);
        moon.update_orbit(paused);
    }

};
