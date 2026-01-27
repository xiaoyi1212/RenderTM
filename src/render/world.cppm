module;

#include "../prelude.hpp"

export module render:world;

import :math;

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

    static constexpr double kPi = std::numbers::pi_v<double>;

    auto update_orbit(const bool paused) -> void
    {
        if (!orbit_enabled)
        {
            return;
        }
        if (!paused)
        {
            orbit_angle += orbit_speed;
            orbit_angle = std::fmod(orbit_angle, kPi);
            if (orbit_angle < 0.0)
            {
                orbit_angle += kPi;
            }
        }
        direction = orbit_direction(orbit_angle);
    }

    [[nodiscard]]
    auto orbit_height(const Vec3& dir) const -> double
    {
        const double latitude_rad = orbit_latitude_deg * kPi / 180.0;
        const double max_y = std::cos(latitude_rad);
        if (max_y <= 0.0)
        {
            return 0.0;
        }
        const double height = dir.y < 0.0 ? (-dir.y) / max_y : 0.0;
        return std::clamp(height, 0.0, 1.0);
    }

    [[nodiscard]]
    auto orbit_direction(const double angle) const -> Vec3
    {
        const double latitude_rad = orbit_latitude_deg * kPi / 180.0;
        const double max_alt = kPi * 0.5 - latitude_rad;
        double wrapped = std::fmod(angle, kPi);
        if (wrapped < 0.0)
        {
            wrapped += kPi;
        }
        const double alt = max_alt * std::sin(wrapped);
        const double az = wrapped - kPi * 0.5;
        const double cos_alt = std::cos(alt);
        const double sin_alt = std::sin(alt);
        const double x = cos_alt * std::sin(az);
        const double z = cos_alt * std::cos(az);
        const double y = -sin_alt;
        return Vec3{x, y, z}.normalize();
    }
};

export struct Atmosphere
{
    double sky_intensity = 0.32;
    double sun_height_power = 0.5;
    double moon_ambient_floor = 0.22;
    double exposure = 1.0;

    LinearColor day_zenith = ColorSrgb::from_hex(0xFF78C2FF).to_linear();
    LinearColor day_horizon = ColorSrgb::from_hex(0xFF172433).to_linear();
    LinearColor dawn_zenith = ColorSrgb::from_hex(0xFFB55A1A).to_linear();
    LinearColor dawn_horizon = ColorSrgb::from_hex(0xFF4A200A).to_linear();

    [[nodiscard]]
    auto sample(const float sun_height) const -> std::pair<LinearColor, LinearColor>
    {
        const float t = std::clamp(sun_height, 0.0f, 1.0f);
        auto zenith = LinearColor::lerp(dawn_zenith, day_zenith, t);
        auto horizon = LinearColor::lerp(dawn_horizon, day_horizon, t);
        return {zenith, horizon};
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
        .direction = {0.0, 1.0, 0.0},
        .color = {1.0f, 1.0f, 1.0f},
        .intensity = 0.2,
        .angular_radius = 0.0,
        .orbit_enabled = false,
        .orbit_angle = 0.0,
        .orbit_speed = 0.0,
        .orbit_latitude_deg = 0.0
    };

    Atmosphere sky{};

    struct SkyGradient
    {
        LinearColor zenith;
        LinearColor horizon;
    };

    auto update_orbits(const bool paused) -> void
    {
        sun.update_orbit(paused);
        moon.update_orbit(paused);
    }

    [[nodiscard]]
    auto sky_gradient() const -> SkyGradient
    {
        const bool sun_orbit = sun.orbit_enabled;
        const Vec3& sun_dir = sun.direction;
        const double sun_height = sun_orbit ? sun.orbit_height(sun_dir) : 1.0;
        const auto [zenith, horizon] = sky.sample(static_cast<float>(sun_height));
        return {zenith, horizon};
    }
};
