module;

#include "../prelude.hpp"

export module settings;

import math;

export struct TaaSettings
{
    bool enabled = true;
    double blend = 0.05;
    bool clamp_enabled = true;
};

export struct GiSettings
{
    bool enabled = false;
    double strength = 0.0;
    int bounce_count = 2;
    double ray_bias = 0.04;
    double max_distance = 12.0;
    int noise_salt = 73;
    int sample_count = 1;
    float clamp = 4.0f;
    float ao_lift = 0.15f;
};

export struct ShadowSettings
{
    double ray_bias = 0.05;
    int sun_salt = 17;
    int moon_salt = 19;
    bool filter_enabled = true;
    float filter_depth_threshold = 1.0f;
    float filter_normal_threshold = 0.5f;
    float filter_center_weight = 4.0f;
    float filter_neighbor_weight = 1.0f;
};

export struct LightingSettings
{
    double sun_intensity_boost = 1.2;
    double hemisphere_bounce_strength = 0.35;
    LinearColor hemisphere_bounce_color{1.0f, 0.9046612f, 0.7758222f};
};

export struct RenderSettings
{
    bool paused = false;
    double ambient_light = 0.13;
    bool ambient_occlusion_enabled = true;
    bool shadow_enabled = true;
    TaaSettings taa{};
    GiSettings gi{};
    ShadowSettings shadow{};
    LightingSettings lighting{};

    auto set_paused(const bool value) -> void
    {
        paused = value;
    }

    [[nodiscard]]
    auto is_paused() const -> bool
    {
        return paused;
    }

    auto toggle_pause() -> void
    {
        paused = !paused;
    }

    auto set_ambient_light(const double value) -> void
    {
        ambient_light = std::max(0.0, value);
    }

    [[nodiscard]]
    auto get_ambient_light() const -> double
    {
        return ambient_light;
    }

    auto set_ambient_occlusion_enabled(const bool value) -> void
    {
        ambient_occlusion_enabled = value;
    }

    [[nodiscard]]
    auto get_ambient_occlusion_enabled() const -> bool
    {
        return ambient_occlusion_enabled;
    }

    auto set_shadow_enabled(const bool value) -> void
    {
        shadow_enabled = value;
    }

    [[nodiscard]]
    auto get_shadow_enabled() const -> bool
    {
        return shadow_enabled;
    }

    auto set_taa_enabled(const bool value) -> void
    {
        taa.enabled = value;
    }

    [[nodiscard]]
    auto get_taa_enabled() const -> bool
    {
        return taa.enabled;
    }

    auto set_taa_blend(const double value) -> void
    {
        taa.blend = std::clamp(value, 0.0, 1.0);
    }

    [[nodiscard]]
    auto get_taa_blend() const -> double
    {
        return taa.blend;
    }

    auto set_taa_clamp_enabled(const bool value) -> void
    {
        taa.clamp_enabled = value;
    }

    [[nodiscard]]
    auto get_taa_clamp_enabled() const -> bool
    {
        return taa.clamp_enabled;
    }

    auto set_gi_enabled(const bool value) -> void
    {
        gi.enabled = value;
    }

    [[nodiscard]]
    auto get_gi_enabled() const -> bool
    {
        return gi.enabled;
    }

    auto set_gi_strength(const double value) -> void
    {
        gi.strength = std::max(0.0, value);
    }

    [[nodiscard]]
    auto get_gi_strength() const -> double
    {
        return gi.strength;
    }

    auto set_gi_bounce_count(const int count) -> void
    {
        gi.bounce_count = count;
    }

    [[nodiscard]]
    auto get_gi_bounce_count() const -> int
    {
        return gi.bounce_count;
    }
};
