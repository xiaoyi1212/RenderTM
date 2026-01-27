module;

#include "../prelude.hpp"

module render;

void RenderEngine::set_paused(const bool paused)
{
    rotationPaused.store(paused, std::memory_order_relaxed);
}

bool RenderEngine::is_paused() const
{
    return rotationPaused.load(std::memory_order_relaxed);
}

void RenderEngine::toggle_pause()
{
    rotationPaused.store(!rotationPaused.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
}

void RenderEngine::set_light_direction(const Vec3 dir)
{
    world.sun.direction = dir;
    mark_state_dirty();
}

Vec3 RenderEngine::get_light_direction() const
{
    return world.sun.direction;
}

void RenderEngine::set_light_intensity(const double intensity)
{
    world.sun.intensity = intensity;
    mark_state_dirty();
}

double RenderEngine::get_light_intensity() const
{
    return world.sun.intensity;
}

void RenderEngine::set_sun_orbit_enabled(const bool enabled)
{
    world.sun.orbit_enabled = enabled;
    mark_state_dirty();
}

bool RenderEngine::get_sun_orbit_enabled() const
{
    return world.sun.orbit_enabled;
}

void RenderEngine::set_sun_orbit_angle(const double angle)
{
    world.sun.orbit_angle = angle;
    mark_state_dirty();
}

double RenderEngine::get_sun_orbit_angle() const
{
    return world.sun.orbit_angle;
}

void RenderEngine::set_moon_direction(const Vec3 dir)
{
    world.moon.direction = dir;
    mark_state_dirty();
}

void RenderEngine::set_moon_intensity(const double intensity)
{
    world.moon.intensity = intensity;
    mark_state_dirty();
}

void RenderEngine::set_sky_top_color(const uint32_t color)
{
    world.sky.day_zenith = ColorSrgb::from_hex(color).to_linear();
    mark_state_dirty();
}

uint32_t RenderEngine::get_sky_top_color() const
{
    return pack_color(world.sky.day_zenith.to_srgb());
}

void RenderEngine::set_sky_bottom_color(const uint32_t color)
{
    world.sky.day_horizon = ColorSrgb::from_hex(color).to_linear();
    mark_state_dirty();
}

uint32_t RenderEngine::get_sky_bottom_color() const
{
    return pack_color(world.sky.day_horizon.to_srgb());
}

void RenderEngine::set_sky_light_intensity(const double intensity)
{
    world.sky.sky_intensity = intensity;
    mark_state_dirty();
}

double RenderEngine::get_sky_light_intensity() const
{
    return world.sky.sky_intensity;
}

void RenderEngine::set_exposure(const double value)
{
    world.sky.exposure = std::max(0.0, value);
    mark_state_dirty();
}

double RenderEngine::get_exposure() const
{
    return world.sky.exposure;
}

void RenderEngine::set_taa_enabled(const bool enabled)
{
    taaEnabled.store(enabled, std::memory_order_relaxed);
    mark_state_dirty();
}

bool RenderEngine::get_taa_enabled() const
{
    return taaEnabled.load(std::memory_order_relaxed);
}

void RenderEngine::set_taa_blend(const double blend)
{
    taaBlend.store(std::clamp(blend, 0.0, 1.0), std::memory_order_relaxed);
    mark_state_dirty();
}

double RenderEngine::get_taa_blend() const
{
    return taaBlend.load(std::memory_order_relaxed);
}

void RenderEngine::set_taa_clamp_enabled(const bool enabled)
{
    taaClampEnabled.store(enabled, std::memory_order_relaxed);
    mark_state_dirty();
}

bool RenderEngine::get_taa_clamp_enabled() const
{
    return taaClampEnabled.load(std::memory_order_relaxed);
}

void RenderEngine::set_gi_enabled(const bool enabled)
{
    giEnabled.store(enabled, std::memory_order_relaxed);
    mark_state_dirty();
}

bool RenderEngine::get_gi_enabled() const
{
    return giEnabled.load(std::memory_order_relaxed);
}

void RenderEngine::set_gi_strength(const double strength)
{
    giStrength.store(std::max(0.0, strength), std::memory_order_relaxed);
    mark_state_dirty();
}

double RenderEngine::get_gi_strength() const
{
    return giStrength.load(std::memory_order_relaxed);
}

void RenderEngine::set_gi_bounce_count(const int count)
{
    giBounceCount.store(count, std::memory_order_relaxed);
    mark_state_dirty();
}

int RenderEngine::get_gi_bounce_count() const
{
    return giBounceCount.load(std::memory_order_relaxed);
}

void RenderEngine::reset_taa_history()
{
    mark_state_dirty();
}

void RenderEngine::set_ambient_occlusion_enabled(const bool enabled)
{
    ambientOcclusionEnabled.store(enabled, std::memory_order_relaxed);
    mark_state_dirty();
}

void RenderEngine::set_shadow_enabled(const bool enabled)
{
    shadowEnabled.store(enabled, std::memory_order_relaxed);
    mark_state_dirty();
}

double RenderEngine::taa_sharpen_strength() const
{
    return taaSharpenStrength.load(std::memory_order_relaxed);
}

double RenderEngine::taa_sharpen_percent() const
{
    const double max_strength = kTaaSharpenMax;
    if (max_strength <= 0.0)
    {
        return 0.0;
    }
    const double strength = taa_sharpen_strength();
    const double ratio = std::clamp(strength / max_strength, 0.0, 1.0);
    return ratio * 100.0;
}

void RenderEngine::mark_state_dirty()
{
    renderStateVersion.fetch_add(1u, std::memory_order_relaxed);
}
