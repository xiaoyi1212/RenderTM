module;

#include "prelude.hpp"

export module noise;

export constexpr int kBlueNoiseSize = 64;
export constexpr int kBlueNoiseMask = kBlueNoiseSize - 1;
export constexpr unsigned char kBlueNoise[4096] = {
#include "blue_noise.inc"
};

constexpr uint32_t hash32(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

export struct BlueNoiseShift
{
    uint32_t x;
    uint32_t y;
};

export BlueNoiseShift blue_noise_shift(int frame, int salt)
{
    const uint32_t seed = static_cast<uint32_t>(frame) ^
                          (static_cast<uint32_t>(salt) * 0x9e3779b9u);
    return {
        hash32(seed) & static_cast<uint32_t>(kBlueNoiseMask),
        hash32(seed ^ 0x85ebca6bu) & static_cast<uint32_t>(kBlueNoiseMask)
    };
}

export float sample_noise_shifted(int x, int y, const BlueNoiseShift& shift)
{
    const int nx = (x + static_cast<int>(shift.x)) & kBlueNoiseMask;
    const int ny = (y + static_cast<int>(shift.y)) & kBlueNoiseMask;
    const int idx = ny * kBlueNoiseSize + nx;
    return (static_cast<float>(kBlueNoise[idx]) + 0.5f) / 256.0f;
}

export float sample_noise(int x, int y, int frame, int salt)
{
    return sample_noise_shifted(x, y, blue_noise_shift(frame, salt));
}
