module;

#include "../prelude.hpp"

export module math;

export template<bool IsLinear>
struct ColorBase {
    float r, g, b;

    using Color = ColorBase;
    using OtherColor = ColorBase<!IsLinear>;

    [[nodiscard]]
    constexpr auto to_srgb() const -> OtherColor
    requires (IsLinear) {
        auto convert = [](float c) -> float {
            c = std::clamp(c, 0.0f, 1.0f);
            float res = (c <= 0.0031308f)
                ? (c * 12.92f)
                : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
            return res * 255.0f;
        };
        return { convert(r), convert(g), convert(b) };
    }

    [[nodiscard]]
    constexpr auto to_linear() const -> OtherColor
    requires (!IsLinear) {
        auto convert = [](float c) -> float {
            c /= 255.0f;
            if (c <= 0.04045f) return c / 12.92f;
            return std::pow((c + 0.055f) / 1.055f, 2.4f);
        };
        return { convert(r), convert(g), convert(b) };
    }

    [[nodiscard]]
    constexpr auto operator+(const Color& rhs) const -> Color
    {
        return {r + rhs.r, g + rhs.g, b + rhs.b};
    }

    [[nodiscard]]
    constexpr auto operator*(const Color& rhs) const -> Color
    {
        return {r * rhs.r, g * rhs.g, b * rhs.b};
    }

    [[nodiscard]]
    constexpr auto operator*(const float scale) const -> Color
    {
        return {r * scale, g * scale, b * scale};
    }

    [[nodiscard]]
    static constexpr auto from_hex(uint32_t hex) -> Color
    requires (!IsLinear) {
        return {
            static_cast<float>((hex >> 16) & 0xFF),
            static_cast<float>((hex >> 8) & 0xFF),
            static_cast<float>(hex & 0xFF)
        };
    }

    [[nodiscard]]
    static constexpr auto lerp(const Color& a, const Color& b, const float t) -> Color
    {
        return {
            a.r + (b.r - a.r) * t,
            a.g + (b.g - a.g) * t,
            a.b + (b.b - a.b) * t
        };
    }
};

export using LinearColor = ColorBase<true>;
export using ColorSrgb   = ColorBase<false>;

export struct Scalar
{
    [[nodiscard]]
    static constexpr auto smoothstep(const float edge0, const float edge1, const float x) -> float
    {
        if (edge0 == edge1)
        {
            return x < edge0 ? 0.0f : 1.0f;
        }
        const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }
};

export struct Vec2
{
    double x, y;

    [[nodiscard]]
    static constexpr auto zero() -> Vec2
    {
        return {0.0, 0.0};
    }
};

export struct Vec3
{
    double x, y, z;

    [[nodiscard]]
    static constexpr auto zero() -> Vec3
    {
        return {0.0, 0.0, 0.0};
    }

    [[nodiscard]]
    constexpr auto operator+(const Vec3& rhs) const -> Vec3
    {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }

    [[nodiscard]]
    constexpr auto operator-(const Vec3& rhs) const -> Vec3
    {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }

    [[nodiscard]]
    constexpr auto operator*(double scalar) const -> Vec3
    {
        return {x * scalar, y * scalar, z * scalar};
    }

    [[nodiscard]]
    constexpr auto dot(const Vec3& rhs) const -> double
    {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    [[nodiscard]]
    constexpr auto cross(const Vec3& rhs) const -> Vec3
    {
        return {
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        };
    }

    [[nodiscard]]
    constexpr auto normalize() const -> Vec3
    {
        const double len = std::sqrt(x * x + y * y + z * z);
        if (len == 0.0)
        {
            return {0.0, 0.0, 0.0};
        }
        return Vec3{x / len, y / len, z / len};
    }

    [[nodiscard]]
    static constexpr auto get_basis(const Vec3& n) -> std::tuple<Vec3, Vec3, Vec3> {
        Vec3 helper = {0.0, 1.0, 0.0};
        if (std::abs(n.dot(helper)) > 0.99) {
            helper = {0.0, 0.0, 1.0};
        }
        Vec3 right = helper.cross(n).normalize();
        Vec3 up = n.cross(right);
        return {right, up, n};
    }

    [[nodiscard]]
    static constexpr auto lerp(const Vec3& a, const Vec3& b, const double t) -> Vec3
    {
        return {
            std::lerp(a.x, b.x, t),
            std::lerp(a.y, b.y, t),
            std::lerp(a.z, b.z, t)
        };
    }
};

export struct Mat4
{
    std::array<std::array<double, 4>, 4> m{};

    [[nodiscard]]
    static constexpr auto identity() -> Mat4
    {
        Mat4 out{};
        out.m[0][0] = 1.0;
        out.m[1][1] = 1.0;
        out.m[2][2] = 1.0;
        out.m[3][3] = 1.0;
        return out;
    }

    [[nodiscard]]
    constexpr auto operator*(const Mat4& rhs) const -> Mat4
    {
        Mat4 out{};
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                double sum = 0.0;
                for (int k = 0; k < 4; ++k)
                {
                    sum += m[i][k] * rhs.m[k][j];
                }
                out.m[i][j] = sum;
            }
        }
        return out;
    }

    [[nodiscard]]
    constexpr auto invert() const -> std::optional<Mat4>
    {
        Mat4 out{};
        std::array<std::array<double, 8>, 4> aug{};
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                aug[i][j] = m[i][j];
            }
            for (int j = 0; j < 4; ++j)
            {
                aug[i][j + 4] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int col = 0; col < 4; ++col)
        {
            int pivot = col;
            double max_abs = std::fabs(aug[col][col]);
            for (int row = col + 1; row < 4; ++row)
            {
                const double value = std::fabs(aug[row][col]);
                if (value > max_abs)
                {
                    max_abs = value;
                    pivot = row;
                }
            }
            if (max_abs < 1e-12)
            {
                return std::nullopt;
            }
            if (pivot != col)
            {
                for (int j = 0; j < 8; ++j)
                {
                    std::swap(aug[col][j], aug[pivot][j]);
                }
            }

            const double inv_pivot = 1.0 / aug[col][col];
            for (int j = 0; j < 8; ++j)
            {
                aug[col][j] *= inv_pivot;
            }

            for (int row = 0; row < 4; ++row)
            {
                if (row == col)
                {
                    continue;
                }
                const double factor = aug[row][col];
                if (factor == 0.0)
                {
                    continue;
                }
                for (int j = 0; j < 8; ++j)
                {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                out.m[i][j] = aug[i][j + 4];
            }
        }
        return out;
    }

};
