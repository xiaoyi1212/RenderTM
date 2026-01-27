module;

#include "../prelude.hpp"

export module render:camera;

import :math;

export struct Camera {
    Vec3 position{16.0, -19.72, -1.93};
    Vec2 rotation{-0.69, -0.60};

    static constexpr double near_plane = 0.05;
    static constexpr double far_plane = 1000.0;

    auto move_local(const Vec3& delta) -> void
    {
        position = position + rotate_yaw_pitch(delta, rotation.x, rotation.y);
    }

    auto rotate(const Vec2& delta) -> void
    {
        rotation.x += delta.x;
        rotation.y = clamp_pitch(rotation.y + delta.y);
    }

    auto set_rotation(const Vec2& rot) -> void
    {
        rotation.x = rot.x;
        rotation.y = clamp_pitch(rot.y);
    }

    [[nodiscard]]
    auto from_camera_space(const Vec3& view) const -> Vec3
    {
        return position + rotate_pitch_yaw(view, rotation.x, rotation.y);
    }

    [[nodiscard]]
    auto to_camera_space(const Vec3& world) const -> Vec3
    {
        return rotate_yaw_pitch(world - position, -rotation.x, -rotation.y);
    }

    [[nodiscard]]
    auto view_matrix() const -> Mat4
    {
        const double yaw = -rotation.x;
        const double pitch = -rotation.y;
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);
        const double cp = std::cos(pitch);
        const double sp = std::sin(pitch);

        Mat4 m = Mat4::identity();
        m.m[0][0] = cy;
        m.m[0][1] = 0.0;
        m.m[0][2] = sy;

        m.m[1][0] = sy * sp;
        m.m[1][1] = cp;
        m.m[1][2] = -cy * sp;

        m.m[2][0] = -sy * cp;
        m.m[2][1] = sp;
        m.m[2][2] = cy * cp;

        m.m[0][3] = -(m.m[0][0] * position.x + m.m[0][1] * position.y + m.m[0][2] * position.z);
        m.m[1][3] = -(m.m[1][0] * position.x + m.m[1][1] * position.y + m.m[1][2] * position.z);
        m.m[2][3] = -(m.m[2][0] * position.x + m.m[2][1] * position.y + m.m[2][2] * position.z);

        return m;
    }

private:
    [[nodiscard]]
    static constexpr auto clamp_pitch(const double pitch) -> double
    {
        return std::clamp(pitch, -1.4, 1.4);
    }

    [[nodiscard]]
    static auto rotate_yaw_pitch(const Vec3& v, const double yaw, const double pitch) -> Vec3
    {
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);
        const double cp = std::cos(pitch);
        const double sp = std::sin(pitch);

        const double x1 = v.x * cy + v.z * sy;
        const double z1 = -v.x * sy + v.z * cy;
        const double y1 = v.y;

        const double y2 = y1 * cp - z1 * sp;
        const double z2 = y1 * sp + z1 * cp;

        return {x1, y2, z2};
    }

    [[nodiscard]]
    static auto rotate_pitch_yaw(const Vec3& v, const double yaw, const double pitch) -> Vec3
    {
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);
        const double cp = std::cos(pitch);
        const double sp = std::sin(pitch);

        const double x1 = v.x;
        const double y1 = v.y * cp - v.z * sp;
        const double z1 = v.y * sp + v.z * cp;

        const double x2 = x1 * cy + z1 * sy;
        const double z2 = -x1 * sy + z1 * cy;

        return {x2, y1, z2};
    }
};
