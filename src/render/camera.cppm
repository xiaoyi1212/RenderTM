module;

#include "../prelude.hpp"

export module camera;

import math;

export struct Camera {
    Vec3 position{17.42, 26.26, -2.76};
    Vec2 rotation{-0.69, 0.60};

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

    [[nodiscard]]
    static auto projection(const double width, const double height,
                           const double proj_scale_x, const double proj_scale_y) -> Mat4
    {
        if (width <= 0.0 || height <= 0.0 || far_plane <= near_plane)
        {
            return Mat4::identity();
        }
        const double sx = 2.0 * proj_scale_x / width;
        const double sy = 2.0 * proj_scale_y / height;
        const double inv_range = 1.0 / (far_plane - near_plane);
        const double a = far_plane * inv_range;
        const double b = -near_plane * far_plane * inv_range;

        Mat4 m{};
        m.m[0][0] = sx;
        m.m[1][1] = -sy;
        m.m[2][2] = a;
        m.m[2][3] = b;
        m.m[3][2] = 1.0;
        return m;
    }

    [[nodiscard]]
    static auto screen_to_world(const double screen_x, const double screen_y, const double depth,
                                const Mat4& inv_vp, const double width, const double height,
                                const double proj_a, const double proj_b) -> Vec3
    {
        const double ndc_x = (screen_x / width - 0.5) * 2.0;
        const double ndc_y = (screen_y / height - 0.5) * 2.0;

        const double view_z = depth;
        const double ndc_z = proj_a + proj_b / view_z;

        const double clip_w = view_z;
        const double clip_x = ndc_x * clip_w;
        const double clip_y = ndc_y * clip_w;
        const double clip_z = ndc_z * clip_w;

        double wx = inv_vp.m[0][0] * clip_x + inv_vp.m[0][1] * clip_y + inv_vp.m[0][2] * clip_z + inv_vp.m[0][3] * clip_w;
        double wy = inv_vp.m[1][0] * clip_x + inv_vp.m[1][1] * clip_y + inv_vp.m[1][2] * clip_z + inv_vp.m[1][3] * clip_w;
        double wz = inv_vp.m[2][0] * clip_x + inv_vp.m[2][1] * clip_y + inv_vp.m[2][2] * clip_z + inv_vp.m[2][3] * clip_w;
        double ww = inv_vp.m[3][0] * clip_x + inv_vp.m[3][1] * clip_y + inv_vp.m[3][2] * clip_z + inv_vp.m[3][3] * clip_w;

        if (std::abs(ww) > 1e-6)
        {
            const double inv_ww = 1.0 / ww;
            wx *= inv_ww;
            wy *= inv_ww;
            wz *= inv_ww;
        }

        return {wx, wy, wz};
    }

    [[nodiscard]]
    static auto world_to_screen(const Mat4& vp, const Vec3& world,
                                const size_t width, const size_t height) -> Vec2
    {
        if (width == 0 || height == 0)
        {
            return {0.0, 0.0};
        }

        const double clip_x = vp.m[0][0] * world.x + vp.m[0][1] * world.y + vp.m[0][2] * world.z + vp.m[0][3];
        const double clip_y = vp.m[1][0] * world.x + vp.m[1][1] * world.y + vp.m[1][2] * world.z + vp.m[1][3];
        const double clip_w = vp.m[3][0] * world.x + vp.m[3][1] * world.y + vp.m[3][2] * world.z + vp.m[3][3];

        if (clip_w <= near_plane)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return {nan, nan};
        }

        const double inv_w = 1.0 / clip_w;
        const double ndc_x = clip_x * inv_w;
        const double ndc_y = clip_y * inv_w;
        const double screen_x_out = (ndc_x * 0.5 + 0.5) * static_cast<double>(width);
        const double screen_y_out = (ndc_y * 0.5 + 0.5) * static_cast<double>(height);
        return {screen_x_out, screen_y_out};
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
