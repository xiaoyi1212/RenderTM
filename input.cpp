#include "input.h"

#include <algorithm>
#include <cctype>
#include <cmath>

InputAction input_map_key(const int ch)
{
    if (ch == 'q' || ch == 'Q') return InputAction::Quit;
    if (ch == 'p' || ch == 'P') return InputAction::TogglePause;
    if (ch == 'g' || ch == 'G') return InputAction::ToggleGI;
    if (ch == 'w' || ch == 'W') return InputAction::MoveForward;
    if (ch == 's' || ch == 'S') return InputAction::MoveBackward;
    if (ch == 'a' || ch == 'A') return InputAction::MoveLeft;
    if (ch == 'd' || ch == 'D') return InputAction::MoveRight;
    if (ch == 'r' || ch == 'R') return InputAction::MoveUp;
    if (ch == 'f' || ch == 'F') return InputAction::MoveDown;
    return InputAction::None;
}

MouseParseResult input_parse_sgr_mouse(const std::string_view buffer, size_t* consumed, MouseEvent* out_event)
{
    if (consumed)
    {
        *consumed = 0;
    }
    if (buffer.size() < 3)
    {
        return MouseParseResult::NeedMore;
    }
    if (buffer[0] != '\x1b' || buffer[1] != '[' || buffer[2] != '<')
    {
        return MouseParseResult::Invalid;
    }

    size_t idx = 3;
    auto parse_number = [&](int& value) -> MouseParseResult {
        if (idx >= buffer.size())
        {
            return MouseParseResult::NeedMore;
        }
        int parsed = 0;
        bool has_digit = false;
        while (idx < buffer.size())
        {
            const unsigned char c = static_cast<unsigned char>(buffer[idx]);
            if (!std::isdigit(c))
            {
                break;
            }
            parsed = parsed * 10 + (c - '0');
            idx++;
            has_digit = true;
        }
        if (!has_digit)
        {
            return MouseParseResult::Invalid;
        }
        value = parsed;
        return MouseParseResult::Parsed;
    };

    int button = 0;
    int x = 0;
    int y = 0;

    MouseParseResult result = parse_number(button);
    if (result != MouseParseResult::Parsed)
    {
        return result;
    }
    if (idx >= buffer.size())
    {
        return MouseParseResult::NeedMore;
    }
    if (buffer[idx] != ';')
    {
        return MouseParseResult::Invalid;
    }
    idx++;

    result = parse_number(x);
    if (result != MouseParseResult::Parsed)
    {
        return result;
    }
    if (idx >= buffer.size())
    {
        return MouseParseResult::NeedMore;
    }
    if (buffer[idx] != ';')
    {
        return MouseParseResult::Invalid;
    }
    idx++;

    result = parse_number(y);
    if (result != MouseParseResult::Parsed)
    {
        return result;
    }
    if (idx >= buffer.size())
    {
        return MouseParseResult::NeedMore;
    }

    const char terminator = buffer[idx];
    if (terminator != 'M' && terminator != 'm')
    {
        return MouseParseResult::Invalid;
    }
    idx++;

    if (consumed)
    {
        *consumed = idx;
    }
    if (out_event)
    {
        out_event->button = button;
        out_event->x = x;
        out_event->y = y;
        out_event->motion = (button & 32) != 0;
        out_event->pressed = (terminator == 'M');
    }
    return MouseParseResult::Parsed;
}

InputParseResult input_parse_csi_key(const std::string_view buffer, size_t* consumed, InputAction* out_action)
{
    if (consumed)
    {
        *consumed = 0;
    }
    if (buffer.size() < 2)
    {
        return InputParseResult::NeedMore;
    }
    if (buffer[0] != '\x1b' || buffer[1] != '[')
    {
        return InputParseResult::Invalid;
    }
    if (buffer.size() < 3)
    {
        return InputParseResult::NeedMore;
    }
    InputAction action = InputAction::None;
    switch (buffer[2])
    {
        case 'A':
            action = InputAction::MoveForward;
            break;
        case 'B':
            action = InputAction::MoveBackward;
            break;
        case 'C':
            action = InputAction::MoveRight;
            break;
        case 'D':
            action = InputAction::MoveLeft;
            break;
        default:
            return InputParseResult::Invalid;
    }
    if (consumed)
    {
        *consumed = 3;
    }
    if (out_action)
    {
        *out_action = action;
    }
    return InputParseResult::Parsed;
}

MouseLookDelta input_mouse_look_velocity(const int mouse_x, const int mouse_y,
                                         const int width, const int height,
                                         const int deadzone_radius, const double max_speed)
{
    if (width <= 0 || height <= 0 || max_speed <= 0.0)
    {
        return {0.0, 0.0};
    }

    const double center_x = (static_cast<double>(width) + 1.0) * 0.5;
    const double center_y = (static_cast<double>(height) + 1.0) * 0.5;
    const double dx = static_cast<double>(mouse_x) - center_x;
    const double dy = static_cast<double>(mouse_y) - center_y;

    if (std::abs(dx) <= deadzone_radius && std::abs(dy) <= deadzone_radius)
    {
        return {0.0, 0.0};
    }

    const double max_dx = std::max(center_x - 1.0, static_cast<double>(width) - center_x);
    const double max_dy = std::max(center_y - 1.0, static_cast<double>(height) - center_y);
    const double avail_x = max_dx - static_cast<double>(deadzone_radius);
    const double avail_y = max_dy - static_cast<double>(deadzone_radius);
    if (avail_x <= 0.0 || avail_y <= 0.0)
    {
        return {0.0, 0.0};
    }

    const double mag_x = std::clamp((std::abs(dx) - deadzone_radius) / avail_x, 0.0, 1.0);
    const double mag_y = std::clamp((std::abs(dy) - deadzone_radius) / avail_y, 0.0, 1.0);
    const double yaw = std::copysign(mag_x * max_speed, dx);
    const double pitch = -std::copysign(mag_y * max_speed, dy);
    return {yaw, pitch};
}
