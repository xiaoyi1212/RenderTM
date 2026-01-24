module;

#include "prelude.hpp"

export module input;

export enum class InputAction
{
    None,
    Quit,
    TogglePause,
    ToggleGI,
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown
};

export enum class MouseParseResult
{
    Parsed,
    NeedMore,
    Invalid
};

export enum class InputParseResult
{
    Parsed,
    NeedMore,
    Invalid
};

export struct MouseEvent
{
    int button;
    int x;
    int y;
    bool motion;
    bool pressed;
};

export struct MouseParse
{
    MouseParseResult result = MouseParseResult::Invalid;
    size_t consumed = 0;
    MouseEvent event{};
};

export struct InputParse
{
    InputParseResult result = InputParseResult::Invalid;
    size_t consumed = 0;
    InputAction action = InputAction::None;
};

export struct MouseLookDelta
{
    double yaw;
    double pitch;
};

export struct InputParser
{
    static InputAction key_to_action(int ch);
    static MouseParse parse_sgr_mouse(std::string_view buffer);
    static InputParse parse_csi_key(std::string_view buffer);
    static MouseLookDelta mouse_look_velocity(int mouse_x, int mouse_y, int width, int height,
                                              int deadzone_radius, double max_speed);
};

InputAction InputParser::key_to_action(const int ch)
{
    if (ch < 0)
    {
        return InputAction::None;
    }
    switch (std::tolower(static_cast<unsigned char>(ch)))
    {
        case 'q': return InputAction::Quit;
        case 'p': return InputAction::TogglePause;
        case 'g': return InputAction::ToggleGI;
        case 'w': return InputAction::MoveForward;
        case 's': return InputAction::MoveBackward;
        case 'a': return InputAction::MoveLeft;
        case 'd': return InputAction::MoveRight;
        case 'r': return InputAction::MoveUp;
        case 'f': return InputAction::MoveDown;
        default: return InputAction::None;
    }
}

namespace {

struct ParseCursor
{
    std::string_view buffer;
    size_t idx = 0;

    bool has(size_t count = 1) const
    {
        return idx + count <= buffer.size();
    }

    MouseParseResult parse_int(int& value)
    {
        if (!has())
        {
            return MouseParseResult::NeedMore;
        }
        const char* begin = buffer.data() + idx;
        const char* end = buffer.data() + buffer.size();
        const auto result = std::from_chars(begin, end, value);
        if (result.ptr == begin)
        {
            return MouseParseResult::Invalid;
        }
        idx = static_cast<size_t>(result.ptr - buffer.data());
        return MouseParseResult::Parsed;
    }

    MouseParseResult expect(char ch)
    {
        if (!has())
        {
            return MouseParseResult::NeedMore;
        }
        if (buffer[idx] != ch)
        {
            return MouseParseResult::Invalid;
        }
        ++idx;
        return MouseParseResult::Parsed;
    }
};

} // namespace

MouseParse InputParser::parse_sgr_mouse(const std::string_view buffer)
{
    MouseParse parsed{};
    if (buffer.size() < 3)
    {
        parsed.result = MouseParseResult::NeedMore;
        return parsed;
    }
    if (buffer[0] != '\x1b' || buffer[1] != '[' || buffer[2] != '<')
    {
        parsed.result = MouseParseResult::Invalid;
        return parsed;
    }

    ParseCursor cursor{buffer, 3};
    int button = 0;
    int x = 0;
    int y = 0;

    MouseParseResult result = cursor.parse_int(button);
    if (result != MouseParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    result = cursor.expect(';');
    if (result != MouseParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    result = cursor.parse_int(x);
    if (result != MouseParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    result = cursor.expect(';');
    if (result != MouseParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    result = cursor.parse_int(y);
    if (result != MouseParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    if (!cursor.has())
    {
        parsed.result = MouseParseResult::NeedMore;
        return parsed;
    }

    const char terminator = buffer[cursor.idx];
    if (terminator != 'M' && terminator != 'm')
    {
        parsed.result = MouseParseResult::Invalid;
        return parsed;
    }
    ++cursor.idx;

    parsed.result = MouseParseResult::Parsed;
    parsed.consumed = cursor.idx;
    parsed.event.button = button;
    parsed.event.x = x;
    parsed.event.y = y;
    parsed.event.motion = (button & 32) != 0;
    parsed.event.pressed = (terminator == 'M');
    return parsed;
}

InputParse InputParser::parse_csi_key(const std::string_view buffer)
{
    InputParse parsed{};
    if (buffer.size() < 2)
    {
        parsed.result = InputParseResult::NeedMore;
        return parsed;
    }
    if (buffer[0] != '\x1b' || buffer[1] != '[')
    {
        parsed.result = InputParseResult::Invalid;
        return parsed;
    }
    if (buffer.size() < 3)
    {
        parsed.result = InputParseResult::NeedMore;
        return parsed;
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
            parsed.result = InputParseResult::Invalid;
            return parsed;
    }
    parsed.result = InputParseResult::Parsed;
    parsed.consumed = 3;
    parsed.action = action;
    return parsed;
}

MouseLookDelta InputParser::mouse_look_velocity(const int mouse_x, const int mouse_y,
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
