module;

#include "prelude.hpp"

export module input;

export enum class InputAction
{
    None,
    Quit,
    TogglePause,
    ToggleGI,
    ToggleAO,
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown
};

export enum class ParseResult
{
    Parsed,
    NeedMore,
    Invalid
};

export struct MouseEvent
{
    int button = 0;
    int x = 0;
    int y = 0;
    bool motion = false;
    bool pressed = false;
};

export struct MouseParse
{
    ParseResult result = ParseResult::Invalid;
    size_t consumed = 0;
    MouseEvent event{};
};

export struct InputParse
{
    ParseResult result = ParseResult::Invalid;
    size_t consumed = 0;
    InputAction action = InputAction::None;
};

export struct MouseLookDelta
{
    double yaw;
    double pitch;
};

export struct MouseLookParams
{
    int mouse_x;
    int mouse_y;
    int term_width;
    int term_height;
    int deadzone_radius;
    double max_speed;
};

export struct InputParser
{
    static auto key_to_action(const int ch) -> InputAction;
    static auto parse_sgr_mouse(const std::string_view buffer) -> MouseParse;
    static auto parse_csi_key(const std::string_view buffer) -> InputParse;
    static auto mouse_look_velocity(const MouseLookParams& params) -> MouseLookDelta;
};

[[nodiscard]]
auto InputParser::key_to_action(const int ch) -> InputAction
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
        case 'o': return InputAction::ToggleAO;
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

    [[nodiscard]]
    constexpr auto has(size_t count = 1) const -> bool
    {
        return idx + count <= buffer.size();
    }

    [[nodiscard]]
    auto parse_int(int& value) -> ParseResult
    {
        if (!has())
        {
            return ParseResult::NeedMore;
        }
        const char* begin = buffer.data() + idx;
        const char* end = buffer.data() + buffer.size();
        const auto result = std::from_chars(begin, end, value);
        if (result.ptr == begin)
        {
            return ParseResult::Invalid;
        }
        idx = static_cast<size_t>(result.ptr - buffer.data());
        return ParseResult::Parsed;
    }

    [[nodiscard]]
    constexpr auto expect(char ch) -> ParseResult
    {
        if (!has())
        {
            return ParseResult::NeedMore;
        }
        if (buffer[idx] != ch)
        {
            return ParseResult::Invalid;
        }
        ++idx;
        return ParseResult::Parsed;
    }
};

} // namespace

[[nodiscard]]
auto InputParser::parse_sgr_mouse(const std::string_view buffer) -> MouseParse
{
    MouseParse parsed{};
    if (buffer.size() < 3)
    {
        parsed.result = ParseResult::NeedMore;
        return parsed;
    }
    if (buffer[0] != '\x1b' || buffer[1] != '[' || buffer[2] != '<')
    {
        parsed.result = ParseResult::Invalid;
        return parsed;
    }

    ParseCursor cursor{buffer, 3};
    int button = 0;
    int x = 0;
    int y = 0;

    ParseResult result = cursor.parse_int(button);
    if (result != ParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    result = cursor.expect(';');
    if (result != ParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    result = cursor.parse_int(x);
    if (result != ParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    result = cursor.expect(';');
    if (result != ParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    result = cursor.parse_int(y);
    if (result != ParseResult::Parsed)
    {
        parsed.result = result;
        return parsed;
    }
    if (!cursor.has())
    {
        parsed.result = ParseResult::NeedMore;
        return parsed;
    }

    const char terminator = buffer[cursor.idx];
    if (terminator != 'M' && terminator != 'm')
    {
        parsed.result = ParseResult::Invalid;
        return parsed;
    }
    ++cursor.idx;

    parsed.result = ParseResult::Parsed;
    parsed.consumed = cursor.idx;
    parsed.event.button = button;
    parsed.event.x = x;
    parsed.event.y = y;
    parsed.event.motion = (button & 32) != 0;
    parsed.event.pressed = (terminator == 'M');
    return parsed;
}

[[nodiscard]]
auto InputParser::parse_csi_key(const std::string_view buffer) -> InputParse
{
    InputParse parsed{};
    if (buffer.size() < 2)
    {
        parsed.result = ParseResult::NeedMore;
        return parsed;
    }
    if (buffer[0] != '\x1b' || buffer[1] != '[')
    {
        parsed.result = ParseResult::Invalid;
        return parsed;
    }
    if (buffer.size() < 3)
    {
        parsed.result = ParseResult::NeedMore;
        return parsed;
    }
    InputAction action = InputAction::None;
    switch (buffer[2])
    {
        case 'A': action = InputAction::MoveForward; break;
        case 'B': action = InputAction::MoveBackward; break;
        case 'C': action = InputAction::MoveRight; break;
        case 'D': action = InputAction::MoveLeft; break;
        default:
            parsed.result = ParseResult::Invalid;
            return parsed;
    }
    parsed.result = ParseResult::Parsed;
    parsed.consumed = 3;
    parsed.action = action;
    return parsed;
}

[[nodiscard]]
auto InputParser::mouse_look_velocity(const MouseLookParams& params) -> MouseLookDelta
{
    if (params.term_width <= 0 || params.term_height <= 0 || params.max_speed <= 0.0)
    {
        return {0.0, 0.0};
    }

    const double center_x = (static_cast<double>(params.term_width) + 1.0) * 0.5;
    const double center_y = (static_cast<double>(params.term_height) + 1.0) * 0.5;
    const double dx = static_cast<double>(params.mouse_x) - center_x;
    const double dy = static_cast<double>(params.mouse_y) - center_y;

    if (std::abs(dx) <= params.deadzone_radius && std::abs(dy) <= params.deadzone_radius)
    {
        return {0.0, 0.0};
    }

    const double max_dx = std::max(center_x - 1.0, static_cast<double>(params.term_width) - center_x);
    const double max_dy = std::max(center_y - 1.0, static_cast<double>(params.term_height) - center_y);
    const double avail_x = max_dx - static_cast<double>(params.deadzone_radius);
    const double avail_y = max_dy - static_cast<double>(params.deadzone_radius);
    
    if (avail_x <= 0.0 || avail_y <= 0.0) return {0.0, 0.0};

    const double mag_x = std::clamp((std::abs(dx) - params.deadzone_radius) / avail_x, 0.0, 1.0);
    const double mag_y = std::clamp((std::abs(dy) - params.deadzone_radius) / avail_y, 0.0, 1.0);
    const double yaw = std::copysign(mag_x * params.max_speed, dx);
    const double pitch = std::copysign(mag_y * params.max_speed, dy);

    return {yaw, pitch};
}
