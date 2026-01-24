#pragma once

#include <cstddef>
#include <string_view>

enum class InputAction
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

enum class MouseParseResult
{
    Parsed,
    NeedMore,
    Invalid
};

enum class InputParseResult
{
    Parsed,
    NeedMore,
    Invalid
};

struct MouseEvent
{
    int button;
    int x;
    int y;
    bool motion;
    bool pressed;
};

struct MouseParse
{
    MouseParseResult result = MouseParseResult::Invalid;
    size_t consumed = 0;
    MouseEvent event{};
};

struct InputParse
{
    InputParseResult result = InputParseResult::Invalid;
    size_t consumed = 0;
    InputAction action = InputAction::None;
};

struct MouseLookDelta
{
    double yaw;
    double pitch;
};

struct InputParser
{
    static InputAction key_to_action(int ch);
    static MouseParse parse_sgr_mouse(std::string_view buffer);
    static InputParse parse_csi_key(std::string_view buffer);
    static MouseLookDelta mouse_look_velocity(int mouse_x, int mouse_y, int width, int height,
                                              int deadzone_radius, double max_speed);
};
