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

InputAction input_map_key(int ch);

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

struct MouseLookDelta
{
    double yaw;
    double pitch;
};

MouseParseResult input_parse_sgr_mouse(std::string_view buffer, size_t* consumed, MouseEvent* out_event);
InputParseResult input_parse_csi_key(std::string_view buffer, size_t* consumed, InputAction* out_action);
MouseLookDelta input_mouse_look_velocity(int mouse_x, int mouse_y, int width, int height,
                                         int deadzone_radius, double max_speed);
