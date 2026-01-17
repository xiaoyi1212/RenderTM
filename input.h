#pragma once

enum class InputAction
{
    None,
    Quit,
    TogglePause,
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    YawLeft,
    YawRight,
    PitchUp,
    PitchDown
};

InputAction input_map_key(int ch);
