#include "input.h"

InputAction input_map_key(const int ch)
{
    if (ch == 'q' || ch == 'Q') return InputAction::Quit;
    if (ch == 'p' || ch == 'P') return InputAction::TogglePause;
    if (ch == 'w' || ch == 'W') return InputAction::MoveForward;
    if (ch == 's' || ch == 'S') return InputAction::MoveBackward;
    if (ch == 'a' || ch == 'A') return InputAction::MoveLeft;
    if (ch == 'd' || ch == 'D') return InputAction::MoveRight;
    if (ch == 'r' || ch == 'R') return InputAction::MoveUp;
    if (ch == 'f' || ch == 'F') return InputAction::MoveDown;
    if (ch == 'j' || ch == 'J') return InputAction::YawLeft;
    if (ch == 'l' || ch == 'L') return InputAction::YawRight;
    if (ch == 'i' || ch == 'I') return InputAction::PitchUp;
    if (ch == 'k' || ch == 'K') return InputAction::PitchDown;
    return InputAction::None;
}
