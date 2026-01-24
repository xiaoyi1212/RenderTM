#include "controls.h"

#include <cmath>

MoveIntent MoveIntent::from_action(const InputAction action, const double step, const double yaw)
{
    switch (action)
    {
        case InputAction::MoveForward:
            return {MoveSpace::Local, {0.0, 0.0, step}};
        case InputAction::MoveBackward:
            return {MoveSpace::Local, {0.0, 0.0, -step}};
        case InputAction::MoveLeft:
        case InputAction::MoveRight:
        {
            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            const double dir = (action == InputAction::MoveRight) ? 1.0 : -1.0;
            return {MoveSpace::World, {dir * step * cy, 0.0, -dir * step * sy}};
        }
        case InputAction::MoveUp:
            return {MoveSpace::World, {0.0, -step, 0.0}};
        case InputAction::MoveDown:
            return {MoveSpace::World, {0.0, step, 0.0}};
        case InputAction::None:
        case InputAction::Quit:
        case InputAction::TogglePause:
            break;
    }
    return {MoveSpace::None, {}};
}
