#include "controls.h"

#include <cmath>

MoveIntent controls_action_to_move(const InputAction action, const double step, const double yaw)
{
    switch (action)
    {
        case InputAction::MoveForward:
            return {MoveSpace::Local, {0.0, 0.0, step}};
        case InputAction::MoveBackward:
            return {MoveSpace::Local, {0.0, 0.0, -step}};
        case InputAction::MoveLeft:
        {
            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            return {MoveSpace::World, {-step * cy, 0.0, step * sy}};
        }
        case InputAction::MoveRight:
        {
            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            return {MoveSpace::World, {step * cy, 0.0, -step * sy}};
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
    return {MoveSpace::None, {0.0, 0.0, 0.0}};
}
