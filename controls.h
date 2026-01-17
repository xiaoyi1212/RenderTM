#pragma once

#include "input.h"
#include "render.h"

enum class MoveSpace
{
    None,
    Local,
    World
};

struct MoveIntent
{
    MoveSpace space;
    Vec3 delta;
};

MoveIntent controls_action_to_move(InputAction action, double step, double yaw);
