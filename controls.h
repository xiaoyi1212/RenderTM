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

    static MoveIntent from_action(InputAction action, double step, double yaw);
};
