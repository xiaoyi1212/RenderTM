#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "controls.h"

TEST_CASE("controls_action_to_move maps vertical movement directions")
{
    constexpr double step = 0.25;
    const MoveIntent up = controls_action_to_move(InputAction::MoveUp, step, 0.0);
    const MoveIntent down = controls_action_to_move(InputAction::MoveDown, step, 0.0);

    REQUIRE(up.space == MoveSpace::World);
    REQUIRE(up.delta.y == Catch::Approx(-step));
    REQUIRE(down.space == MoveSpace::World);
    REQUIRE(down.delta.y == Catch::Approx(step));
}

TEST_CASE("controls_action_to_move maps forward and backward in local space")
{
    constexpr double step = 0.3;
    const MoveIntent forward = controls_action_to_move(InputAction::MoveForward, step, 0.0);
    const MoveIntent backward = controls_action_to_move(InputAction::MoveBackward, step, 0.0);

    REQUIRE(forward.space == MoveSpace::Local);
    REQUIRE(forward.delta.z == Catch::Approx(step));
    REQUIRE(backward.space == MoveSpace::Local);
    REQUIRE(backward.delta.z == Catch::Approx(-step));
}

TEST_CASE("controls_action_to_move strafes using yaw only")
{
    constexpr double step = 0.4;
    constexpr double half_pi = 1.5707963267948966;
    const MoveIntent right = controls_action_to_move(InputAction::MoveRight, step, half_pi);
    const MoveIntent left = controls_action_to_move(InputAction::MoveLeft, step, half_pi);

    REQUIRE(right.space == MoveSpace::World);
    REQUIRE(right.delta.x == Catch::Approx(0.0).margin(1e-6));
    REQUIRE(right.delta.z == Catch::Approx(-step));
    REQUIRE(left.space == MoveSpace::World);
    REQUIRE(left.delta.x == Catch::Approx(0.0).margin(1e-6));
    REQUIRE(left.delta.z == Catch::Approx(step));
}
