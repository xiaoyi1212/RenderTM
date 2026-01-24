#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "controls.h"

TEST_CASE("MoveIntent::from_action maps vertical movement directions")
{
    constexpr double step = 0.25;
    const MoveIntent up = MoveIntent::from_action(InputAction::MoveUp, step, 0.0);
    const MoveIntent down = MoveIntent::from_action(InputAction::MoveDown, step, 0.0);

    REQUIRE(up.space == MoveSpace::World);
    REQUIRE(up.delta.y == Catch::Approx(-step));
    REQUIRE(down.space == MoveSpace::World);
    REQUIRE(down.delta.y == Catch::Approx(step));
}

TEST_CASE("MoveIntent::from_action maps forward and backward in local space")
{
    constexpr double step = 0.3;
    const MoveIntent forward = MoveIntent::from_action(InputAction::MoveForward, step, 0.0);
    const MoveIntent backward = MoveIntent::from_action(InputAction::MoveBackward, step, 0.0);

    REQUIRE(forward.space == MoveSpace::Local);
    REQUIRE(forward.delta.z == Catch::Approx(step));
    REQUIRE(backward.space == MoveSpace::Local);
    REQUIRE(backward.delta.z == Catch::Approx(-step));
}

TEST_CASE("MoveIntent::from_action strafes using yaw only")
{
    constexpr double step = 0.4;
    constexpr double half_pi = 1.5707963267948966;
    const MoveIntent right = MoveIntent::from_action(InputAction::MoveRight, step, half_pi);
    const MoveIntent left = MoveIntent::from_action(InputAction::MoveLeft, step, half_pi);

    REQUIRE(right.space == MoveSpace::World);
    REQUIRE(right.delta.x == Catch::Approx(0.0).margin(1e-6));
    REQUIRE(right.delta.z == Catch::Approx(-step));
    REQUIRE(left.space == MoveSpace::World);
    REQUIRE(left.delta.x == Catch::Approx(0.0).margin(1e-6));
    REQUIRE(left.delta.z == Catch::Approx(step));
}
