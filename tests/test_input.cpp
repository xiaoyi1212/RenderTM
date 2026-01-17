#include <catch2/catch_test_macros.hpp>

#include "input.h"

TEST_CASE("input_map_key maps quit keys")
{
    REQUIRE(input_map_key('q') == InputAction::Quit);
    REQUIRE(input_map_key('Q') == InputAction::Quit);
    REQUIRE(input_map_key('x') == InputAction::None);
    REQUIRE(input_map_key(-1) == InputAction::None);
}

TEST_CASE("input_map_key maps pause keys")
{
    REQUIRE(input_map_key('p') == InputAction::TogglePause);
    REQUIRE(input_map_key('P') == InputAction::TogglePause);
}

TEST_CASE("input_map_key maps camera movement keys")
{
    REQUIRE(input_map_key('w') == InputAction::MoveForward);
    REQUIRE(input_map_key('s') == InputAction::MoveBackward);
    REQUIRE(input_map_key('a') == InputAction::MoveLeft);
    REQUIRE(input_map_key('d') == InputAction::MoveRight);
    REQUIRE(input_map_key('r') == InputAction::MoveUp);
    REQUIRE(input_map_key('f') == InputAction::MoveDown);
}

TEST_CASE("input_map_key maps camera rotation keys")
{
    REQUIRE(input_map_key('j') == InputAction::YawLeft);
    REQUIRE(input_map_key('l') == InputAction::YawRight);
    REQUIRE(input_map_key('i') == InputAction::PitchUp);
    REQUIRE(input_map_key('k') == InputAction::PitchDown);
}
