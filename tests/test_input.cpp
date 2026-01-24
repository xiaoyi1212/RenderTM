#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string_view>

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

TEST_CASE("input_map_key maps GI toggle keys")
{
    REQUIRE(input_map_key('g') == InputAction::ToggleGI);
    REQUIRE(input_map_key('G') == InputAction::ToggleGI);
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

TEST_CASE("input_parse_sgr_mouse parses motion event")
{
    MouseEvent event{};
    size_t consumed = 0;
    const auto result = input_parse_sgr_mouse(std::string_view("\x1b[<35;12;8M"), &consumed, &event);
    REQUIRE(result == MouseParseResult::Parsed);
    REQUIRE(consumed == 11);
    REQUIRE(event.x == 12);
    REQUIRE(event.y == 8);
    REQUIRE(event.motion);
}

TEST_CASE("input_parse_sgr_mouse detects incomplete sequence")
{
    MouseEvent event{};
    size_t consumed = 0;
    const auto result = input_parse_sgr_mouse(std::string_view("\x1b[<35;12;"), &consumed, &event);
    REQUIRE(result == MouseParseResult::NeedMore);
}

TEST_CASE("input_parse_sgr_mouse rejects non-mouse input")
{
    MouseEvent event{};
    size_t consumed = 0;
    const auto result = input_parse_sgr_mouse(std::string_view("abc"), &consumed, &event);
    REQUIRE(result == MouseParseResult::Invalid);
}

TEST_CASE("input_parse_csi_key parses arrow keys")
{
    InputAction action = InputAction::None;
    size_t consumed = 0;
    REQUIRE(input_parse_csi_key(std::string_view("\x1b[A"), &consumed, &action) == InputParseResult::Parsed);
    REQUIRE(consumed == 3);
    REQUIRE(action == InputAction::MoveForward);

    REQUIRE(input_parse_csi_key(std::string_view("\x1b[B"), &consumed, &action) == InputParseResult::Parsed);
    REQUIRE(action == InputAction::MoveBackward);

    REQUIRE(input_parse_csi_key(std::string_view("\x1b[C"), &consumed, &action) == InputParseResult::Parsed);
    REQUIRE(action == InputAction::MoveRight);

    REQUIRE(input_parse_csi_key(std::string_view("\x1b[D"), &consumed, &action) == InputParseResult::Parsed);
    REQUIRE(action == InputAction::MoveLeft);
}

TEST_CASE("input_parse_csi_key handles incomplete sequence")
{
    InputAction action = InputAction::None;
    size_t consumed = 0;
    REQUIRE(input_parse_csi_key(std::string_view("\x1b["), &consumed, &action) == InputParseResult::NeedMore);
}

TEST_CASE("input_mouse_look_velocity respects deadzone")
{
    const auto delta = input_mouse_look_velocity(40, 12, 80, 24, 2, 1.0);
    REQUIRE(delta.yaw == Catch::Approx(0.0));
    REQUIRE(delta.pitch == Catch::Approx(0.0));
}

TEST_CASE("input_mouse_look_velocity scales with distance and direction")
{
    const auto near = input_mouse_look_velocity(38, 8, 80, 24, 2, 1.0);
    const auto far = input_mouse_look_velocity(1, 1, 80, 24, 2, 1.0);

    REQUIRE(near.yaw < 0.0);
    REQUIRE(near.pitch > 0.0);
    REQUIRE(far.yaw < 0.0);
    REQUIRE(far.pitch > 0.0);

    REQUIRE(std::abs(far.yaw) > std::abs(near.yaw));
    REQUIRE(std::abs(far.pitch) > std::abs(near.pitch));
}
