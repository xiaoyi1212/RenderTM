#include "test_prelude.hpp"

import input;

TEST_CASE("key_to_action maps quit keys")
{
    REQUIRE(InputParser::key_to_action('q') == InputAction::Quit);
    REQUIRE(InputParser::key_to_action('Q') == InputAction::Quit);
    REQUIRE(InputParser::key_to_action('x') == InputAction::None);
    REQUIRE(InputParser::key_to_action(-1) == InputAction::None);
}

TEST_CASE("key_to_action maps pause keys")
{
    REQUIRE(InputParser::key_to_action('p') == InputAction::TogglePause);
    REQUIRE(InputParser::key_to_action('P') == InputAction::TogglePause);
}

TEST_CASE("key_to_action maps GI toggle keys")
{
    REQUIRE(InputParser::key_to_action('g') == InputAction::ToggleGI);
    REQUIRE(InputParser::key_to_action('G') == InputAction::ToggleGI);
}

TEST_CASE("key_to_action maps camera movement keys")
{
    REQUIRE(InputParser::key_to_action('w') == InputAction::MoveForward);
    REQUIRE(InputParser::key_to_action('s') == InputAction::MoveBackward);
    REQUIRE(InputParser::key_to_action('a') == InputAction::MoveLeft);
    REQUIRE(InputParser::key_to_action('d') == InputAction::MoveRight);
    REQUIRE(InputParser::key_to_action('r') == InputAction::MoveUp);
    REQUIRE(InputParser::key_to_action('f') == InputAction::MoveDown);
}

TEST_CASE("parse_sgr_mouse parses motion event")
{
    const auto result = InputParser::parse_sgr_mouse(std::string_view("\x1b[<35;12;8M"));
    REQUIRE(result.result == ParseResult::Parsed);
    REQUIRE(result.consumed == 11);
    REQUIRE(result.event.x == 12);
    REQUIRE(result.event.y == 8);
    REQUIRE(result.event.motion);
}

TEST_CASE("parse_sgr_mouse detects incomplete sequence")
{
    const auto result = InputParser::parse_sgr_mouse(std::string_view("\x1b[<35;12;"));
    REQUIRE(result.result == ParseResult::NeedMore);
}

TEST_CASE("parse_sgr_mouse rejects non-mouse input")
{
    const auto result = InputParser::parse_sgr_mouse(std::string_view("abc"));
    REQUIRE(result.result == ParseResult::Invalid);
}

TEST_CASE("parse_csi_key parses arrow keys")
{
    const auto up = InputParser::parse_csi_key(std::string_view("\x1b[A"));
    REQUIRE(up.result == ParseResult::Parsed);
    REQUIRE(up.consumed == 3);
    REQUIRE(up.action == InputAction::MoveForward);

    const auto down = InputParser::parse_csi_key(std::string_view("\x1b[B"));
    REQUIRE(down.result == ParseResult::Parsed);
    REQUIRE(down.action == InputAction::MoveBackward);

    const auto right = InputParser::parse_csi_key(std::string_view("\x1b[C"));
    REQUIRE(right.result == ParseResult::Parsed);
    REQUIRE(right.action == InputAction::MoveRight);

    const auto left = InputParser::parse_csi_key(std::string_view("\x1b[D"));
    REQUIRE(left.result == ParseResult::Parsed);
    REQUIRE(left.action == InputAction::MoveLeft);
}

TEST_CASE("parse_csi_key handles incomplete sequence")
{
    const auto result = InputParser::parse_csi_key(std::string_view("\x1b["));
    REQUIRE(result.result == ParseResult::NeedMore);
}

TEST_CASE("mouse_look_velocity respects deadzone")
{
    const auto delta = InputParser::mouse_look_velocity({
        .mouse_x = 40,
        .mouse_y = 12,
        .term_width = 80,
        .term_height = 24,
        .deadzone_radius = 2,
        .max_speed = 1.0
    });

    REQUIRE(delta.yaw == Catch::Approx(0.0));
    REQUIRE(delta.pitch == Catch::Approx(0.0));
}

TEST_CASE("mouse_look_velocity scales with distance and direction")
{
    MouseLookParams params {
        .mouse_x = 0,
        .mouse_y = 0,
        .term_width = 80,
        .term_height = 24,
        .deadzone_radius = 2,
        .max_speed = 1.0
    };

    params.mouse_x = 38;
    params.mouse_y = 8;
    const auto near = InputParser::mouse_look_velocity(params);

    params.mouse_x = 1;
    params.mouse_y = 1;
    const auto far = InputParser::mouse_look_velocity(params);

    REQUIRE(near.yaw < 0.0);
    REQUIRE(near.pitch < 0.0);
    REQUIRE(far.yaw < 0.0);
    REQUIRE(far.pitch < 0.0);

    REQUIRE(std::abs(far.yaw) > std::abs(near.yaw));
    REQUIRE(std::abs(far.pitch) > std::abs(near.pitch));
}
