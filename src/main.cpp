#include "prelude.hpp"

import controls;
import input;
import keyboard;
import render;
import terminal;

namespace {

struct SignalState
{
    static auto install() -> void
    {
        std::signal(SIGINT, &SignalState::handle);
        std::signal(SIGTERM, &SignalState::handle);
        std::signal(SIGWINCH, &SignalState::handle);
    }

    [[nodiscard]]
    static auto shutdown() -> bool
    {
        return shutdown_req != 0;
    }

    [[nodiscard]]
    static auto take_resize() -> bool
    {
        if (resize_req == 0)
        {
            return false;
        }
        resize_req = 0;
        return true;
    }

private:
    static auto handle(int sig) -> void
    {
        if (sig == SIGWINCH)
        {
            resize_req = 1;
            return;
        }
        shutdown_req = 1;
    }

    static inline volatile std::sig_atomic_t shutdown_req = 0;
    static inline volatile std::sig_atomic_t resize_req = 0;
};

struct TerminalSession
{
    TerminalSession()
    {
        TerminalRender::init();
    }

    ~TerminalSession()
    {
        TerminalRender::shutdown();
    }

    TerminalSession(const TerminalSession&) = delete;
    auto operator=(const TerminalSession&) -> TerminalSession& = delete;

    [[nodiscard]]
    auto read_char() const -> std::optional<unsigned char>
    {
        return keyboard.read_char();
    }

private:
    KeyboardMode keyboard;
};

struct RenderThreads
{
    explicit RenderThreads(RenderEngine& engine):
        output_thread([](std::stop_token token) {
            TerminalRender::output_loop(token);
        }),
        render_thread([&engine](std::stop_token token) {
            while (!token.stop_requested())
            {
                TerminalRender::submit_frame(engine);
            }
        })
    {}

    ~RenderThreads()
    {
        render_thread.request_stop();
        output_thread.request_stop();
    }

    RenderThreads(const RenderThreads&) = delete;
    auto operator=(const RenderThreads&) -> RenderThreads& = delete;

private:
    std::jthread output_thread;
    std::jthread render_thread;
};

struct MousePos
{
    int x = 0;
    int y = 0;
};

struct App
{
    auto run() -> int
    {
        auto read_keyboard = [&]() -> void
        {
            auto ch = session.read_char();
            while (ch.has_value())
            {
                input_buffer.push_back(static_cast<char>(*ch));
                ch = session.read_char();
            }
        };

        bool running = true;
        while (running)
        {
            if (SignalState::shutdown())
            {
                running = false;
                break;
            }
            if (SignalState::take_resize())
            {
                TerminalRender::update_size(0);
            }

            read_keyboard();
            running = process_input();
            if (!running) break;

            const double dt = sample_dt();
            if (dt > 0.0) update_mouse_look(dt);
        }
        return 0;
    }

private:
    static constexpr double kMoveStep = 0.2;
    static constexpr double kMouseMaxSpeed = 1.2;
    static constexpr int kMouseDeadzone = 8;

    [[nodiscard]]
    auto process_input() -> bool
    {
        size_t offset = 0;
        while (offset < input_buffer.size())
        {
            if (input_buffer[offset] == '\x1b')
            {
                size_t consumed = 0;
                std::string_view view(input_buffer);
                view.remove_prefix(offset);

                const MouseParse mouse_result = InputParser::parse_sgr_mouse(view);
                if (mouse_result.result == ParseResult::NeedMore) break;
                if (mouse_result.result == ParseResult::Parsed)
                {
                    offset += mouse_result.consumed;
                    mouse_pos = MousePos{mouse_result.event.x, mouse_result.event.y};
                    continue;
                }

                const InputParse key_result = InputParser::parse_csi_key(view);
                if (key_result.result == ParseResult::NeedMore) break;
                if (key_result.result == ParseResult::Parsed)
                {
                    offset += key_result.consumed;
                    if (!handle_action(key_result.action)) return false;
                    continue;
                }
            }

            const int key = static_cast<unsigned char>(input_buffer[offset]);
            offset++;
            if (!handle_action(InputParser::key_to_action(key))) return false;
        }

        if (offset > 0)
        {
            input_buffer.erase(0, offset);
        }
        return true;
    }

    [[nodiscard]]
    auto handle_action(InputAction action) -> bool
    {
        auto perform_movement = [&](const MoveIntent& move)
        {
            if (move.space == MoveSpace::Local)
                engine.move_camera_local(move.delta);
            else if (move.space == MoveSpace::World)
                engine.move_camera(move.delta);
        };

        switch (action)
        {
            case InputAction::Quit: return false;
            case InputAction::None: return true;
            case InputAction::TogglePause: engine.toggle_pause(); return true;
            case InputAction::ToggleGI: toggle_gi(); return true;
            default: break;
        }

        const Vec2 rot = engine.get_camera_rotation();
        perform_movement(MoveIntent::from_action(action, kMoveStep, rot.x));
        return true;
    }

    auto sample_dt() -> double
    {
        const auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_look_time).count();
        last_look_time = now;
        return std::clamp(dt, 0.0, 0.1);
    }

    auto update_mouse_look(double dt) -> void
    {
        if (!mouse_pos) return;

        const TerminalSize term_size = TerminalRender::size();
        if (term_size.width == 0 || term_size.height <= 1) return;

        auto clamp_pos = [](int val, int min_v, int max_v) -> int {
            return std::clamp(val, min_v, max_v);
        };

        const int max_x = static_cast<int>(term_size.width);
        const int max_y = static_cast<int>(term_size.height - 1);

        const MouseLookParams look_params{
            .mouse_x = clamp_pos(mouse_pos->x, 1, max_x),
            .mouse_y = clamp_pos(mouse_pos->y - 1, 1, max_y),
            .term_width = max_x,
            .term_height = max_y,
            .deadzone_radius = kMouseDeadzone,
            .max_speed = kMouseMaxSpeed
        };

        const MouseLookDelta velocity = InputParser::mouse_look_velocity(look_params);

        if (velocity.yaw != 0.0 || velocity.pitch != 0.0)
        {
            engine.rotate_camera({velocity.yaw * dt, velocity.pitch * dt});
        }
    }

    auto toggle_gi() -> void
    {
        const bool enabled = engine.get_gi_enabled();
        engine.set_gi_enabled(!enabled);
        if (!enabled && engine.get_gi_strength() <= 0.0)
        {
            engine.set_gi_strength(1.0);
        }
    }

    RenderEngine engine;
    TerminalSession session;
    RenderThreads threads{engine};
    std::string input_buffer;
    std::optional<MousePos> mouse_pos;
    std::chrono::steady_clock::time_point last_look_time = std::chrono::steady_clock::now();
};

} // namespace

auto main() -> int
{
    SignalState::install();
    App app;
    return app.run();
}
