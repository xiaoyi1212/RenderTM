#include "prelude.hpp"

import controls;
import input;
import keyboard;
import render;
import terminal;

namespace {

struct SignalState
{
    static void install()
    {
        std::signal(SIGINT, &SignalState::handle);
        std::signal(SIGTERM, &SignalState::handle);
        std::signal(SIGWINCH, &SignalState::handle);
    }

    static bool shutdown()
    {
        return shutdown_req != 0;
    }

    static bool take_resize()
    {
        if (resize_req == 0)
        {
            return false;
        }
        resize_req = 0;
        return true;
    }

private:
    static void handle(int sig)
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
    TerminalSession& operator=(const TerminalSession&) = delete;

    std::optional<unsigned char> read_char() const
    {
        return keyboard.read_char();
    }

private:
    KeyboardMode keyboard;
};

struct RenderThreads
{
    RenderThreads():
        output_thread([](std::stop_token token) {
            TerminalRender::output_loop(token);
        }),
        render_thread([](std::stop_token token) {
            while (!token.stop_requested()) TerminalRender::submit_frame();
        })
    {}

    ~RenderThreads()
    {
        render_thread.request_stop();
        output_thread.request_stop();
    }

    RenderThreads(const RenderThreads&) = delete;
    RenderThreads& operator=(const RenderThreads&) = delete;

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
    int run()
    {
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

    void read_keyboard()
    {
        for (auto ch = session.read_char(); ch.has_value(); ch = session.read_char())
        {
            input_buffer.push_back(static_cast<char>(*ch));
        }
    }

    bool process_input()
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
                if (mouse_result.result == MouseParseResult::NeedMore) break;
                if (mouse_result.result == MouseParseResult::Parsed)
                {
                    offset += mouse_result.consumed;
                    mouse_pos = MousePos{mouse_result.event.x, mouse_result.event.y};
                    continue;
                }

                const InputParse key_result = InputParser::parse_csi_key(view);
                if (key_result.result == InputParseResult::NeedMore) break;
                if (key_result.result == InputParseResult::Parsed)
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

    bool handle_action(InputAction action)
    {
        switch (action)
        {
            case InputAction::None:
                return true;
            case InputAction::Quit:
                return false;
            case InputAction::TogglePause:
                render_toggle_pause();
                return true;
            case InputAction::ToggleGI:
                toggle_gi();
                return true;
            default:
                break;
        }

        const Vec2 rot = render_get_camera_rotation();
        const MoveIntent move = MoveIntent::from_action(action, kMoveStep, rot.x);
        if (move.space == MoveSpace::Local)
        {
            render_move_camera_local(move.delta);
        }
        else if (move.space == MoveSpace::World)
        {
            render_move_camera(move.delta);
        }
        return true;
    }

    double sample_dt()
    {
        const auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_look_time).count();
        last_look_time = now;
        return std::clamp(dt, 0.0, 0.1);
    }

    void update_mouse_look(double dt)
    {
        if (!mouse_pos) return;

        const TerminalSize term_size = TerminalRender::size();
        const size_t term_width = term_size.width;
        const size_t term_height = term_size.height;
        if (term_width == 0 || term_height <= 1) return;

        const size_t view_height = term_height - 1;
        const int max_x = static_cast<int>(term_width);
        const int max_y = static_cast<int>(view_height);
        const int mouse_x = std::clamp(mouse_pos->x, 1, max_x);
        const int mouse_y = std::clamp(mouse_pos->y - 1, 1, max_y);

        const MouseLookDelta velocity = InputParser::mouse_look_velocity(
            mouse_x, mouse_y,
            max_x,
            max_y,
            kMouseDeadzone,
            kMouseMaxSpeed
        );

        if (velocity.yaw != 0.0 || velocity.pitch != 0.0)
        {
            render_rotate_camera({velocity.yaw * dt, velocity.pitch * dt});
        }
    }

    void toggle_gi() const
    {
        const bool enabled = render_get_gi_enabled();
        render_set_gi_enabled(!enabled);
        if (!enabled && render_get_gi_strength() <= 0.0)
        {
            render_set_gi_strength(1.0);
        }
    }

    TerminalSession session;
    RenderThreads threads;
    std::string input_buffer;
    std::optional<MousePos> mouse_pos;
    std::chrono::steady_clock::time_point last_look_time = std::chrono::steady_clock::now();
};

} // namespace

int main()
{
    SignalState::install();

    App app;
    return app.run();
}
