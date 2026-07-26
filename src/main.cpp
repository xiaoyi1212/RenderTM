#include "prelude.hpp"

import input;
import keyboard;
import render;
import terminal;

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
    RenderThreads(RenderEngine& engine, RenderInputMailbox& mailbox):
        output_thread([](std::stop_token token) {
            TerminalRender::output_loop(token);
        }),
        render_thread([&engine, &mailbox](std::stop_token token) {
            while (!token.stop_requested())
            {
                TerminalRender::submit_frame(engine, mailbox);
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

struct App
{
    auto run() -> int
    {
        while (true)
        {
            if (SignalState::shutdown())
            {
                return 0;
            }
            if (SignalState::take_resize())
            {
                TerminalRender::update_size();
            }

            wait_for_input(kPollIntervalMs);
            read_keyboard();
            if (!process_input()) return 0;

            const double dt = sample_dt();
            if (dt > 0.0) update_mouse_look(dt);
        }
    }

private:
    static constexpr int kPollIntervalMs = 8;
    static constexpr double kMoveStep = 0.2;
    static constexpr double kMouseMaxSpeed = 1.2;
    static constexpr int kMouseDeadzone = 8;
    static constexpr auto kEscTimeout = std::chrono::milliseconds(50);

    RenderEngine engine;
    RenderInputMailbox mailbox;
    TerminalSession session;
    RenderThreads threads{engine, mailbox};
    std::string input_buffer;
    std::optional<MousePosition> mouse_pos;
    std::chrono::steady_clock::time_point last_input_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_look_time = std::chrono::steady_clock::now();

    static auto wait_for_input(const int timeout_ms) -> void
    {
        pollfd pfd{STDIN_FILENO, POLLIN, 0};
        ::poll(&pfd, 1, timeout_ms);
    }

    auto read_keyboard() -> void
    {
        const size_t before = input_buffer.size();
        auto ch = session.read_char();
        while (ch.has_value())
        {
            input_buffer.push_back(static_cast<char>(*ch));
            ch = session.read_char();
        }
        if (input_buffer.size() > before)
        {
            last_input_time = std::chrono::steady_clock::now();
        }
    }

    [[nodiscard]]
    auto process_input() -> bool
    {
        size_t offset = 0;
        while (offset < input_buffer.size())
        {
            std::string_view view(input_buffer);
            view.remove_prefix(offset);

            const InputEvent event = InputParser::parse(view);
            if (event.consumed == 0)
            {
                const auto now = std::chrono::steady_clock::now();
                if (now - last_input_time < kEscTimeout)
                {
                    break;
                }
                offset += 1;
                continue;
            }

            offset += event.consumed;
            if (event.mouse)
            {
                mouse_pos = event.mouse;
            }
            if (!handle_action(event.action)) return false;
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
        switch (action)
        {
            case InputAction::Quit: return false;
            case InputAction::None: return true;
            case InputAction::ToggleAO: mailbox.push_toggle_ao(); return true;
            case InputAction::ToggleGI: mailbox.push_toggle_gi(); return true;
            case InputAction::TogglePause: mailbox.push_toggle_pause(); return true;
            default: break;
        }

        mailbox.push_move(InputParser::movement_intent(action) * kMoveStep);
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

        const int max_x = static_cast<int>(term_size.width);
        const int max_y = static_cast<int>(term_size.height - 1);

        const MouseLookParams look_params{
            .mouse_x = std::clamp(mouse_pos->x, 1, max_x),
            .mouse_y = std::clamp(mouse_pos->y - 1, 1, max_y),
            .term_width = max_x,
            .term_height = max_y,
            .deadzone_radius = kMouseDeadzone,
            .max_speed = kMouseMaxSpeed
        };

        const MouseLookDelta velocity = InputParser::mouse_look_velocity(look_params);

        if (velocity.yaw != 0.0 || velocity.pitch != 0.0)
        {
            mailbox.push_rotate({velocity.yaw * dt, velocity.pitch * dt});
        }
    }
};

auto main() -> int
{
    SignalState::install();
    App app;
    return app.run();
}
