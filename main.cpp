#include <csignal>
#include <chrono>
#include <cmath>

#include "terminal_render.h"
#include "keyboard.h"
#include "input.h"
#include "render.h"
#include "controls.h"

#include <atomic>
#include <pthread.h>
#include <string>
#include <string_view>
#include <unistd.h>

static std::atomic<bool> running{true};
static volatile std::sig_atomic_t shutdownRequested = 0;
static volatile std::sig_atomic_t resizeRequested = 0;

static void handle_signal(int sig)
{
    if (sig == SIGWINCH)
    {
        resizeRequested = 1;
        return;
    }
    shutdownRequested = 1;
}

void* render_thread(void*)
{
    while (running.load(std::memory_order_relaxed))
    {
        render_print();
    }
    return nullptr;
}

void* output_thread(void*)
{
    render_output_run();
    return nullptr;
}

int main()
{
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGWINCH, handle_signal);
    keyboard_setup();
    render_init();
    const double move_step = 0.2;
    const double mouse_max_speed = 1.2;
    const int mouse_deadzone = 8;
    pthread_t rthread;
    pthread_t othread;
    pthread_create(&othread, nullptr, output_thread, nullptr);
    pthread_create(&rthread, nullptr, render_thread, nullptr);
    std::string input_buffer;
    struct MouseLookState
    {
        int x = 0;
        int y = 0;
        bool has_position = false;
    } mouse_state;
    auto last_look_time = std::chrono::steady_clock::now();
    auto toggle_gi = []() {
        const bool enabled = render_get_gi_enabled();
        render_set_gi_enabled(!enabled);
        if (!enabled && render_get_gi_strength() <= 0.0)
        {
            render_set_gi_strength(1.0);
        }
    };
    while (running.load(std::memory_order_relaxed))
    {
        if (shutdownRequested)
        {
            running.store(false, std::memory_order_relaxed);
            break;
        }
        if (resizeRequested)
        {
            resizeRequested = 0;
            render_update_size(0);
        }
        int ch = keyboard_read_char();
        while (ch != -1)
        {
            input_buffer.push_back(static_cast<char>(ch));
            ch = keyboard_read_char();
        }

        size_t offset = 0;
        while (offset < input_buffer.size())
        {
            if (input_buffer[offset] == '\x1b')
            {
                MouseEvent event{};
                size_t consumed = 0;
                const auto result = input_parse_sgr_mouse(std::string_view(input_buffer).substr(offset), &consumed, &event);
                if (result == MouseParseResult::NeedMore)
                {
                    break;
                }
                if (result == MouseParseResult::Parsed)
                {
                    offset += consumed;
                    mouse_state.x = event.x;
                    mouse_state.y = event.y;
                    mouse_state.has_position = true;
                    continue;
                }
                InputAction action = InputAction::None;
                const auto key_result = input_parse_csi_key(std::string_view(input_buffer).substr(offset), &consumed, &action);
                if (key_result == InputParseResult::NeedMore)
                {
                    break;
                }
                if (key_result == InputParseResult::Parsed)
                {
                    offset += consumed;
                    if (action == InputAction::Quit)
                    {
                        running.store(false, std::memory_order_relaxed);
                    }
                    else if (action == InputAction::TogglePause)
                    {
                        render_toggle_pause();
                    }
                    else if (action == InputAction::ToggleGI)
                    {
                        toggle_gi();
                    }
                    else
                    {
                        const Vec2 rot = render_get_camera_rotation();
                        const MoveIntent move = controls_action_to_move(action, move_step, rot.x);
                        if (move.space == MoveSpace::Local)
                        {
                            render_move_camera_local(move.delta);
                        }
                        else if (move.space == MoveSpace::World)
                        {
                            render_move_camera(move.delta);
                        }
                    }
                    continue;
                }
            }

            const int key = static_cast<unsigned char>(input_buffer[offset]);
            offset++;
            const InputAction action = input_map_key(key);
            if (action == InputAction::Quit)
            {
                running.store(false, std::memory_order_relaxed);
            }
            else if (action == InputAction::TogglePause)
            {
                render_toggle_pause();
            }
            else if (action == InputAction::ToggleGI)
            {
                toggle_gi();
            }
            else
            {
                const Vec2 rot = render_get_camera_rotation();
                const MoveIntent move = controls_action_to_move(action, move_step, rot.x);
                if (move.space == MoveSpace::Local)
                {
                    render_move_camera_local(move.delta);
                }
                else if (move.space == MoveSpace::World)
                {
                    render_move_camera(move.delta);
                }
            }
        }
        if (offset > 0)
        {
            input_buffer.erase(0, offset);
        }

        const auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_look_time).count();
        last_look_time = now;
        if (dt > 0.1)
        {
            dt = 0.1;
        }
        if (dt <= 0.0)
        {
            continue;
        }

        if (mouse_state.has_position)
        {
            const size_t term_width = render_get_terminal_width();
            const size_t term_height = render_get_terminal_height();
            if (term_width > 0 && term_height > 1)
            {
                const size_t view_height = term_height - 1;
                int mouse_x = mouse_state.x;
                int mouse_y = mouse_state.y - 1;
                if (mouse_x < 1) mouse_x = 1;
                if (mouse_y < 1) mouse_y = 1;
                if (mouse_x > static_cast<int>(term_width)) mouse_x = static_cast<int>(term_width);
                if (mouse_y > static_cast<int>(view_height)) mouse_y = static_cast<int>(view_height);

                const MouseLookDelta velocity = input_mouse_look_velocity(
                    mouse_x, mouse_y,
                    static_cast<int>(term_width),
                    static_cast<int>(view_height),
                    mouse_deadzone,
                    mouse_max_speed
                );

                if (velocity.yaw != 0.0 || velocity.pitch != 0.0)
                {
                    render_rotate_camera({velocity.yaw * dt, velocity.pitch * dt});
                }
            }
        }
    }
    render_output_request_stop();
    pthread_join(othread, nullptr);
    pthread_join(rthread, nullptr);
    render_shutdown();
    keyboard_restore();
    return 0;
}
