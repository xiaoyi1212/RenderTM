#include <csignal>

#include "terminal_render.h"
#include "keyboard.h"
#include "input.h"
#include "render.h"

#include <atomic>
#include <pthread.h>
#include <unistd.h>

static std::atomic<bool> running{true};
static volatile std::sig_atomic_t shutdownRequested = 0;

static void handle_signal(int)
{
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

int main()
{
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    keyboard_setup();
    render_init();
    const double move_step = 0.1;
    const double rotate_step = 0.03;
    pthread_t rthread;
    pthread_create(&rthread, nullptr, render_thread, nullptr);
    while (running.load(std::memory_order_relaxed))
    {
        if (shutdownRequested)
        {
            running.store(false, std::memory_order_relaxed);
            break;
        }
        const int ch = keyboard_read_char();
        switch (input_map_key(ch))
        {
            case InputAction::Quit:
                running.store(false, std::memory_order_relaxed);
                break;
            case InputAction::TogglePause:
                render_toggle_pause();
                break;
            case InputAction::MoveForward:
                render_move_camera_local({0.0, 0.0, move_step});
                break;
            case InputAction::MoveBackward:
                render_move_camera_local({0.0, 0.0, -move_step});
                break;
            case InputAction::MoveLeft:
                render_move_camera_local({-move_step, 0.0, 0.0});
                break;
            case InputAction::MoveRight:
                render_move_camera_local({move_step, 0.0, 0.0});
                break;
            case InputAction::MoveUp:
                render_move_camera({0.0, move_step, 0.0});
                break;
            case InputAction::MoveDown:
                render_move_camera({0.0, -move_step, 0.0});
                break;
            case InputAction::YawLeft:
                render_rotate_camera({-rotate_step, 0.0});
                break;
            case InputAction::YawRight:
                render_rotate_camera({rotate_step, 0.0});
                break;
            case InputAction::PitchUp:
                render_rotate_camera({0.0, -rotate_step});
                break;
            case InputAction::PitchDown:
                render_rotate_camera({0.0, rotate_step});
                break;
            case InputAction::None:
                break;
        }
    }
    pthread_join(rthread, nullptr);
    render_shutdown();
    keyboard_restore();
    return 0;
}
