#include "terminal_render.h"
#include "render.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

static char buffer[65536];
std::atomic<size_t> raw_width, raw_height;
static auto lastTime = std::chrono::high_resolution_clock::now();
static int frameCount = 0;
static double fps = 0.0;

static const std::string render_char = "\u2580";

static int stdout_flags = -1;
static std::string pending_output;
static size_t pending_offset = 0;
static std::mutex frame_mutex;
static std::condition_variable frame_cv;
static std::string queued_frame;
static bool frame_ready = false;
static bool output_shutdown = false;

static void stdout_init()
{
    stdout_flags = fcntl(STDOUT_FILENO, F_GETFL, 0);
    if (stdout_flags >= 0)
    {
        fcntl(STDOUT_FILENO, F_SETFL, stdout_flags | O_NONBLOCK);
    }
}

static void stdout_restore()
{
    if (stdout_flags >= 0)
    {
        fcntl(STDOUT_FILENO, F_SETFL, stdout_flags);
    }
}

static void stdout_write(const char* data, size_t len)
{
    (void)::write(STDOUT_FILENO, data, len);
}

static bool stdout_flush_pending()
{
    if (pending_output.empty()) return true;
    const size_t remaining = pending_output.size() - pending_offset;
    if (remaining == 0)
    {
        pending_output.clear();
        pending_offset = 0;
        return true;
    }

    const ssize_t n = ::write(STDOUT_FILENO, pending_output.data() + pending_offset, remaining);
    if (n > 0)
    {
        pending_offset += static_cast<size_t>(n);
        if (pending_offset >= pending_output.size())
        {
            pending_output.clear();
            pending_offset = 0;
            return true;
        }
        return false;
    }
    return false;
}

static void render_submit_frame(std::string frame)
{
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        queued_frame = std::move(frame);
        frame_ready = true;
    }
    frame_cv.notify_one();
}

static bool render_wait_for_frame(std::string& out)
{
    std::unique_lock<std::mutex> lock(frame_mutex);
    frame_cv.wait(lock, [] { return frame_ready || output_shutdown; });
    if (output_shutdown)
    {
        return false;
    }
    out = std::move(queued_frame);
    frame_ready = false;
    return true;
}

void render_update_size(int sig)
{
    winsize w{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) != 0 || w.ws_col == 0 || w.ws_row == 0)
    {
        if (raw_width.load(std::memory_order_relaxed) == 0 || raw_height.load(std::memory_order_relaxed) == 0)
        {
            raw_width.store(80, std::memory_order_relaxed);
            raw_height.store(24, std::memory_order_relaxed);
        }
        return;
    }
    raw_width.store(w.ws_col, std::memory_order_relaxed);
    raw_height.store(w.ws_row, std::memory_order_relaxed);
}

void render_init()
{
    setvbuf(stdout, buffer, _IOFBF, sizeof(buffer));
    render_update_size(0);
    stdout_init();
    const char* enter_alt = "\033[?1049h\033[?25l\033[?1003h\033[?1006h\033[0m\033[2J\033[H";
    stdout_write(enter_alt, std::strlen(enter_alt));
}

void render_print()
{
    render_update_size(0);
    const size_t term_width = raw_width.load(std::memory_order_relaxed);
    const size_t term_height = raw_height.load(std::memory_order_relaxed);
    if (term_width == 0 || term_height < 2)
    {
        return;
    }

    const size_t display_rows = term_height - 1;
    const size_t width = term_width;
    const size_t height = display_rows * 2;

    frameCount++;
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = currentTime - lastTime;
    if (elapsed.count() >= 1.0)
    {
        fps = frameCount / elapsed.count();
        frameCount = 0;
        lastTime = currentTime;
    }

    std::string frame_buffer;
    frame_buffer.reserve(width * display_rows * 20);

    frame_buffer += "\033[H\033[7m";
    char info[256];
    const Vec3 cam_pos = render_get_camera_position();
    const Vec2 cam_rot = render_get_camera_rotation();
    const double rad_to_deg = 180.0 / 3.14159265358979323846;
    const double yaw_deg = cam_rot.x * rad_to_deg;
    const double pitch_deg = cam_rot.y * rad_to_deg;
    const int len = snprintf(info, sizeof(info),
                             " RenderTM v0.0.1 Terminal:%lux%lu | Pixel:%lux%lu | FPS:%.2f | Cam(%.2f,%.2f,%.2f) Rot(%.1f,%.1f)",
                             width, static_cast<size_t>(term_height), width, height, fps,
                             cam_pos.x, cam_pos.y, cam_pos.z, yaw_deg, pitch_deg);
    frame_buffer += info;
    if (len < static_cast<int>(width)) frame_buffer.append(width - len, ' ');
    frame_buffer += "\033[0m";

    static std::vector<uint32_t> framebuffer;
    if (framebuffer.size() != width * height) framebuffer.resize(width * height);
    render_update_array(framebuffer.data(), width, height);

    uint32_t last_fg = 0xFFFFFFFF, last_bg = 0xFFFFFFFF;
    char color_buf[64];

    for (size_t y = 0; y < height; y += 2)
    {
        frame_buffer += "\033[" + std::to_string(y / 2 + 2) + ";1H";

        for (size_t x = 0; x < width; ++x)
        {
            const uint32_t top = framebuffer[y * width + x];
            const uint32_t bot = framebuffer[(y + 1) * width + x];

            if (top != last_fg)
            {
                const int n = snprintf(color_buf, sizeof(color_buf), "\033[38;2;%d;%d;%dm",
                                 top >> 16 & 0xFF, top >> 8 & 0xFF, top & 0xFF);
                frame_buffer.append(color_buf, n);
                last_fg = top;
            }
            if (bot != last_bg)
            {
                const int n = snprintf(color_buf, sizeof(color_buf), "\033[48;2;%d;%d;%dm",
                                 bot >> 16 & 0xFF, bot >> 8 & 0xFF, bot & 0xFF);
                frame_buffer.append(color_buf, n);
                last_bg = bot;
            }
            frame_buffer += render_char;
        }
        frame_buffer += "\033[0m";
        last_fg = 0xFFFFFFFF;
        last_bg = 0xFFFFFFFF;
    }
    render_submit_frame(std::move(frame_buffer));
}

void render_shutdown()
{
    const char* leave_alt = "\033[?1003l\033[?1006l\033[?25h\033[0m\033[?1049l";
    stdout_write(leave_alt, std::strlen(leave_alt));
    stdout_restore();
}

void render_output_run()
{
    std::string frame;
    while (true)
    {
        if (!stdout_flush_pending())
        {
            usleep(1000);
            continue;
        }
        if (!render_wait_for_frame(frame))
        {
            break;
        }
        pending_output = std::move(frame);
        pending_offset = 0;
    }
}

void render_output_request_stop()
{
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        output_shutdown = true;
    }
    frame_cv.notify_all();
}

size_t render_get_terminal_width()
{
    return raw_width.load(std::memory_order_relaxed);
}

size_t render_get_terminal_height()
{
    return raw_height.load(std::memory_order_relaxed);
}
