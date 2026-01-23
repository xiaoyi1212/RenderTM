#include "terminal_render.h"
#include "render.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <utility>
#include <string>
#include <vector>
#include <sys/ioctl.h>
#include <unistd.h>

static char buffer[65536];
std::atomic<size_t> raw_width, raw_height;
static auto lastTime = std::chrono::high_resolution_clock::now();
static int frameCount = 0;
static double fps = 0.0;
static auto outputLastTime = std::chrono::high_resolution_clock::now();
static int outputFrameCount = 0;
static std::atomic<double> outputFps{0.0};

struct RenderFrame
{
    size_t width = 0;
    size_t height = 0;
    double render_fps = 0.0;
    Vec3 cam_pos{};
    Vec2 cam_rot{};
    double sharpen = 0.0;
    double sharpen_pct = 0.0;
    std::vector<uint32_t> pixels;
};

static const std::string render_char = "\u2580";

static int stdout_flags = -1;
static std::string pending_output;
static size_t pending_offset = 0;
static std::mutex frame_mutex;
static std::condition_variable frame_cv;
static RenderFrame queued_frame;
static std::vector<uint32_t> recycled_pixels;
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

static void render_submit_frame(RenderFrame frame)
{
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        if (frame_ready && !queued_frame.pixels.empty())
        {
            if (recycled_pixels.capacity() < queued_frame.pixels.capacity())
            {
                recycled_pixels = std::move(queued_frame.pixels);
            }
            else
            {
                queued_frame.pixels.clear();
            }
        }
        queued_frame = std::move(frame);
        frame_ready = true;
    }
    frame_cv.notify_one();
}

static bool render_wait_for_frame(RenderFrame& out)
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

static std::vector<uint32_t> render_take_recycled_pixels()
{
    std::lock_guard<std::mutex> lock(frame_mutex);
    return std::move(recycled_pixels);
}

static void render_recycle_pixels(std::vector<uint32_t> pixels)
{
    if (pixels.empty())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(frame_mutex);
    if (recycled_pixels.capacity() < pixels.capacity())
    {
        recycled_pixels = std::move(pixels);
    }
}

static std::string render_format_frame(const RenderFrame& frame, const RenderFrame* prev)
{
    const size_t width = frame.width;
    const size_t height = frame.height;
    const size_t display_rows = height / 2;
    const size_t term_height = display_rows + 1;

    std::string frame_buffer;
    frame_buffer.reserve(width * display_rows * 20);

    frame_buffer += "\033[0m\033[H\033[7m";
    char info[256];
    const double rad_to_deg = 180.0 / 3.14159265358979323846;
    const double yaw_deg = frame.cam_rot.x * rad_to_deg;
    const double pitch_deg = frame.cam_rot.y * rad_to_deg;
    const double out_fps = outputFps.load(std::memory_order_relaxed);
    const int len = snprintf(info, sizeof(info),
                             " RenderTM v0.0.1 | Terminal:%lux%lu Pixel:%lux%lu | Render:%.2f Output:%.2f | Sharpen:%.3f (%.0f%%) | Cam(%.2f,%.2f,%.2f) Rot(%.1f,%.1f)",
                             width, static_cast<size_t>(term_height), width, height, frame.render_fps,
                             out_fps, frame.sharpen, frame.sharpen_pct,
                             frame.cam_pos.x, frame.cam_pos.y, frame.cam_pos.z, yaw_deg, pitch_deg);
    frame_buffer += info;
    if (len < static_cast<int>(width)) frame_buffer.append(width - len, ' ');
    frame_buffer += "\033[0m";

    const bool have_prev = prev && prev->width == width && prev->height == height &&
                           prev->pixels.size() == frame.pixels.size();
    uint32_t last_fg = 0xFFFFFFFF;
    uint32_t last_bg = 0xFFFFFFFF;
    char color_buf[64];

    auto append_fg = [&](uint32_t color) {
        const int n = snprintf(color_buf, sizeof(color_buf), "\033[38;2;%d;%d;%dm",
                               (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF);
        frame_buffer.append(color_buf, n);
    };
    auto append_bg = [&](uint32_t color) {
        const int n = snprintf(color_buf, sizeof(color_buf), "\033[48;2;%d;%d;%dm",
                               (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF);
        frame_buffer.append(color_buf, n);
    };
    auto append_cursor = [&](size_t row, size_t col) {
        frame_buffer += "\033[";
        frame_buffer += std::to_string(row);
        frame_buffer += ";";
        frame_buffer += std::to_string(col);
        frame_buffer += "H";
    };

    if (!have_prev)
    {
        for (size_t y = 0; y < height; y += 2)
        {
            append_cursor(y / 2 + 2, 1);
            last_fg = 0xFFFFFFFF;
            last_bg = 0xFFFFFFFF;

            for (size_t x = 0; x < width; ++x)
            {
                const uint32_t top = frame.pixels[y * width + x];
                const uint32_t bot = frame.pixels[(y + 1) * width + x];

                if (top != last_fg)
                {
                    append_fg(top);
                    last_fg = top;
                }
                if (bot != last_bg)
                {
                    append_bg(bot);
                    last_bg = bot;
                }
                frame_buffer += render_char;
            }
            frame_buffer += "\033[0m";
        }
        return frame_buffer;
    }

    for (size_t y = 0; y < height; y += 2)
    {
        const size_t row = y / 2 + 2;
        size_t run_start = 0;
        bool in_run = false;

        for (size_t x = 0; x < width; ++x)
        {
            const size_t idx = y * width + x;
            const uint32_t top = frame.pixels[idx];
            const uint32_t bot = frame.pixels[idx + width];
            const uint32_t prev_top = prev->pixels[idx];
            const uint32_t prev_bot = prev->pixels[idx + width];
            const bool changed = top != prev_top || bot != prev_bot;

            if (changed)
            {
                if (!in_run)
                {
                    in_run = true;
                    run_start = x;
                }
            }
            else if (in_run)
            {
                append_cursor(row, run_start + 1);
                last_fg = 0xFFFFFFFF;
                last_bg = 0xFFFFFFFF;
                for (size_t rx = run_start; rx < x; ++rx)
                {
                    const size_t ridx = y * width + rx;
                    const uint32_t rtop = frame.pixels[ridx];
                    const uint32_t rbot = frame.pixels[ridx + width];
                    if (last_fg != rtop || last_bg != rbot)
                    {
                        append_fg(rtop);
                        append_bg(rbot);
                        last_fg = rtop;
                        last_bg = rbot;
                    }
                    frame_buffer += render_char;
                }
                in_run = false;
            }
        }

        if (in_run)
        {
            append_cursor(row, run_start + 1);
            last_fg = 0xFFFFFFFF;
            last_bg = 0xFFFFFFFF;
            for (size_t rx = run_start; rx < width; ++rx)
            {
                const size_t ridx = y * width + rx;
                const uint32_t rtop = frame.pixels[ridx];
                const uint32_t rbot = frame.pixels[ridx + width];
                if (last_fg != rtop || last_bg != rbot)
                {
                    append_fg(rtop);
                    append_bg(rbot);
                    last_fg = rtop;
                    last_bg = rbot;
                }
                frame_buffer += render_char;
            }
        }
    }
    frame_buffer += "\033[0m";
    return frame_buffer;
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

    std::vector<uint32_t> framebuffer = render_take_recycled_pixels();
    if (framebuffer.size() != width * height) framebuffer.resize(width * height);
    render_update_array(framebuffer.data(), width, height);

    RenderFrame frame;
    frame.width = width;
    frame.height = height;
    frame.render_fps = fps;
    frame.cam_pos = render_get_camera_position();
    frame.cam_rot = render_get_camera_rotation();
    frame.sharpen = render_debug_get_taa_sharpen_strength();
    frame.sharpen_pct = render_debug_get_taa_sharpen_percent();
    frame.pixels = std::move(framebuffer);
    render_submit_frame(std::move(frame));
}

void render_shutdown()
{
    const char* leave_alt = "\033[?1003l\033[?1006l\033[?25h\033[0m\033[?1049l";
    stdout_write(leave_alt, std::strlen(leave_alt));
    stdout_restore();
}

void render_output_run()
{
    RenderFrame frame;
    RenderFrame last_frame;
    bool have_last = false;
    while (true)
    {
        const bool had_pending = !pending_output.empty();
        if (!stdout_flush_pending())
        {
            usleep(1000);
            continue;
        }
        if (had_pending)
        {
            outputFrameCount++;
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = currentTime - outputLastTime;
            if (elapsed.count() >= 1.0)
            {
                outputFps.store(outputFrameCount / elapsed.count(), std::memory_order_relaxed);
                outputFrameCount = 0;
                outputLastTime = currentTime;
            }
        }
        if (!render_wait_for_frame(frame))
        {
            break;
        }
        pending_output = render_format_frame(frame, have_last ? &last_frame : nullptr);
        pending_offset = 0;
        if (have_last)
        {
            std::vector<uint32_t> recycle = std::move(last_frame.pixels);
            last_frame = std::move(frame);
            render_recycle_pixels(std::move(recycle));
        }
        else
        {
            last_frame = std::move(frame);
            have_last = true;
        }
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
