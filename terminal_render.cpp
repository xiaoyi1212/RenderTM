#include "terminal_render.h"
#include "render.h"
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <ostream>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>

static char buffer[65536];
std::atomic<size_t> raw_width, raw_height;
static auto lastTime = std::chrono::high_resolution_clock::now();
static int frameCount = 0;
static double fps = 0.0;

extern double tps;

static const std::string render_char = "\u2580";

static int stdout_flags = -1;
static bool stdout_nonblock = true;
static std::string pending_output;
static size_t pending_offset = 0;

static void stdout_init()
{
    if (stdout_nonblock)
    {
        stdout_flags = fcntl(STDOUT_FILENO, F_GETFL, 0);
        if (stdout_flags >= 0)
        {
            fcntl(STDOUT_FILENO, F_SETFL, stdout_flags | O_NONBLOCK);
        }
    }
}

static void stdout_restore()
{
    if (stdout_nonblock && stdout_flags >= 0)
    {
        fcntl(STDOUT_FILENO, F_SETFL, stdout_flags);
    }
}

static void stdout_write_best_effort(const char* data, size_t len)
{
    if (!stdout_nonblock)
    {
        size_t written = 0;
        while (written < len)
        {
            const ssize_t n = ::write(STDOUT_FILENO, data + written, len - written);
            if (n <= 0) break;
            written += static_cast<size_t>(n);
        }
        return;
    }

    const ssize_t n = ::write(STDOUT_FILENO, data, len);
    if (n < 0)
    {
        return;
    }
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
    const char* enter_alt = "\033[?1049h\033[?25l\033[0m\033[2J\033[H";
    stdout_write_best_effort(enter_alt, std::strlen(enter_alt));
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
    const int len = snprintf(info, sizeof(info), " RenderTM v0.0.1 Terminal:%lux%lu | Pixel:%lux%lu | FPS:%.2f TPS:%.2f",
                             width, static_cast<size_t>(term_height), width, height, fps, tps);
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
    if (stdout_nonblock)
    {
        if (!stdout_flush_pending())
        {
            return;
        }
        pending_output = std::move(frame_buffer);
        pending_offset = 0;
        if (!stdout_flush_pending())
        {
            return;
        }
    }
    else
    {
        stdout_write_best_effort(frame_buffer.data(), frame_buffer.size());
    }
}

void render_shutdown()
{
    const char* leave_alt = "\033[?25h\033[0m\033[?1049l";
    stdout_write_best_effort(leave_alt, std::strlen(leave_alt));
    stdout_restore();
}
