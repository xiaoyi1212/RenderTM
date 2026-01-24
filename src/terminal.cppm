module;

#include "prelude.hpp"

export module terminal;

import render;

export struct TerminalSize
{
    size_t width = 0;
    size_t height = 0;
};

export struct TerminalRender
{
    static void init();
    static void shutdown();
    static void update_size(int sig);
    static void submit_frame();
    static void output_loop(std::stop_token token);
    static TerminalSize size();
};

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::string_view kRenderChar = "\u2580";
constexpr std::string_view kEnterAlt = "\033[?1049h\033[?25l\033[?1003h\033[?1006h";
constexpr std::string_view kLeaveAlt = "\033[?1003l\033[?1006l\033[?25h\033[?1049l";

std::array<char, 65536> stdout_buffer{};
std::atomic_size_t raw_width{0};
std::atomic_size_t raw_height{0};
std::atomic<double> output_fps{0.0};

struct RenderFrame {
    size_t width = 0;
    size_t height = 0;
    double render_fps = 0.0;
    Vec3 cam_pos{};
    Vec2 cam_rot{};
    double sharpen = 0.0;
    double sharpen_pct = 0.0;
    std::vector<uint32_t> pixels;
};

struct FpsCounter {
    Clock::time_point last = Clock::now();
    int frame_count = 0;
    double fps = 0.0;

    double tick() {
        ++frame_count;
        const auto now = Clock::now();
        const std::chrono::duration<double> elapsed = now - last;
        if (elapsed.count() >= 1.0) {
            fps = frame_count / elapsed.count();
            frame_count = 0;
            last = now;
        }
        return fps;
    }
};

struct StdoutMode {
    int flags = -1;

    void enable_nonblock() {
        flags = fcntl(STDOUT_FILENO, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK);
        }
    }

    void restore() {
        if (flags >= 0) {
            fcntl(STDOUT_FILENO, F_SETFL, flags);
        }
    }
};

struct OutputWriter {
    std::string pending;
    size_t offset = 0;

    bool has_pending() const { return !pending.empty(); }

    void set(std::string text) {
        pending = std::move(text);
        offset = 0;
    }

    bool flush() {
        if (pending.empty()) return true;
        const size_t remaining = pending.size() - offset;
        if (remaining == 0) {
            pending.clear();
            offset = 0;
            return true;
        }

        const ssize_t n = ::write(STDOUT_FILENO, pending.data() + offset, remaining);
        if (n > 0) {
            offset += static_cast<size_t>(n);
            if (offset >= pending.size()) {
                pending.clear();
                offset = 0;
                return true;
            }
            return false;
        }
        return false;
    }
};

struct FrameQueue {
    std::mutex mutex;
    std::condition_variable cv;
    std::optional<RenderFrame> queued;
    std::vector<uint32_t> recycled;
    bool shutdown = false;

    void submit(RenderFrame frame) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (queued && !queued->pixels.empty()) {
                if (recycled.capacity() < queued->pixels.capacity()) {
                    recycled = std::move(queued->pixels);
                } else {
                    queued->pixels.clear();
                }
            }
            queued = std::move(frame);
        }
        cv.notify_one();
    }

    bool wait(RenderFrame& out) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return queued.has_value() || shutdown; });
        if (shutdown) return false;
        out = std::move(*queued);
        queued.reset();
        return true;
    }

    std::vector<uint32_t> take_recycled() {
        std::lock_guard<std::mutex> lock(mutex);
        return std::move(recycled);
    }

    void recycle(std::vector<uint32_t> pixels) {
        if (pixels.empty()) return;
        std::lock_guard<std::mutex> lock(mutex);
        if (recycled.capacity() < pixels.capacity()) {
            recycled = std::move(pixels);
        }
    }

    void request_stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown = true;
        }
        cv.notify_all();
    }
};

StdoutMode stdout_mode;
OutputWriter output_writer;
FrameQueue frame_queue;
FpsCounter render_fps_counter;
FpsCounter output_fps_counter;

std::string render_format_frame(const RenderFrame& frame, const RenderFrame* prev) {
    const size_t width = frame.width;
    const size_t height = frame.height;
    const size_t display_rows = height / 2;
    const size_t term_height = display_rows + 1;

    std::string frame_buffer;
    frame_buffer.reserve(width * display_rows * 20);

    frame_buffer += "\033[0m\033[H\033[7m";
    const double rad_to_deg = 180.0 / 3.14159265358979323846;
    const double yaw_deg = frame.cam_rot.x * rad_to_deg;
    const double pitch_deg = frame.cam_rot.y * rad_to_deg;
    const double out_fps = output_fps.load(std::memory_order_relaxed);
    const size_t info_start = frame_buffer.size();
    std::format_to(std::back_inserter(frame_buffer),
                   " RenderTM v0.0.1 | Terminal:{}x{} Pixel:{}x{} | "
                   "Render:{:.2f} Output:{:.2f} | Sharpen:{:.3f} ({:.0f}%) | "
                   "Cam({:.2f},{:.2f},{:.2f}) Rot({:.1f},{:.1f})",
                   width, term_height, width, height, frame.render_fps, out_fps,
                   frame.sharpen, frame.sharpen_pct, frame.cam_pos.x, frame.cam_pos.y,
                   frame.cam_pos.z, yaw_deg, pitch_deg);
    const size_t len = frame_buffer.size() - info_start;
    if (len < width) frame_buffer.append(width - len, ' ');
    frame_buffer += "\033[0m";

    const bool have_prev = prev && prev->width == width && prev->height == height &&
                           prev->pixels.size() == frame.pixels.size();
    uint32_t last_fg = 0xFFFFFFFF;
    uint32_t last_bg = 0xFFFFFFFF;

    auto append_ansi_rgb = [&](bool fg, uint32_t color) {
        char buf[40];
        char* it = buf;
        char* const buf_end = buf + sizeof(buf) - 1;
        *it++ = '\033';
        *it++ = '[';
        *it++ = fg ? '3' : '4';
        *it++ = '8';
        *it++ = ';';
        *it++ = '2';
        *it++ = ';';

        auto append_u8 = [&](uint32_t v, char end) {
            auto result = std::to_chars(it, buf_end, v);
            it = result.ptr;
            *it++ = end;
        };

        append_u8((color >> 16) & 0xFF, ';');
        append_u8((color >> 8) & 0xFF, ';');
        auto result = std::to_chars(it, buf_end, color & 0xFF);
        it = result.ptr;
        *it++ = 'm';

        frame_buffer.append(buf, static_cast<size_t>(it - buf));
    };

    auto append_cursor = [&](size_t row, size_t col) {
        frame_buffer += "\033[";
        frame_buffer += std::to_string(row);
        frame_buffer += ";";
        frame_buffer += std::to_string(col);
        frame_buffer += "H";
    };

    if (!have_prev) {
        for (size_t y = 0; y < height; y += 2) {
            append_cursor(y / 2 + 2, 1);
            last_fg = 0xFFFFFFFF;
            last_bg = 0xFFFFFFFF;

            for (size_t x = 0; x < width; ++x) {
                const uint32_t top = frame.pixels[y * width + x];
                const uint32_t bot = frame.pixels[(y + 1) * width + x];

                if (top != last_fg) {
                    append_ansi_rgb(true, top);
                    last_fg = top;
                }
                if (bot != last_bg) {
                    append_ansi_rgb(false, bot);
                    last_bg = bot;
                }
                frame_buffer += kRenderChar;
            }
            frame_buffer += "\033[0m";
        }
        return frame_buffer;
    }

    for (size_t y = 0; y < height; y += 2) {
        const size_t row = y / 2 + 2;
        size_t run_start = 0;
        bool in_run = false;

        for (size_t x = 0; x < width; ++x) {
            const size_t idx = y * width + x;
            const uint32_t top = frame.pixels[idx];
            const uint32_t bot = frame.pixels[idx + width];
            const uint32_t prev_top = prev->pixels[idx];
            const uint32_t prev_bot = prev->pixels[idx + width];
            const bool changed = top != prev_top || bot != prev_bot;

            if (changed) {
                if (!in_run) {
                    in_run = true;
                    run_start = x;
                }
            } else if (in_run) {
                append_cursor(row, run_start + 1);
                last_fg = 0xFFFFFFFF;
                last_bg = 0xFFFFFFFF;
                for (size_t rx = run_start; rx < x; ++rx) {
                    const size_t ridx = y * width + rx;
                    const uint32_t rtop = frame.pixels[ridx];
                    const uint32_t rbot = frame.pixels[ridx + width];
                    if (last_fg != rtop || last_bg != rbot) {
                        append_ansi_rgb(true, rtop);
                        append_ansi_rgb(false, rbot);
                        last_fg = rtop;
                        last_bg = rbot;
                    }
                    frame_buffer += kRenderChar;
                }
                in_run = false;
            }
        }

        if (in_run) {
            append_cursor(row, run_start + 1);
            last_fg = 0xFFFFFFFF;
            last_bg = 0xFFFFFFFF;
            for (size_t rx = run_start; rx < width; ++rx) {
                const size_t ridx = y * width + rx;
                const uint32_t rtop = frame.pixels[ridx];
                const uint32_t rbot = frame.pixels[ridx + width];
                if (last_fg != rtop || last_bg != rbot) {
                    append_ansi_rgb(true, rtop);
                    append_ansi_rgb(false, rbot);
                    last_fg = rtop;
                    last_bg = rbot;
                }
                frame_buffer += kRenderChar;
            }
        }
    }
    frame_buffer += "\033[0m";
    return frame_buffer;
}

} // namespace

void TerminalRender::init() {
    setvbuf(stdout, stdout_buffer.data(), _IOFBF, stdout_buffer.size());
    update_size(0);
    stdout_mode.enable_nonblock();
    (void)::write(STDOUT_FILENO, kEnterAlt.data(), kEnterAlt.size());
}

void TerminalRender::submit_frame() {
    const size_t term_width = raw_width.load(std::memory_order_relaxed);
    const size_t term_height = raw_height.load(std::memory_order_relaxed);
    if (term_width == 0 || term_height < 2) return;

    const size_t display_rows = term_height - 1;
    const size_t width = term_width;
    const size_t height = display_rows * 2;

    const double render_fps = render_fps_counter.tick();

    std::vector<uint32_t> framebuffer = frame_queue.take_recycled();
    if (framebuffer.size() != width * height) framebuffer.resize(width * height);
    render_update_array(framebuffer.data(), width, height);

    RenderFrame frame;
    frame.width = width;
    frame.height = height;
    frame.render_fps = render_fps;
    frame.cam_pos = render_get_camera_position();
    frame.cam_rot = render_get_camera_rotation();
    frame.sharpen = render_debug_get_taa_sharpen_strength();
    frame.sharpen_pct = render_debug_get_taa_sharpen_percent();
    frame.pixels = std::move(framebuffer);
    frame_queue.submit(std::move(frame));
}

void TerminalRender::shutdown() {
    (void)::write(STDOUT_FILENO, kLeaveAlt.data(), kLeaveAlt.size());
    stdout_mode.restore();
}

TerminalSize TerminalRender::size() {
    return {
        raw_width.load(std::memory_order_relaxed),
        raw_height.load(std::memory_order_relaxed)
    };
}

void TerminalRender::update_size(int sig) {
    winsize w{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) != 0 || w.ws_col == 0 || w.ws_row == 0) {
        if (raw_width.load(std::memory_order_relaxed) == 0 || raw_height.load(std::memory_order_relaxed) == 0) {
            raw_width.store(80, std::memory_order_relaxed);
            raw_height.store(24, std::memory_order_relaxed);
        }
        return;
    }
    raw_width.store(w.ws_col, std::memory_order_relaxed);
    raw_height.store(w.ws_row, std::memory_order_relaxed);
}

void TerminalRender::output_loop(std::stop_token token) {
    std::stop_callback stop_cb(token, [] { frame_queue.request_stop(); });
    RenderFrame frame;
    RenderFrame last_frame;
    bool have_last = false;
    while (true) {
        const bool had_pending = output_writer.has_pending();
        if (!output_writer.flush()) {
            usleep(1000);
            continue;
        }
        if (had_pending) {
            output_fps.store(output_fps_counter.tick(), std::memory_order_relaxed);
        }
        if (!frame_queue.wait(frame)) {
            break;
        }
        output_writer.set(render_format_frame(frame, have_last ? &last_frame : nullptr));
        if (have_last) {
            std::vector<uint32_t> recycle = std::move(last_frame.pixels);
            last_frame = std::move(frame);
            frame_queue.recycle(std::move(recycle));
        } else {
            last_frame = std::move(frame);
            have_last = true;
        }
    }
}
