#pragma once

#include <cstddef>
#include <stop_token>

struct TerminalSize
{
    size_t width = 0;
    size_t height = 0;
};

struct TerminalRender
{
    static void init();
    static void shutdown();
    static void update_size(int sig);
    static void submit_frame();
    static void output_loop(std::stop_token token);
    static TerminalSize size();
};
