#pragma once

#include <cstddef>
#include <string>

void render_init();
void render_print();
void render_output_run();
void render_output_request_stop();
void render_shutdown();
size_t render_get_terminal_width();
size_t render_get_terminal_height();
