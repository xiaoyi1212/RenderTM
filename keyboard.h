#pragma once

#include <optional>
#include <termios.h>

struct KeyboardMode
{
    KeyboardMode();
    ~KeyboardMode();

    KeyboardMode(const KeyboardMode&) = delete;
    KeyboardMode& operator=(const KeyboardMode&) = delete;

    std::optional<unsigned char> read_char() const;

    struct termios original_termios_{};
    int original_flags_ = -1;
    bool configured_ = false;
};
