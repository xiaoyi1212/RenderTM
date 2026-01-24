#include "keyboard.h"

#include <cerrno>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

KeyboardMode::KeyboardMode()
{
    if (tcgetattr(STDIN_FILENO, &original_termios_) != 0)
    {
        return;
    }
    termios raw = original_termios_;
    raw.c_lflag &= ~(ECHO | ECHONL | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    (void)tcsetattr(STDIN_FILENO, TCSANOW, &raw);

    original_flags_ = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (original_flags_ >= 0)
    {
        (void)fcntl(STDIN_FILENO, F_SETFL, original_flags_ | O_NONBLOCK);
    }

    configured_ = true;
}

KeyboardMode::~KeyboardMode()
{
    if (!configured_)
    {
        return;
    }
    (void)tcsetattr(STDIN_FILENO, TCSANOW, &original_termios_);
    if (original_flags_ >= 0)
    {
        (void)fcntl(STDIN_FILENO, F_SETFL, original_flags_);
    }
    configured_ = false;
}

std::optional<unsigned char> KeyboardMode::read_char() const
{
    unsigned char ch = 0;
    const ssize_t count = read(STDIN_FILENO, &ch, 1);
    if (count == 1)
    {
        return ch;
    }
    if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
    {
        return std::nullopt;
    }
    return std::nullopt;
}
