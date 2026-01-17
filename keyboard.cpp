#include "keyboard.h"

#include <cerrno>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

static termios original_termios{};
static int original_flags = 0;
static bool configured = false;

void keyboard_setup()
{
    if (configured) return;

    tcgetattr(STDIN_FILENO, &original_termios);
    termios raw = original_termios;
    raw.c_lflag &= ~(ECHO | ECHONL | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);

    original_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, original_flags | O_NONBLOCK);

    configured = true;
}

void keyboard_restore()
{
    if (!configured) return;
    tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);
    fcntl(STDIN_FILENO, F_SETFL, original_flags);
    configured = false;
}

int keyboard_read_char()
{
    unsigned char ch = 0;
    const ssize_t count = read(STDIN_FILENO, &ch, 1);
    if (count == 1) return ch;
    if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) return -1;
    return -1;
}
