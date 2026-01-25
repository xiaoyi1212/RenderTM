# RenderTM

A stupid Minecraft-like game (just a render currently) running in terminal.

## Prepare

Install CMake v4.0+, Ninja, and your favorite C++ compiler.

## Compile & Run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/render_tm
```
