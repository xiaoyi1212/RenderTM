# RenderTM

A stupid Minecraft-like game (just a render currently) running in terminal.

## Prepare

Install CMake v4.0+ and your prefer C++ compiler.

## Compile & Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -ffast-math"
cmake --build build -j
./build/untitled
```
