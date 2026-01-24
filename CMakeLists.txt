cmake_minimum_required(VERSION 4.0)
project(RenderTM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
include(FetchContent)

add_library(core STATIC)
target_sources(core
    PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES FILES
    src/controls.cppm
    src/input.cppm
    src/keyboard.cppm
    src/render.cppm
    src/terminal.cppm
    src/noise.cppm
)
target_compile_features(core PUBLIC cxx_std_23)
target_compile_options(core PRIVATE
    "$<$<CONFIG:Release>:-O3>"
    "$<$<CONFIG:Release>:-ffast-math>"
)

add_executable(render_tm src/main.cpp)
target_link_libraries(render_tm PRIVATE core)
target_precompile_headers(render_tm PRIVATE src/prelude.hpp)

if(BUILD_TESTING)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.12.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(tests
        tests/test_controls.cpp
        tests/test_input.cpp
        tests/test_render.cpp
    )
    target_link_libraries(tests PRIVATE core Catch2::Catch2WithMain)
    target_precompile_headers(tests PRIVATE tests/test_prelude.hpp)
    target_compile_options(tests PRIVATE -fno-omit-frame-pointer)

    include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
    catch_discover_tests(tests)
endif()
