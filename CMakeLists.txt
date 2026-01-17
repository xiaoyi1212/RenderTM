cmake_minimum_required(VERSION 4.0)
project(untitled)

set(CMAKE_CXX_STANDARD 20)

include(CTest)
enable_testing()
include(FetchContent)

add_executable(untitled main.cpp
        terminal_render.cpp
        terminal_render.h
        render.cpp
        render.h
        input.cpp
        input.h
        vector.cpp
        vector.h
        keyboard.h
        keyboard.cpp)

    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.12.0
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
    )
FetchContent_MakeAvailable(Catch2)

add_executable(untitled_tests
        tests/test_input.cpp
        tests/test_vector.cpp
        tests/test_render.cpp
        input.cpp
        vector.cpp
        render.cpp)
target_link_libraries(untitled_tests PRIVATE Catch2::Catch2WithMain)
target_include_directories(untitled_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(untitled_tests PRIVATE cxx_std_20)

include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
catch_discover_tests(untitled_tests)
