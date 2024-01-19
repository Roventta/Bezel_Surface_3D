cmake_minimum_required(VERSION 3.10)
project(BezierSurface)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include ./include)

add_executable(BezierSurface main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp Shader.hpp)

target_link_libraries(BezierSurface ${OpenCV_LIBRARIES})
