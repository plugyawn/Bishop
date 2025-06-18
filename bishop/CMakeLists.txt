cmake_minimum_required(VERSION 3.0)
project(bishop VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)