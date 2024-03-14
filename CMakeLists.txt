cmake_minimum_required(VERSION 3.22)
project(savitzky_golay)

set(CMAKE_CXX_FLAGS "-O3")

# add_subdirectory(pybind11)
find_package(pybind11 CONFIG)
pybind11_add_module(savitzky_golay src/main.cpp)

find_package(PkgConfig)
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

find_package(Boost REQUIRED)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(savitzky_golay
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

