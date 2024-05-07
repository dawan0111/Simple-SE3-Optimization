cmake_minimum_required(VERSION 3.21)

set(PACKAGE_NAME SIMPLE_SE3_OPTIMIZATION)
set(PACKAGE_VERSION 0.0.1)

project(${PACKAGE_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Sophus REQUIRED)

include_directories("/usr/include/eigen3")
add_executable(MAIN main.cpp)
target_link_libraries(MAIN Sophus::Sophus)