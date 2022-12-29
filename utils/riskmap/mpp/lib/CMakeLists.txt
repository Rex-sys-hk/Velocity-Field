cmake_minimum_required(VERSION 3.4.0)
project(frenet LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Eigen3 REQUIRED)
include_directories(AFTER ${EIGEN3_INCLUDE_DIR})
include_directories(
  ${PROJECT_SOURCE_DIR}
  thirdparty
  thirdparty/pybind11/include
)

add_subdirectory(utils)
add_subdirectory(solver)
add_subdirectory(smoothing)
add_subdirectory(thirdparty/pybind11)

find_package(Python3 COMPONENTS Development NumPy)

# add_executable(test
# # c++
#   frenet/polynomials.cc
#   frenet/frenet_sampler.cc
#   frenet/reference_line.cc
#   frenet/frenet_trajectory.cc
#   frenet/frenet_transform.cc
#   test.cc
# )

# target_include_directories(test PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
# target_link_libraries(test PRIVATE smoothing solver utils Python3::Python Python3::NumPy)

pybind11_add_module(frenet
    # c++
    frenet/polynomials.cc
    frenet/frenet_sampler.cc
    frenet/reference_line.cc
    frenet/frenet_transform.cc
    frenet/box2d.cc
    # wrapper
    wrapper.cc
)

target_link_libraries(frenet PRIVATE smoothing solver utils)