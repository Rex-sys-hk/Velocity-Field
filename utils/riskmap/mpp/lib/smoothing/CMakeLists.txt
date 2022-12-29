set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(smoothing STATIC)

target_sources(smoothing
  PRIVATE
    affine_constraint.cc
    discrete_points_math.cc
    osqp_spline1d_solver.cc
    osqp_spline2d_solver.cc
    polynomialxd.cc
    spline1d_constraint.cc
    spline1d_kernel_helper.cc
    spline1d_kernel.cc
    spline1d_seg.cc
    spline1d.cc
    spline2d_constraint.cc
    spline2d_kernel.cc
    spline2d_seg
    spline2d.cc
)

target_link_libraries(smoothing 
  solver
  utils
)