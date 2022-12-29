set(osqp_DIR ${PROJECT_SOURCE_DIR}/thirdparty/osqp/install/lib/cmake/osqp)

find_package(osqp REQUIRED)

add_library(solver STATIC)

target_sources(solver PRIVATE
  osqp/osqp_interface.cc
)
target_include_directories(solver PUBLIC
  ${PROJECT_SOURCE_DIR}/thirdparty/osqp/install/include
)

target_link_libraries(solver 
  osqp::osqpstatic
  ${CMAKE_DL_LIBS} 
  utils
)
