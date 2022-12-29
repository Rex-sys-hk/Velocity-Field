#pragma once

#include <vector>
#include <Eigen/Dense>

template<typename T>
using vector_Eigen = std::vector<T, Eigen::aligned_allocator<T>>;

using vector_Eigen2d = vector_Eigen<Eigen::Vector2d>;
using vector_Eigen3d = vector_Eigen<Eigen::Vector3d>;