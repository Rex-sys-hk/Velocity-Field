#pragma once

#include <Eigen/Core>
#include <vector>

class Box2D {
 public:
  Box2D() = default;
  Box2D(const Eigen::Vector2d& center, const double angle, const double length,
        const double width);
  ~Box2D() = default;

  bool HasOverlapWith(const Box2D& box);

  inline double angle() const { return angle_; }
  inline double length() const { return 2 * half_length_; }
  inline double width() const { return 2 * half_width_; }
  inline const Eigen::Vector2d& center() const { return center_; }
  inline const std::vector<Eigen::Vector2d>& GetArchorPoints() const {
    return corners_;
  }

  inline const std::array<double, 2>& xlim() const { return xlim_; }
  inline const std::array<double, 2>& ylim() const { return ylim_; }

 private:
  void CalculateCorners();

 private:
  Eigen::Vector2d center_;
  double half_length_ = 0.0;
  double half_width_ = 0.0;
  double angle_ = 0.0;

  double cos_angle_ = 0.0;
  double sin_angle_ = 0.0;

  // right-bottom cornor is the start point, clock-wise
  std::vector<Eigen::Vector2d> corners_;

  // <min, max> of aabb
  std::array<double, 2> xlim_{1e9, -1e9};
  std::array<double, 2> ylim_{1e9, -1e9};
  // for efficiency
  std::array<double, 4> helper_{0, 0, 0, 0};
};
