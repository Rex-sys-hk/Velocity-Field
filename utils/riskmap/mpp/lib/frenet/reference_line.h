#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

#include "frenet/frenet_transform.h"
#include "smoothing/spline2d.h"
#include "tk_spline/spline.h"
#include "utils/type.h"

class ReferenceLine {
 public:
  ReferenceLine() = default;

  ReferenceLine(const std::vector<double>& x, const std::vector<double>& y,
                const std::vector<double>& s);

  void Sample(const std::vector<double> s, std::vector<double>& sample_x,
              std::vector<double>& sample_y) const;

  double GetProjection(const Eigen::Vector2d& pos,
                       const double init_guess = 0.0) const;

  double GetArcLength(const Eigen::Vector2d& pos, const double epsilon = 1e-3,
                      const double upper_bound = 20) const;

  double GetMaxCurvature(const std::vector<double>& s) const;

  ReferencePoint GetReferencePoint(const double s) const;

  Eigen::MatrixXd GetPoints() const;

  inline double length() const { return length_; }

  Eigen::VectorXd get_meta() const;
  void load_meta(Eigen::VectorXd meta);

 private:
  double length_ = 0.0;

  Spline2d spline_;
  vector_Eigen2d refs_;
};
