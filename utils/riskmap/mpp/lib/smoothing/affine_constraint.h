

#pragma once

#include <Eigen/Core>
#include <vector>

#define udrive_inf ((double)1e30)  // NOLINT

/**
 * @class AffineConstraint
 * @brief constraint in form of "l <= Ax <= u", equality constraint means l = u
 */
class AffineConstraint {
 public:
  AffineConstraint() = default;

  AffineConstraint(const Eigen::MatrixXd& constraint_matrix,
                   const std::vector<double>& lower_bound,
                   const std::vector<double>& upper_bound);

  const Eigen::MatrixXd& constraint_matrix() const;

  const std::vector<double>& lower_bound() const;

  const std::vector<double>& upper_bound() const;

  bool AddConstraint(const Eigen::MatrixXd& constraint_matrix,
                     const std::vector<double>& lower_bound,
                     const std::vector<double>& upper_bound);

  void Print() const;

 private:
  Eigen::MatrixXd constraint_matrix_;

  std::vector<double> lower_bound_;

  std::vector<double> upper_bound_;
};
