

#include "smoothing/osqp_spline2d_solver.h"

// #include <glog/logging.h>

#include <algorithm>

#include "solver/osqp/osqp_interface.h"



OsqpSpline2dSolver::OsqpSpline2dSolver(const std::vector<double>& t_knots,
                                       const uint32_t order)
    : Spline2dSolver(t_knots, order) {}

void OsqpSpline2dSolver::Reset(const std::vector<double>& t_knots,
                               const uint32_t order) {
  spline_ = Spline2d(t_knots, order);
  kernel_ = Spline2dKernel(t_knots, order);
  constraint_ = Spline2dConstraint(t_knots, order);
}

Spline2dConstraint* OsqpSpline2dSolver::mutable_constraint() {
  return &constraint_;
}

Spline2dKernel* OsqpSpline2dSolver::mutable_kernel() { return &kernel_; }

Spline2d* OsqpSpline2dSolver::mutable_spline() { return &spline_; }

bool OsqpSpline2dSolver::Solve() {
  const Eigen::MatrixXd& P = kernel_.kernel_matrix();
  const Eigen::MatrixXd& A = constraint_.constraint_matrix();

  Eigen::VectorXd q = kernel_.gradient();

  const size_t n = A.rows();
  // have to copy once anyway
  std::vector<double> lower_bound(constraint_.lower_bound());
  std::vector<double> upper_bound(constraint_.upper_bound());

  // Eigen::Map does not create its own memory
  Eigen::VectorXd l =
      Eigen::Map<Eigen::VectorXd, Eigen::Aligned>(lower_bound.data(), n);
  Eigen::VectorXd u =
      Eigen::Map<Eigen::VectorXd, Eigen::Aligned>(upper_bound.data(), n);

  Eigen::VectorXd solution;

  if (!OsqpInterface::Solve(P, q, A, l, u, &solution)) {
    // std::cout << "fail to solve the spline" << std::endl;
    return false;
  }

  return spline_.set_splines(solution, spline_.spline_order());
}

// extract
const Spline2d& OsqpSpline2dSolver::spline() const { return spline_; }

