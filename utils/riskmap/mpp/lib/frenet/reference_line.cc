#include "reference_line.h"

#include "smoothing/osqp_spline2d_solver.h"

ReferenceLine::ReferenceLine(const std::vector<double>& x,
                             const std::vector<double>& y,
                             const std::vector<double>& s) {
  // spline_x_ = tk::spline(s, x);
  // spline_y_ = tk::spline(s, y);

  length_ = s.back();
  refs_.resize(x.size());

  for (int i = 0; i < x.size(); ++i) {
    refs_[i] = Eigen::Vector2d(x[i], y[i]);
  }

  constexpr int segments = 3;
  double segment_len = (s.back() - s.front()) / segments;

  std::vector<double> s_knots;
  for (int i = 0; i < segments + 1; ++i) {
    s_knots.emplace_back(s.front() + i * segment_len);
    // std::cout << s_knots.back() << std::endl;
  }

  OsqpSpline2dSolver spline2d_solver(s_knots, 4);
  auto mutable_kernel = spline2d_solver.mutable_kernel();
  auto mutable_constraint = spline2d_solver.mutable_constraint();

  mutable_kernel->Add2dReferenceLineKernelMatrix(s, refs_, 10);
  mutable_kernel->AddRegularization(1e-5);
  mutable_kernel->Add2dSecondOrderDerivativeMatrix(20);
  mutable_kernel->Add2dThirdOrderDerivativeMatrix(100);
  mutable_constraint->Add2dThirdDerivativeSmoothConstraint();

  spline2d_solver.Solve();
  spline_ = spline2d_solver.spline();
}

void ReferenceLine::Sample(const std::vector<double> s,
                           std::vector<double>& sample_x,
                           std::vector<double>& sample_y) const {
  for (const auto location : s) {
    auto p = spline_.pos(location);
    // printf("%f, (%f, %f)\n", location, p.x(), p.y());
    sample_x.emplace_back(p.x());
    sample_y.emplace_back(p.y());
  }
}

double ReferenceLine::GetArcLength(const Eigen::Vector2d& pos,
                                   const double epsilon,
                                   const double upper_bound) const {
  double lb = 0;
  double ub = upper_bound;
  double step = (ub - lb) / 2.0;
  double mid = lb + step;

  double l_value = (spline_.pos(lb) - pos).squaredNorm();
  double r_value = (spline_.pos(ub) - pos).squaredNorm();
  double m_value = (spline_.pos(mid) - pos).squaredNorm();

  while (std::fabs(ub - lb) > epsilon) {
    double min = std::min(std::min(l_value, m_value), r_value);
    step *= 0.5;
    if (min == l_value) {
      ub = mid;
      mid = lb + step;
      m_value = (spline_.pos(mid) - pos).squaredNorm();
      r_value = (spline_.pos(ub) - pos).squaredNorm();
    } else if (min == m_value) {
      lb = mid - step;
      ub = mid + step;
      l_value = (spline_.pos(lb) - pos).squaredNorm();
      r_value = (spline_.pos(ub) - pos).squaredNorm();
    } else {
      lb = mid;
      mid = ub - step;
      l_value = (spline_.pos(lb) - pos).squaredNorm();
      m_value = (spline_.pos(mid) - pos).squaredNorm();
    }
  }
  return (lb + ub) / 2;
}

double ReferenceLine::GetProjection(const Eigen::Vector2d& pos,
                                    const double init_guess) const {
  constexpr double max_iter = 20;
  constexpr double tol = 1e-5;

  double s = init_guess;
  double jac, hess, delta_s;
  Eigen::Vector2d ds, dds, diff;
  for (int i = 0; i < max_iter; ++i) {
    ds(0) = spline_.DerivativeX(s);
    ds(1) = spline_.DerivativeY(s);
    dds(0) = spline_.SecondDerivativeX(s);
    dds(1) = spline_.SecondDerivativeY(s);

    diff = spline_.pos(s) - pos;

    jac = diff.dot(ds);
    hess = ds.dot(ds) + diff.dot(dds);

    delta_s = -jac / hess;
    s += delta_s;

    if (std::fabs(delta_s) < tol) break;
  }

  return s;
}

ReferencePoint ReferenceLine::GetReferencePoint(const double s) const {
  ReferencePoint ref;
  ref.s = s;
  ref.point = spline_.pos(s);
  ref.theta = spline_.theta(s);
  spline_.GetCurvature(s, &ref.kappa, &ref.dkappa);
  return ref;
}

double ReferenceLine::GetMaxCurvature(const std::vector<double>& s) const {
  double curvature = 0.0;
  double max_curvature = -1e9;
  for (const auto p : s) {
    curvature = std::fabs(spline_.GetCurvature(0.0));
    if (max_curvature < curvature) {
      max_curvature = curvature;
    }
  }
  return max_curvature;
}

Eigen::MatrixXd ReferenceLine::GetPoints() const {
  Eigen::MatrixXd ref_line(35, 2);
  for (int i = 0; i < 35; ++i) {
    ref_line(i, 0) = spline_.x(2 * i);
    ref_line(i, 1) = spline_.y(2 * i);
  }
  return ref_line;
}

Eigen::VectorXd ReferenceLine::get_meta() const {
  auto order = spline_.spline_order();
  auto t_knots = spline_.t_knots();
  auto spline2d_params = spline_.get_splines();

  auto knots_size = t_knots.size();
  auto param_size = spline2d_params.size();

  Eigen::VectorXd meta(3 + knots_size + param_size);

  meta(0) = order;
  meta(1) = knots_size;
  meta(2) = param_size;

  for (int i = 0; i < t_knots.size(); ++i) {
    meta(3 + i) = t_knots[i];
  }

  meta.block(3 + knots_size, 0, param_size, 1) = spline2d_params;

  return meta;
}

void ReferenceLine::load_meta(Eigen::VectorXd meta) {
  auto order = meta(0);
  auto knots_size = meta(1);
  auto param_size = meta(2);

  std::vector<double> t_knots(knots_size);
  for (int i = 0; i < knots_size; ++i) {
    t_knots[i] = meta(3 + i);
  }

  Eigen::VectorXd spline2d_params =
      meta.block(3 + knots_size, 0, param_size, 1);

  spline_ = Spline2d(t_knots, order);
  spline_.set_splines(spline2d_params, order);
}