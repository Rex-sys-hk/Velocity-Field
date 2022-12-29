#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

class QuinticPolynomial {
 public:
  QuinticPolynomial() = default;
  QuinticPolynomial(const double x0, const double v0, const double a0,
                    const double xt, const double vt, const double at,
                    const double t, const double offset = 0.0);

  QuinticPolynomial(const double x0);

  double x(const double t);
  double dx(const double t);
  double ddx(const double t);

  Eigen::Vector3d get_derivertives(const double t);

  std::vector<double> x_vec(const std::vector<double>& t);
  // std::vector<double> dx_vec(const std::vector<double>& t);
  // std::vector<double> ddx_vec(const std::vector<double>& t);

 private:
  double offset_ = 0.0;  // ! s may start from nonzero
  std::array<double, 6> a_;
};

class QuadraticPolynomial {
 public:
  QuadraticPolynomial() = default;
  QuadraticPolynomial(const double x0, const double v0, const double a0,
                      const double vt, const double at, const double t);

  double x(const double t);
  double dx(const double t);
  double ddx(const double t);

  Eigen::Vector3d get_derivertives(const double t);

  std::vector<double> x_vec(const std::vector<double>& t);
  // std::vector<double> dx_vec(const std::vector<double>& t);
  // std::vector<double> ddx_vec(const std::vector<double>& t);

 private:
  std::array<double, 5> a_;
};