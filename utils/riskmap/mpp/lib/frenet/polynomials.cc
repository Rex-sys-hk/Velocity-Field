#include "frenet/polynomials.h"

#include <Eigen/Dense>
#include <iostream>

static std::vector<double> GetPowers(const double t, const int order) {
  std::vector<double> p(order, t);
  for (int i = 1; i < order; ++i) {
    p[i] = p[i - 1] * t;
  }
  return p;
}

static void GetPowers(const double t, const int order,
                      std::vector<double>& t_powers) {
  t_powers[0] = t;
  for (int i = 1; i < order; ++i) {
    t_powers[i] = t_powers[i - 1] * t;
  }
}

QuinticPolynomial::QuinticPolynomial(const double x0) {
  a_[0] = x0;
  for (int i = 1; i < 6; ++i) {
    a_[i] = 0.0;
  }
}

QuinticPolynomial::QuinticPolynomial(const double x0, const double v0,
                                     const double a0, const double xt,
                                     const double vt, const double at,
                                     const double t, const double offset)
    : offset_(offset) {
  a_[0] = x0;
  a_[1] = v0;
  a_[2] = a0 / 2;

  std::vector<double> t_pows = GetPowers(t - offset_, 5);
  Eigen::Matrix3d A;
  A(0, 0) = t_pows[2];
  A(0, 1) = t_pows[3];
  A(0, 2) = t_pows[4];
  A(1, 0) = 3 * t_pows[1];
  A(1, 1) = 4 * t_pows[2];
  A(1, 2) = 5 * t_pows[3];
  A(2, 0) = 6 * t_pows[0];
  A(2, 1) = 12 * t_pows[1];
  A(2, 2) = 20 * t_pows[2];

  Eigen::Vector3d b;
  b(0) = xt - a_[0] - a_[1] * t_pows[0] - a_[2] * t_pows[1];
  b(1) = vt - a_[1] - 2 * a_[2] * t_pows[0];
  b(2) = at - 2 * a_[2];

  Eigen::Vector3d x = A.householderQr().solve(b);
  a_[3] = x[0];
  a_[4] = x[1];
  a_[5] = x[2];
}

double QuinticPolynomial::x(const double t) {
  auto t_pows = GetPowers(t - offset_, 5);
  return a_[0] + a_[1] * t_pows[0] + a_[2] * t_pows[1] + a_[3] * t_pows[2] +
         a_[4] * t_pows[3] + a_[5] * t_pows[4];
}

double QuinticPolynomial::dx(const double t) {
  auto t_pows = GetPowers(t - offset_, 4);
  return a_[1] + 2 * a_[2] * t_pows[0] + 3 * a_[3] * t_pows[1] +
         4 * a_[4] * t_pows[2] + 5 * a_[5] * t_pows[3];
}

double QuinticPolynomial::ddx(const double t) {
  auto t_pows = GetPowers(t - offset_, 3);
  return 2 * a_[2] + 6 * a_[3] * t_pows[0] + 12 * a_[4] * t_pows[1] +
         20 * a_[5] * t_pows[2];
}

Eigen::Vector3d QuinticPolynomial::get_derivertives(const double t) {
  Eigen::Vector3d derivertives;

  auto t_pows = GetPowers(t - offset_, 5);
  derivertives[0] = a_[0] + a_[1] * t_pows[0] + a_[2] * t_pows[1] +
                    a_[3] * t_pows[2] + a_[4] * t_pows[3] + a_[5] * t_pows[4];
  derivertives[1] = a_[1] + 2 * a_[2] * t_pows[0] + 3 * a_[3] * t_pows[1] +
                    4 * a_[4] * t_pows[2] + 5 * a_[5] * t_pows[3];
  derivertives[2] = 2 * a_[2] + 6 * a_[3] * t_pows[0] + 12 * a_[4] * t_pows[1] +
                    20 * a_[5] * t_pows[2];

  return derivertives;
}

std::vector<double> QuinticPolynomial::x_vec(const std::vector<double>& ts) {
  int dim = ts.size();

  std::vector<double> xs(dim, 0.0);
  std::vector<double> t_pows(5, 0);

  for (int i = 0; i < dim; ++i) {
    GetPowers(ts[i] - offset_, 5, t_pows);
    xs[i] = a_[0] + a_[1] * t_pows[0] + a_[2] * t_pows[1] + a_[3] * t_pows[2] +
            a_[4] * t_pows[3] + a_[5] * t_pows[4];
  }

  return xs;
}

QuadraticPolynomial::QuadraticPolynomial(const double x0, const double v0,
                                         const double a0, const double vt,
                                         const double at, const double t) {
  a_[0] = x0;
  a_[1] = v0;
  a_[2] = a0 / 2;

  std::vector<double> t_pows = GetPowers(t, 3);
  Eigen::Matrix2d A;
  A(0, 0) = 3 * t_pows[1];
  A(0, 1) = 4 * t_pows[2];
  A(1, 0) = 6 * t_pows[0];
  A(1, 1) = 12 * t_pows[1];

  Eigen::Vector2d b;
  b(0) = vt - a_[1] - 2 * a_[2] * t_pows[0];
  b(1) = at - 2 * a_[2];

  Eigen::Vector2d x = A.householderQr().solve(b);
  a_[3] = x[0];
  a_[4] = x[1];
}

double QuadraticPolynomial::x(const double t) {
  auto t_pows = GetPowers(t, 4);
  return a_[0] + a_[1] * t_pows[0] + a_[2] * t_pows[1] + a_[3] * t_pows[2] +
         a_[4] * t_pows[3];
}

double QuadraticPolynomial::dx(const double t) {
  auto t_pows = GetPowers(t, 3);
  return a_[1] + 2 * a_[2] * t_pows[0] + 3 * a_[3] * t_pows[1] +
         4 * a_[4] * t_pows[2];
}

double QuadraticPolynomial::ddx(const double t) {
  auto t_pows = GetPowers(t, 2);
  return 2 * a_[2] + 6 * a_[3] * t_pows[0] + 12 * a_[4] * t_pows[1];
}

Eigen::Vector3d QuadraticPolynomial::get_derivertives(const double t) {
  Eigen::Vector3d derivertives;

  auto t_pows = GetPowers(t, 4);
  derivertives[0] = a_[0] + a_[1] * t_pows[0] + a_[2] * t_pows[1] +
                    a_[3] * t_pows[2] + a_[4] * t_pows[3];
  derivertives[1] = a_[1] + 2 * a_[2] * t_pows[0] + 3 * a_[3] * t_pows[1] +
                    4 * a_[4] * t_pows[2];
  derivertives[2] = 2 * a_[2] + 6 * a_[3] * t_pows[0] + 12 * a_[4] * t_pows[1];

  return derivertives;
}

std::vector<double> QuadraticPolynomial::x_vec(const std::vector<double>& ts) {
  int dim = ts.size();

  std::vector<double> xs(dim, 0.0);
  std::vector<double> t_pows(4, 0);

  for (int i = 0; i < dim; ++i) {
    GetPowers(ts[i], 4, t_pows);
    xs[i] = a_[0] + a_[1] * t_pows[0] + a_[2] * t_pows[1] + a_[3] * t_pows[2] +
            a_[4] * t_pows[3];
  }

  return xs;
}