#pragma once

#include <Eigen/Dense>
#include <vector>

#include "frenet/frenet_transform.h"
#include "frenet/polynomials.h"
#include "frenet/reference_line.h"

constexpr int kTrajFeatureDim = 12;

struct FrenetPoint {
  double t = 0.0;
  Eigen::Vector3d s;
  Eigen::Vector3d d;
};

enum class FrenetTrajectoryType { kArcLength = 0, kTime = 1 };

template <typename T>
class FrenetTrajectory {
 public:
  FrenetTrajectory() = default;
  FrenetTrajectory(
      const T& lon, const QuinticPolynomial lat,
      const FrenetTrajectoryType type = FrenetTrajectoryType::kArcLength)
      : lon_(lon), lat_(lat), type_(type) {}

  void CalGlobalState(const ReferenceLine& ref_line,
                      const std::vector<double>& ts);

  bool IsValid(const double max_curvature) const;

  inline const std::vector<State>& GetTraj() { return traj_; }

  Eigen::MatrixXd ToNumpy();

  static int feature_size() { return kTrajFeatureDim; }

 private:
  FrenetTrajectoryType type_ = FrenetTrajectoryType::kArcLength;

  T lon_;
  QuinticPolynomial lat_;

  std::vector<State> traj_;
};

template <typename T>
void FrenetTrajectory<T>::CalGlobalState(const ReferenceLine& ref_line,
                                         const std::vector<double>& ts) {
  if (!traj_.empty()) {
    traj_.clear();
  }

  State state;

  if (type_ == FrenetTrajectoryType::kArcLength) {
    for (const double t : ts) {
      state.s = lon_.get_derivertives(t);
      state.d = lat_.get_derivertives(state.s[0]);
      auto ref_point = ref_line.GetReferencePoint(state.s[0]);
      FrenetTransform::CalGlobalState(ref_point, state);
      traj_.emplace_back(state);
    }
  } else if (type_ == FrenetTrajectoryType::kTime) {
    for (const double t : ts) {
      state.s = lon_.get_derivertives(t);
      state.d_t = lat_.get_derivertives(t);

      state.d[0] = state.d_t[0];
      state.d[1] = state.d_t[1] / state.s[1];
      state.d[2] = (state.d_t[2] - state.d_t[1] * state.s[2]) /
                   (state.s[1] * state.s[1]);

      auto ref_point = ref_line.GetReferencePoint(state.s[0]);
      FrenetTransform::CalGlobalState(ref_point, state);
      traj_.emplace_back(state);
    }
  }
}

template <typename T>
Eigen::MatrixXd FrenetTrajectory<T>::ToNumpy() {
  const double init_s = traj_[0].s[0];
  Eigen::MatrixXd numpy(traj_.size(), kTrajFeatureDim);

  for (int i = 0; i < traj_.size(); ++i) {
    numpy(i, 0) = traj_[i].pos[0];
    numpy(i, 1) = traj_[i].pos[1];
    numpy(i, 2) = std::cos(traj_[i].yaw);
    numpy(i, 3) = std::sin(traj_[i].yaw);
    numpy(i, 4) = traj_[i].s[0] - init_s;
    numpy(i, 5) = traj_[i].s[1];
    numpy(i, 6) = traj_[i].s[2];
    numpy(i, 7) = traj_[i].d[0];
    numpy(i, 8) = traj_[i].d[1];
    numpy(i, 9) = traj_[i].d[2];
    numpy(i, 10) = traj_[i].vel;
    numpy(i, 11) = traj_[i].acc;
  }
  return numpy;
}

template <typename T>
bool FrenetTrajectory<T>::IsValid(const double max_curvature) const {
  assert(traj_.size() > 0);

  for (const auto& state : traj_) {
    if (fabs(state.kappa) > max_curvature) {
      return false;
    }
  }

  return true;
}