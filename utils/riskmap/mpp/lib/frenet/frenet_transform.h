#pragma once

#include <Eigen/Dense>

struct State {
  //  frenet
  Eigen::Vector3d s;
  Eigen::Vector3d d;  // d_s by default
  Eigen::Vector3d d_t;

  // global
  Eigen::Vector2d pos;
  double yaw = 0.0;
  double vel = 0.0;
  double acc = 0.0;
  double kappa = 0.0;

  std::string DebugString() const {
    std::ostringstream os;
    os << "[FrenetReferencePoint]:  "
       << "s: (" << s(0) << ", " << s(1) << ", " << s(2) << ")\t"
       << "d: (" << d(0) << ", " << d(1) << ", " << d(2) << ")\t"
       << "point: (" << pos.x() << ", " << pos.y() << ")\t";
    return os.str();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ReferencePoint {
  Eigen::Vector2d point;
  double s = 0.0;
  double theta = 0.0;
  double kappa = 0.0;
  double dkappa = 0.0;

  std::string DebugString() const {
    std::ostringstream os;
    os << "[FrenetReferencePoint]:  "
       << "point: (" << point.x() << ", " << point.y() << ")\t"
       << "s: " << s << "\t"
       << "theta: " << theta << "\t"
       << "kappa: " << kappa << "\t"
       << "dkappa: " << dkappa;
    return os.str();
  }
};

class FrenetTransform {
 public:
  static void CalFrenetState(const ReferencePoint& ref, State& state);

  static void CalGlobalState(const ReferencePoint& ref, State& state);

  static void CalGlobalStateTimeParameterized(const ReferencePoint& ref,
                                              State& state);
};
