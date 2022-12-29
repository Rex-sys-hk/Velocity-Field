#include "frenet/frenet_transform.h"

#include "utils/math.h"

void FrenetTransform::CalFrenetState(const ReferencePoint& ref, State& state) {
  Eigen::Vector2d normal(-std::sin(ref.theta), std::cos(ref.theta));
  state.d[0] = (state.pos - ref.point).dot(normal);

  const double one_minus_kappa_rd = 1 - ref.kappa * state.d[0];

  const double delta_theta = NormalizeAngle(state.yaw - ref.theta);
  const double tan_delta_theta = std::tan(delta_theta);
  const double cos_delta_theta = std::cos(delta_theta);
  state.d[1] = one_minus_kappa_rd * tan_delta_theta;
  const double dkappa_d_kappa_dd =
      ref.dkappa * state.d[0] + ref.kappa * state.d[1];
  state.d[2] =
      -dkappa_d_kappa_dd * tan_delta_theta +
      one_minus_kappa_rd / (cos_delta_theta * cos_delta_theta) *
          (ref.kappa * one_minus_kappa_rd / cos_delta_theta - ref.kappa);

  state.s[0] = ref.s;
  state.s[1] = state.vel * cos_delta_theta / one_minus_kappa_rd;

  const double d_delta_theta =
      1.0 / (1.0 + tan_delta_theta * tan_delta_theta) *
      (state.d[2] * one_minus_kappa_rd + state.d[1] * state.d[1] * ref.kappa) /
      (one_minus_kappa_rd * one_minus_kappa_rd);

  state.s[2] = (state.acc * cos_delta_theta -
                state.s[1] * state.s[1] *
                    (state.d[1] * d_delta_theta - dkappa_d_kappa_dd)) /
               one_minus_kappa_rd;

  state.d_t[0] = state.d[0];
  state.d_t[1] = state.s[1] * state.d[1];
  state.d_t[2] = state.d[2] * state.s[1] * state.s[1] + state.d[1] * state.s[2];
}

void FrenetTransform::CalGlobalState(const ReferencePoint& ref, State& state) {
  Eigen::Vector2d normal(-std::sin(ref.theta), std::cos(ref.theta));

  const double one_minus_kappa_rd = 1 - ref.kappa * state.d[0];

  const double tan_delta_theta = state.d[1] / one_minus_kappa_rd;
  const double delta_theta = std::atan2(state.d[1], one_minus_kappa_rd);
  const double cos_delta_theta = std::cos(delta_theta);

  state.pos = normal * state.d[0] + ref.point;
  state.vel = state.s[1] * one_minus_kappa_rd / cos_delta_theta;
  state.yaw = NormalizeAngle(delta_theta + ref.theta);

  const double dkappa_d_kappa_dd =
      ref.dkappa * state.d[0] + ref.kappa * state.d[1];
  state.kappa = (((state.d[2] + dkappa_d_kappa_dd * tan_delta_theta) *
                  (cos_delta_theta * cos_delta_theta) / one_minus_kappa_rd) +
                 ref.kappa) *
                cos_delta_theta / one_minus_kappa_rd;

  const double d_delta_theta =
      1.0 / (1.0 + tan_delta_theta * tan_delta_theta) *
      (state.d[2] * one_minus_kappa_rd + state.d[1] * state.d[1] * ref.kappa) /
      (one_minus_kappa_rd * one_minus_kappa_rd);

  state.acc = state.s[2] * one_minus_kappa_rd / cos_delta_theta +
              state.s[1] * state.s[1] / cos_delta_theta *
                  (state.d[1] * d_delta_theta - dkappa_d_kappa_dd);
}

void FrenetTransform::CalGlobalStateTimeParameterized(const ReferencePoint& ref,
                                                      State& state) {
  state.d[0] = state.d_t[0];
  state.d[1] = state.d_t[1] / state.s[1];
  state.d[2] =
      (state.d_t[2] - state.d_t[1] * state.s[2]) / state.s[1] / state.s[1];

  CalGlobalState(ref, state);
}
