#include "frenet/frenet_sampler.h"

#include <fstream>
#include <iostream>

#include "frenet/box2d.h"
#include "frenet/frenet_transform.h"
#include "frenet/polynomials.h"

constexpr double EGO_LENGTH = 4.87;
constexpr double EGO_WIDTH = 1.85;
static std::vector<double> CHECK_STEP{5, 8, 11, 14, 17, 20, 23, 26, 29};

FrenetSampler::FrenetSampler(std::string conf) {
  std::ifstream ifs(conf);
  json j = json::parse(ifs);

  conf_.from_json(j, conf_);

  double delta_lat = conf_.width / conf_.num_lat_sample;
  for (int i = 0; i < conf_.num_lat_sample * 2 + 1; ++i) {
    lat_samples_.emplace_back(-conf_.width + i * delta_lat);
  }
}

FrenetSampler::FrenetSampler(const double max_vel, const double width,
                             const int num_lon_sample, const int num_lat_sample,
                             const double max_deceleration,
                             const double max_acceleration,
                             const double max_curvature) {
  conf_.max_vel = max_vel;
  conf_.width = width;
  conf_.num_lon_sample = num_lon_sample;
  conf_.num_lat_sample = num_lat_sample;
  conf_.max_deceleration = max_deceleration;
  conf_.max_acceleration = max_acceleration;
  conf_.max_curvature = max_curvature;

  double delta_lat = conf_.width / conf_.num_lat_sample;
  for (int i = 0; i < conf_.num_lat_sample * 2 + 1; ++i) {
    lat_samples_.emplace_back(-conf_.width + i * delta_lat);
  }
}

void FrenetSampler::SetInitState(const double x, const double y,
                                 const double yaw, const double vel,
                                 const double acc) {
  init_state_.pos = Eigen::Vector2d(x, y);
  init_state_.yaw = yaw;
  init_state_.vel = vel;
  init_state_.acc = acc;
}

void FrenetSampler::SetFrenetInitState(const Eigen::VectorXd& frenet_state) {
  init_state_.s = frenet_state.block(0, 0, 3, 1);
  init_state_.d = frenet_state.block(3, 0, 3, 1);
}

void FrenetSampler::SetReferenceLine(const std::vector<double>& ref_x,
                                     const std::vector<double>& ref_y,
                                     const std::vector<double>& ref_s) {
  ref_line_ = ReferenceLine(ref_x, ref_y, ref_s);

  init_state_.s[0] = ref_line_.GetArcLength(init_state_.pos, 1e-5, 40);
  FrenetTransform::CalFrenetState(ref_line_.GetReferencePoint(init_state_.s[0]),
                                  init_state_);
}

void FrenetSampler::Sample(const double T) {
  if (!traj_set_.empty()) {
    traj_set_.clear();
  }

  // ref_line_ = ReferenceLine(ref_x, ref_y, ref_s);
  // init_state_.s[0] = ref_line_.GetArcLength(init_state_.pos, 1e-5, 40);
  // // printf("init pos (%f, %f), init_s : %f\n", init_state_.pos.x(),
  // //        init_state_.pos.y(), init_state_.s[0]);

  // FrenetTransform::CalFrenetState(ref_line_.GetReferencePoint(init_state_.s[0]),
  //                                 init_state_);

  const double min_vel =
      std::max(0.0, init_state_.s[1] + conf_.max_deceleration * T);
  const double max_vel =
      std::min(conf_.max_vel, init_state_.s[1] + conf_.max_acceleration * T);
  const double delta_vel = (max_vel - min_vel) / conf_.num_lon_sample;
  // printf("min_vel: %f, max_vel: %f, delta_vel: %f\n", min_vel, max_vel,
  //        delta_vel);
  // printf("init_s: (%f,%f,%f), init_d: (%f, %f, %f)\n", init_state_.s[0],
  //        init_state_.s[1], init_state_.s[2], init_state_.d[0],
  //        init_state_.d[1], init_state_.d[2]);

  for (int i = 0; i < conf_.num_lon_sample; ++i) {
    QuadraticPolynomial lon(init_state_.s[0], init_state_.s[1],
                            init_state_.s[2], min_vel + delta_vel * i, 0.0, T);
    double s_at_T = lon.x(T);

    for (const auto target_d : lat_samples_) {
      if (std::fabs(s_at_T - init_state_.s[0]) < 1e-3) {
        // only keep one stop traj
        QuinticPolynomial lat(init_state_.d[0]);
        traj_set_.emplace_back(FrenetTrajectory<QuadraticPolynomial>(lon, lat));
        break;
      } else {
        QuinticPolynomial lat(init_state_.d[0], init_state_.d[1],
                              init_state_.d[2], target_d, 0.0, 0.0, s_at_T,
                              init_state_.s[0]);
        traj_set_.emplace_back(FrenetTrajectory<QuadraticPolynomial>(lon, lat));
      }
    }
  }
}

void FrenetSampler::SampleByTime(const double T) {
  if (!traj_set_.empty()) {
    traj_set_.clear();
  }

  const double min_vel =
      std::max(0.0, init_state_.s[1] + conf_.max_deceleration * T);
  const double max_vel =
      std::min(conf_.max_vel, init_state_.s[1] + conf_.max_acceleration * T);
  const double delta_vel = (max_vel - min_vel) / conf_.num_lon_sample;

  for (int i = 0; i < conf_.num_lon_sample; ++i) {
    QuadraticPolynomial lon(init_state_.s[0], init_state_.s[1],
                            init_state_.s[2], min_vel + delta_vel * i, 0.0, T);
    double s_at_T = lon.x(T);

    QuinticPolynomial lat;
    if (std::fabs(s_at_T - init_state_.s[0]) < 1e-3) {
      lat = QuinticPolynomial(init_state_.d[0]);
      traj_set_.emplace_back(FrenetTrajectory<QuadraticPolynomial>(
          lon, lat, FrenetTrajectoryType::kArcLength));
      continue;  // * only keep one stopping trajectory
    }

    for (const double lateral : lat_samples_) {
      if (lon.dx(T) < 2.0) {
        // * avoid singularity at s_dot = 0, this happens when the vehicle is
        // * decelerating to zero
        lat = QuinticPolynomial(init_state_.d[0], init_state_.d[1],
                                init_state_.d[2], lateral, 0.0, 0.0, s_at_T,
                                init_state_.s[0]);
        traj_set_.emplace_back(FrenetTrajectory<QuadraticPolynomial>(
            lon, lat, FrenetTrajectoryType::kArcLength));
      } else {
        lat = QuinticPolynomial(init_state_.d_t[0], init_state_.d_t[1],
                                init_state_.d_t[2], lateral, 0.0, 0.0, T);
        traj_set_.emplace_back(FrenetTrajectory<QuadraticPolynomial>(
            lon, lat, FrenetTrajectoryType::kTime));
      }
    }
  }
}

void FrenetSampler::SampleByTimeWithInitialSequence(const double T,
                                                    Eigen::MatrixXd init_seq) {
  assert(init_seq.rows() > 0);

  int seq_len = init_seq.rows();
}

void FrenetSampler::SampleGlobalState(std::vector<double> ts) {
  for (auto& traj : traj_set_) {
    traj.CalGlobalState(ref_line_, ts);
  }
}

int FrenetSampler::GetMaxSamplesNumber() const {
  return conf_.num_lon_sample * lat_samples_.size();
}

Eigen::MatrixXd FrenetSampler::GenerateTrajectory(
    const Eigen::VectorXd target_state, const double T,
    const std::vector<double>& ts) {
  QuinticPolynomial lon(init_state_.s[0], init_state_.s[1], init_state_.s[2],
                        std::max(init_state_.s[0], target_state[0]),
                        target_state[1], target_state[2], T, 0);

  QuinticPolynomial lat;
  if (fabs(lon.x(T) - init_state_.s[0]) <= 1e-3) {
    lat = QuinticPolynomial(target_state[3]);  // avoid singularity
  } else {
    lat = QuinticPolynomial(init_state_.d[0], init_state_.d[1],
                            init_state_.d[2], target_state[3], target_state[4],
                            target_state[5], lon.x(T), init_state_.s[0]);
  }

  auto traj = FrenetTrajectory<QuinticPolynomial>(lon, lat);
  traj.CalGlobalState(ref_line_, ts);

  return traj.ToNumpy();
}

Eigen::VectorXd FrenetSampler::GetFrenetState(const double x, const double y,
                                              const double yaw,
                                              const double vel,
                                              const double acc) const {
  State state;
  state.pos = Eigen::Vector2d(x, y);
  state.yaw = yaw;
  state.vel = vel;
  state.acc = acc;

  double arc_length = ref_line_.GetArcLength(state.pos, 1e-5, 80);
  arc_length = std::max(arc_length, init_state_.s[0]);

  FrenetTransform::CalFrenetState(ref_line_.GetReferencePoint(arc_length),
                                  state);

  Eigen::VectorXd frenet_state(6);
  frenet_state(0) = state.s(0);
  frenet_state(1) = state.s(1);
  frenet_state(2) = state.s(2);
  frenet_state(3) = state.d(0);
  frenet_state(4) = state.d(1);
  frenet_state(5) = state.d(2);

  return frenet_state;
}

Eigen::VectorXd FrenetSampler::GetGlobalState(
    const Eigen::VectorXd& frenet_state) const {
  State state;
  state.s = Eigen::Vector3d(frenet_state(0), frenet_state(1), frenet_state(2));
  state.d = Eigen::Vector3d(frenet_state(3), frenet_state(4), frenet_state(5));

  FrenetTransform::CalGlobalState(ref_line_.GetReferencePoint(frenet_state(0)),
                                  state);

  Eigen::VectorXd global_state(5);
  global_state(0) = state.pos.x();
  global_state(1) = state.pos.y();
  global_state(2) = state.yaw;
  global_state(3) = state.vel;
  global_state(4) = state.acc;

  return global_state;
}

std::vector<Eigen::MatrixXd> FrenetSampler::GetNumpyTrajectories(
    const bool exclude_invalid) {
  std::vector<Eigen::MatrixXd> np_trajs;

  for (auto& traj : traj_set_) {
    if (!exclude_invalid || traj.IsValid(conf_.max_curvature)) {
      np_trajs.emplace_back(traj.ToNumpy());
    }
  }

  if (np_trajs.size() == 0) {
    np_trajs.emplace_back(traj_set_.front().ToNumpy());
  }

  return np_trajs;
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<double>>
FrenetSampler::GetNumpyTrajectoriesWithLabel(const bool exclude_invalid) {
  std::vector<Eigen::MatrixXd> np_trajs;

  auto labels = GenerateSoftLabels();

  std::vector<double> valid_sample_labels;
  valid_sample_labels.reserve(labels.size());

  for (int i = 0; i < traj_set_.size(); ++i) {
    if (!exclude_invalid || traj_set_[i].IsValid(conf_.max_curvature)) {
      np_trajs.emplace_back(traj_set_[i].ToNumpy());
      valid_sample_labels.emplace_back(labels[i]);
    }
  }

  if (np_trajs.size() == 0) {
    np_trajs.emplace_back(traj_set_.front().ToNumpy());
    valid_sample_labels.emplace_back(labels.front());
  }

  return std::make_pair(np_trajs, valid_sample_labels);
}

std::vector<std::vector<State>> FrenetSampler::GetTrajectories(
    const std::vector<double>& ts) {
  std::vector<std::vector<State>> global_trajs;

  for (auto& traj : traj_set_) {
    traj.CalGlobalState(ref_line_, ts);
    if (traj.IsValid(conf_.max_curvature)) {
      global_trajs.emplace_back(traj.GetTraj());
    }
  }

  if (global_trajs.size() == 0) {
    global_trajs.emplace_back(traj_set_.front().GetTraj());
  }

  return global_trajs;
}

Eigen::VectorXd FrenetSampler::get_init_state() const {
  Eigen::VectorXd state(6);
  state << init_state_.s, init_state_.d;
  return state;
}

Eigen::MatrixXd FrenetSampler::get_reference_line() const {
  return ref_line_.GetPoints();
}

Eigen::VectorXd FrenetSampler::get_meta() const {
  auto ref_line_meta = ref_line_.get_meta();

  Eigen::VectorXd sampler_meta(6 + ref_line_meta.size());

  sampler_meta.block(0, 0, 3, 1) = init_state_.s;
  sampler_meta.block(3, 0, 3, 1) = init_state_.d;
  sampler_meta.block(6, 0, ref_line_meta.size(), 1) = ref_line_meta;

  return sampler_meta;
}

void FrenetSampler::load_meta(Eigen::VectorXd meta) {
  init_state_.s = meta.block(0, 0, 3, 1);
  init_state_.d = meta.block(3, 0, 3, 1);

  ref_line_.load_meta(meta.block(6, 0, meta.size() - 6, 1));
}

// void FrenetSampler::SetAgents(std::vector<Eigen::MatrixXd>& agents,
//                               const int num_agents,
//                               const double safe_distance) {
//   agents_boxes_.clear();
//   agents_boxes_.resize(CHECK_STEP.size());
//   const int mask_idx = agents[0].cols() - 1;

//   for (int idx = 0; idx < CHECK_STEP.size(); ++idx) {
//     const double t = CHECK_STEP[idx];
//     for (int i = 0; i < num_agents; ++i) {
//       Eigen::Vector2d pos(agents[i](t,0), agents[i](t,1));
//       if (agents[i](t, mask_idx) > 0) {
//         // * notice: agents (x, y, yaw, length, width, ..., mask)
//         agents_boxes_[idx].emplace_back(Box2D(
//             pos, agents[i](t, 2),
//             agents[i](t, 3) + safe_distance, agents[i](t, 4) + safe_distance));
//       }
//     }
//   }
// }

std::vector<double> FrenetSampler::GenerateSoftLabels() {
  std::vector<double> labels;
  labels.resize(traj_set_.size());

  if (agents_boxes_.size() == 0) return labels;

  for (int traj_idx = 0; traj_idx < traj_set_.size(); ++traj_idx) {
    const std::vector<State>& ego_traj = traj_set_[traj_idx].GetTraj();
    for (int i = 0; i < CHECK_STEP.size(); ++i) {
      bool collide = false;
      auto ego_box = Box2D(ego_traj[CHECK_STEP[i]].pos,
                           ego_traj[CHECK_STEP[i]].yaw, EGO_WIDTH, EGO_WIDTH);

      for (const auto& agent_box : agents_boxes_[i]) {
        if (ego_box.HasOverlapWith(agent_box)) {
          labels[traj_idx] = -1;
          collide = true;
          break;
        }
      }

      if (collide) break;
    }
  }

  return labels;
}