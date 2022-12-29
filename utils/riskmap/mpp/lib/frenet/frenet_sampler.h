#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "frenet/box2d.h"
#include "frenet/frenet_trajectory.hpp"
#include "frenet/frenet_transform.h"
#include "frenet/reference_line.h"
#include "nlohmann/json.h"

using nlohmann::json;

struct SamplerConfig {
  double max_vel = 30;
  double width = 3;
  int num_lon_sample = 50;
  int num_lat_sample = 5;

  double max_deceleration = -6;
  double max_acceleration = 6;

  double max_curvature = 0.5;

  void from_json(const json& j, SamplerConfig& p) {
    j.at("max_vel").get_to(p.max_vel);
    j.at("width").get_to(p.width);
    j.at("max_deceleration").get_to(p.max_deceleration);
    j.at("max_acceleration").get_to(p.max_acceleration);
    j.at("num_lon_sample").get_to(p.num_lon_sample);
    j.at("num_lat_sample").get_to(p.num_lat_sample);
    j.at("max_curvature").get_to(p.max_curvature);
  }
};

class FrenetSampler {
 public:
  FrenetSampler() = default;
  FrenetSampler(std::string conf);
  FrenetSampler(const double max_vel, const double width,
                const int num_lon_sample, const int num_lat_sample,
                const double max_deceleration, const double max_acceleration,
                const double max_curvature);

  ~FrenetSampler() = default;

  void SetInitState(const double x, const double y, const double yaw,
                    const double v, const double a);

  void SetFrenetInitState(const Eigen::VectorXd& frenet_state);

  void SetReferenceLine(const std::vector<double>& ref_x,
                        const std::vector<double>& ref_y,
                        const std::vector<double>& ref_s);

  void Sample(const double T);

  void SampleByTime(const double T);

  void SampleByTimeWithInitialSequence(const double T,
                                       Eigen::MatrixXd init_seq);

  void SampleGlobalState(std::vector<double> ts);

  int GetMaxSamplesNumber() const;

  Eigen::VectorXd GetFrenetState(const double x, const double y,
                                 const double yaw, const double vel,
                                 const double acc) const;

  Eigen::VectorXd GetGlobalState(const Eigen::VectorXd& frenet_state) const;

  std::vector<std::vector<State>> GetTrajectories(
      const std::vector<double>& ts);

  std::vector<Eigen::MatrixXd> GetNumpyTrajectories(
      const bool exclude_invalid = true);

  std::pair<std::vector<Eigen::MatrixXd>, std::vector<double>>
  GetNumpyTrajectoriesWithLabel(const bool exclude_invalid = true);

  void SetAgents(std::vector<Eigen::MatrixXd>& agents, const int num_agents,
                 const double safe_distance);

  std::vector<double> GenerateSoftLabels();

  Eigen::MatrixXd GenerateTrajectory(const Eigen::VectorXd target_state,
                                     const double T,
                                     const std::vector<double>& ts);

  Eigen::VectorXd get_init_state() const;

  Eigen::MatrixXd get_reference_line() const;

  Eigen::VectorXd get_meta() const;

  int get_traj_feature_size() const {
    return FrenetTrajectory<QuadraticPolynomial>::feature_size();
  }

  void load_meta(Eigen::VectorXd meta);

 private:
  State init_state_;
  SamplerConfig conf_;

  ReferenceLine ref_line_;

  std::vector<double> lat_samples_;
  std::vector<FrenetTrajectory<QuadraticPolynomial>> traj_set_;

  std::vector<std::vector<Box2D>> agents_boxes_;
};
