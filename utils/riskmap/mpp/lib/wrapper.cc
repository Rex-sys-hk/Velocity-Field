#include "frenet/frenet_sampler.h"
#include "frenet/polynomials.h"
#include "frenet/reference_line.h"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(frenet, m) {
  py::class_<QuinticPolynomial> quintic_polynomial(m, "QuinticPolynomial");
  quintic_polynomial
      .def(py::init<const double, const double, const double, const double,
                    const double, const double, const double, const double>(),
           py::arg("x0"), py::arg("v0"), py::arg("a0"), py::arg("xt"),
           py::arg("vt"), py::arg("at"), py::arg("t"), py::arg("offset"))
      .def("x", &QuinticPolynomial::x, py::arg("t"))
      .def("dx", &QuinticPolynomial::dx, py::arg("t"))
      .def("ddx", &QuinticPolynomial::ddx, py::arg("t"))
      .def("x_vec", &QuinticPolynomial::x_vec, py::arg("ts"));

  py::class_<QuadraticPolynomial> quadratic_polynomial(m,
                                                       "QuadraticPolynomial");
  quadratic_polynomial
      .def(py::init<const double, const double, const double, const double,
                    const double, const double>(),
           py::arg("x0"), py::arg("v0"), py::arg("a0"), py::arg("vt"),
           py::arg("at"), py::arg("t"))
      .def("x", &QuadraticPolynomial::x, py::arg("t"))
      .def("dx", &QuadraticPolynomial::dx, py::arg("t"))
      .def("ddx", &QuadraticPolynomial::ddx, py::arg("t"))
      .def("x_vec", &QuadraticPolynomial::x_vec, py::arg("ts"));

  py::class_<ReferenceLine> reference_line(m, "ReferenceLine");
  reference_line
      .def(py::init<const std::vector<double>&, const std::vector<double>&,
                    const std::vector<double>&>(),
           py::arg("x"), py::arg("y"), py::arg("s"))
      .def("get_max_curvature", &ReferenceLine::GetMaxCurvature, py::arg("s"));

  py::class_<FrenetSampler> frenet_sampler(m, "FrenetSampler");
  frenet_sampler.def(py::init())
      .def(py::init<std::string>(), py::arg("conf"))
      .def(py::init<double, double, double, double, double, double, double>(),
           py::arg("max_vel"), py::arg("width"), py::arg("num_lon_samples"),
           py::arg("num_lat_samples"), py::arg("max_deceleration"),
           py::arg("max_acceleration"), py::arg("max_curvature"))
      .def("set_reference_line", &FrenetSampler::SetReferenceLine,
           py::arg("ref_x"), py::arg("ref_y"), py::arg("ref_s"))
      .def("sample", &FrenetSampler::Sample, py::arg("T"))
      .def("sample_by_time", &FrenetSampler::SampleByTime, py::arg("T"))
      .def("sample_by_time_with_init_sequence",
           &FrenetSampler::SampleByTimeWithInitialSequence, py::arg("T"),
           py::arg("init_sequence"))
      .def("set_init_state", &FrenetSampler::SetInitState, py::arg("x"),
           py::arg("y"), py::arg("yaw"), py::arg("vel"), py::arg("acc"))
      .def("set_frenet_init_state", &FrenetSampler::SetFrenetInitState,
           py::arg("frenet_state"))
      .def("get_frenet_state", &FrenetSampler::GetFrenetState, py::arg("x"),
           py::arg("y"), py::arg("yaw"), py::arg("vel"), py::arg("acc"))
      .def("get_global_state", &FrenetSampler::GetGlobalState,
           py::arg("frenet_state"))
      .def("get_max_num_of_samples", &FrenetSampler::GetMaxSamplesNumber)
      .def("generate_trajectory", &FrenetSampler::GenerateTrajectory,
           py::arg("target_state"), py::arg("T"), py::arg("ts"))
      .def("get_init_state", &FrenetSampler::get_init_state)
      .def("get_ref_line", &FrenetSampler::get_reference_line)
      .def("get_meta", &FrenetSampler::get_meta)
      .def("load_meta", &FrenetSampler::load_meta)
      .def("sample_global_state", &FrenetSampler::SampleGlobalState,
           py::arg("ts"))
      .def("get_trajectories", &FrenetSampler::GetNumpyTrajectories,
           py::arg("exclude_invalid") = true)
      .def("get_trajectory_feature_size", &FrenetSampler::get_traj_feature_size)
     //  .def("set_agents", &FrenetSampler::SetAgents, py::arg("agents"),
     //       py::arg("agents_num"), py::arg("safety_distance"))
      .def("get_trajectories_with_label",
           &FrenetSampler::GetNumpyTrajectoriesWithLabel,
           py::arg("exclude_invalid") = true);
}