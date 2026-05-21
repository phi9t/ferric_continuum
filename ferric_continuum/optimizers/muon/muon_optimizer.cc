#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <string>

namespace {

namespace py = pybind11;

bool IsCContiguousDouble(const py::buffer_info& info) {
  if (info.itemsize != static_cast<ssize_t>(sizeof(double))) {
    return false;
  }
  if (info.ndim == 0) {
    return false;
  }
  ssize_t expected_stride = static_cast<ssize_t>(sizeof(double));
  for (ssize_t dim = info.ndim - 1; dim >= 0; --dim) {
    if (info.strides[dim] != expected_stride) {
      return false;
    }
    expected_stride *= info.shape[dim];
  }
  return true;
}

void EnsureArrayCompat(const py::array& array, const std::string& name) {
  if (!array.dtype().is(py::dtype::of<double>())) {
    throw py::type_error(name + " must have dtype float64");
  }
  if (!(array.flags() & py::array::c_style)) {
    throw py::value_error(name + " must be C-contiguous");
  }
  if (!(array.flags() & py::array::writeable)) {
    throw py::value_error(name + " must be writeable");
  }
  const py::buffer_info info = array.request();
  if (!IsCContiguousDouble(info)) {
    throw py::value_error(name + " must be a C-contiguous float64 array");
  }
}

void EnsureSameShape(const py::buffer_info& lhs, const py::buffer_info& rhs,
                     const std::string& lhs_name, const std::string& rhs_name) {
  if (lhs.ndim != rhs.ndim) {
    throw py::value_error(lhs_name + " and " + rhs_name +
                          " must have the same number of dimensions");
  }
  for (ssize_t dim = 0; dim < lhs.ndim; ++dim) {
    if (lhs.shape[dim] != rhs.shape[dim]) {
      throw py::value_error(lhs_name + " and " + rhs_name + " must have identical shapes");
    }
  }
}

void EnsureTwoDim(const py::buffer_info& info, const std::string& name) {
  if (info.ndim != 2) {
    throw py::value_error(name + " must be a 2D array");
  }
}

Eigen::MatrixXd OrthogonalizeNewtonSchulz(const Eigen::MatrixXd& update, int ns_steps,
                                          double coeff_a, double coeff_b, double coeff_c) {
  Eigen::MatrixXd x = update;
  const double frob_norm = x.norm();
  if (frob_norm > 0.0) {
    x *= std::sqrt(static_cast<double>(x.rows())) / frob_norm;
  }

  for (int i = 0; i < ns_steps; ++i) {
    const Eigen::MatrixXd gram = x * x.transpose();
    const Eigen::MatrixXd gram2 = gram * gram;
    x = coeff_a * x + (coeff_b * gram + coeff_c * gram2) * x;
  }
  return x;
}

py::array_t<double> MuonUpdate(py::array_t<double> params, py::array_t<double> grads,
                               py::array_t<double> momentum, double learning_rate, double beta,
                               bool nesterov, int ns_steps, double ns_coeff_a, double ns_coeff_b,
                               double ns_coeff_c, double weight_decay) {
  EnsureArrayCompat(params, "params");
  EnsureArrayCompat(grads, "grads");
  EnsureArrayCompat(momentum, "momentum");

  if (learning_rate <= 0.0) {
    throw py::value_error("learning_rate must be > 0");
  }
  if (beta <= 0.0 || beta >= 1.0) {
    throw py::value_error("beta must be in (0, 1)");
  }
  if (ns_steps <= 0) {
    throw py::value_error("ns_steps must be >= 1");
  }

  const py::buffer_info params_info = params.request();
  const py::buffer_info grads_info = grads.request();
  const py::buffer_info momentum_info = momentum.request();

  EnsureSameShape(params_info, grads_info, "params", "grads");
  EnsureSameShape(params_info, momentum_info, "params", "momentum");
  EnsureTwoDim(params_info, "params");

  auto* params_ptr = static_cast<double*>(params_info.ptr);
  auto* grads_ptr = static_cast<double*>(grads_info.ptr);
  auto* momentum_ptr = static_cast<double*>(momentum_info.ptr);

  using MatrixRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const auto rows = static_cast<Eigen::Index>(params_info.shape[0]);
  const auto cols = static_cast<Eigen::Index>(params_info.shape[1]);
  Eigen::Map<MatrixRM> params_map(params_ptr, rows, cols);
  Eigen::Map<MatrixRM> grads_map(grads_ptr, rows, cols);
  Eigen::Map<MatrixRM> momentum_map(momentum_ptr, rows, cols);

  const double one_minus_beta = 1.0 - beta;
  momentum_map = beta * momentum_map + one_minus_beta * grads_map;
  Eigen::MatrixXd update =
      nesterov ? (beta * momentum_map + one_minus_beta * grads_map) : momentum_map;

  const Eigen::MatrixXd ortho_update =
      OrthogonalizeNewtonSchulz(update, ns_steps, ns_coeff_a, ns_coeff_b, ns_coeff_c);

  params_map -= learning_rate * (ortho_update + weight_decay * params_map);

  return params;
}

}  // namespace

PYBIND11_MODULE(_muon_cc, m) {
  m.doc() = "Muon optimizer (momentum + Newton-Schulz orthogonalization).";
  m.def("muon_update", &MuonUpdate, py::arg("params"), py::arg("grads"), py::arg("momentum"),
        py::arg("learning_rate"), py::arg("beta") = 0.9, py::arg("nesterov") = true,
        py::arg("ns_steps") = 5, py::arg("ns_coeff_a") = 3.4445, py::arg("ns_coeff_b") = -4.775,
        py::arg("ns_coeff_c") = 2.0315, py::arg("weight_decay") = 0.0,
        "Applies an in-place Muon update to 2D parameters.");
}
