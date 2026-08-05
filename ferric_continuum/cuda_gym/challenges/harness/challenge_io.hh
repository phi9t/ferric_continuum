#ifndef FERRIC_CONTINUUM_CUDA_GYM_CHALLENGES_HARNESS_CHALLENGE_IO_HH_
#define FERRIC_CONTINUUM_CUDA_GYM_CHALLENGES_HARNESS_CHALLENGE_IO_HH_

// Tiny I/O helpers shared by challenge binaries.
//
// Each challenge binary receives one JSON object as argv[1] of the form
//   {"case": {<flat int/float fields>}, "seed": <int>}
// and prints a JSON object
//   {"status": <int>, "shape": [...], "data": [...], "elapsed_ms": <float>}
//
// The grader only ever emits flat objects with integer/float scalar fields, so
// a minimal parser (no nested arrays/strings beyond what we need) suffices and
// keeps challenge binaries dependency-free.

#include <cuda_runtime.h>

#include <cctype>
#include <chrono>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ferric_continuum::cuda_gym::challenges {

// Throws std::runtime_error if a CUDA runtime call fails. Use in challenge
// reference (and student) binaries so device failures never look like success.
inline void CheckCuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

// After a kernel launch: check launch error, then synchronize.
inline void CheckCudaKernel(const char* what) {
  CheckCuda(cudaGetLastError(), what);
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

// Extremely small JSON reader for the flat case object. It looks up integer or
// double fields by key inside the "case" sub-object of the argv payload.
class CaseSpec {
 public:
  explicit CaseSpec(const std::string& json) : json_(json) {}

  int GetInt(const std::string& key, int fallback = 0) const {
    double v;
    return Find(key, &v) ? static_cast<int>(v) : fallback;
  }

  double GetDouble(const std::string& key, double fallback = 0.0) const {
    double v;
    return Find(key, &v) ? v : fallback;
  }

 private:
  // Finds `"key" : number` anywhere in the payload and parses the number.
  bool Find(const std::string& key, double* out) const {
    const std::string needle = "\"" + key + "\"";
    std::size_t pos = json_.find(needle);
    if (pos == std::string::npos) {
      return false;
    }
    pos = json_.find(':', pos + needle.size());
    if (pos == std::string::npos) {
      return false;
    }
    ++pos;
    while (pos < json_.size() &&
           (json_[pos] == ' ' || json_[pos] == '\t')) {
      ++pos;
    }
    std::size_t end = pos;
    while (end < json_.size() &&
           (std::isdigit(static_cast<unsigned char>(json_[end])) ||
            json_[end] == '-' || json_[end] == '+' || json_[end] == '.' ||
            json_[end] == 'e' || json_[end] == 'E')) {
      ++end;
    }
    if (end == pos) {
      return false;
    }
    *out = std::stod(json_.substr(pos, end - pos));
    return true;
  }

  std::string json_;
};

// Deterministic uniform inputs in [-1, 1], seeded per key so different tensors
// (e.g. Q/K/V) get independent but reproducible data.
inline std::vector<float> RandomInput(std::size_t n, unsigned seed,
                                      unsigned key = 0) {
  std::mt19937 rng(seed * 2654435761u + key * 40503u + 1u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) {
    x = dist(rng);
  }
  return v;
}

// Times `fn` (a kernel launch + copy) in milliseconds.
template <typename Fn>
double TimeMs(Fn&& fn) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  fn();
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Prints the challenge result JSON to stdout.
inline void PrintResult(int status, const std::vector<int>& shape,
                        const std::vector<float>& data, double elapsed_ms) {
  std::string out = "{\"status\": " + std::to_string(status) + ", \"shape\": [";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    out += std::to_string(shape[i]);
    if (i + 1 < shape.size()) out += ", ";
  }
  out += "], \"data\": [";
  char buf[64];
  for (std::size_t i = 0; i < data.size(); ++i) {
    std::snprintf(buf, sizeof(buf), "%.9g", data[i]);
    out += buf;
    if (i + 1 < data.size()) out += ", ";
  }
  out += "], \"elapsed_ms\": ";
  std::snprintf(buf, sizeof(buf), "%.6f", elapsed_ms);
  out += buf;
  out += "}";
  std::printf("%s\n", out.c_str());
}

}  // namespace ferric_continuum::cuda_gym::challenges

#endif  // FERRIC_CONTINUUM_CUDA_GYM_CHALLENGES_HARNESS_CHALLENGE_IO_HH_
