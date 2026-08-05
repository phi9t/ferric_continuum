#ifndef FERRIC_CONTINUUM_CUDA_GYM_LESSONS_01_HELLO_GPU_HELLO_GPU_HH_
#define FERRIC_CONTINUUM_CUDA_GYM_LESSONS_01_HELLO_GPU_HELLO_GPU_HH_

#include <cstddef>
#include <string>
#include <vector>

namespace ferric_continuum::cuda_gym::hello_gpu {

// Properties of the current CUDA device, gathered from cudaGetDeviceProperties.
struct DeviceInfo {
  int device_id = 0;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  int multiprocessor_count = 0;
  std::size_t total_global_mem_bytes = 0;
};

// Returns properties of device 0. Throws std::runtime_error if no CUDA device is
// available or the query fails.
DeviceInfo QueryDevice();

// The "first launch": computes y = a * x + y on the GPU, in place over y.
// Throws std::invalid_argument on length mismatch and std::runtime_error on any
// CUDA runtime failure. This is the SAXPY primitive, kept here as the canonical
// first kernel launch of the gym.
void Saxpy(float a, const std::vector<float>& x, std::vector<float>& y);

}  // namespace ferric_continuum::cuda_gym::hello_gpu

#endif  // FERRIC_CONTINUUM_CUDA_GYM_LESSONS_01_HELLO_GPU_HELLO_GPU_HH_
