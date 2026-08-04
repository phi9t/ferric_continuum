#ifndef FERRIC_CONTINUUM_CUDA_SAXPY_HH_
#define FERRIC_CONTINUUM_CUDA_SAXPY_HH_

#include <vector>

namespace ferric_continuum::cuda {

// Computes y = a * x + y on the GPU, in place over `y`.
//
// Throws std::invalid_argument if x and y differ in length, and
// std::runtime_error if any CUDA runtime call fails (e.g. no device present).
void Saxpy(float a, const std::vector<float>& x, std::vector<float>& y);

}  // namespace ferric_continuum::cuda

#endif  // FERRIC_CONTINUUM_CUDA_SAXPY_HH_
