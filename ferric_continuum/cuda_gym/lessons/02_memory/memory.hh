#ifndef FERRIC_CONTINUUM_CUDA_GYM_LESSONS_02_MEMORY_MEMORY_HH_
#define FERRIC_CONTINUUM_CUDA_GYM_LESSONS_02_MEMORY_MEMORY_HH_

#include <vector>

namespace ferric_continuum::cuda_gym::memory {

// Copies `host_in` to the device and straight back into the returned vector
// (H2D then D2H, no compute). Demonstrates cudaMalloc / cudaMemcpy / cudaFree
// and that a plain round-trip is bit-exact. Throws std::runtime_error on any
// CUDA failure.
std::vector<float> RoundTrip(const std::vector<float>& host_in);

// Round-trips through device memory while scaling each element by `factor` on
// the device, so the copy is observable. Throws std::runtime_error on failure.
std::vector<float> RoundTripScaled(const std::vector<float>& host_in,
                                   float factor);

}  // namespace ferric_continuum::cuda_gym::memory

#endif  // FERRIC_CONTINUUM_CUDA_GYM_LESSONS_02_MEMORY_MEMORY_HH_
