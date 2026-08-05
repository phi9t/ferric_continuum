#ifndef FERRIC_CONTINUUM_CUDA_GYM_LESSONS_04_REDUCTION_REDUCTION_HH_
#define FERRIC_CONTINUUM_CUDA_GYM_LESSONS_04_REDUCTION_REDUCTION_HH_

#include <vector>

namespace ferric_continuum::cuda_gym::reduction {

// Sums all elements of `host_in` on the GPU using a two-level reduction:
// each block reduces its slice in shared memory (a tree reduction), then the
// per-block partial sums are combined into the final result with atomicAdd.
// Returns the total. Throws std::runtime_error on any CUDA failure.
float Sum(const std::vector<float>& host_in);

}  // namespace ferric_continuum::cuda_gym::reduction

#endif  // FERRIC_CONTINUUM_CUDA_GYM_LESSONS_04_REDUCTION_REDUCTION_HH_
