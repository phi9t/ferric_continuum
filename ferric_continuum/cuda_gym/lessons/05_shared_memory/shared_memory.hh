#ifndef FERRIC_CONTINUUM_CUDA_GYM_LESSONS_05_SHARED_MEMORY_SHARED_MEMORY_HH_
#define FERRIC_CONTINUUM_CUDA_GYM_LESSONS_05_SHARED_MEMORY_SHARED_MEMORY_HH_

#include <vector>

namespace ferric_continuum::cuda_gym::shared_memory {

// Matrix-vector product y = M x, where M is row-major (rows x cols) and x has
// `cols` entries; y has `rows` entries. Each block handles one row and stages
// tiles of x in shared memory so the whole block reuses each loaded chunk,
// illustrating the shared-memory tiling pattern. Throws std::invalid_argument on
// a size mismatch and std::runtime_error on CUDA failure.
std::vector<float> MatVec(int rows, int cols, const std::vector<float>& matrix,
                          const std::vector<float>& x);

}  // namespace ferric_continuum::cuda_gym::shared_memory

#endif  // FERRIC_CONTINUUM_CUDA_GYM_LESSONS_05_SHARED_MEMORY_SHARED_MEMORY_HH_
