#ifndef FERRIC_CONTINUUM_CUDA_GYM_LESSONS_03_INDEXING_INDEXING_HH_
#define FERRIC_CONTINUUM_CUDA_GYM_LESSONS_03_INDEXING_INDEXING_HH_

#include <cstddef>
#include <vector>

namespace ferric_continuum::cuda_gym::indexing {

// Fills `out[i] = i` using the 1D global thread index
// (blockIdx.x*blockDim.x + threadIdx.x), demonstrating the flat index map and
// the bounds guard that prevents out-of-range writes. `n` may not be a multiple
// of the block size. Throws std::runtime_error on CUDA failure.
std::vector<float> Iota(std::size_t n);

// Fills a row-major (rows x cols) matrix with out[r,c] = r*cols + c using a 2D
// grid/block, demonstrating the (x,y) → (col,row) mapping. Throws
// std::runtime_error on CUDA failure.
std::vector<float> RowMajorIndices(int rows, int cols);

}  // namespace ferric_continuum::cuda_gym::indexing

#endif  // FERRIC_CONTINUUM_CUDA_GYM_LESSONS_03_INDEXING_INDEXING_HH_
