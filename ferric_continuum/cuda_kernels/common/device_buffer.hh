#ifndef FERRIC_CONTINUUM_CUDA_KERNELS_COMMON_DEVICE_BUFFER_HH_
#define FERRIC_CONTINUUM_CUDA_KERNELS_COMMON_DEVICE_BUFFER_HH_

// RAII owner for a slab of device memory.
//
// DeviceBuffer<T> allocates `count` elements with cudaMalloc on construction and
// frees them on destruction, so kernel entry points can early-return on error
// (via FERRIC_CUDA_CHECK) without leaking. Allocation failure is reported via
// `ok()` rather than exceptions, keeping the kernels usable from the C ABI where
// exceptions must not cross the boundary.

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ferric_continuum::cuda_kernels {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) { Allocate(count); }

  ~DeviceBuffer() { Reset(); }

  // Movable but not copyable: the buffer uniquely owns its device pointer.
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_), status_(other.status_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
    other.status_ = cudaSuccess;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      ptr_ = other.ptr_;
      count_ = other.count_;
      status_ = other.status_;
      other.ptr_ = nullptr;
      other.count_ = 0;
      other.status_ = cudaSuccess;
    }
    return *this;
  }

  // True if the last allocation succeeded (or the buffer is empty).
  bool ok() const { return status_ == cudaSuccess; }
  cudaError_t status() const { return status_; }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

 private:
  void Allocate(std::size_t count) {
    count_ = count;
    if (count == 0) {
      ptr_ = nullptr;
      status_ = cudaSuccess;
      return;
    }
    status_ = cudaMalloc(&ptr_, count * sizeof(T));
    if (status_ != cudaSuccess) {
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  void Reset() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
    status_ = cudaSuccess;
  }

  T* ptr_ = nullptr;
  std::size_t count_ = 0;
  cudaError_t status_ = cudaSuccess;
};

}  // namespace ferric_continuum::cuda_kernels

#endif  // FERRIC_CONTINUUM_CUDA_KERNELS_COMMON_DEVICE_BUFFER_HH_
