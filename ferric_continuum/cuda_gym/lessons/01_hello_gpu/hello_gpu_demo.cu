// Lesson 01 demo: print device properties, then run the first kernel launch.

#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "ferric_continuum/cuda_gym/lessons/01_hello_gpu/hello_gpu.hh"

namespace hg = ferric_continuum::cuda_gym::hello_gpu;

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Lesson 01: Hello GPU ===";

  const hg::DeviceInfo info = hg::QueryDevice();
  LOG(INFO) << absl::StrCat("Device ", info.device_id, ": ", info.name);
  LOG(INFO) << absl::StrCat("  Compute capability: ", info.compute_major, ".",
                            info.compute_minor);
  LOG(INFO) << absl::StrCat("  SM count: ", info.multiprocessor_count);
  LOG(INFO) << absl::StrCat(
      "  Global memory: ",
      info.total_global_mem_bytes / (1024ULL * 1024ULL), " MiB");

  // First launch: y = 2*x + y.
  const float a = 2.0F;
  const std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> y = {10.0F, 20.0F, 30.0F, 40.0F};
  hg::Saxpy(a, x, y);

  LOG(INFO) << "SAXPY y = 2*x + y result:";
  for (std::size_t i = 0; i < y.size(); ++i) {
    LOG(INFO) << absl::StrCat("  y[", i, "] = ", y[i]);
  }

  return 0;
}

