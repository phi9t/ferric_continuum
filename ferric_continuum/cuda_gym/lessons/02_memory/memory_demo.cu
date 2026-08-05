// Lesson 02 demo: round-trip a buffer through device memory, then scale it.

#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "ferric_continuum/cuda_gym/lessons/02_memory/memory.hh"

namespace mem = ferric_continuum::cuda_gym::memory;

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Lesson 02: Memory ===";

  const std::vector<float> in = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
  const std::vector<float> back = mem::RoundTrip(in);
  const std::vector<float> scaled = mem::RoundTripScaled(in, 10.0F);

  for (std::size_t i = 0; i < in.size(); ++i) {
    LOG(INFO) << absl::StrCat("  in=", in[i], " round_trip=", back[i],
                              " x10=", scaled[i]);
  }

  return 0;
}
