#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include <utility>

#include "move_semantics.hh"

namespace ff = ferric::foundation;

int main() {
  // Initialize Abseil logging
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Move Semantics Demo ===";
  LOG(INFO) << "Demonstrating the efficiency of move operations";

  // Scenario 1: Return by value (move optimization)
  LOG(INFO) << "1. Creating buffer (return by value)...";
  ff::LargeBuffer::reset_counts();
  ff::LargeBuffer buf1 = ff::create_buffer(10000);
  LOG(INFO) << "   Copies: " << ff::LargeBuffer::copy_count()
            << ", Moves: " << ff::LargeBuffer::move_count();
  LOG(INFO) << "   Result: Efficient! Move or RVO used.";

  // Scenario 2: Passing lvalue (triggers copy)
  LOG(INFO) << "2. Passing lvalue by value (expensive copy)...";
  ff::LargeBuffer::reset_counts();
  ff::LargeBuffer buf2 = ff::create_buffer(10000);
  ff::LargeBuffer buf3 = ff::process_copy(buf2);  // buf2 is lvalue, must copy
  LOG(INFO) << "   Copies: " << ff::LargeBuffer::copy_count()
            << ", Moves: " << ff::LargeBuffer::move_count();
  LOG(INFO) << "   Result: Expensive! Copy needed to preserve buf2.";

  // Scenario 3: Passing rvalue with std::move (efficient)
  LOG(INFO) << "3. Passing rvalue with std::move (efficient)...";
  ff::LargeBuffer::reset_counts();
  ff::LargeBuffer buf4 = ff::create_buffer(10000);
  ff::LargeBuffer buf5 = ff::process_move(std::move(buf4));  // Explicit move
  LOG(INFO) << "   Copies: " << ff::LargeBuffer::copy_count()
            << ", Moves: " << ff::LargeBuffer::move_count();
  LOG(INFO) << "   Result: Efficient! Move used instead of copy.";
  LOG(INFO) << "   buf4 size after move: " << buf4.size() << " (moved-from state)";

  // Scenario 4: Temporary (automatic move)
  LOG(INFO) << "4. Passing temporary (automatic move)...";
  ff::LargeBuffer::reset_counts();
  ff::LargeBuffer buf6 = ff::process_move(ff::create_buffer(10000));
  LOG(INFO) << "   Copies: " << ff::LargeBuffer::copy_count()
            << ", Moves: " << ff::LargeBuffer::move_count();
  LOG(INFO) << "   Result: Efficient! Temporary is automatically moved.";

  LOG(INFO) << "Key Insight:";
  LOG(INFO) << "- Move semantics enable efficient transfer of resources";
  LOG(INFO) << "- Use std::move() to explicitly indicate you're done with an object";
  LOG(INFO) << "- Compiler optimizes automatically for temporaries";

  return 0;
}
