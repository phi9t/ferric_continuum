#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include <utility>

#include "constructor_rules.hh"

namespace ff = ferric::foundation;

int main() {
  // Initialize Abseil logging
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Constructor Rules (Rule of Five) Demo ===";

  // ===== Example 1: Full Rule of Five =====
  LOG(INFO) << "1. Full Rule of Five - ResourceManager";
  ff::ResourceManager::reset_stats();

  {
    LOG(INFO) << "Creating resource with default constructor...";
    ff::ResourceManager r1(100);
    LOG(INFO) << "   Default constructions: " << ff::ResourceManager::default_constructions();

    LOG(INFO) << "Copying resource (copy constructor)...";
    ff::ResourceManager r2 = r1;  // Copy constructor
    LOG(INFO) << "   Copy constructions: " << ff::ResourceManager::copy_constructions();
    LOG(INFO) << "   Both resources valid: r1=" << r1.is_valid() << " r2=" << r2.is_valid();

    LOG(INFO) << "Moving resource (move constructor)...";
    ff::ResourceManager r3 = std::move(r1);  // Move constructor
    LOG(INFO) << "   Move constructions: " << ff::ResourceManager::move_constructions();
    LOG(INFO) << "   After move: r1=" << r1.is_valid() << " (moved-from), r3=" << r3.is_valid();

    LOG(INFO) << "Leaving scope...";
  }

  LOG(INFO) << "After scope: destructions=" << ff::ResourceManager::destructions();
  LOG(INFO) << "All resources automatically cleaned up!";

  // ===== Example 2: Move-Only Type =====
  LOG(INFO) << "2. Move-Only Type - Cannot be copied";
  {
    ff::MoveOnlyResource res1("resource1");
    LOG(INFO) << "   Created: " << res1.name() << " valid=" << res1.is_valid();

    // This would NOT compile - copy is deleted:
    // ff::MoveOnlyResource res2 = res1;  // ERROR!

    LOG(INFO) << "Moving resource (only way to transfer ownership)...";
    ff::MoveOnlyResource res2 = std::move(res1);
    LOG(INFO) << "   After move:";
    LOG(INFO) << "   - res1 valid=" << res1.is_valid() << " (moved-from)";
    LOG(INFO) << "   - res2: " << res2.name() << " valid=" << res2.is_valid();

    LOG(INFO) << "Use case: File handles, unique pointers, exclusive resources";
  }

  // ===== Example 3: Copyable Type with Defaults =====
  LOG(INFO) << "3. Copyable Type with = default";
  {
    ff::Point p1(3.0, 4.0);
    LOG(INFO) << "   Created p1: (" << p1.x() << ", " << p1.y() << ")";

    ff::Point p2 = p1;  // Copy (= default)
    LOG(INFO) << "   Copied to p2: (" << p2.x() << ", " << p2.y() << ")";

    ff::Point p3 = std::move(p1);  // Move (= default)
    LOG(INFO) << "   Moved to p3: (" << p3.x() << ", " << p3.y() << ")";

    LOG(INFO) << "Compiler-generated versions work perfectly for simple types!";
  }

  // ===== Example 4: Rule of Zero (BEST PRACTICE) =====
  LOG(INFO) << "4. Rule of Zero - Use RAII Types";
  {
    ff::RuleOfZeroExample example;
    example.set_name("Best Practice");
    example.add_value(1);
    example.add_value(2);
    example.add_value(3);

    LOG(INFO) << "   Name: " << example.name();
    LOG(INFO) << "   Values: " << example.data().size() << " items";

    // Copy and move work automatically - no special member functions needed!
    ff::RuleOfZeroExample copy = example;
    LOG(INFO) << "   After copy: " << copy.name() << " has " << copy.data().size() << " items";

    ff::RuleOfZeroExample moved = std::move(example);
    LOG(INFO) << "   After move: " << moved.name() << " has " << moved.data().size() << " items";

    LOG(INFO) << "No manual resource management - std::vector and std::string "
              << "handle everything!";
  }

  // ===== Performance Comparison =====
  LOG(INFO) << "5. Performance: Return by Value";
  ff::ResourceManager::reset_stats();

  {
    LOG(INFO) << "Creating resource via factory function...";
    ff::ResourceManager r = ff::create_resource(1000);
    LOG(INFO) << "   Default constructions: " << ff::ResourceManager::default_constructions();
    LOG(INFO) << "   Move constructions: " << ff::ResourceManager::move_constructions();
    LOG(INFO) << "   Copy constructions: " << ff::ResourceManager::copy_constructions();
    LOG(INFO) << "Efficient! Move or RVO used (no copies)";
  }

  {
    LOG(INFO) << "Creating vector of resources...";
    ff::ResourceManager::reset_stats();
    auto resources = ff::create_multiple_resources(5, 100);
    LOG(INFO) << "   Created " << resources.size() << " resources";
    LOG(INFO) << "   Default constructions: " << ff::ResourceManager::default_constructions();
    LOG(INFO) << "   Move constructions: " << ff::ResourceManager::move_constructions();
    LOG(INFO) << "Move semantics enable efficient container operations!";
  }

  LOG(INFO) << "Key Principles:";
  LOG(INFO) << "- Rule of Five: Define all special members when managing resources";
  LOG(INFO) << "- Rule of Zero: Prefer RAII types (std::vector, std::unique_ptr)";
  LOG(INFO) << "- = default: Use compiler-generated versions when correct";
  LOG(INFO) << "- = delete: Prevent unwanted operations (move-only types)";
  LOG(INFO) << "- Move semantics: Enable efficient resource transfer";

  return 0;
}
