#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "smart_pointers.hh"

namespace ff = ferric::foundation;

int main() {
  // Initialize Abseil logging
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Smart Pointers & RAII Demo ===";

  // ===== unique_ptr - Exclusive Ownership =====
  LOG(INFO) << "1. unique_ptr - Exclusive Ownership";
  {
    auto list = ff::create_list({1, 2, 3, 4, 5});
    LOG(INFO) << "   Created list with " << ff::count_nodes(list.get()) << " nodes";
    LOG(INFO) << "   First value: " << list->value();

    // Transfer ownership
    auto list2 = std::move(list);
    LOG(INFO) << "   After move:";
    LOG(INFO) << "   - list is nullptr: " << (list == nullptr);
    LOG(INFO) << "   - list2 owns the data: " << (list2 != nullptr);

    LOG(INFO) << "   Leaving scope - automatic cleanup!";
  }  // list2 destroyed here, all nodes automatically deleted
  LOG(INFO) << "   All memory cleaned up automatically.";

  // ===== shared_ptr - Shared Ownership =====
  LOG(INFO) << "2. shared_ptr - Shared Ownership";
  {
    LOG(INFO) << "   Resources alive: " << ff::Resource::instance_count();

    auto resource = ff::create_shared_resource(42);
    LOG(INFO) << "   Created resource " << resource->id();
    LOG(INFO) << "   Use count: " << resource.use_count();
    LOG(INFO) << "   Resources alive: " << ff::Resource::instance_count();

    {
      // Share ownership
      auto shared = ff::share_resource(resource, 3);
      LOG(INFO) << "   After sharing with 3 more owners:";
      LOG(INFO) << "   Use count: " << resource.use_count();
      LOG(INFO) << "   Resources alive: " << ff::Resource::instance_count();

      LOG(INFO) << "   Leaving inner scope...";
    }  // shared vector destroyed, but resource still alive

    LOG(INFO) << "   Back to outer scope:";
    LOG(INFO) << "   Use count: " << resource.use_count();
    LOG(INFO) << "   Resources alive: " << ff::Resource::instance_count();

    LOG(INFO) << "   Leaving outer scope...";
  }  // resource destroyed when last shared_ptr goes away
  LOG(INFO) << "   Resources alive: " << ff::Resource::instance_count();

  // ===== RAII Pattern =====
  LOG(INFO) << "3. RAII Pattern";
  {
    ff::FileGuard file("data.txt");
    LOG(INFO) << "   File opened: " << file.filename();
    LOG(INFO) << "   Is open: " << file.is_open();

    // Transfer ownership
    ff::FileGuard file2 = std::move(file);
    LOG(INFO) << "   After move:";
    LOG(INFO) << "   - file is closed: " << (!file.is_open());
    LOG(INFO) << "   - file2 owns resource: " << file2.is_open();

    LOG(INFO) << "   Leaving scope...";
  }  // file2 destroyed, file automatically closed
  LOG(INFO) << "   File automatically closed in destructor.";

  LOG(INFO) << "Key Benefits:";
  LOG(INFO) << "- No manual delete needed - automatic cleanup";
  LOG(INFO) << "- Exception-safe - cleanup happens even if exception thrown";
  LOG(INFO) << "- Clear ownership semantics - who owns what?";
  LOG(INFO) << "- unique_ptr: exclusive ownership (can't be copied)";
  LOG(INFO) << "- shared_ptr: shared ownership (reference counted)";
  LOG(INFO) << "- RAII: resource lifetime tied to object lifetime";

  return 0;
}
