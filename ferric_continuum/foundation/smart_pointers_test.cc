#include "smart_pointers.hh"

#include <gtest/gtest.h>

namespace ferric::foundation {
namespace {

// =============================================================================
// Node and unique_ptr tests
// =============================================================================

TEST(SmartPointersTest, NodeConstruction) {
  Node node(42);
  EXPECT_EQ(node.value(), 42);
  EXPECT_EQ(node.next(), nullptr);
}

TEST(SmartPointersTest, NodeAppend) {
  Node node1(1);
  auto node2 = std::make_unique<Node>(2);
  auto node3 = std::make_unique<Node>(3);

  node1.append(std::move(node2));
  ASSERT_NE(node1.next(), nullptr);
  EXPECT_EQ(node1.next()->value(), 2);

  node1.append(std::move(node3));
  ASSERT_NE(node1.next()->next(), nullptr);
  EXPECT_EQ(node1.next()->next()->value(), 3);
}

TEST(SmartPointersTest, CreateListEmpty) {
  auto list = create_list({});
  EXPECT_EQ(list, nullptr);
}

TEST(SmartPointersTest, CreateListSingle) {
  auto list = create_list({42});
  ASSERT_NE(list, nullptr);
  EXPECT_EQ(list->value(), 42);
  EXPECT_EQ(list->next(), nullptr);
  EXPECT_EQ(count_nodes(list.get()), 1);
}

TEST(SmartPointersTest, CreateListMultiple) {
  auto list = create_list({1, 2, 3, 4, 5});
  ASSERT_NE(list, nullptr);
  EXPECT_EQ(list->value(), 1);
  EXPECT_EQ(count_nodes(list.get()), 5);

  // Verify chain
  ASSERT_NE(list->next(), nullptr);
  EXPECT_EQ(list->next()->value(), 2);
  ASSERT_NE(list->next()->next(), nullptr);
  EXPECT_EQ(list->next()->next()->value(), 3);
}

TEST(SmartPointersTest, CountNodesEmpty) {
  EXPECT_EQ(count_nodes(nullptr), 0);
}

TEST(SmartPointersTest, CountNodesMultiple) {
  auto list = create_list({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  EXPECT_EQ(count_nodes(list.get()), 10);
}

TEST(SmartPointersTest, UniquePtrOwnershipTransfer) {
  auto list = create_list({1, 2, 3});
  auto list2 = std::move(list);

  EXPECT_EQ(list, nullptr);   // Moved from
  ASSERT_NE(list2, nullptr);  // Moved to
  EXPECT_EQ(list2->value(), 1);
}

TEST(SmartPointersTest, UniquePtrAutomaticCleanup) {
  {
    auto list = create_list({1, 2, 3, 4, 5});
    EXPECT_EQ(count_nodes(list.get()), 5);
  }  // list destroyed, all nodes automatically deleted
  // Test passes if no memory leaks
}

// =============================================================================
// Resource and shared_ptr tests
// =============================================================================

TEST(SmartPointersTest, ResourceConstruction) {
  int count_before = Resource::instance_count();
  {
    Resource r(123);
    EXPECT_EQ(r.id(), 123);
    EXPECT_EQ(Resource::instance_count(), count_before + 1);
  }
  EXPECT_EQ(Resource::instance_count(), count_before);
}

TEST(SmartPointersTest, CreateSharedResource) {
  auto resource = create_shared_resource(42);
  EXPECT_EQ(resource.use_count(), 1);
  EXPECT_EQ(resource->id(), 42);
}

TEST(SmartPointersTest, SharedPtrSharing) {
  auto resource = create_shared_resource(42);
  EXPECT_EQ(resource.use_count(), 1);
  EXPECT_EQ(resource->id(), 42);

  {
    auto shared = share_resource(resource, 3);
    EXPECT_EQ(resource.use_count(), 4);  // Original + 3 copies
    EXPECT_EQ(Resource::instance_count(), Resource::instance_count());
  }

  EXPECT_EQ(resource.use_count(), 1);  // Back to 1
}

TEST(SmartPointersTest, ShareResourceMultipleTimes) {
  auto resource = create_shared_resource(99);

  auto shared1 = share_resource(resource, 2);
  EXPECT_EQ(resource.use_count(), 3);

  auto shared2 = share_resource(resource, 3);
  EXPECT_EQ(resource.use_count(), 6);
}

TEST(SmartPointersTest, SharedPtrCopy) {
  auto resource = create_shared_resource(42);
  auto resource2 = resource;  // Copy

  EXPECT_EQ(resource.use_count(), 2);
  EXPECT_EQ(resource2.use_count(), 2);
  EXPECT_EQ(resource->id(), resource2->id());
}

TEST(SmartPointersTest, SharedPtrMove) {
  auto resource = create_shared_resource(42);
  auto resource2 = std::move(resource);

  EXPECT_EQ(resource.use_count(), 0);
  EXPECT_EQ(resource2.use_count(), 1);
  EXPECT_EQ(resource2->id(), 42);
}

// =============================================================================
// FileGuard and RAII tests
// =============================================================================

TEST(SmartPointersTest, FileGuardConstruction) {
  FileGuard file("test.txt");
  EXPECT_TRUE(file.is_open());
  EXPECT_EQ(file.filename(), "test.txt");
}

TEST(SmartPointersTest, FileGuardDestructor) {
  {
    FileGuard file("test.txt");
    EXPECT_TRUE(file.is_open());
  }  // File closed automatically in destructor
  // Test passes if no crash
}

TEST(SmartPointersTest, FileGuardMoveConstructor) {
  FileGuard file1("test.txt");
  EXPECT_TRUE(file1.is_open());

  FileGuard file2 = std::move(file1);
  EXPECT_FALSE(file1.is_open());  // Moved from
  EXPECT_TRUE(file2.is_open());   // Moved to
  EXPECT_EQ(file2.filename(), "test.txt");
}

TEST(SmartPointersTest, FileGuardMoveAssignment) {
  FileGuard file1("file1.txt");
  FileGuard file2("file2.txt");

  EXPECT_TRUE(file1.is_open());
  EXPECT_TRUE(file2.is_open());

  file2 = std::move(file1);

  EXPECT_FALSE(file1.is_open());  // Moved from
  EXPECT_TRUE(file2.is_open());   // Moved to
  EXPECT_EQ(file2.filename(), "file1.txt");
}

TEST(SmartPointersTest, FileGuardMultipleFiles) {
  FileGuard file1("file1.txt");
  FileGuard file2("file2.txt");
  FileGuard file3("file3.txt");

  EXPECT_TRUE(file1.is_open());
  EXPECT_TRUE(file2.is_open());
  EXPECT_TRUE(file3.is_open());

  EXPECT_EQ(file1.filename(), "file1.txt");
  EXPECT_EQ(file2.filename(), "file2.txt");
  EXPECT_EQ(file3.filename(), "file3.txt");
}

}  // namespace
}  // namespace ferric::foundation
