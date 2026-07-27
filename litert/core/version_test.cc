// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/core/version.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"

namespace litert::internal {
namespace {

TEST(VersionTest, IsSameVersion) {
  EXPECT_TRUE(IsSameVersion({1, 2, 3}, {1, 2, 3}));
  EXPECT_FALSE(IsSameVersion({1, 2, 3}, {1, 2, 4}));
  EXPECT_FALSE(IsSameVersion({1, 2, 3}, {1, 3, 3}));
  EXPECT_FALSE(IsSameVersion({1, 2, 3}, {2, 2, 3}));
}

TEST(VersionTest, IsCompatibleVersionMajorMismatch) {
  // Major mismatch is always incompatible.
  EXPECT_FALSE(IsCompatibleVersion({1, 0, 0}, {2, 0, 0}));
  EXPECT_FALSE(IsCompatibleVersion({2, 0, 0}, {1, 0, 0}));
  EXPECT_FALSE(IsCompatibleVersion({0, 1, 0}, {1, 1, 0}));
}

TEST(VersionTest, IsCompatibleVersionPreOneZero) {
  // For major == 0, minor must match exactly.
  EXPECT_TRUE(IsCompatibleVersion({0, 1, 0}, {0, 1, 0}));
  EXPECT_TRUE(
      IsCompatibleVersion({0, 1, 0}, {0, 1, 1}));  // Patch mismatch allowed
  EXPECT_TRUE(
      IsCompatibleVersion({0, 1, 1}, {0, 1, 0}));  // Patch mismatch allowed
  EXPECT_FALSE(IsCompatibleVersion({0, 1, 0}, {0, 2, 0}));
  EXPECT_FALSE(IsCompatibleVersion({0, 2, 0}, {0, 1, 0}));
}

TEST(VersionTest, IsCompatibleVersionPostOneZero) {
  // For major >= 1, runtime minor must be >= vendor minor.
  // Format: IsCompatibleVersion(vendor, runtime)

  // Same version is compatible.
  EXPECT_TRUE(IsCompatibleVersion({1, 1, 0}, {1, 1, 0}));

  // Runtime is newer -> Compatible (backward compatibility).
  EXPECT_TRUE(IsCompatibleVersion({1, 1, 0}, {1, 2, 0}));
  EXPECT_TRUE(IsCompatibleVersion({1, 1, 5}, {1, 2, 0}));

  // Runtime is older -> Incompatible.
  EXPECT_FALSE(IsCompatibleVersion({1, 2, 0}, {1, 1, 0}));

  // Patch version is ignored.
  EXPECT_TRUE(IsCompatibleVersion({1, 1, 1}, {1, 1, 2}));
  EXPECT_TRUE(IsCompatibleVersion({1, 1, 2}, {1, 1, 1}));
}

TEST(VersionTest, IsCompatibleVersionAsRuntime) {
  // Test compatibility against current runtime version defined in headers.
  LiteRtApiVersion current_runtime = {LITERT_API_VERSION_MAJOR,
                                      LITERT_API_VERSION_MINOR,
                                      LITERT_API_VERSION_PATCH};

  EXPECT_TRUE(IsCompatibleVersionAsRuntime(current_runtime));

  // Create a compatible version by changing only the patch version.
  LiteRtApiVersion compatible_version = current_runtime;
  compatible_version.patch += 1;
  EXPECT_TRUE(IsCompatibleVersionAsRuntime(compatible_version));

  // Create an incompatible version by changing the major version.
  LiteRtApiVersion incompatible_version_major = current_runtime;
  incompatible_version_major.major += 1;
  EXPECT_FALSE(IsCompatibleVersionAsRuntime(incompatible_version_major));
}

struct DummyInterface {
  size_t size;
  int (*func1)();
  int (*func2)();
  int (*func3)();
};

TEST(VersionTest, MemberSupported) {
  // Scenario 1: Older interface that only supports func1 and func2.
  size_t old_size =
      offsetof(DummyInterface, func2) + sizeof(DummyInterface::func2);

  std::vector<uint8_t> buffer(sizeof(DummyInterface), 0);
  DummyInterface* old_interface =
      reinterpret_cast<DummyInterface*>(buffer.data());
  old_interface->size = old_size;
  old_interface->func1 = []() { return 1; };
  old_interface->func2 = []() { return 2; };

  EXPECT_TRUE(LITERT_IS_MEMBER_SUPPORTED(old_interface, func1));
  EXPECT_TRUE(LITERT_IS_MEMBER_SUPPORTED(old_interface, func2));
  EXPECT_FALSE(LITERT_IS_MEMBER_SUPPORTED(old_interface, func3));

  // Scenario 2: Newer interface that supports all functions.
  DummyInterface new_interface;
  new_interface.size = sizeof(DummyInterface);
  new_interface.func1 = []() { return 1; };
  new_interface.func2 = []() { return 2; };
  new_interface.func3 = []() { return 3; };

  EXPECT_TRUE(LITERT_IS_MEMBER_SUPPORTED(&new_interface, func1));
  EXPECT_TRUE(LITERT_IS_MEMBER_SUPPORTED(&new_interface, func2));
  EXPECT_TRUE(LITERT_IS_MEMBER_SUPPORTED(&new_interface, func3));

  // Scenario 3: Null pointer
  DummyInterface* null_interface = nullptr;
  EXPECT_FALSE(LITERT_IS_MEMBER_SUPPORTED(null_interface, func1));
}

}  // namespace
}  // namespace litert::internal
