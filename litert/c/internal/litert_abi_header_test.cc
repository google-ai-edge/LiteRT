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

#include "litert/c/internal/litert_abi_header.h"

#include <cstddef>

#include <gtest/gtest.h>

namespace {

int DummyFunc() { return 0; }

// Initial API version.
struct DummyStructV1 {
  LiteRtAbiHeader abi_header;
  int (*func1)();
  int (*func2)();
};


// Deprecated func1 and added func3 in V2.
struct DummyStructV2 {
  LiteRtAbiHeader abi_header;
  int (*func2)();
  int (*func3)();
};

TEST(LiteRtAbiHeaderTest, IsCompatible) {
  DummyStructV1 instance;
  instance.abi_header.struct_size = sizeof(DummyStructV1);
  instance.abi_header.major_version = 1;
  instance.abi_header.minor_version = 2;
  instance.abi_header.reserved = 0;

  // Same major, lower or equal minor => compatible
  EXPECT_TRUE(
      LITERT_ABI_IS_COMPATIBLE(&instance, /*req_major=*/1, /*req_minor=*/0));
  EXPECT_TRUE(
      LITERT_ABI_IS_COMPATIBLE(&instance, /*req_major=*/1, /*req_minor=*/2));

  // Same major, higher minor => incompatible
  EXPECT_FALSE(
      LITERT_ABI_IS_COMPATIBLE(&instance, /*req_major=*/1, /*req_minor=*/3));

  // Different major => incompatible
  EXPECT_FALSE(
      LITERT_ABI_IS_COMPATIBLE(&instance, /*req_major=*/2, /*req_minor=*/0));
  EXPECT_FALSE(
      LITERT_ABI_IS_COMPATIBLE(&instance, /*req_major=*/0, /*req_minor=*/1));
}

TEST(LiteRtAbiHeaderTest, HasApiV1) {
  DummyStructV1 instance;
  instance.abi_header.struct_size = sizeof(DummyStructV1);
  instance.abi_header.major_version = 1;
  instance.abi_header.minor_version = 0;
  instance.abi_header.reserved = 0;
  instance.func1 = DummyFunc;
  instance.func2 = DummyFunc;

  // Valid instance and non-null members
  EXPECT_TRUE(LITERT_ABI_HAS_API(&instance, /*req_major=*/1, func1));
  EXPECT_TRUE(LITERT_ABI_HAS_API(&instance, /*req_major=*/1, func2));

  // Null instance pointer
  EXPECT_FALSE(LITERT_ABI_HAS_API(static_cast<DummyStructV1*>(nullptr),
                                  /*req_major=*/1, func1));

  // Major version mismatch
  EXPECT_FALSE(LITERT_ABI_HAS_API(&instance, /*req_major=*/2, func1));

  // Function pointer is null
  instance.func2 = nullptr;
  EXPECT_FALSE(LITERT_ABI_HAS_API(&instance, /*req_major=*/1, func2));
  instance.func2 = DummyFunc;

  // Simulate an older instance that only contains func1 in memory
  instance.abi_header.struct_size = offsetof(DummyStructV1, func2);
  EXPECT_TRUE(LITERT_ABI_HAS_API(&instance, /*req_major=*/1, func1));
  EXPECT_FALSE(LITERT_ABI_HAS_API(&instance, /*req_major=*/1, func2));
}


TEST(LiteRtAbiHeaderTest, HasApiV2) {
  DummyStructV1 instance_v1;
  instance_v1.abi_header.struct_size = sizeof(DummyStructV1);
  instance_v1.abi_header.major_version = 1;
  instance_v1.abi_header.minor_version = 0;
  instance_v1.abi_header.reserved = 0;
  instance_v1.func1 = DummyFunc;
  instance_v1.func2 = DummyFunc;

  DummyStructV2 instance_v2;
  instance_v2.abi_header.struct_size = sizeof(DummyStructV2);
  instance_v2.abi_header.major_version = 2;
  instance_v2.abi_header.minor_version = 0;
  instance_v2.abi_header.reserved = 0;
  instance_v2.func2 = DummyFunc;
  instance_v2.func3 = DummyFunc;

  // Valid instance and non-null members
  EXPECT_TRUE(LITERT_ABI_HAS_API(&instance_v2, /*req_major=*/2, func2));
  EXPECT_TRUE(LITERT_ABI_HAS_API(&instance_v2, /*req_major=*/2, func3));

  // Compilation error to access deprecated func1.
  // EXPECT_FALSE(LITERT_ABI_HAS_API(&instance_v2, /*req_major=*/2, func1));

  // Major version mismatch
  EXPECT_FALSE(LITERT_ABI_HAS_API(&instance_v1, /*req_major=*/2, func2));
}

}  // namespace
