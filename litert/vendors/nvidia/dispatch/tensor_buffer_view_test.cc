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

#include "litert/vendors/nvidia/dispatch/tensor_buffer_view.h"

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"

namespace litert::nvidia {
namespace {

TEST(TensorBufferViewTest, AppliesOffsetToAllocationAddress) {
  void* allocation = reinterpret_cast<void*>(uintptr_t{0x100000});

  auto view = ResolveCudaTensorBufferView(allocation, /*allocation_size=*/4096,
                                          /*offset=*/512,
                                          /*packed_size=*/1024);

  ASSERT_TRUE(view);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(*view), uintptr_t{0x100200});
}

TEST(TensorBufferViewTest, RejectsViewBeyondAllocation) {
  auto view = ResolveCudaTensorBufferView(
      reinterpret_cast<void*>(uintptr_t{0x100000}),
      /*allocation_size=*/1024, /*offset=*/768, /*packed_size=*/512);

  ASSERT_FALSE(view);
  EXPECT_EQ(view.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(TensorBufferViewTest, RejectsAddressOverflow) {
  void* allocation = reinterpret_cast<void*>(
      std::numeric_limits<uintptr_t>::max() - uintptr_t{255});

  auto view = ResolveCudaTensorBufferView(allocation, /*allocation_size=*/512,
                                          /*offset=*/256,
                                          /*packed_size=*/256);

  ASSERT_FALSE(view);
  EXPECT_EQ(view.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(TensorBufferViewTest, RejectsNullAllocation) {
  auto view = ResolveCudaTensorBufferView(
      /*allocation=*/nullptr, /*allocation_size=*/1024, /*offset=*/0,
      /*packed_size=*/1024);

  ASSERT_FALSE(view);
  EXPECT_EQ(view.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::nvidia
