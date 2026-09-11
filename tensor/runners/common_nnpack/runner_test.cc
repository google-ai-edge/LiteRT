/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/runners/common_nnpack/runner.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using absl_testing::StatusIs;
using common_nnpack::internal::Reserve;
using ::testing::ElementsAreArray;
using ::testing::SizeIs;

TEST(ReserveTest, AllocatesWhenNullptr) {
  std::shared_ptr<Buffer> buffer = nullptr;
  ASSERT_THAT(Reserve(buffer, 16), IsOk());
  ASSERT_NE(buffer, nullptr);
  EXPECT_TRUE(buffer->IsA(OwningCpuBuffer::TypeId()));
  EXPECT_EQ(buffer->Lock().size(), 16);
}

TEST(ReserveTest, NoOpWhenAlreadyLargeEnough) {
  std::shared_ptr<Buffer> buffer = OwningCpuBuffer::Allocate<Type::kI8>(32);
  const Buffer* original_ptr = buffer.get();
  ASSERT_THAT(Reserve(buffer, 16), IsOk());
  EXPECT_EQ(buffer.get(), original_ptr);
  EXPECT_EQ(buffer->Lock().size(), 32);
}

TEST(ReserveTest, ReallocatesOwningBufferAndPreservesData) {
  const std::vector<uint8_t> initial = {10, 20, 30, 40};
  std::shared_ptr<Buffer> buffer = OwningCpuBuffer::Copy<Type::kI8>(initial);

  ASSERT_THAT(Reserve(buffer, 8, /*preserve_data=*/true), IsOk());
  auto lock = buffer->Lock().As<const uint8_t>();
  EXPECT_THAT(lock, SizeIs(8));
  EXPECT_THAT(lock.SubSpan(0, 4), ElementsAreArray(initial));
}

TEST(ReserveTest, FailsForNonOwningViewSmallerThanRequired) {
  const std::vector<uint8_t> data = {1, 2, 3, 4};
  std::shared_ptr<Buffer> buffer = std::make_shared<SpanCpuBuffer>(
      reinterpret_cast<const std::byte*>(data.data()), data.size());

  EXPECT_THAT(Reserve(buffer, 8), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ByteSizeTest, ComputesCorrectSize) {
  graph::TensorInformation info{
      .type = Type::kFP32,
      .shape = {2, 3, 4},
  };
  EXPECT_EQ(ByteSize(info), 2 * 3 * 4 * sizeof(float));
}

}  // namespace
}  // namespace litert::tensor
