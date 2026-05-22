/* Copyright 2025 Google LLC.

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

#include "tensor/buffer.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/internal/matchers.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::litert::tensor::IsOk;
using ::testing::Address;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::Pointer;
using ::testing::ResultOf;
using ::testing::SizeIs;
using ::testing::litert::AlignmentIs;

// Helps building a `LockedBufferSpan` with side effects.
struct Lockable {
  static constexpr int kData[] = {1, 2, 3, 4};
  int i = 0;
  LockedBufferSpan<std::byte> LockMutable() {
    ++i;
    return LockedBufferSpan<std::byte>(
        reinterpret_cast<std::byte*>(const_cast<int*>(kData)),
        [this](std::byte*) { --i; }, sizeof(kData));
  }
};

TEST(LockedBufferSpanTest, ManagesLocking) {
  Lockable l;
  ASSERT_EQ(l.i, 0);
  {
    LockedBufferSpan<std::byte> span = l.LockMutable();
    ASSERT_EQ(l.i, 1);
  }
  ASSERT_EQ(l.i, 0);
}

TEST(LockedBufferSpanTest, CanBeCastAndUsedAsAContainer) {
  Lockable l;
  LockedBufferSpan<int> span = l.LockMutable().As<int>();
  EXPECT_THAT(span, ElementsAreArray(Lockable::kData));
  EXPECT_THAT(span, SizeIs(std::size(Lockable::kData)));
  EXPECT_THAT(const_cast<const LockedBufferSpan<int>&>(span),
              ElementsAreArray(Lockable::kData));
  EXPECT_THAT(const_cast<const LockedBufferSpan<int>&>(span),
              SizeIs(std::size(Lockable::kData)));
}

TEST(SpanCpuBufferTest, BuildFromRawData) {
  const int32_t backing_array[] = {1, 2, 3, 4, 5};
  SpanCpuBuffer bv(
      reinterpret_cast<std::byte*>(const_cast<int32_t*>(backing_array)),
      sizeof(backing_array));
  LockedBufferSpan<const int32_t> data = bv.Lock().As<const int32_t>();
  EXPECT_THAT(data, ElementsAreArray(backing_array));
}

TEST(SpanCpuBufferTest, BuildFromCArray) {
  const int32_t backing_array[] = {1, 2, 3, 4, 5};
  SpanCpuBuffer buffer(backing_array);
  LockedBufferSpan<const int32_t> data = buffer.Lock().As<const int32_t>();
  EXPECT_THAT(data, ElementsAreArray(backing_array));
}

TEST(OwningCpuBuffer, IsAlignedWorks) {
  const char data[kCpuBufferAlignment] = {};
  ASSERT_THAT(data,
              Contains(Address(AlignmentIs(kCpuBufferAlignment))).Times(1));

  for (size_t i = 0; i < kCpuBufferAlignment; ++i) {
    EXPECT_EQ(OwningCpuBuffer::IsAligned(data + i),
              testing::Value(data + i, AlignmentIs(kCpuBufferAlignment)));
  }
}

TEST(OwningCpuBuffer, AlignedAllocationIsAligned) {
  // We allocate severals blocks to avoid an allocation that would be
  // aligned by chance. We keep them around to avoid the system reusing a block
  // that was just freed.
  std::vector<OwningCpuBuffer::CustomAllocPtr> allocations;
  for (size_t i = 0; i < 100; ++i) {
    allocations.push_back(
        OwningCpuBuffer::AlignedAlloc(kCpuBufferAlignment + 1));
  }
  EXPECT_THAT(allocations, Each(AlignmentIs(kCpuBufferAlignment)));
  EXPECT_THAT(allocations,
              Each(Pointer(ResultOf(&OwningCpuBuffer::IsAligned, Eq(true)))));
}

TEST(OwningCpuBuffer, TransferOwnershipFromRawData) {
  const int32_t reference_data[] = {1, 2, 3, 4, 5};
  OwningCpuBuffer::CustomAllocPtr data =
      OwningCpuBuffer::AlignedAlloc(sizeof(reference_data));
  absl::c_copy(reference_data, reinterpret_cast<int32_t*>(data.get()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<OwningCpuBuffer> buffer,
      OwningCpuBuffer::Own(std::move(data),
                           std::size(reference_data) * sizeof(int32_t)));

  ASSERT_THAT(buffer, NotNull());
  ASSERT_THAT(buffer->LockMutable().As<int32_t>(),
              ElementsAreArray(reference_data));
}

TEST(OwningCpuBuffer, TransferOwnershipFromRawDataFailsIfMisaligned) {
  // We create a purposefully misaligned pointer.
  std::byte* raw_data = new std::byte[2];
  OwningCpuBuffer::CustomAllocPtr data(
      OwningCpuBuffer::IsAligned(raw_data) ? raw_data + 1 : raw_data,
      [](std::byte* p) {
        if (OwningCpuBuffer::IsAligned(p - 1)) {
          p = p - 1;
        }
        delete[] p;
      });
  EXPECT_THAT(OwningCpuBuffer::Own(std::move(data), 2), Not(IsOk()));
}

TEST(OwningCpuBuffer, CopyFromRawData) {
  const int32_t reference_data[] = {1, 2, 3, 4, 5};

  std::shared_ptr<OwningCpuBuffer> buffer = OwningCpuBuffer::Copy(
      reinterpret_cast<const char*>(reference_data), sizeof(reference_data));
  ASSERT_THAT(buffer, NotNull());
  ASSERT_THAT(buffer->LockMutable().As<int32_t>(),
              ElementsAreArray(reference_data));
}

TEST(OwningCpuBuffer, CopyFromSequence) {
  const std::list<int32_t> reference_data{1, 2, 3, 4, 5};

  std::shared_ptr<OwningCpuBuffer> buffer =
      OwningCpuBuffer::Copy<Type::kI32>(reference_data);
  ASSERT_THAT(buffer, NotNull());
  ASSERT_THAT(buffer->LockMutable().As<int32_t>(),
              ElementsAreArray(reference_data));
}

TEST(OwningCpuBuffer, CopyFromInitializerList) {
  std::initializer_list<float> reference_data{1, 2, 3, 4, 5};

  std::shared_ptr<OwningCpuBuffer> buffer =
      OwningCpuBuffer::Copy<Type::kFP32>(reference_data);
  ASSERT_THAT(buffer, NotNull());
  ASSERT_THAT(buffer->LockMutable().As<float>(),
              ElementsAreArray(reference_data));
}

TEST(OwningCpuBuffer, CopyToBF16) {
  std::initializer_list<int> reference_data{1, 2, 3, 4, 5};

  std::shared_ptr<OwningCpuBuffer> buffer =
      OwningCpuBuffer::Copy<Type::kBF16>(reference_data);
  ASSERT_THAT(buffer, NotNull());
  ASSERT_THAT(buffer->LockMutable().As<bf16_t>(),
              ElementsAreArray(reference_data));
}

TEST(OwningCpuBuffer, CopyFromBF16) {
  std::initializer_list<bf16_t> reference_data{1, 2, 3, 4, 5};

  std::shared_ptr<OwningCpuBuffer> buffer =
      OwningCpuBuffer::Copy<Type::kI32>(reference_data);
  ASSERT_THAT(buffer, NotNull());
  ASSERT_THAT(buffer->LockMutable().As<int>(),
              ElementsAreArray(reference_data));
}

}  // namespace
}  // namespace litert::tensor
