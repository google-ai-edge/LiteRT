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

#include "tensor/runners/litert/litert_buffer.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "tensor/buffer.h"
#include "tensor/internal/type_id.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::litert::tensor::IsOk;
using ::testing::Not;

TEST(LitertBufferTest, LockAndWrite) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::move(*env_or);

  std::vector<int32_t> shape = {1, 4};
  LiteRtRankedTensorType c_type;
  c_type.element_type = kLiteRtElementTypeFloat32;
  c_type.layout.rank = 2;
  c_type.layout.dimensions[0] = 1;
  c_type.layout.dimensions[1] = 4;
  RankedTensorType tensor_type(c_type);

  auto tb_or = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                           tensor_type, 4 * sizeof(float));
  ASSERT_TRUE(tb_or.HasValue());
  auto tb = std::move(*tb_or);

  LitertBuffer buffer(std::move(tb));

  {
    auto locked_span = buffer.LockMutable();
    auto* data = reinterpret_cast<float*>(locked_span.data());
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
  }  // Unlocks!

  {
    auto locked_span = buffer.Lock();
    const auto* data = reinterpret_cast<const float*>(locked_span.data());
    EXPECT_EQ(data[0], 1.0f);
    EXPECT_EQ(data[1], 2.0f);
    EXPECT_EQ(data[2], 3.0f);
    EXPECT_EQ(data[3], 4.0f);
  }
}

TEST(LitertBufferTest, GetTypeId) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::move(*env_or);

  LiteRtRankedTensorType c_type;
  c_type.element_type = kLiteRtElementTypeFloat32;
  c_type.layout.rank = 2;
  c_type.layout.dimensions[0] = 1;
  c_type.layout.dimensions[1] = 4;
  RankedTensorType tensor_type(c_type);

  auto tb_or = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                           tensor_type, 4 * sizeof(float));
  ASSERT_TRUE(tb_or.HasValue());
  auto tb = std::move(*tb_or);

  LitertBuffer buffer(std::move(tb));
  EXPECT_EQ(buffer.GetTypeId(), internal::TypeId::Get<LitertBuffer>());
}

TEST(LitertBufferTest, IsA) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::move(*env_or);

  LiteRtRankedTensorType c_type;
  c_type.element_type = kLiteRtElementTypeFloat32;
  c_type.layout.rank = 2;
  c_type.layout.dimensions[0] = 1;
  c_type.layout.dimensions[1] = 4;
  RankedTensorType tensor_type(c_type);

  auto tb_or = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                           tensor_type, 4 * sizeof(float));
  ASSERT_TRUE(tb_or.HasValue());
  auto tb = std::move(*tb_or);

  LitertBuffer buffer(std::move(tb));
  EXPECT_TRUE(buffer.IsA(internal::TypeId::Get<LitertBuffer>()));
  EXPECT_FALSE(buffer.IsA(internal::TypeId::Get<SpanCpuBuffer>()));
}

TEST(LitertBufferTest, As) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::move(*env_or);

  LiteRtRankedTensorType c_type;
  c_type.element_type = kLiteRtElementTypeFloat32;
  c_type.layout.rank = 2;
  c_type.layout.dimensions[0] = 1;
  c_type.layout.dimensions[1] = 4;
  RankedTensorType tensor_type(c_type);

  auto tb_or = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                           tensor_type, 4 * sizeof(float));
  ASSERT_TRUE(tb_or.HasValue());
  auto tb = std::move(*tb_or);

  LitertBuffer buffer(std::move(tb));
  Buffer& buffer_ref = buffer;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LitertBuffer & buffer_as,
                                  buffer_ref.As<LitertBuffer>());
  EXPECT_EQ(&buffer_as, &buffer);
  EXPECT_THAT(buffer_ref.As<SpanCpuBuffer>(), Not(IsOk()));

  const Buffer& const_buffer_ref = buffer;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const LitertBuffer& const_buffer_as,
                                  const_buffer_ref.As<LitertBuffer>());
  EXPECT_EQ(&const_buffer_as, &buffer);
  EXPECT_THAT(const_buffer_ref.As<SpanCpuBuffer>(), Not(IsOk()));
}

TEST(LitertBufferTest, CreateManagedHostAndProperties) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::move(*env_or);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      LitertBuffer::CreateManagedHost(env, {2, 3}, ElementType::Float32,
                                      2 * 3 * sizeof(float)));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto buffer_type, buffer->BufferType());
  EXPECT_EQ(buffer_type, TensorBufferType::kHostMemory);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto size, buffer->Size());
  EXPECT_EQ(size, 2 * 3 * sizeof(float));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto packed_size, buffer->PackedSize());
  EXPECT_EQ(packed_size, 2 * 3 * sizeof(float));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tensor_type, buffer->TensorType());
  EXPECT_EQ(tensor_type.ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor_type.Layout().Rank(), 2);
  EXPECT_EQ(tensor_type.Layout().Dimensions()[0], 2);
  EXPECT_EQ(tensor_type.Layout().Dimensions()[1], 3);

  EXPECT_FALSE(buffer->HasEvent());
}

TEST(LitertBufferTest, CreateFromHostMemory) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::move(*env_or);

  alignas(64) float raw_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  Layout layout(Dimensions({1, 4}));
  RankedTensorType tensor_type(ElementType::Float32, std::move(layout));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto buffer, LitertBuffer::CreateFromHostMemory(
                       env, tensor_type, raw_data, 4 * sizeof(float)));

  {
    auto span = buffer->Lock();
    const auto* ptr = reinterpret_cast<const float*>(span.data());
    EXPECT_EQ(ptr[0], 10.0f);
    EXPECT_EQ(ptr[1], 20.0f);
    EXPECT_EQ(ptr[2], 30.0f);
    EXPECT_EQ(ptr[3], 40.0f);
  }
}

TEST(LitertBufferTest, DuplicateSharesUnderlyingBuffer) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::move(*env_or);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      LitertBuffer::CreateManagedHost(env, {2}, ElementType::Float32,
                                      2 * sizeof(float)));

  {
    auto span = buffer->LockMutable();
    auto* ptr = reinterpret_cast<float*>(span.data());
    ptr[0] = 42.0f;
    ptr[1] = 84.0f;
  }

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto dup_buffer, buffer->Duplicate());

  {
    auto span = dup_buffer->Lock();
    const auto* ptr = reinterpret_cast<const float*>(span.data());
    EXPECT_EQ(ptr[0], 42.0f);
    EXPECT_EQ(ptr[1], 84.0f);
  }
}

}  // namespace
}  // namespace litert::tensor
