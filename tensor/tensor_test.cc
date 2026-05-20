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

#include "tensor/tensor.h"

#include <memory>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::litert::tensor::IsOk;
using ::litert::tensor::IsOkAndHolds;
using ::testing::Address;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Not;
using ::testing::StrEq;

MATCHER(IsValidTensor, "") {
  return ExplainMatchResult(IsOk(), GetStatus(arg.GetRaw()), result_listener);
}

MATCHER_P(LockedPtr, matcher, "") {
  return ExplainMatchResult(matcher, arg.lock(), result_listener);
}

// Helper function for tests. Assumes `IsValidTensor()` returns true.
auto GetInfo(TensorHandle& tensor) { return GetInfo(tensor.GetRaw()); }

TEST(QuantizationTest, CanCastQuantization) {
  PerChannelAffineQuantization per_channel_quantization;
  const Quantization* const_quantization = &per_channel_quantization;
  EXPECT_THAT(const_quantization->As<PerChannelAffineQuantization>(), IsOk());
  static_assert(
      std::is_const_v<std::remove_reference_t<
          decltype(const_quantization->As<PerChannelAffineQuantization>()
                       .value())>>);
  EXPECT_THAT(const_quantization->As<const PerChannelAffineQuantization>(),
              IsOk());
  Quantization* mutable_quantization = &per_channel_quantization;
  EXPECT_THAT(mutable_quantization->As<PerChannelAffineQuantization>(), IsOk());
  EXPECT_THAT(mutable_quantization->As<const PerChannelAffineQuantization>(),
              IsOk());
}

TEST(TensorTest, DefaultConstructedTensorIsValid) {
  Tensor a;
  EXPECT_THAT(a, IsValidTensor());
}

TEST(TensorTest, DefaultConstructedTensorIsNameless) {
  Tensor a;
  EXPECT_THAT(a.GetName(), StrEq(""));
}

TEST(TensorTest, SetNameWorks) {
  Tensor a;
  a.SetName("input1");
  EXPECT_THAT(a.GetName(), StrEq("input1"));
}

TEST(TensorTest, SetNameOnRValueWorks) {
  Tensor a = TensorHandle().SetName("input1");
  EXPECT_THAT(a.GetName(), StrEq("input1"));
}

TEST(TensorTest, SetBufferWorks) {
  auto expected_buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  Tensor a;
  a.SetBuffer(expected_buffer);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, a.GetBuffer());
  EXPECT_THAT(buffer.Lock(), ElementsAreArray(expected_buffer->Lock()));
}

TEST(TensorTest, SetBufferOnRValueWorks) {
  auto expected_buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  Tensor a = TensorHandle().SetBuffer(expected_buffer);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, a.GetBuffer());
  EXPECT_THAT(buffer.Lock(), ElementsAreArray(expected_buffer->Lock()));
}

TEST(TensorTest, DefaultConstructedTensorDontHaveAProducer) {
  Tensor a;
  // The input tensors don't have a producer.
  EXPECT_EQ(a.GetRaw().group->producer, nullptr);
}

TEST(TensorTest, InitConstructorWorks) {
  std::shared_ptr<OwningCpuBuffer> buffer =
      OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  Tensor a(
      {.name = "a", .type = Type::kI32, .shape = {2, 2}, .buffer = buffer});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& a_info,
                                  GetInfo(a));
  EXPECT_EQ(a_info.type, Type::kI32);
  EXPECT_THAT(a_info.shape, ElementsAre(2, 2));
  EXPECT_THAT(a_info.name, StrEq("a"));
  EXPECT_THAT(a_info.buffer, Eq(buffer));
}

TEST(TensorTest, FullyConnectedKeepDims) {
  Tensor input({.type = Type::kFP32, .shape = {2, 2, 3, 4}});
  Tensor weights({.type = Type::kFP32, .shape = {5, 4}});
  Tensor bias({.type = Type::kFP32, .shape = {5}});
  Tensor output = FullyConnected(input, weights, bias, kActNone,
                                 /*keep_num_dims=*/true);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& output_info, GetInfo(output));
  EXPECT_THAT(output_info.shape, ElementsAre(2, 2, 3, 5));
}

TEST(TensorTest, FullyConnectedFlatten) {
  Tensor input({.type = Type::kFP32, .shape = {2, 2, 3, 4}});
  Tensor weights({.type = Type::kFP32, .shape = {5, 4}});
  Tensor bias({.type = Type::kFP32, .shape = {5}});
  Tensor output = FullyConnected(input, weights, bias, kActNone,
                                 /*keep_num_dims=*/false);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& output_info, GetInfo(output));
  EXPECT_THAT(output_info.shape, ElementsAre(2, 5));
}

TEST(TensorTest, SetQuantizationWorks) {
  Tensor a;
  auto quantization = std::make_shared<PerChannelAffineQuantization>();
  quantization->scales = {0.5f};
  quantization->zero_points = {128};
  quantization->quantized_dimension = 1;

  a.SetQuantization(quantization);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& a_info, GetInfo(a));
  EXPECT_THAT(a_info.quantization, Eq(quantization));
}

TEST(TensorTest, ShallowClone) {
  auto buffer = OwningCpuBuffer::Copy<Type::kFP32>({1, 2, 3, 4});
  const Tensor model({.name = "test_tensor",
                      .type = Type::kFP32,
                      .shape = {2, 2, 3, 4},
                      .buffer = buffer});
  const Tensor clone = model.ShallowClone();

  EXPECT_THAT(clone, Not(Eq(model)));
  EXPECT_THAT(clone.GetName(), Eq(model.GetName()));
  EXPECT_THAT(clone.GetType(), Eq(model.GetType()));
  EXPECT_THAT(clone.GetShape(), Eq(model.GetShape()));
  EXPECT_THAT(clone.GetBuffer(), IsOkAndHolds(Address(Eq(buffer.get()))));
}

TEST(TensorTest, ShallowCloneTo) {
  auto buffer = OwningCpuBuffer::Copy<Type::kFP32>({1, 2, 3, 4});
  const Tensor model({.name = "test_tensor",
                      .type = Type::kFP32,
                      .shape = {2, 2, 3, 4},
                      .buffer = buffer});
  const Tensor original_clone;
  Tensor clone = original_clone;
  model.ShallowCloneTo(clone);

  EXPECT_THAT(clone, Eq(original_clone));
  EXPECT_THAT(clone, Not(Eq(model)));
  EXPECT_THAT(clone.GetName(), Eq(model.GetName()));
  EXPECT_THAT(clone.GetType(), Eq(model.GetType()));
  EXPECT_THAT(clone.GetShape(), Eq(model.GetShape()));
  EXPECT_THAT(clone.GetBuffer(), IsOkAndHolds(Address(Eq(buffer.get()))));
}

}  // namespace
}  // namespace litert::tensor
