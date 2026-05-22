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

#include "tensor/arithmetic.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/optional.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::litert::tensor::IsOk;
using ::litert::tensor::IsOkAndHolds;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

MATCHER(IsValidTensor, "") {
  return ExplainMatchResult(IsOk(), GetStatus(arg.GetRaw()), result_listener);
}

MATCHER_P(LockedPtr, matcher, "") {
  return ExplainMatchResult(matcher, arg.lock(), result_listener);
}

// Helper function for tests. Assumes `IsValidTensor()` returns true.
auto GetConsumers(TensorHandle& tensor) {
  return GetConsumers(tensor.GetRaw());
}

// Helper function for tests. Assumes `IsValidTensor()` returns true.
auto GetProducer(TensorHandle& tensor) { return GetProducer(tensor.GetRaw()); }

// Helper function for tests. Assumes `IsValidTensor()` returns true.
auto GetInfo(TensorHandle& tensor) { return GetInfo(tensor.GetRaw()); }

TEST(ArithmeticTest, ChainingOpsKeepsTrackOfProducersAndConsumers) {
  Tensor a, b, c;
  Tensor d = Mul(a, b);
  auto e = Add(c, d);

  // The Mul op is the producer of d...
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr mul_op, GetProducer(d));
  EXPECT_NE(mul_op, nullptr);
  // ... and a consumer of a and b.
  EXPECT_THAT(GetConsumers(a), IsOkAndHolds(Contains(LockedPtr(mul_op))));
  EXPECT_THAT(GetConsumers(b), IsOkAndHolds(Contains(LockedPtr(mul_op))));

  // The Add op is the producer of e...
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr add_op, GetProducer(e));
  EXPECT_NE(add_op, nullptr);
  // ... and a consumer of c and d.
  EXPECT_THAT(GetConsumers(c), IsOkAndHolds(Contains(LockedPtr(add_op))));
  EXPECT_THAT(GetConsumers(d), IsOkAndHolds(Contains(LockedPtr(add_op))));
}

void IsAUnaryElementwiseOp(absl::string_view op_name, TensorHandle in,
                           TensorHandle out) {
  ASSERT_THAT(in, IsValidTensor());
  ASSERT_THAT(out, IsValidTensor());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& in_info,
                                  GetInfo(in));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& out_info,
                                  GetInfo(out));
  EXPECT_EQ(out_info.type, in_info.type);
  EXPECT_EQ(out_info.shape, in_info.shape);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr op, GetProducer(out));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->GetName(), op_name);
  EXPECT_THAT(op->inputs, UnorderedElementsAre(in.GetRaw()));
  EXPECT_THAT(op->outputs_group.lock(), out.GetRaw().group);
  EXPECT_THAT(GetConsumers(in), IsOkAndHolds(Contains(LockedPtr(op))));
}

void IsABinaryElementwiseOp(absl::string_view op_name, TensorHandle in_1,
                            TensorHandle in_2, TensorHandle out,
                            absl::optional<Type>
                              expected_output_type = absl::nullopt) {
  ASSERT_THAT(in_1, IsValidTensor());
  ASSERT_THAT(in_2, IsValidTensor());
  ASSERT_THAT(out, IsValidTensor());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& in_1_info,
                                  GetInfo(in_1));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& out_info,
                                  GetInfo(out));
  EXPECT_EQ(out_info.type, expected_output_type.has_value()
                               ? *expected_output_type
                               : in_1_info.type);
  EXPECT_EQ(out_info.shape, in_1_info.shape);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr op, GetProducer(out));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->GetName(), op_name);
  EXPECT_THAT(op->inputs, UnorderedElementsAre(in_1.GetRaw(), in_2.GetRaw()));
  EXPECT_THAT(op->outputs_group.lock(), out.GetRaw().group);
  ASSERT_THAT(GetConsumers(in_1), IsOkAndHolds(Contains(LockedPtr(op))));
  ASSERT_THAT(GetConsumers(in_2), IsOkAndHolds(Contains(LockedPtr(op))));
}

TEST(ArithmeticTest, AddWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Add", a, b, Add(a, b)));
}

TEST(ArithmeticTest, MulWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Mul", a, b, Mul(a, b)));
}

TEST(ArithmeticTest, AbsWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Abs", a, Abs(a)));
}

TEST(ArithmeticTest, ReluWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Relu", a, Relu(a)));
}

TEST(ArithmeticTest, Relu6Works) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Relu6", a, Relu6(a)));
}

TEST(ArithmeticTest, LeakyReluWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("LeakyRelu", a, LeakyRelu(a)));
}

TEST(ArithmeticTest, EluWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Elu", a, Elu(a)));
}

TEST(ArithmeticTest, HardSwishWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("HardSwish", a, HardSwish(a)));
}

TEST(ArithmeticTest, PReluWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  Tensor b({.type = Type::kFP32, .shape = {3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("PRelu", a, b, PRelu(a, b)));
}

TEST(ArithmeticTest, L2NormalizationWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp(
      "L2Normalization", a, L2Normalization(a)));
}

TEST(ArithmeticTest, SubWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Sub", a, b, Sub(a, b)));
}

TEST(ArithmeticTest, DivWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Div", a, b, Div(a, b)));
}

TEST(ArithmeticTest, SquareWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Square", a, Square(a)));
}

TEST(ArithmeticTest, RsqrtWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Rsqrt", a, Rsqrt(a)));
}

TEST(ArithmeticTest, PowWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Pow", a, b, Pow(a, b)));
}

TEST(ArithmeticTest, NegWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Neg", a, Neg(a)));
}

TEST(ArithmeticTest, SqrtWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Sqrt", a, Sqrt(a)));
}

TEST(ArithmeticTest, ExpWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Exp", a, Exp(a)));
}

TEST(ArithmeticTest, LogWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Log", a, Log(a)));
}

TEST(ArithmeticTest, CeilWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Ceil", a, Ceil(a)));
}

TEST(ArithmeticTest, FloorWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Floor", a, Floor(a)));
}

TEST(ArithmeticTest, FloorDivWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp(
      "FloorDiv", a, b, FloorDiv(a, b)));
}

TEST(ArithmeticTest, FloorModWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp(
      "FloorMod", a, b, FloorMod(a, b)));
}

TEST(ArithmeticTest, SignWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Sign", a, Sign(a)));
}

TEST(ArithmeticTest, RoundWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Round", a, Round(a)));
}

TEST(ArithmeticTest, TransposeWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3, 4}});
  const std::vector<int32_t> p_data = {2, 0, 1};
  Tensor p({.type = Type::kI32, .shape = {3}, .buffer = p_data});
  Tensor b = Transpose(a, p);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(4, 2, 3));
}

TEST(ArithmeticTest, TransposeWithVectorWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3, 4}});
  Tensor b = Transpose(a, {2, 0, 1});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(4, 2, 3));
}

TEST(ArithmeticTest, TransposeConv2DWorks) {
  Tensor filter({.type = Type::kFP32, .shape = {1, 3, 3, 1}});
  Tensor input({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  Tensor bias({.type = Type::kFP32, .shape = {1}});
  Tensor b =
      TransposeConv2D(filter, input, bias, {1, 10, 10, 1}, kPaddingSame, 2, 2);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 10, 10, 1));
}

TEST(ArithmeticTest, SoftmaxWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Softmax(a, 1.0);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, UnorderedElementsAre(2, 4));
}

TEST(ArithmeticTest, LogSoftmaxWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = LogSoftmax(a);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, UnorderedElementsAre(2, 4));
}

TEST(ArithmeticTest, AveragePool2DWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  Tensor b = AveragePool2D(a, /*filter_height=*/2, /*filter_width=*/2,
                           /*stride_h=*/2, /*stride_w=*/2, kPaddingSame);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 3, 3, 1));
}

TEST(ArithmeticTest, MaxPool2DWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  Tensor b = MaxPool2D(a, /*filter_height=*/2, /*filter_width=*/2,
                       /*stride_h=*/2, /*stride_w=*/2, kPaddingSame);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 3, 3, 1));
}

TEST(ArithmeticTest, FullyConnectedWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 8}});
  Tensor w({.type = Type::kFP32, .shape = {4, 8}});
  Tensor bias({.type = Type::kFP32, .shape = {4}});
  Tensor b = FullyConnected(a, w, bias, kActNone, false);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, FullyConnectedWithoutBiasWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 8}});
  Tensor w({.type = Type::kFP32, .shape = {4, 8}});
  Tensor b = FullyConnected(a, w, kActNone, false);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr op, GetProducer(b));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->inputs.size(), 2);
}

TEST(ArithmeticTest, Conv2DWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  Tensor filter({.type = Type::kFP32, .shape = {1, 3, 3, 1}});
  Tensor bias({.type = Type::kFP32, .shape = {1}});
  Tensor b =
      Conv2D(a, filter, bias, /*stride_h=*/2, /*stride_w=*/2, kPaddingSame);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 3, 3, 1));
}

TEST(ArithmeticTest, DepthwiseConv2DWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  Tensor filter({.type = Type::kFP32, .shape = {1, 3, 3, 1}});
  Tensor bias({.type = Type::kFP32, .shape = {1}});
  Tensor b = DepthwiseConv2D(a, filter, bias, 2, 2, kPaddingSame);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 3, 3, 1));
}

TEST(ArithmeticTest, PadWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {0, 0, 1, 1, 2, 2, 0, 0};
  Tensor p({.type = Type::kI32, .shape = {4, 2}, .buffer = p_data});
  Tensor b = Pad(a, p);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 4, 7, 1));
}

TEST(ArithmeticTest, PadV2Works) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const int32_t p_data[] = {0, 0, 1, 1, 2, 2, 0, 0};
  Tensor p({.type = Type::kI32,
            .shape = {4, 2},
            .buffer = OwningCpuBuffer::Copy<Type::kI32>(p_data)});
  Tensor c({.type = Type::kFP32,
            .shape = {1},
            .buffer = OwningCpuBuffer::Copy<Type::kFP32>({0.0f})});
  Tensor b = PadV2(a, p, c);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 4, 7, 1));
}

TEST(ArithmeticTest, SumKeepDimsWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {1, 2};
  Tensor p({.type = Type::kI32, .shape = {2}, .buffer = p_data});
  Tensor b = Sum(a, p, true);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 1, 1, 1));
}

TEST(ArithmeticTest, SumWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {1, 2};
  Tensor p({.type = Type::kI32, .shape = {2}, .buffer = p_data});
  Tensor b = Sum(a, p, false);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 1));
}

TEST(ArithmeticTest, ReduceMaxKeepDimsWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {1, 2};
  Tensor p({.type = Type::kI32, .shape = {2}, .buffer = p_data});
  Tensor b = ReduceMax(a, p, true);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 1, 1, 1));
}

TEST(ArithmeticTest, ReduceMaxWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {1, 2};
  Tensor p({.type = Type::kI32, .shape = {2}, .buffer = p_data});
  Tensor b = ReduceMax(a, p, false);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 1));
}

TEST(ArithmeticTest, MeanKeepDimsWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {1, 2};
  Tensor p({.type = Type::kI32, .shape = {2}, .buffer = p_data});
  Tensor b = Mean(a, p, true);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 1, 1, 1));
}

TEST(ArithmeticTest, MeanWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {1, 2};
  Tensor p({.type = Type::kI32, .shape = {2}, .buffer = p_data});
  Tensor b = Mean(a, p, false);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 1));
}

TEST(ArithmeticTest, BatchMatMulWorks) {
  Tensor x({.type = Type::kFP32, .shape = {2, 3, 4}});
  Tensor y({.type = Type::kFP32, .shape = {2, 4, 5}});
  Tensor z = BatchMatMul(x, y, false, false);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& z_info, GetInfo(z));
  EXPECT_THAT(z_info.shape, ElementsAre(2, 3, 5));
}

TEST(ArithmeticTest, BatchMatMulFailsWithMismatchedDimensions) {
  Tensor x({.type = Type::kFP32, .shape = {2, 3, 4}});
  Tensor y({.type = Type::kFP32, .shape = {2, 5, 5}});
  Tensor z = BatchMatMul(x, y, false, false);
  EXPECT_THAT(GetStatus(z.GetRaw()), ::testing::Not(IsOk()));
}

TEST(ArithmeticTest, ConcatenationWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3}});
  Tensor b({.type = Type::kFP32, .shape = {2, 3}});
  Tensor c = Concatenation({a, b}, 1, kActNone);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 6));
}

TEST(ArithmeticTest, UnpackWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3, 4}});
  std::vector<Tensor<>> b = Unpack(a, 2, 0);
  ASSERT_EQ(b.size(), 2);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b0_info, GetInfo(b[0]));
  EXPECT_THAT(b0_info.shape, ElementsAre(3, 4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b1_info, GetInfo(b[1]));
  EXPECT_THAT(b1_info.shape, ElementsAre(3, 4));
}

TEST(ArithmeticTest, SplitWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4, 6}});
  std::vector<Tensor<>> b = Split(a, 1, 2);
  ASSERT_EQ(b.size(), 2);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b0_info, GetInfo(b[0]));
  EXPECT_THAT(b0_info.shape, ElementsAre(2, 2, 6));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b1_info, GetInfo(b[1]));
  EXPECT_THAT(b1_info.shape, ElementsAre(2, 2, 6));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr op, GetProducer(b[0]));
  ASSERT_NE(op, nullptr);
  auto* split_op = dynamic_cast<graph::SplitOperation<>*>(op.get());
  ASSERT_NE(split_op, nullptr);
  EXPECT_EQ(split_op->num_splits, 2);
}

TEST(ArithmeticTest, GeluWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Gelu(a, false);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, TanhWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Tanh", a, Tanh(a)));
}

TEST(ArithmeticTest, CastWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Cast(a, Type::kI32);
}

TEST(ArithmeticTest, SelectWorks) {
  Tensor condition({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = Select(condition, a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, SelectV2Works) {
  Tensor condition({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = SelectV2(condition, a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, SliceWorks) {
  Tensor a({.type = Type::kFP32, .shape = {4, 4}});
  Tensor b = Slice(a, {1, 1}, {2, 2});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 2));
}

TEST(ArithmeticTest, LessWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = Less(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, GreaterWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = Greater(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, GreaterEqualWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = GreaterEqual(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, NotEqualWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = NotEqual(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, MinimumWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = Minimum(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, MaximumWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = Maximum(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, LogicalAndWorks) {
  Tensor a({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor b({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor c = LogicalAnd(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, LogicalOrWorks) {
  Tensor a({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor b({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor c = LogicalOr(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, BitwiseXorWorks) {
  Tensor a({.type = Type::kI32, .shape = {2, 4}});
  Tensor b({.type = Type::kI32, .shape = {2, 4}});
  Tensor c = BitwiseXor(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, RightShiftWorks) {
  Tensor a({.type = Type::kI32, .shape = {2, 4}});
  Tensor b({.type = Type::kI32, .shape = {2, 4}});
  Tensor c = RightShift(a, b);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, CosWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Cos(a);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, SinWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Sin(a);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, ReshapeWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3}});
  Tensor b = Reshape(a, {3, 2, 1});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(3, 2, 1));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, ExpandDimsWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3}});
  Tensor b = ExpandDims(a, 1);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 1, 2, 3));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, ExpandDimsNegativeAxisWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3}});
  Tensor b = ExpandDims(a, -1);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 2, 3, 1));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, SqueezeWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 1, 3, 1}});
  Tensor b = Squeeze(a);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 3));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, SqueezeSpecificDimsWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 1, 3, 1}});
  Tensor b = Squeeze(a, {0, -1});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 1, 3));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, LogisticWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Logistic(a);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, UnorderedElementsAre(2, 4));
}

TEST(ArithmeticTest, EmbeddingLookupWorks) {
  Tensor value({.type = Type::kFP32, .shape = {10, 4}});
  Tensor lookup({.type = Type::kI32, .shape = {2}});
  Tensor result = EmbeddingLookup(lookup, value);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& result_info, GetInfo(result));
  EXPECT_THAT(result_info.shape, ElementsAre(2, 4));
  EXPECT_EQ(result_info.type, Type::kFP32);
}

TEST(ArithmeticTest, EmbeddingLookupWithVectorWorks) {
  Tensor value({.type = Type::kFP32, .shape = {10, 4}});
  Tensor result = EmbeddingLookup({1, 2}, value);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& result_info, GetInfo(result));
  EXPECT_THAT(result_info.shape, ElementsAre(2, 4));
  EXPECT_EQ(result_info.type, Type::kFP32);
}

TEST(ArithmeticTest, DynamicUpdateSliceWorks) {
  Tensor operand({.type = Type::kFP32, .shape = {10, 10}});
  Tensor update({.type = Type::kFP32, .shape = {2, 2}});
  Tensor start_indices({.type = Type::kI32, .shape = {2}});
  Tensor result = DynamicUpdateSlice(operand, update, start_indices);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& result_info, GetInfo(result));
  EXPECT_THAT(result_info.shape, ElementsAre(10, 10));
  EXPECT_EQ(result_info.type, Type::kFP32);
}

TEST(ArithmeticTest, DynamicUpdateSliceWithVectorWorks) {
  Tensor operand({.type = Type::kFP32, .shape = {10, 10}});
  Tensor update({.type = Type::kFP32, .shape = {2, 2}});
  Tensor result = DynamicUpdateSlice(operand, update, {0, 0});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& result_info, GetInfo(result));
  EXPECT_THAT(result_info.shape, ElementsAre(10, 10));
  EXPECT_EQ(result_info.type, Type::kFP32);
}

TEST(ArithmeticTest, CustomWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  std::vector<Tensor<>> outputs =
      Custom({a}, "MyCustomOp", {1, 2, 3}, {{2, 4}}, {Type::kFP32});
  ASSERT_EQ(outputs.size(), 1);
  Tensor b = outputs[0];
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, UnorderedElementsAre(2, 4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr op, GetProducer(b));
  ASSERT_NE(op, nullptr);
  auto* custom_op = dynamic_cast<graph::CustomOperationData*>(op.get());
  ASSERT_NE(custom_op, nullptr);
  EXPECT_EQ(custom_op->custom_code, "MyCustomOp");
  EXPECT_THAT(custom_op->custom_options, ElementsAre(1, 2, 3));
}

TEST(ArithmeticTest, TileWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3}});
  const std::vector<int32_t> multiples_data = {2, 1};
  Tensor multiples(
      {.type = Type::kI32, .shape = {2}, .buffer = multiples_data});
  Tensor b = Tile(a, multiples);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(4, 3));
}

TEST(ArithmeticTest, TileWithVectorWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3}});
  Tensor b = Tile(a, {2, 1});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(4, 3));
}

TEST(ArithmeticTest, TopKWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 8}});
  std::vector<Tensor<>> outputs = TopK(a, 2);
  ASSERT_EQ(outputs.size(), 2);
  Tensor values = outputs[0];
  Tensor indices = outputs[1];
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& values_info, GetInfo(values));
  EXPECT_THAT(values_info.shape, ElementsAre(1, 2));
  EXPECT_EQ(values_info.type, Type::kFP32);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& indices_info, GetInfo(indices));
  EXPECT_THAT(indices_info.shape, ElementsAre(1, 2));
  EXPECT_EQ(indices_info.type, Type::kI32);
}

TEST(ArithmeticTest, TopKWorksWithDifferentShapes) {
  Tensor a({.type = Type::kFP32, .shape = {3, 5}});
  std::vector<Tensor<>> outputs = TopK(a, 3);
  ASSERT_EQ(outputs.size(), 2);
  Tensor values = outputs[0];
  Tensor indices = outputs[1];
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& values_info, GetInfo(values));
  EXPECT_THAT(values_info.shape, ElementsAre(3, 3));
  EXPECT_EQ(values_info.type, Type::kFP32);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& indices_info, GetInfo(indices));
  EXPECT_THAT(indices_info.shape, ElementsAre(3, 3));
  EXPECT_EQ(indices_info.type, Type::kI32);
}

TEST(ArithmeticTest, ArgMaxWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3, 4}});
  Tensor b = ArgMax(a, 1);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
  EXPECT_EQ(b_info.type, Type::kI64);
}

TEST(ArithmeticTest, ArgMaxNegativeAxisWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3, 4}});
  Tensor b = ArgMax(a, -1);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 3));
  EXPECT_EQ(b_info.type, Type::kI64);
}

TEST(ArithmeticTest, SpaceToDepthWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 4, 4, 1}});
  Tensor b = SpaceToDepth(a, 2);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 2, 2, 4));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, DepthToSpaceWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 2, 4}});
  Tensor b = DepthToSpace(a, 2);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 4, 4, 1));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, CumsumWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor axis({.type = Type::kI32,
               .shape = {},
               .buffer = OwningCpuBuffer::Copy<Type::kI32>({1})});
  Tensor b = Cumsum(a, axis, /*exclusive=*/true, /*reverse=*/true);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, ReverseWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor axes({.type = Type::kI32,
               .shape = {1},
               .buffer = OwningCpuBuffer::Copy<Type::kI32>({1})});
  Tensor b = Reverse(a, axes);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, GatherWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor indices({.type = Type::kI32, .shape = {2}});
  Tensor b = Gather(a, indices, 0);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, GatherShapeWorks) {
  Tensor a({.type = Type::kFP32, .shape = {5, 6, 7}});
  Tensor indices({.type = Type::kI32, .shape = {3, 4}});
  Tensor b = Gather(a, indices, 1);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(5, 3, 4, 7));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, GatherBatchDimsWorks) {
  Tensor a({.type = Type::kFP32, .shape = {5, 6, 7}});
  Tensor indices({.type = Type::kI32, .shape = {5, 4}});
  Tensor b = Gather(a, indices, 1, 1);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(5, 4, 7));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, GatherNdWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 2, 2}});
  Tensor indices({.type = Type::kI32, .shape = {2, 2}});
  Tensor b = GatherNd(a, indices);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 2));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, OneHotWorks) {
  Tensor indices({.type = Type::kI32, .shape = {4}});
  Tensor depth(
      {.type = Type::kI32, .shape = {}, .buffer = std::vector<int32_t>{3}});
  Tensor on_value(
      {.type = Type::kFP32, .shape = {}, .buffer = std::vector<float>{1.0f}});
  Tensor off_value(
      {.type = Type::kFP32, .shape = {}, .buffer = std::vector<float>{0.0f}});
  Tensor output = OneHot(indices, depth, on_value, off_value, -1);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& out_info, GetInfo(output));
  EXPECT_THAT(out_info.shape, ElementsAre(4, 3));
  EXPECT_EQ(out_info.type, Type::kFP32);
}

TEST(ArithmeticTest, NonMaxSuppressionV5Works) {
  Tensor boxes({.type = Type::kFP32, .shape = {1, 6, 4}});
  Tensor scores({.type = Type::kFP32, .shape = {1, 6}});
  Tensor max_output_size({.type = Type::kI32, .shape = {},
    .buffer = std::vector<int32_t>{6}});
  Tensor iou_threshold({.type = Type::kFP32, .shape = {}});
  Tensor score_threshold({.type = Type::kFP32, .shape = {}});
  Tensor soft_nms_sigma({.type = Type::kFP32, .shape = {}});

  auto outputs =
      NonMaxSuppressionV5(boxes, scores, max_output_size, iou_threshold,
                          score_threshold, soft_nms_sigma);
  auto selected_indices = outputs[0];
  auto selected_scores = outputs[1];
  auto valid_outputs = outputs[2];

  ASSERT_THAT(selected_indices, IsValidTensor());
  ASSERT_THAT(selected_scores, IsValidTensor());
  ASSERT_THAT(valid_outputs, IsValidTensor());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& indices_info,
                                  GetInfo(selected_indices));
  EXPECT_EQ(indices_info.type, Type::kI32);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& scores_info,
                                  GetInfo(selected_scores));
  EXPECT_EQ(scores_info.type, Type::kFP32);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& valid_outputs_info,
                                  GetInfo(valid_outputs));
  EXPECT_EQ(valid_outputs_info.type, Type::kI32);
  EXPECT_THAT(valid_outputs_info.shape, testing::IsEmpty());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr op,
                                  GetProducer(selected_indices));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->GetName(), "NonMaxSuppressionV5");
  EXPECT_THAT(
      op->inputs,
      UnorderedElementsAre(boxes.GetRaw(), scores.GetRaw(),
                           max_output_size.GetRaw(), iou_threshold.GetRaw(),
                           score_threshold.GetRaw(), soft_nms_sigma.GetRaw()));

  // Verify all outputs come from the same op
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr op_scores,
                                  GetProducer(selected_scores));
  ASSERT_EQ(op, op_scores);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::shared_ptr op_valid,
                                  GetProducer(valid_outputs));
  ASSERT_EQ(op, op_valid);
}

TEST(ArithmeticTest, NonMaxSuppressionV5ShapeInference) {
  Tensor boxes({.type = Type::kFP32, .shape = {1, 6, 4}});
  Tensor scores({.type = Type::kFP32, .shape = {1, 6}});
  Tensor max_output_size(
      {.type = Type::kI32, .shape = {}, .buffer = std::vector<int32_t>{3}});
  Tensor iou_threshold({.type = Type::kFP32, .shape = {}});
  Tensor score_threshold({.type = Type::kFP32, .shape = {}});
  Tensor soft_nms_sigma({.type = Type::kFP32, .shape = {}});

  auto outputs =
      NonMaxSuppressionV5(boxes, scores, max_output_size, iou_threshold,
                          score_threshold, soft_nms_sigma);
  auto selected_indices = outputs[0];
  auto selected_scores = outputs[1];

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& indices_info,
                                  GetInfo(selected_indices));
  EXPECT_THAT(indices_info.shape, ElementsAre(3));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& scores_info,
                                  GetInfo(selected_scores));
  EXPECT_THAT(scores_info.shape, ElementsAre(3));
}

TEST(ArithmeticTest, NonMaxSuppressionV5IntOverload) {
  Tensor boxes({.type = Type::kFP32, .shape = {1, 6, 4}});
  Tensor scores({.type = Type::kFP32, .shape = {1, 6}});
  Tensor iou_threshold({.type = Type::kFP32, .shape = {}});
  Tensor score_threshold({.type = Type::kFP32, .shape = {}});
  Tensor soft_nms_sigma({.type = Type::kFP32, .shape = {}});

  auto outputs = NonMaxSuppressionV5(boxes, scores, 3, iou_threshold,
                                     score_threshold, soft_nms_sigma);
  auto selected_indices = outputs[0];
  auto selected_scores = outputs[1];

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& indices_info,
                                  GetInfo(selected_indices));
  EXPECT_THAT(indices_info.shape, ElementsAre(3));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& scores_info,
                                  GetInfo(selected_scores));
  EXPECT_THAT(scores_info.shape, ElementsAre(3));
}

TEST(ArithmeticTest, NonMaxSuppressionV5DynamicMaxOutputSizeFails) {
  Tensor boxes({.type = Type::kFP32, .shape = {2, 10, 4}});
  Tensor scores({.type = Type::kFP32, .shape = {2, 10}});
  // max_output_size is not a constant tensor (no buffer).
  Tensor max_output_size({.type = Type::kI32, .shape = {}});
  Tensor iou_threshold({.type = Type::kFP32, .shape = {}});
  Tensor score_threshold({.type = Type::kFP32, .shape = {}});
  Tensor soft_nms_sigma({.type = Type::kFP32, .shape = {}});

  auto outputs =
      NonMaxSuppressionV5(boxes, scores, max_output_size, iou_threshold,
                          score_threshold, soft_nms_sigma);
  ASSERT_EQ(outputs.size(), 3);
  for (const auto& output : outputs) {
    ASSERT_TRUE(!output.GetStatus().ok());
    EXPECT_THAT(output.GetStatus().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(
        output.GetStatus().message(),
        ::testing::HasSubstr("max_output_size must be a constant tensor"));
  }
}

}  // namespace
}  // namespace litert::tensor
