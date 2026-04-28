// Copyright 2025 Google LLC.
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

#include "litert/test/generators/unary.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"
#include "litert/test/simple_buffer.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert {
namespace testing {
namespace {

using ::litert::internal::GetTflOptions;
using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Types;

template <typename D>
class UnaryTest : public RngTest {};

template <size_t Rank, typename T, LiteRtOpCode OpCode, typename OutputT = T>
using GenForTest = Unary<SizeC<Rank>, T, OpCodeC<OpCode>, SizeC<64>, OutputT>;

template <typename In, typename Out, LiteRtOpCode OpCode>
Out ExpectedUnaryValue(In value) {
  if constexpr (OpCode == kLiteRtOpCodeTflFloor) {
    return FloorReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflCast) {
    return CastReference<In, Out>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflLogistic) {
    return LogisticReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflRelu) {
    return ReluReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflReluN1To1) {
    return ReluN1To1Reference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflRelu6) {
    return Relu6Reference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflLeakyRelu) {
    return LeakyReluReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflTanh) {
    return TanhReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflExp) {
    return ExpReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflNeg) {
    return NegReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflSin) {
    return SinReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflLog) {
    return LogReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflSqrt) {
    return SqrtReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflRsqrt) {
    return RsqrtReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflSquare) {
    return SquareReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflZerosLike) {
    return ZerosLikeReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflAbs) {
    return AbsReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflCeil) {
    return CeilReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflCos) {
    return CosReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflElu) {
    return EluReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflRound) {
    return RoundReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflHardSwish) {
    return HardSwishReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflGelu) {
    return GeluReference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflRelu0To1) {
    return Relu0To1Reference<In>()(value);
  } else if constexpr (OpCode == kLiteRtOpCodeTflSign) {
    return SignReference<In>()(value);
  }
}

template <typename GraphT>
void ExpectReferenceOutput(
    const typename GraphT::Traits::Params& params,
    std::initializer_list<typename GraphT::Traits::template InputDataType<0>>
        input_data,
    std::initializer_list<typename GraphT::Traits::template OutputDataType<0>>
        expected_data) {
  using InputDataType = typename GraphT::Traits::template InputDataType<0>;
  using OutputDataType = typename GraphT::Traits::template OutputDataType<0>;

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, GraphT::Create(params));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input, SimpleBuffer::Create<InputDataType>(params.shape));
  LITERT_ASSERT_OK(input.Write(input_data));

  VarBuffers inputs;
  inputs.push_back(std::move(input));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs,
      SimpleBuffer::LikeSignature(gen->Graph().Subgraph(0).Outputs().begin(),
                                  gen->Graph().Subgraph(0).Outputs().end()));
  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));
  EXPECT_THAT(outputs[0].template Span<OutputDataType>(),
              ElementsAreArray(expected_data));
}

template <typename GraphT>
void ExpectCastOptionsAndTensorTypes(
    const typename GraphT::Traits::Params& params,
    ElementType expected_input_type, ElementType expected_output_type,
    tflite::TensorType expected_in_data_type,
    tflite::TensorType expected_out_data_type) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, GraphT::Create(params));
  const auto& sg = gen->Graph().Subgraph(0);
  EXPECT_EQ(static_cast<ElementType>(
                sg.Tensor(0).Type().second.ranked_tensor_type.element_type),
            expected_input_type);
  EXPECT_EQ(static_cast<ElementType>(
                sg.Tensor(1).Type().second.ranked_tensor_type.element_type),
            expected_output_type);

  const auto& op = sg.Op(0);
  const auto& tfl_opts = GetTflOptions(op);
  ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_CastOptions);
  const auto* options = tfl_opts.AsCastOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(options->in_data_type, expected_in_data_type);
  EXPECT_EQ(options->out_data_type, expected_out_data_type);
}

// clang-format off
using UnaryTypes = Types<
    GenForTest<1, float, kLiteRtOpCodeTflFloor>,
    GenForTest<1, float, kLiteRtOpCodeTflCast, int32_t>,
    GenForTest<1, float, kLiteRtOpCodeTflCast, int64_t>,
    GenForTest<1, float, kLiteRtOpCodeTflLogistic>,
    GenForTest<1, float, kLiteRtOpCodeTflRelu>,
    GenForTest<2, int32_t, kLiteRtOpCodeTflRelu>,
    GenForTest<1, float, kLiteRtOpCodeTflReluN1To1>,
    GenForTest<1, float, kLiteRtOpCodeTflRelu6>,
    GenForTest<1, float, kLiteRtOpCodeTflLeakyRelu>,
    GenForTest<1, float, kLiteRtOpCodeTflTanh>,
    GenForTest<1, float, kLiteRtOpCodeTflExp>,
    GenForTest<1, float, kLiteRtOpCodeTflNeg>,
    GenForTest<1, float, kLiteRtOpCodeTflSin>,
    GenForTest<1, float, kLiteRtOpCodeTflLog>,
    GenForTest<2, float, kLiteRtOpCodeTflSqrt>,
    GenForTest<2, float, kLiteRtOpCodeTflRsqrt>,
    GenForTest<1, float, kLiteRtOpCodeTflSquare>,
    GenForTest<3, int32_t, kLiteRtOpCodeTflZerosLike>,
    GenForTest<3, float, kLiteRtOpCodeTflZerosLike>,
    GenForTest<1, float, kLiteRtOpCodeTflAbs>,
    GenForTest<1, int32_t, kLiteRtOpCodeTflAbs>,
    GenForTest<1, float, kLiteRtOpCodeTflCeil>,
    GenForTest<1, float, kLiteRtOpCodeTflCos>,
    GenForTest<2, float, kLiteRtOpCodeTflElu>,
    GenForTest<1, float, kLiteRtOpCodeTflRound>,
    GenForTest<1, float, kLiteRtOpCodeTflHardSwish>,
    GenForTest<3, float, kLiteRtOpCodeTflGelu>,
    GenForTest<1, float, kLiteRtOpCodeTflRelu0To1>,
    GenForTest<1, float, kLiteRtOpCodeTflSign>,
    GenForTest<1, int32_t, kLiteRtOpCodeTflSign>,
    GenForTest<1, int32_t, kLiteRtOpCodeTflNeg>,
    GenForTest<1, int32_t, kLiteRtOpCodeTflSquare>,
    GenForTest<2, int32_t, kLiteRtOpCodeTflCast, float>,
    GenForTest<2, int32_t, kLiteRtOpCodeTflCast, int64_t>,
    GenForTest<2, int64_t, kLiteRtOpCodeTflCast, int32_t>,
    GenForTest<2, int64_t, kLiteRtOpCodeTflCast, float>
>;
// clang-format on

TYPED_TEST_SUITE(UnaryTest, UnaryTypes);

TYPED_TEST(UnaryTest, TestLogic) {
  using Traits = typename TypeParam::Traits;
  using DataType = typename Traits::template InputDataType<0>;
  using OutputDataType = typename Traits::template OutputDataType<0>;
  auto device = this->TracedDevice();
  typename Traits::Params params;
  params.shape.fill(2);
  const auto rank = params.shape.size();

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, TypeParam::Create(params));
  EXPECT_EQ(gen->Graph().NumSubgraphs(), 1);
  const auto& sg = gen->Graph().Subgraph(0);
  EXPECT_EQ(sg.Ops().size(), 1);
  EXPECT_EQ(sg.Tensors().size(), 2);
  EXPECT_EQ(sg.Inputs().size(), 1);
  EXPECT_EQ(sg.Outputs().size(), 1);

  const auto& tensor1_type = sg.Tensor(0).Type().second.ranked_tensor_type;
  EXPECT_EQ(tensor1_type.layout.rank, rank);
  EXPECT_THAT(absl::MakeConstSpan(tensor1_type.layout.dimensions, rank),
              Each(Eq(2)));

  const auto& tensor2_type = sg.Tensor(1).Type().second.ranked_tensor_type;
  EXPECT_EQ(static_cast<ElementType>(tensor2_type.element_type),
            TypeParam::kOutputElementTypeValue);
  EXPECT_THAT(absl::MakeConstSpan(tensor2_type.layout.dimensions, rank),
              Each(Eq(2)));

  RandomTensorDataBuilder data_builder;
  data_builder.SetIntDummy();
  data_builder.SetFloatDummy();
  LITERT_ASSERT_OK_AND_ASSIGN(const auto inputs,
                              gen->MakeInputs(device, data_builder));
  EXPECT_EQ(inputs.size(), 1);

  auto input = inputs[0].template AsView<DataType>();

  std::vector<DataType> expected_input_data(input.NumElements());
  if constexpr (std::is_same_v<DataType, float> ||
                std::is_same_v<DataType, int32_t>) {
    std::iota(expected_input_data.begin(), expected_input_data.end(), 0);
    EXPECT_THAT(input.data, ElementsAreArray(expected_input_data));
  } else {
    expected_input_data.assign(input.data.begin(), input.data.end());
  }

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs,
      SimpleBuffer::LikeSignature(sg.Outputs().begin(), sg.Outputs().end()));
  EXPECT_EQ(outputs.size(), 1);

  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  std::vector<OutputDataType> expected_output_data(input.NumElements());
  std::transform(expected_input_data.begin(), expected_input_data.end(),
                 expected_output_data.begin(), [](DataType value) {
                   return ExpectedUnaryValue<DataType, OutputDataType,
                                             TypeParam::kOperationCode>(value);
                 });
  EXPECT_THAT(outputs[0].template Span<OutputDataType>(),
              ElementsAreArray(expected_output_data));
}

TEST(UnaryGeneratorTest, LeakyReluUsesConfiguredAlpha) {
  using GraphT = GenForTest<1, float, kLiteRtOpCodeTflLeakyRelu>;
  GraphT::Traits::Params params;
  params.shape = {4};

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, GraphT::Create(params));
  const auto& op = gen->Graph().Subgraph(0).Op(0);
  const auto& tfl_opts = GetTflOptions(op);
  ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_LeakyReluOptions);
  const auto* options = tfl_opts.AsLeakyReluOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_FLOAT_EQ(options->alpha, 0.25f);

  ExpectReferenceOutput<GraphT>(params, {-4.0f, -2.0f, 0.0f, 8.0f},
                                {-1.0f, -0.5f, 0.0f, 8.0f});
}

TEST(UnaryGeneratorTest, CastUsesDistinctTensorTypesAndTruncatesTowardZero) {
  using GraphT = GenForTest<1, float, kLiteRtOpCodeTflCast, int32_t>;
  GraphT::Traits::Params params;
  params.shape = {4};

  ExpectCastOptionsAndTensorTypes<GraphT>(
      params, ElementType::Float32, ElementType::Int32,
      tflite::TensorType_FLOAT32, tflite::TensorType_INT32);

  ExpectReferenceOutput<GraphT>(params, {-1.75f, -0.25f, 1.5f, 2.9f},
                                {-1, 0, 1, 2});
}

TEST(UnaryGeneratorTest, CastCoversFloatAndInt64Endpoints) {
  {
    using GraphT = GenForTest<1, float, kLiteRtOpCodeTflCast, int64_t>;
    GraphT::Traits::Params params;
    params.shape = {4};
    ExpectCastOptionsAndTensorTypes<GraphT>(
        params, ElementType::Float32, ElementType::Int64,
        tflite::TensorType_FLOAT32, tflite::TensorType_INT64);
    ExpectReferenceOutput<GraphT>(params, {-7.9f, -0.25f, 3.5f, 9.0f},
                                  {-7, 0, 3, 9});
  }
  {
    using GraphT = GenForTest<1, int64_t, kLiteRtOpCodeTflCast, float>;
    GraphT::Traits::Params params;
    params.shape = {4};
    ExpectCastOptionsAndTensorTypes<GraphT>(
        params, ElementType::Int64, ElementType::Float32,
        tflite::TensorType_INT64, tflite::TensorType_FLOAT32);
    ExpectReferenceOutput<GraphT>(params, {-7, -1, 0, 9},
                                  {-7.0f, -1.0f, 0.0f, 9.0f});
  }
}

TEST(UnaryGeneratorTest, CastCoversInt32AndInt64WidthChanges) {
  {
    using GraphT = GenForTest<1, int32_t, kLiteRtOpCodeTflCast, int64_t>;
    GraphT::Traits::Params params;
    params.shape = {4};
    ExpectCastOptionsAndTensorTypes<GraphT>(
        params, ElementType::Int32, ElementType::Int64,
        tflite::TensorType_INT32, tflite::TensorType_INT64);
    ExpectReferenceOutput<GraphT>(params, {-7, -1, 0, 9}, {-7, -1, 0, 9});
  }
  {
    using GraphT = GenForTest<1, int64_t, kLiteRtOpCodeTflCast, int32_t>;
    GraphT::Traits::Params params;
    params.shape = {4};
    ExpectCastOptionsAndTensorTypes<GraphT>(
        params, ElementType::Int64, ElementType::Int32,
        tflite::TensorType_INT64, tflite::TensorType_INT32);
    ExpectReferenceOutput<GraphT>(params, {-7, -1, 0, 9}, {-7, -1, 0, 9});
  }
}

TEST(UnaryGeneratorTest, IntegerUnaryOpsHandleNegativeInputs) {
  {
    using GraphT = GenForTest<1, int32_t, kLiteRtOpCodeTflAbs>;
    GraphT::Traits::Params params;
    params.shape = {4};
    ExpectReferenceOutput<GraphT>(params, {-3, -1, 0, 2}, {3, 1, 0, 2});
  }
  {
    using GraphT = GenForTest<1, int32_t, kLiteRtOpCodeTflNeg>;
    GraphT::Traits::Params params;
    params.shape = {4};
    ExpectReferenceOutput<GraphT>(params, {-3, -1, 0, 2}, {3, 1, 0, -2});
  }
  {
    using GraphT = GenForTest<1, int32_t, kLiteRtOpCodeTflSquare>;
    GraphT::Traits::Params params;
    params.shape = {4};
    ExpectReferenceOutput<GraphT>(params, {-3, -1, 0, 2}, {9, 1, 0, 4});
  }
  {
    using GraphT = GenForTest<1, int32_t, kLiteRtOpCodeTflSign>;
    GraphT::Traits::Params params;
    params.shape = {4};
    ExpectReferenceOutput<GraphT>(params, {-3, -1, 0, 2}, {-1, -1, 0, 1});
  }
}

}  // namespace
}  // namespace testing
}  // namespace litert
