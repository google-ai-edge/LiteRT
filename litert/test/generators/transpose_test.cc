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

#include "litert/test/generators/transpose.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"
#include "litert/test/simple_buffer.h"
#include "tflite/kernels/internal/portable_tensor_utils.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Types;
using ::testing::litert::HasTypeAspect;
using ::litert::internal::GetTflOptions;

template <typename D>
class TransposeTest : public RngTest {};

template <size_t Rank, typename T>
using GenForTest = Transpose<SizeC<Rank>, T, SizeC<64>>;

template <typename T, size_t Rank>
std::vector<T> ExpectedTranspose(
    const std::array<Layout::Dim, Rank>& shape,
    const std::array<int32_t, Rank>& permutation, absl::Span<const T> input) {
  std::array<Layout::Dim, Rank> output_shape;
  for (size_t i = 0; i < Rank; ++i) {
    output_shape[i] = shape[permutation[i]];
  }

  auto make_strides = [](absl::Span<const Layout::Dim> dims) {
    std::array<size_t, Rank> strides;
    size_t stride = 1;
    for (size_t i = Rank; i-- > 0;) {
      strides[i] = stride;
      stride *= dims[i];
    }
    return strides;
  };

  const auto input_strides = make_strides(shape);
  const auto output_strides = make_strides(output_shape);
  std::vector<T> output(input.size());
  std::array<size_t, Rank> output_indices;
  std::array<size_t, Rank> input_indices;

  for (size_t output_offset = 0; output_offset < output.size();
       ++output_offset) {
    size_t remaining = output_offset;
    for (size_t axis = 0; axis < Rank; ++axis) {
      output_indices[axis] = remaining / output_strides[axis];
      remaining %= output_strides[axis];
    }
    for (size_t axis = 0; axis < Rank; ++axis) {
      input_indices[permutation[axis]] = output_indices[axis];
    }

    size_t input_offset = 0;
    for (size_t axis = 0; axis < Rank; ++axis) {
      input_offset += input_indices[axis] * input_strides[axis];
    }
    output[output_offset] = input[input_offset];
  }
  return output;
}

// clang-format off
using TransposeTypes = Types<
    GenForTest<1, float>,
    GenForTest<2, int8_t>,
    GenForTest<3, uint8_t>,
    GenForTest<3, int32_t>,
    GenForTest<4, float>
>;
// clang-format on

TYPED_TEST_SUITE(TransposeTest, TransposeTypes);

TYPED_TEST(TransposeTest, TestLogic) {
  using Traits = typename TypeParam::Traits;
  using DataType = typename Traits::template InputDataType<0>;

  typename Traits::Params params;
  params.shape.fill(2);
  for (size_t axis = 0; axis < params.permutation.size(); ++axis) {
    params.permutation[axis] = params.permutation.size() - axis - 1;
  }
  const auto rank = params.shape.size();
  const auto expected_output_shape = TypeParam::OutputShape(params);
  const auto expected_input_shape =
      absl::MakeConstSpan(params.shape.data(), params.shape.size());
  const auto expected_output_shape_span =
      absl::MakeConstSpan(expected_output_shape.data(), expected_output_shape.size());
  const Layout::Dim permutation_rank = rank;

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, TypeParam::Create(params));
  EXPECT_EQ(gen->GetConformanceSpec().comparator_kind,
            ConformanceComparatorKind::kExact);
  EXPECT_EQ(gen->Graph().NumSubgraphs(), 1);
  const auto& sg = gen->Graph().Subgraph(0);
  EXPECT_EQ(sg.Ops().size(), 1);
  EXPECT_EQ(sg.Tensors().size(), 3);
  EXPECT_EQ(sg.Inputs().size(), 1);
  EXPECT_EQ(sg.Outputs().size(), 1);

  EXPECT_THAT(sg.Tensor(0).Type().second.ranked_tensor_type,
              HasTypeAspect(TypeParam::kElementTypeValue, expected_input_shape));
  EXPECT_THAT(sg.Tensor(1).Type().second.ranked_tensor_type,
              HasTypeAspect(ElementType::Int32,
                            absl::MakeConstSpan(&permutation_rank, 1)));
  EXPECT_THAT(sg.Tensor(2).Type().second.ranked_tensor_type,
              HasTypeAspect(TypeParam::kElementTypeValue,
                            expected_output_shape_span));

  const auto& op = sg.Op(0);
  EXPECT_EQ(op.OpCode(), kLiteRtOpCodeTflTranspose);
  const auto& tfl_opts = GetTflOptions(op);
  ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_TransposeOptions);
  ASSERT_NE(tfl_opts.AsTransposeOptions(), nullptr);

  const auto& perm_tensor = sg.Tensor(1);
  ASSERT_EQ(perm_tensor.Weights().Buffer().Size(), rank * sizeof(int32_t));
  const auto* perm_data = reinterpret_cast<const int32_t*>(
      perm_tensor.Weights().Buffer().Data());
  EXPECT_THAT(absl::MakeConstSpan(perm_data, rank),
              ElementsAreArray(params.permutation));

  auto device = this->TracedDevice();
  RandomTensorDataBuilder data_builder;
  data_builder.SetFloatDummy();
  data_builder.SetIntDummy();
  LITERT_ASSERT_OK_AND_ASSIGN(const auto inputs,
                              gen->MakeInputs(device, data_builder));
  ASSERT_EQ(inputs.size(), 1);
  const auto input = inputs[0].template AsView<DataType>();

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs,
      SimpleBuffer::LikeSignature(sg.Outputs().begin(), sg.Outputs().end()));
  ASSERT_EQ(outputs.size(), 1);

  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));
  EXPECT_THAT(outputs[0].template Span<DataType>(),
              ElementsAreArray(ExpectedTranspose<DataType>(
                  params.shape, params.permutation, input.data)));
}

TEST(TransposeTest, TwoDimensionalReferenceMatchesKernelExample) {
  using GraphT = GenForTest<2, float>;
  GraphT::Traits::Params params;
  params.shape = {3, 2};
  params.permutation = {1, 0};

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, GraphT::Create(params));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input, SimpleBuffer::Create<float>({3, 2}));
  LITERT_ASSERT_OK(input.Write({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}));

  VarBuffers inputs;
  inputs.push_back(std::move(input));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs, SimpleBuffer::LikeSignature(
                        gen->Graph().Subgraph(0).Outputs().begin(),
                        gen->Graph().Subgraph(0).Outputs().end()));
  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  EXPECT_THAT(outputs[0].template Span<float>(),
              ElementsAreArray({0.f, 2.f, 4.f, 1.f, 3.f, 5.f}));
}

TEST(TransposeTest, ThreeDimensionalReferenceMatchesKernelExample) {
  using GraphT = GenForTest<3, uint8_t>;
  GraphT::Traits::Params params;
  params.shape = {2, 3, 4};
  params.permutation = {2, 0, 1};

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, GraphT::Create(params));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input,
                              SimpleBuffer::Create<uint8_t>({2, 3, 4}));
  std::vector<uint8_t> input_data(24);
  std::iota(input_data.begin(), input_data.end(), 0);
  LITERT_ASSERT_OK(input.Write(input_data));

  VarBuffers inputs;
  inputs.push_back(std::move(input));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs, SimpleBuffer::LikeSignature(
                        gen->Graph().Subgraph(0).Outputs().begin(),
                        gen->Graph().Subgraph(0).Outputs().end()));
  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  EXPECT_THAT(outputs[0].template Span<uint8_t>(),
              ElementsAreArray({0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21,
                                2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));
}

TEST(TransposeTest, RejectsInvalidPermutation) {
  using GraphT = GenForTest<2, float>;
  GraphT::Traits::Params params;
  params.shape = {2, 3};
  params.permutation = {0, 0};

  auto result = GraphT::Create(params);
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(TransposeTest, Int4ReferenceMatchesKernelExample) {
  using GraphT = TransposeInt4<SizeC<2>, SizeC<64>>;
  GraphT::Params params;
  params.shape = {3, 2};
  params.permutation = {1, 0};

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, GraphT::Create(params));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input, SimpleBuffer::Create(
                      RankedTensorType(ElementType::Int4, Layout(Dimensions({3, 2})))));
  const std::vector<int8_t> unpacked_input = {0, 1, 2, 3, 4, 5};
  tflite::tensor_utils::PackInt8IntoDenseInt(
      unpacked_input.data(), unpacked_input.size(), /*bit_width=*/4,
      input.template Span<int8_t>().data());

  VarBuffers inputs;
  inputs.push_back(std::move(input));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs, SimpleBuffer::LikeSignature(
                        gen->Graph().Subgraph(0).Outputs().begin(),
                        gen->Graph().Subgraph(0).Outputs().end()));
  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  std::vector<int8_t> unpacked_output(6);
  tflite::tensor_utils::UnpackPackedIntToInt8(
      outputs[0].template Span<int8_t>().data(), unpacked_output.size(),
      /*bit_width=*/4, unpacked_output.data());
  EXPECT_THAT(unpacked_output, ElementsAreArray({0, 2, 4, 1, 3, 5}));
}

}  // namespace
}  // namespace litert::testing
