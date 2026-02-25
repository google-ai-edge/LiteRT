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

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/test/generators/common.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"
#include "litert/test/simple_buffer.h"

namespace litert {
namespace testing {
namespace {

using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Types;

template <typename D>
class UnaryTest : public RngTest {};

template <size_t Rank, typename T, LiteRtOpCode OpCode>
using GenForTest = Unary<SizeC<Rank>, T, OpCodeC<OpCode>, SizeC<64>>;

// clang-format off
using UnaryTypes = Types<
    GenForTest<1, float, kLiteRtOpCodeTflFloor>,
    GenForTest<1, float, kLiteRtOpCodeTflLogistic>,
    GenForTest<1, float, kLiteRtOpCodeTflRelu>,
    GenForTest<2, int32_t, kLiteRtOpCodeTflRelu>,
    GenForTest<1, float, kLiteRtOpCodeTflReluN1To1>,
    GenForTest<1, float, kLiteRtOpCodeTflRelu6>,
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
    GenForTest<1, float, kLiteRtOpCodeTflCeil>,
    GenForTest<1, float, kLiteRtOpCodeTflCos>,
    GenForTest<2, float, kLiteRtOpCodeTflElu>,
    GenForTest<1, float, kLiteRtOpCodeTflRound>,
    GenForTest<1, float, kLiteRtOpCodeTflHardSwish>,
    GenForTest<3, float, kLiteRtOpCodeTflGelu>,
    GenForTest<1, float, kLiteRtOpCodeTflRelu0To1>,
    GenForTest<1, float, kLiteRtOpCodeTflSign>
>;
// clang-format on

TYPED_TEST_SUITE(UnaryTest, UnaryTypes);

TYPED_TEST(UnaryTest, TestLogic) {
  using Traits = typename TypeParam::Traits;
  using DataType = Traits::template InputDataType<0>;
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
  std::iota(expected_input_data.begin(), expected_input_data.end(), 0);

  EXPECT_THAT(input.data, ElementsAreArray(expected_input_data));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs,
      SimpleBuffer::LikeSignature(sg.Outputs().begin(), sg.Outputs().end()));
  EXPECT_EQ(outputs.size(), 1);

  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));
}

}  // namespace
}  // namespace testing
}  // namespace litert
