// Copyright 2024 Google LLC.
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

#include "litert/test/generators/no_op.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/cc/internal/litert_rng.h"
#include "litert/test/generators/common.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"
#include "litert/test/simple_buffer.h"

namespace litert {
namespace testing {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Types;
using ::testing::litert::HasTypeAspect;

template <typename D>
class ExampleTestLogicTest : public RngTest {};

// clang-format off
using ExampleTestLogicTestTypes = Types<
    NoOp<SizeC<1>, float>,
    NoOp<SizeC<2>, int32_t>,
    NoOp<SizeC<2>, float>
>;
// clang-format on

TYPED_TEST_SUITE(ExampleTestLogicTest, ExampleTestLogicTestTypes);

TYPED_TEST(ExampleTestLogicTest, TestLogic) {
  using Traits = typename TypeParam::Traits;
  using DataType = Traits::template InputDataType<0>;
  typename Traits::Params params;
  params.shape.fill(2);
  const auto rank = params.shape.size();
  const auto expected_shape = absl::MakeConstSpan(params.shape.data(), rank);

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, TypeParam::Create(params));
  EXPECT_EQ(gen->Graph().NumSubgraphs(), 1);
  const auto& sg = gen->Graph().Subgraph(0);
  EXPECT_EQ(sg.Ops().size(), 1);
  EXPECT_EQ(sg.Tensors().size(), 3);
  EXPECT_EQ(sg.Inputs().size(), 1);
  EXPECT_EQ(sg.Outputs().size(), 1);

  EXPECT_THAT(sg.Tensor(0).Type().second.ranked_tensor_type,
              HasTypeAspect(TypeParam::kElementType, expected_shape));

  EXPECT_THAT(sg.Tensor(1).Type().second.ranked_tensor_type,
              HasTypeAspect(TypeParam::kElementType, {}));

  EXPECT_THAT(sg.Tensor(2).Type().second.ranked_tensor_type,
              HasTypeAspect(TypeParam::kElementType, expected_shape));

  auto device = this->TracedDevice();
  RandomTensorDataBuilder data_builder;
  data_builder.SetFloatDummy();
  data_builder.SetIntDummy();
  LITERT_ASSERT_OK_AND_ASSIGN(const auto inputs,
                              gen->MakeInputs(device, data_builder));
  EXPECT_EQ(inputs.size(), 1);

  auto lhs = inputs[0].template AsView<DataType>();
  std::vector<DataType> expected_lhs(lhs.NumElements());
  std::iota(expected_lhs.begin(), expected_lhs.end(), 0);
  EXPECT_THAT(lhs.data, ElementsAreArray(expected_lhs));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs,
      SimpleBuffer::LikeSignature(sg.Outputs().begin(), sg.Outputs().end()));
  EXPECT_EQ(outputs.size(), 1);

  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  EXPECT_THAT(outputs[0].template Span<DataType>(),
              ElementsAreArray(expected_lhs));
}

}  // namespace
}  // namespace testing
}  // namespace litert
