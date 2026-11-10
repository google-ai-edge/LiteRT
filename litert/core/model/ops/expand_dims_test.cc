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

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"

namespace litert::internal {
namespace {

LiteRtStatus RunExpandDimsShapeInference(
    const std::vector<uint8_t>& axis_bytes,
    std::vector<int32_t>* inferred_shape = nullptr) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflExpandDims);

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 2}));

  auto& axis_tensor = subgraph.EmplaceTensor();
  SetWeightsFromOwnedBuffer(
      axis_tensor.Weights(),
      OwningBufferRef<uint8_t>(axis_bytes.data(), axis_bytes.size()));
  axis_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, op);
  AttachInput(&axis_tensor, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  LiteRtStatus status = engine.InferShapes();
  if (status == kLiteRtStatusOk && inferred_shape != nullptr) {
    const auto& shape = output.Type().second.ranked_tensor_type.layout;
    inferred_shape->assign(shape.dimensions, shape.dimensions + shape.rank);
  }
  return status;
}

std::vector<uint8_t> MakeAxisBytes(int32_t axis) {
  std::vector<uint8_t> bytes(sizeof(int32_t));
  std::memcpy(bytes.data(), &axis, sizeof(int32_t));
  return bytes;
}

TEST(ExpandDimsOpTest, BasicExpand) {
  std::vector<int32_t> shape;
  ASSERT_EQ(RunExpandDimsShapeInference(MakeAxisBytes(1), &shape),
            kLiteRtStatusOk);
  EXPECT_THAT(shape, testing::ElementsAre(2, 1, 2));
}

TEST(ExpandDimsOpTest, NegativeAxis) {
  std::vector<int32_t> shape;
  ASSERT_EQ(RunExpandDimsShapeInference(MakeAxisBytes(-1), &shape),
            kLiteRtStatusOk);
  EXPECT_THAT(shape, testing::ElementsAre(2, 2, 1));
}

struct ExpandDimsErrorCase {
  std::string test_name;
  std::vector<uint8_t> axis_bytes;
};

using ExpandDimsErrorTest = ::testing::TestWithParam<ExpandDimsErrorCase>;

TEST_P(ExpandDimsErrorTest, RejectsShortAxisBufferAndAxisUnderflow) {
  EXPECT_EQ(RunExpandDimsShapeInference(GetParam().axis_bytes),
            kLiteRtStatusErrorInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    ExpandDimsErrorTests, ExpandDimsErrorTest,
    ::testing::Values(ExpandDimsErrorCase{"ShortAxisBuffer", {0, 0}},
                      ExpandDimsErrorCase{
                          "AxisUnderflow",
                          MakeAxisBytes(std::numeric_limits<int32_t>::min())}),
    [](const ::testing::TestParamInfo<ExpandDimsErrorCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace litert::internal
