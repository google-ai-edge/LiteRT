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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"

namespace litert::internal {
namespace {

TEST(ExpandDimsOpTest, BasicExpand) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflExpandDims);

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 2}));

  auto& axis_tensor = subgraph.EmplaceTensor();
  int32_t axis_data = 1;
  SetWeightsFromOwnedBuffer(
      axis_tensor.Weights(),
      OwningBufferRef<uint8_t>(absl::string_view(
          reinterpret_cast<const char*>(&axis_data), sizeof(int32_t))));
  axis_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, op);
  AttachInput(&axis_tensor, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_THAT(
      std::vector<int32_t>(shape.dimensions, shape.dimensions + shape.rank),
      testing::ElementsAre(2, 1, 2));
}

TEST(ExpandDimsOpTest, NegativeAxis) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflExpandDims);

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 2}));

  auto& axis_tensor = subgraph.EmplaceTensor();
  int32_t axis_data = -1;
  SetWeightsFromOwnedBuffer(
      axis_tensor.Weights(),
      OwningBufferRef<uint8_t>(absl::string_view(
          reinterpret_cast<const char*>(&axis_data), sizeof(int32_t))));
  axis_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, op);
  AttachInput(&axis_tensor, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_THAT(
      std::vector<int32_t>(shape.dimensions, shape.dimensions + shape.rank),
      testing::ElementsAre(2, 2, 1));
}

}  // namespace
}  // namespace litert::internal
