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

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"

namespace litert::internal {
namespace {

TEST(ShapeOpTest, ShapeOp) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflShape);

  // Input: [2, 3, 4]
  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 3, 4}));

  // Output: Shape [3]
  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));

  AttachInput(&input, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  // Output shape should be [3] (rank of input)
  EXPECT_EQ(output.Type().first, kLiteRtRankedTensorType);
  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 1);
  EXPECT_EQ(shape.dimensions[0], 3);

  // Output weights should be populated with [2, 3, 4]
  ASSERT_GT(output.Weights().Buffer().Size(), 0);
  const int32_t* dims =
      reinterpret_cast<const int32_t*>(output.Weights().Buffer().Data());
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[1], 3);
  EXPECT_EQ(dims[2], 4);
}

TEST(ShapeOpTest, RankOp) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflRank);

  // Input: [2, 3, 4]
  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 3, 4}));

  // Output: Scalar
  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));

  AttachInput(&input, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  // Output shape should be 0-D.
  EXPECT_EQ(output.Type().first, kLiteRtRankedTensorType);
  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 0);

  // Output weights should be populated with [3] (rank).
  ASSERT_GT(output.Weights().Buffer().Size(), 0);
  const int32_t* val =
      reinterpret_cast<const int32_t*>(output.Weights().Buffer().Data());
  EXPECT_EQ(val[0], 3);
}

}  // namespace
}  // namespace litert::internal
