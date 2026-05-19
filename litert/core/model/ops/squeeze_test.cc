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
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

TEST(SqueezeOpTest, SqueezeSpecifiedDim) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSqueeze);

  auto options = std::make_unique<tflite::SqueezeOptionsT>();
  options->squeeze_dims = {1};
  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SqueezeOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 1, 3}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_THAT(
      std::vector<int32_t>(shape.dimensions, shape.dimensions + shape.rank),
      testing::ElementsAre(2, 3));
}

TEST(SqueezeOpTest, SqueezeAllUnitDims) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSqueeze);

  auto& input = subgraph.EmplaceTensor();
  input.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2, 1, 3, 1}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_THAT(
      std::vector<int32_t>(shape.dimensions, shape.dimensions + shape.rank),
      testing::ElementsAre(2, 3));
}

}  // namespace
}  // namespace litert::internal
