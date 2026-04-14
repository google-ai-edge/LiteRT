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

#include "litert/core/model/shape_inference.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

TEST(ShapeInferenceTest, AddStaticShapes) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& input0 = subgraph.EmplaceTensor();
  input0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2, 3}));

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2, 3}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input0, op);
  AttachInput(&input1, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  EXPECT_EQ(output.Type().first, kLiteRtRankedTensorType);
  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 3);
  EXPECT_EQ(shape.dimensions[0], 1);
  EXPECT_EQ(shape.dimensions[1], 2);
  EXPECT_EQ(shape.dimensions[2], 3);
}

TEST(ShapeInferenceTest, AddBroadcast) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& input0 = subgraph.EmplaceTensor();
  input0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2, 3}));

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 1}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input0, op);
  AttachInput(&input1, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  EXPECT_EQ(output.Type().first, kLiteRtRankedTensorType);
  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 3);
  EXPECT_EQ(shape.dimensions[0], 1);
  EXPECT_EQ(shape.dimensions[1], 2);
  EXPECT_EQ(shape.dimensions[2], 3);
}

TEST(ShapeInferenceTest, AddDynamic) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& input0 = subgraph.EmplaceTensor();
  input0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 128}));

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 128}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input0, op);
  AttachInput(&input1, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  EXPECT_EQ(output.Type().first, kLiteRtRankedTensorType);
  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 2);
  EXPECT_EQ(shape.dimensions[0], -1);
  EXPECT_EQ(shape.dimensions[1], 128);
}

TEST(ShapeInferenceTest, ReshapeWithOptions) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflReshape);

  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, 4, 4, 3};
  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 48}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  EXPECT_EQ(output.Type().first, kLiteRtRankedTensorType);
  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 4);
  EXPECT_EQ(shape.dimensions[0], 1);
  EXPECT_EQ(shape.dimensions[1], 4);
  EXPECT_EQ(shape.dimensions[2], 4);
  EXPECT_EQ(shape.dimensions[3], 3);
}

TEST(ShapeInferenceTest, ReshapeWithShapeTensor) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflReshape);

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 48}));

  auto& shape_tensor = subgraph.EmplaceTensor();
  int32_t shape_data[] = {1, 4, 4, 3};
  absl::string_view data_view(reinterpret_cast<const char*>(shape_data),
                              sizeof(shape_data));
  SetWeightsFromOwnedBuffer(shape_tensor.Weights(),
                            OwningBufferRef<uint8_t>(data_view));
  shape_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, op);
  AttachInput(&shape_tensor, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  EXPECT_EQ(output.Type().first, kLiteRtRankedTensorType);
  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 4);
  EXPECT_EQ(shape.dimensions[0], 1);
  EXPECT_EQ(shape.dimensions[1], 4);
  EXPECT_EQ(shape.dimensions[2], 4);
  EXPECT_EQ(shape.dimensions[3], 3);
}

TEST(ShapeInferenceTest, ValidateShapes) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& input0 = subgraph.EmplaceTensor();
  input0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2, 3}));

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2, 3}));

  auto& output = subgraph.EmplaceTensor();
  // Set incorrect shape.
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2, 4}));

  AttachInput(&input0, op);
  AttachInput(&input1, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  LiteRtOp failing_op = nullptr;
  ASSERT_EQ(engine.InferShapes(/*validation_only=*/true, &failing_op),
            kLiteRtStatusErrorShapeInferenceFailed);
  EXPECT_EQ(failing_op, &op);
}

TEST(ShapeInferenceTest, SpecializeSubgraph) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& input0 = subgraph.EmplaceTensor();
  input0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 2, 3}));
  subgraph.Inputs().push_back(&input0);

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 2, 3}));
  subgraph.Inputs().push_back(&input1);

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 2, 3}));
  subgraph.Outputs().push_back(&output);

  AttachInput(&input0, op);
  AttachInput(&input1, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  LiteRtSubgraphT* specialized_subgraph = nullptr;

  std::vector<Dims> input_shapes = {{1, 2, 3}, {1, 2, 3}};

  ASSERT_EQ(engine.SpecializeSubgraph(&subgraph, absl::MakeSpan(input_shapes),
                                      &specialized_subgraph),
            kLiteRtStatusOk);

  ASSERT_NE(specialized_subgraph, nullptr);
  EXPECT_EQ(specialized_subgraph->Inputs().size(), 2);

  auto& spec_output = specialized_subgraph->Output(0);
  EXPECT_EQ(spec_output.Type().first, kLiteRtRankedTensorType);
  const auto& shape = spec_output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 3);
  EXPECT_EQ(shape.dimensions[0], 1);
  EXPECT_EQ(shape.dimensions[1], 2);
  EXPECT_EQ(shape.dimensions[2], 3);
}

TEST(ShapeInferenceTest, ComplexGraphValidation) {
  // Construct a graph:
  // 1. RefInput (2, 3, 4) -> Add -> RefSum (2, 3, 4)
  // 2. RefSum -> Shape -> RefShape (3) [2, 3, 4]
  // 3. FlatInput (24) -> Add -> FlatSum (24)
  // 4. FlatSum, RefShape -> Reshape -> Reshaped (2, 3, 4)
  // 5. Reshaped, RefSum -> Add -> Output (2, 3, 4)

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  // Tensors
  auto& ref_input = subgraph.EmplaceTensor();
  ref_input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 3, 4}));

  auto& ref_sum = subgraph.EmplaceTensor();
  ref_sum.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 3, 4}));

  auto& ref_shape = subgraph.EmplaceTensor();
  ref_shape.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));

  auto& flat_input = subgraph.EmplaceTensor();
  flat_input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {24}));

  auto& flat_sum = subgraph.EmplaceTensor();
  flat_sum.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {24}));

  auto& reshaped = subgraph.EmplaceTensor();
  reshaped.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 3, 4}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 3, 4}));

  // Ops
  // 1. Add (Ref)
  auto& add1 = subgraph.EmplaceOp();
  add1.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&ref_input, add1);
  AttachInput(&ref_input, add1);
  AttachOutput(&ref_sum, add1);

  // 2. Shape
  auto& shape_op = subgraph.EmplaceOp();
  shape_op.SetOpCode(kLiteRtOpCodeTflShape);
  AttachInput(&ref_sum, shape_op);
  AttachOutput(&ref_shape, shape_op);

  // 3. Add (Flat)
  auto& add2 = subgraph.EmplaceOp();
  add2.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&flat_input, add2);
  AttachInput(&flat_input, add2);
  AttachOutput(&flat_sum, add2);

  // 4. Reshape
  auto& reshape_op = subgraph.EmplaceOp();
  reshape_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&flat_sum, reshape_op);
  AttachInput(&ref_shape, reshape_op);
  AttachOutput(&reshaped, reshape_op);

  // 5. Add (Final)
  auto& add3 = subgraph.EmplaceOp();
  add3.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&reshaped, add3);
  AttachInput(&ref_sum, add3);
  AttachOutput(&output, add3);

  ShapeInferenceEngine engine(&model);
  // Run inference. This should propagate shapes and data through the graph.
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  auto get_shape = [](const LiteRtTensorT& t) -> std::vector<int32_t> {
    const auto& l = t.Type().second.ranked_tensor_type.layout;
    return {l.dimensions, l.dimensions + l.rank};
  };

  EXPECT_THAT(get_shape(reshaped), testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(get_shape(output), testing::ElementsAre(2, 3, 4));
}

TEST(ShapeInferenceTest, CheckSupportedOps) {
  std::vector<LiteRtOpCode> supported_ops = {
      kLiteRtOpCodeTflAbs,
      kLiteRtOpCodeTflCeil,
      kLiteRtOpCodeTflCos,
      kLiteRtOpCodeTflDequantize,
      kLiteRtOpCodeTflElu,
      kLiteRtOpCodeTflExp,
      kLiteRtOpCodeTflFloor,
      kLiteRtOpCodeTflGelu,
      kLiteRtOpCodeTflHardSwish,
      kLiteRtOpCodeTflLeakyRelu,
      kLiteRtOpCodeTflLog,
      kLiteRtOpCodeTflLogicalNot,
      kLiteRtOpCodeTflLogistic,
      kLiteRtOpCodeTflNeg,
      kLiteRtOpCodeTflQuantize,
      kLiteRtOpCodeTflRelu,
      kLiteRtOpCodeTflRelu0To1,
      kLiteRtOpCodeTflRelu6,
      kLiteRtOpCodeTflReluN1To1,
      kLiteRtOpCodeTflRound,
      kLiteRtOpCodeTflRsqrt,
      kLiteRtOpCodeTflSign,
      kLiteRtOpCodeTflSin,
      kLiteRtOpCodeTflSoftmax,
      kLiteRtOpCodeTflSqrt,
      kLiteRtOpCodeTflSquare,
      kLiteRtOpCodeTflTanh,
      kLiteRtOpCodeTflEqual,
      kLiteRtOpCodeTflFloorDiv,
      kLiteRtOpCodeTflGreater,
      kLiteRtOpCodeTflGreaterEqual,
      kLiteRtOpCodeTflLess,
      kLiteRtOpCodeTflLessEqual,
      kLiteRtOpCodeTflLogicalAnd,
      kLiteRtOpCodeTflLogicalOr,
      kLiteRtOpCodeTflMaximum,
      kLiteRtOpCodeTflMinimum,
      kLiteRtOpCodeTflNotEqual,
      kLiteRtOpCodeTflPow,
      kLiteRtOpCodeTflPrelu,
      kLiteRtOpCodeTflSquaredDifference,
      kLiteRtOpCodeTflAdd,
      kLiteRtOpCodeTflArgMax,
      kLiteRtOpCodeTflArgMin,
      kLiteRtOpCodeTflAveragePool2d,
      kLiteRtOpCodeTflBatchMatmul,
      kLiteRtOpCodeTflBroadcastTo,
      kLiteRtOpCodeTflCast,
      kLiteRtOpCodeTflConcatenation,
      kLiteRtOpCodeTflConv2d,
      kLiteRtOpCodeTflConv3d,
      kLiteRtOpCodeTflConv3dTranspose,
      kLiteRtOpCodeTflDepthToSpace,
      kLiteRtOpCodeTflDepthwiseConv2d,
      kLiteRtOpCodeTflDiv,
      kLiteRtOpCodeTflDynamicUpdateSlice,
      kLiteRtOpCodeTflEmbeddingLookup,
      kLiteRtOpCodeTflFullyConnected,
      kLiteRtOpCodeTflGather,
      kLiteRtOpCodeTflGatherNd,
      kLiteRtOpCodeTflL2Pool2d,
      kLiteRtOpCodeTflMaxPool2d,
      kLiteRtOpCodeTflMean,
      kLiteRtOpCodeTflMirrorPad,
      kLiteRtOpCodeTflMul,
      kLiteRtOpCodeTflPack,
      kLiteRtOpCodeTflPad,
      kLiteRtOpCodeTflPadv2,
      kLiteRtOpCodeTflReduceAll,
      kLiteRtOpCodeTflReduceAny,
      kLiteRtOpCodeTflReduceMax,
      kLiteRtOpCodeTflReduceMin,
      kLiteRtOpCodeTflSum,
      kLiteRtOpCodeTflReshape,
      kLiteRtOpCodeTflResizeBilinear,
      kLiteRtOpCodeTflResizeNearestNeighbor,
      kLiteRtOpCodeTflSelectV2,
      kLiteRtOpCodeTflSpaceToDepth,
      kLiteRtOpCodeTflTranspose,
      kLiteRtOpCodeTflTransposeConv,
      kLiteRtOpCodeTflUnpack,
      kLiteRtOpCodeTflCumsum,
      kLiteRtOpCodeTflL2Normalization,
      kLiteRtOpCodeTflReverseV2,
      kLiteRtOpCodeTflTopkV2,
      kLiteRtOpCodeTflShape,
      kLiteRtOpCodeTflRank,
  };

  ShapeInferenceEngine engine;
  for (auto op_code : supported_ops) {
    LiteRtOpT op;
    op.SetOpCode(op_code);
    auto status = engine.InferOpShapes(&op);
    EXPECT_NE(status, kLiteRtStatusErrorUnsupportedOpShapeInferer)
        << "Op code " << op_code << " is not supported.";
  }
}

TEST(ShapeInferenceTest, TransientDataClearedBetweenSubgraphs) {
  LiteRtModelT model;
  ShapeInferenceEngine engine(&model);

  // Subgraph 1: Produces transient data for a tensor.
  auto& sg1 = model.EmplaceSubgraph();
  auto& in1 = sg1.EmplaceTensor();
  in1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 2}));
  auto& out1 = sg1.EmplaceTensor();
  out1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));
  auto& shape1 = sg1.EmplaceOp();
  shape1.SetOpCode(kLiteRtOpCodeTflShape);
  AttachInput(&in1, shape1);
  AttachOutput(&out1, shape1);

  // Use validation_only=true so that out1 weights are not updated in the model,
  // only transient_data_ is populated.
  ASSERT_EQ(engine.InferSubgraphShapes(&sg1, /*validation_only=*/true),
            kLiteRtStatusOk);

  // Subgraph 2: Has a custom op that should NOT see transient data from sg1.
  auto& sg2 = model.EmplaceSubgraph();
  // Reuse the same tensor object in a different subgraph to verify clearing by
  // pointer.
  sg2.Inputs().push_back(&out1);

  auto& custom_op = sg2.EmplaceOp();
  custom_op.SetOpCode(kLiteRtOpCodeTflCustom);
  AttachInput(&out1, custom_op);
  auto& custom_out = sg2.EmplaceTensor();
  AttachOutput(&custom_out, custom_op);

  bool found_stale_data = false;
  engine.RegisterInferrer(kLiteRtOpCodeTflCustom,
                          [&found_stale_data](const ShapeInferenceContext& ctx,
                                              InferenceResult& result) {
                            if (!ctx.GetInputData(0).empty()) {
                              found_stale_data = true;
                            }
                            return kLiteRtStatusOk;
                          });

  ASSERT_EQ(engine.InferSubgraphShapes(&sg2), kLiteRtStatusOk);
  EXPECT_FALSE(found_stale_data)
      << "Found stale transient data from previous subgraph run";
}

TEST(ShapeInferenceTest, AddUnranked) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& input0 = subgraph.EmplaceTensor();
  TensorType type0;
  type0.first = kLiteRtUnrankedTensorType;
  type0.second.unranked_tensor_type.element_type = kLiteRtElementTypeFloat32;
  input0.SetType(type0);

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2, 3}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(type0);

  AttachInput(&input0, op);
  AttachInput(&input1, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  EXPECT_EQ(output.Type().first, kLiteRtRankedTensorType);
  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  // Currently unranked is treated as scalar {}, so max(0, 3) = 3.
  // And it broadcasts as if it was {1, 1, 1}.
  EXPECT_EQ(shape.rank, 3);
  EXPECT_EQ(shape.dimensions[0], 1);
  EXPECT_EQ(shape.dimensions[1], 2);
  EXPECT_EQ(shape.dimensions[2], 3);
}

}  // namespace
}  // namespace litert::internal
