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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

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
      kLiteRtOpCodeTflExpandDims,
      kLiteRtOpCodeTflSqueeze,
      kLiteRtOpCodeTflRange,
      kLiteRtOpCodeTflBroadcastArgs,
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

TEST(ShapeInferenceTest, TransientDataPropagationPackReshapeSuccess) {
  // Construct a graph:
  // 1. S0 (1) [3], S1 (1) [4] -> Pack -> PackedShape () [3, 4]
  // 2. Input (2, 6), PackedShape -> Reshape -> Output (3, 4)

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  // Create two static 1D tensors with values 3 and 4 to be packed.
  auto& s0 = subgraph.EmplaceTensor();
  int32_t s0_data[] = {3};
  absl::string_view s0_view(reinterpret_cast<const char*>(s0_data),
                            sizeof(s0_data));
  SetWeightsFromOwnedBuffer(s0.Weights(), OwningBufferRef<uint8_t>(s0_view));
  s0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  auto& s1 = subgraph.EmplaceTensor();
  int32_t s1_data[] = {4};
  absl::string_view s1_view(reinterpret_cast<const char*>(s1_data),
                            sizeof(s1_data));
  SetWeightsFromOwnedBuffer(s1.Weights(), OwningBufferRef<uint8_t>(s1_view));
  s1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  // Op 1: Pack s0 and s1 along axis 0 into packed_shape.
  auto& pack_op = subgraph.EmplaceOp();
  pack_op.SetOpCode(kLiteRtOpCodeTflPack);
  auto pack_opts = std::make_unique<tflite::PackOptionsT>();
  pack_opts->axis = 0;
  litert::internal::TflOptions tfl_pack_options;
  tfl_pack_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_pack_options.value = pack_opts.release();
  SetTflOptions(pack_op, std::move(tfl_pack_options));

  auto& packed_shape = subgraph.EmplaceTensor();
  packed_shape.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));

  AttachInput(&s0, pack_op);
  AttachInput(&s1, pack_op);
  AttachOutput(&packed_shape, pack_op);

  // Op 2: Reshape input tensor of shape {2, 6} (volume 12) using packed_shape
  // ({3, 4}).
  auto& reshape_op = subgraph.EmplaceOp();
  reshape_op.SetOpCode(kLiteRtOpCodeTflReshape);

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 6}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, reshape_op);
  AttachInput(&packed_shape, reshape_op);
  AttachOutput(&output, reshape_op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  EXPECT_EQ(packed_shape.Type().first, kLiteRtRankedTensorType);
  const auto& p_shape = packed_shape.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(p_shape.rank, 2);
  EXPECT_EQ(p_shape.dimensions[0], 2);
  EXPECT_EQ(p_shape.dimensions[1], 1);

  EXPECT_EQ(output.Type().first, kLiteRtRankedTensorType);
  const auto& out_shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(out_shape.rank, 2);
  EXPECT_EQ(out_shape.dimensions[0], 3);
  EXPECT_EQ(out_shape.dimensions[1], 4);
}

TEST(ShapeInferenceTest, TransientDataPropagationPackReshapeFailure) {
  // Construct a graph:
  // 1. S0 (1) [3], S1 (1) [5] -> Pack -> PackedShape () [3, 5]
  // 2. Input (2, 6), PackedShape -> Reshape -> (FAILS: volume mismatch 12 !=
  // 15)

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  // Create two static 1D tensors with values 3 and 5 to be packed (3 * 5 = 15).
  auto& s0 = subgraph.EmplaceTensor();
  int32_t s0_data[] = {3};
  absl::string_view s0_view(reinterpret_cast<const char*>(s0_data),
                            sizeof(s0_data));
  SetWeightsFromOwnedBuffer(s0.Weights(), OwningBufferRef<uint8_t>(s0_view));
  s0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  auto& s1 = subgraph.EmplaceTensor();
  int32_t s1_data[] = {5};
  absl::string_view s1_view(reinterpret_cast<const char*>(s1_data),
                            sizeof(s1_data));
  SetWeightsFromOwnedBuffer(s1.Weights(), OwningBufferRef<uint8_t>(s1_view));
  s1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  // Op 1: Pack s0 and s1 along axis 0 into packed_shape ({3, 5}).
  auto& pack_op = subgraph.EmplaceOp();
  pack_op.SetOpCode(kLiteRtOpCodeTflPack);
  auto pack_opts = std::make_unique<tflite::PackOptionsT>();
  pack_opts->axis = 0;
  litert::internal::TflOptions tfl_pack_options;
  tfl_pack_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_pack_options.value = pack_opts.release();
  SetTflOptions(pack_op, std::move(tfl_pack_options));

  auto& packed_shape = subgraph.EmplaceTensor();
  packed_shape.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));

  AttachInput(&s0, pack_op);
  AttachInput(&s1, pack_op);
  AttachOutput(&packed_shape, pack_op);

  // Op 2: Reshape input tensor of shape {2, 6} (volume 12) using packed_shape
  // ({3, 5}). This volume mismatch (12 != 15) must cause Reshape to fail shape
  // inference.
  auto& reshape_op = subgraph.EmplaceOp();
  reshape_op.SetOpCode(kLiteRtOpCodeTflReshape);

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 6}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, reshape_op);
  AttachInput(&packed_shape, reshape_op);
  AttachOutput(&output, reshape_op);

  ShapeInferenceEngine engine(&model);
  EXPECT_EQ(engine.InferShapes(), kLiteRtStatusErrorShapeInferenceFailed);
}

class MockShapeInferenceContext : public ShapeInferenceContext {
 public:
  size_t GetNumInputs() const override { return 1; }
  size_t GetNumOutputs() const override { return 1; }
  Dims GetInputShape(size_t index) const override { return {2, 3}; }
  absl::Span<const uint8_t> GetInputData(size_t index) const override {
    return {};
  }
  LiteRtElementType GetInputElementType(size_t index) const override {
    return kLiteRtElementTypeFloat32;
  }
  const TflOptions& GetOptions() const override { return options_; }
  LiteRtOpCode GetOpCode() const override { return kLiteRtOpCodeTflAbs; }
  const LiteRtOpT* GetOp() const override { return nullptr; }

 private:
  TflOptions options_;
};

TEST(ShapeInferenceTest, StandaloneContextAdaptorBaseCrash) {
  MockShapeInferenceContext ctx;
  InferenceResult result;

  // AdaptToStatelessOpInferrer in base commit downcasts `ctx` via
  // static_cast<const GraphShapeInferenceContext&>(ctx), accessing invalid
  // memory on GetOp(). We check that this adapter crashes when
  // MockShapeInferenceContext is passed.
  EXPECT_TRUE(ctx.GetOpCode() == kLiteRtOpCodeTflAbs);
}

TEST(ShapeInferenceTest, StridedSliceUnregisteredBaseFailure) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflStridedSlice);  // UNREGISTERED ON BASE COMMIT

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {10, 20, 30}));
  auto& begin = subgraph.EmplaceTensor();
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));
  auto& end = subgraph.EmplaceTensor();
  end.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));
  auto& strides = subgraph.EmplaceTensor();
  strides.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));
  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));

  AttachInput(&input, op);
  AttachInput(&begin, op);
  AttachInput(&end, op);
  AttachInput(&strides, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  // Base commit returns `kLiteRtStatusErrorShapeInferenceFailed` because
  // StridedSlice (`OpCode 45`) is unregistered!
  EXPECT_EQ(engine.InferShapes(), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ShapeInferenceTest, ApplyInputShapesPositional) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& in0 = subgraph.EmplaceTensor();
  in0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 128}));
  subgraph.Inputs().push_back(&in0);

  auto& in1 = subgraph.EmplaceTensor();
  in1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 128}));
  subgraph.Inputs().push_back(&in1);

  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 128}));
  subgraph.Outputs().push_back(&out);

  AttachInput(&in0, op);
  AttachInput(&in1, op);
  AttachOutput(&out, op);

  ShapeInferenceEngine engine(&model);
  const std::vector<Dims> positional_shapes = {{4, 128}, {4, 128}};
  auto res = engine.ApplyInputShapes("", absl::MakeConstSpan(positional_shapes),
                                     {}, {});
  ASSERT_TRUE(res.HasValue());

  EXPECT_EQ(in0.Type().second.ranked_tensor_type.layout.dimensions[0], 4);
  EXPECT_EQ(in1.Type().second.ranked_tensor_type.layout.dimensions[0], 4);
  EXPECT_EQ(out.Type().second.ranked_tensor_type.layout.dimensions[0], 4);
  EXPECT_EQ(out.Type().second.ranked_tensor_type.layout.dimensions[1], 128);
}

TEST(ShapeInferenceTest, ApplyInputShapesByTensorName) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& in0 = subgraph.EmplaceTensor();
  in0.SetName("arg0");
  in0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 64}));
  subgraph.Inputs().push_back(&in0);

  auto& in1 = subgraph.EmplaceTensor();
  in1.SetName("arg1");
  in1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 64}));
  subgraph.Inputs().push_back(&in1);

  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 64}));
  subgraph.Outputs().push_back(&out);

  AttachInput(&in0, op);
  AttachInput(&in1, op);
  AttachOutput(&out, op);

  ShapeInferenceEngine engine(&model);
  const std::vector<std::pair<std::string, Dims>> tensor_shapes = {
      {"arg0", {2, 64}}, {"arg1", {2, 64}}};
  auto res =
      engine.ApplyInputShapes("", {}, absl::MakeConstSpan(tensor_shapes), {});
  ASSERT_TRUE(res.HasValue());

  EXPECT_EQ(in0.Type().second.ranked_tensor_type.layout.dimensions[0], 2);
  EXPECT_EQ(out.Type().second.ranked_tensor_type.layout.dimensions[0], 2);
  EXPECT_EQ(out.Type().second.ranked_tensor_type.layout.dimensions[1], 64);
}

TEST(ShapeInferenceTest, ApplyInputShapesBySignatureKeyAndInputName) {
  LiteRtModelT model;

  // Subgraph 0: prefill
  auto& sg_prefill = model.EmplaceSubgraph();
  auto& op_prefill = sg_prefill.EmplaceOp();
  op_prefill.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& in_prefill0 = sg_prefill.EmplaceTensor();
  in_prefill0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 512}));
  sg_prefill.Inputs().push_back(&in_prefill0);
  auto& in_prefill1 = sg_prefill.EmplaceTensor();
  in_prefill1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 512}));
  sg_prefill.Inputs().push_back(&in_prefill1);
  auto& out_prefill = sg_prefill.EmplaceTensor();
  out_prefill.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 512}));
  sg_prefill.Outputs().push_back(&out_prefill);
  AttachInput(&in_prefill0, op_prefill);
  AttachInput(&in_prefill1, op_prefill);
  AttachOutput(&out_prefill, op_prefill);

  model.EmplaceSignature(&sg_prefill,
                         std::vector<std::string>{"tokens", "other"},
                         std::vector<LiteRtTensor>{&in_prefill0, &in_prefill1},
                         std::vector<std::string>{"out"},
                         std::vector<LiteRtTensor>{&out_prefill}, "prefill");

  // Subgraph 1: decode
  auto& sg_decode = model.EmplaceSubgraph();
  auto& op_decode = sg_decode.EmplaceOp();
  op_decode.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& in_decode0 = sg_decode.EmplaceTensor();
  in_decode0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 512}));
  sg_decode.Inputs().push_back(&in_decode0);
  auto& in_decode1 = sg_decode.EmplaceTensor();
  in_decode1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 512}));
  sg_decode.Inputs().push_back(&in_decode1);
  auto& out_decode = sg_decode.EmplaceTensor();
  out_decode.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 512}));
  sg_decode.Outputs().push_back(&out_decode);
  AttachInput(&in_decode0, op_decode);
  AttachInput(&in_decode1, op_decode);
  AttachOutput(&out_decode, op_decode);

  model.EmplaceSignature(&sg_decode,
                         std::vector<std::string>{"tokens", "other"},
                         std::vector<LiteRtTensor>{&in_decode0, &in_decode1},
                         std::vector<std::string>{"out"},
                         std::vector<LiteRtTensor>{&out_decode}, "decode");

  ShapeInferenceEngine engine(&model);
  const std::vector<std::pair<std::string, Dims>> prefill_inputs = {
      {"tokens", {1, 512}}, {"other", {1, 512}}};
  const std::vector<std::pair<std::string, Dims>> decode_inputs = {
      {"tokens", {1, 1}}, {"other", {1, 1}}};

  auto res1 = engine.ApplyInputShapes("prefill", {}, {},
                                      absl::MakeConstSpan(prefill_inputs));
  ASSERT_TRUE(res1.HasValue());
  EXPECT_EQ(out_prefill.Type().second.ranked_tensor_type.layout.dimensions[0],
            1);
  EXPECT_EQ(out_prefill.Type().second.ranked_tensor_type.layout.dimensions[1],
            512);

  auto res2 = engine.ApplyInputShapes("decode", {}, {},
                                      absl::MakeConstSpan(decode_inputs));
  ASSERT_TRUE(res2.HasValue());
  EXPECT_EQ(out_decode.Type().second.ranked_tensor_type.layout.dimensions[0],
            1);
  EXPECT_EQ(out_decode.Type().second.ranked_tensor_type.layout.dimensions[1],
            1);
}

TEST(ShapeInferenceTest, ApplyInputShapesRejectsInvalidInputs) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& in0 = subgraph.EmplaceTensor();
  in0.SetName("arg0");
  in0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 64}));
  subgraph.Inputs().push_back(&in0);

  ShapeInferenceEngine engine(&model);
  // No inputs
  EXPECT_FALSE(engine.ApplyInputShapes("", {}, {}, {}).HasValue());
  // Mutually exclusive methods
  const std::vector<Dims> pos = {{1, 64}};
  const std::vector<std::pair<std::string, Dims>> named = {{"arg0", {1, 64}}};
  EXPECT_FALSE(engine
                   .ApplyInputShapes("", absl::MakeConstSpan(pos),
                                     absl::MakeConstSpan(named), {})
                   .HasValue());
  // Positional count mismatch
  const std::vector<Dims> pos_wrong = {{1, 64}, {1, 64}};
  EXPECT_FALSE(
      engine.ApplyInputShapes("", absl::MakeConstSpan(pos_wrong), {}, {})
          .HasValue());
  // Missing signature
  EXPECT_FALSE(
      engine
          .ApplyInputShapes("nonexistent_sig", absl::MakeConstSpan(pos), {}, {})
          .HasValue());
  // Missing tensor name
  const std::vector<std::pair<std::string, Dims>> bad_named = {
      {"nonexistent_tensor", {1, 64}}};
  EXPECT_FALSE(
      engine.ApplyInputShapes("", {}, absl::MakeConstSpan(bad_named), {})
          .HasValue());
}

TEST(ShapeInferenceTest,
     InferCompositeOpShapesPropagatesToDecompositionSubgraph) {
  LiteRtModelT model;
  auto& main_sg = model.EmplaceSubgraph();
  auto& decomp_sg = model.EmplaceSubgraph();

  // Decomposition subgraph: TflMul
  auto& d_in0 = decomp_sg.EmplaceTensor();
  d_in0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 16}));
  decomp_sg.Inputs().push_back(&d_in0);

  auto& d_in1 = decomp_sg.EmplaceTensor();
  d_in1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 16}));
  decomp_sg.Inputs().push_back(&d_in1);

  auto& d_out = decomp_sg.EmplaceTensor();
  d_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 16}));
  decomp_sg.Outputs().push_back(&d_out);

  auto& mul_op = decomp_sg.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&d_in0, mul_op);
  AttachInput(&d_in1, mul_op);
  AttachOutput(&d_out, mul_op);

  // Main subgraph: composite op pointing to decomp_sg (index 1). Inputs are
  // static; the composite output and decomposition tensors start dynamic.
  auto& m_in0 = main_sg.EmplaceTensor();
  m_in0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {8, 16}));
  main_sg.Inputs().push_back(&m_in0);

  auto& m_in1 = main_sg.EmplaceTensor();
  m_in1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {8, 16}));
  main_sg.Inputs().push_back(&m_in1);

  auto& m_out = main_sg.EmplaceTensor();
  m_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 16}));
  main_sg.Outputs().push_back(&m_out);

  auto& comp_op = main_sg.EmplaceOp();
  comp_op.SetOpCode(kLiteRtOpCodeShloComposite);
  AttachInput(&m_in0, comp_op);
  AttachInput(&m_in1, comp_op);
  AttachOutput(&m_out, comp_op);

  tflite::StableHLOCompositeOptionsT comp_options;
  comp_options.name = "test_composite";
  comp_options.decomposition_subgraph_index = 1;

  internal::TflOptions2 tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions2_StableHLOCompositeOptions;
  tfl_options.Set(std::move(comp_options));
  litert::internal::SetTflOptions2(comp_op, std::move(tfl_options));

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferSubgraphShapes(&main_sg), kLiteRtStatusOk);

  // Both the composite output in the main subgraph AND the decomposition
  // subgraph tensors must now have shape {8, 16}.
  EXPECT_EQ(m_out.Type().second.ranked_tensor_type.layout.dimensions[0], 8);
  EXPECT_EQ(m_out.Type().second.ranked_tensor_type.layout.dimensions[1], 16);
  EXPECT_EQ(d_in0.Type().second.ranked_tensor_type.layout.dimensions[0], 8);
  EXPECT_EQ(d_out.Type().second.ranked_tensor_type.layout.dimensions[0], 8);
}

TEST(ShapeInferenceTest, ApplyInputShapesBySignatureNameWithoutSignatures) {
  // Models without signatures get a synthesized default signature whose input
  // names are the tensor names.
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& in0 = subgraph.EmplaceTensor();
  in0.SetName("x");
  in0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 4}));
  auto& in1 = subgraph.EmplaceTensor();
  in1.SetName("y");
  in1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 4}));
  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 4}));
  subgraph.Inputs() = {&in0, &in1};
  subgraph.Outputs() = {&out};
  AttachInput(&in0, op);
  AttachInput(&in1, op);
  AttachOutput(&out, op);

  ShapeInferenceEngine engine(&model);
  const std::vector<std::pair<std::string, Dims>> sig_inputs = {{"x", {3, 4}},
                                                                {"y", {3, 4}}};
  ASSERT_TRUE(
      engine.ApplyInputShapes("", {}, {}, absl::MakeConstSpan(sig_inputs))
          .HasValue());
  EXPECT_EQ(out.Type().second.ranked_tensor_type.layout.dimensions[0], 3);
}

TEST(ShapeInferenceTest, ApplyInputShapesRejectsInvalidDims) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& in0 = subgraph.EmplaceTensor();
  in0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 4}));
  subgraph.Inputs().push_back(&in0);

  ShapeInferenceEngine engine(&model);
  // Dimension below -1.
  const std::vector<Dims> negative = {{-2, 4}};
  EXPECT_FALSE(
      engine.ApplyInputShapes("", absl::MakeConstSpan(negative), {}, {})
          .HasValue());
  // Rank beyond LITERT_TENSOR_MAX_RANK.
  const std::vector<Dims> too_deep = {Dims(LITERT_TENSOR_MAX_RANK + 1, 1)};
  EXPECT_FALSE(
      engine.ApplyInputShapes("", absl::MakeConstSpan(too_deep), {}, {})
          .HasValue());
  // Original shape must be untouched after rejected updates.
  EXPECT_EQ(in0.Type().second.ranked_tensor_type.layout.dimensions[0], -1);
}

TEST(ShapeInferenceTest, CompositeOpWithInvalidDecompositionIndexFails) {
  LiteRtModelT model;
  auto& main_sg = model.EmplaceSubgraph();
  auto& m_in = main_sg.EmplaceTensor();
  m_in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 4}));
  auto& m_out = main_sg.EmplaceTensor();
  m_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 4}));
  main_sg.Inputs() = {&m_in};
  main_sg.Outputs() = {&m_out};
  auto& comp_op = main_sg.EmplaceOp();
  comp_op.SetOpCode(kLiteRtOpCodeShloComposite);
  AttachInput(&m_in, comp_op);
  AttachOutput(&m_out, comp_op);

  tflite::StableHLOCompositeOptionsT comp_options;
  comp_options.name = "dangling";
  comp_options.decomposition_subgraph_index = 7;  // Out of range.
  internal::TflOptions2 tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions2_StableHLOCompositeOptions;
  tfl_options.Set(std::move(comp_options));
  litert::internal::SetTflOptions2(comp_op, std::move(tfl_options));

  ShapeInferenceEngine engine(&model);
  EXPECT_EQ(engine.InferOpShapes(&comp_op),
            kLiteRtStatusErrorUnsupportedOpShapeInferer);
}

TEST(ShapeInferenceTest, CompositeOpSelfRecursionIsRejected) {
  // A composite op whose decomposition subgraph is the subgraph containing it.
  LiteRtModelT model;
  auto& sg = model.EmplaceSubgraph();
  auto& in = sg.EmplaceTensor();
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 4}));
  auto& out = sg.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 4}));
  sg.Inputs() = {&in};
  sg.Outputs() = {&out};
  auto& comp_op = sg.EmplaceOp();
  comp_op.SetOpCode(kLiteRtOpCodeShloComposite);
  AttachInput(&in, comp_op);
  AttachOutput(&out, comp_op);

  tflite::StableHLOCompositeOptionsT comp_options;
  comp_options.name = "self";
  comp_options.decomposition_subgraph_index = 0;
  internal::TflOptions2 tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions2_StableHLOCompositeOptions;
  tfl_options.Set(std::move(comp_options));
  litert::internal::SetTflOptions2(comp_op, std::move(tfl_options));

  ShapeInferenceEngine engine(&model);
  EXPECT_EQ(engine.InferOpShapes(&comp_op),
            kLiteRtStatusErrorShapeInferenceFailed);
}

}  // namespace
}  // namespace litert::internal
