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

#include "litert/cc/internal/litert_op_options.h"

#include <cstdint>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace {

TEST(OpOptionsTest, GetCompositeOptions) {
  static constexpr auto kOptsType =
      ::tflite::BuiltinOptions2_StableHLOCompositeOptions;
  static constexpr absl::string_view kName = "test.composite";
  static constexpr int kSubgraph = 1;

  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloComposite);

  tflite::StableHLOCompositeOptionsT options;
  options.name = kName;
  options.decomposition_subgraph_index = kSubgraph;

  internal::TflOptions2 tfl_options;
  tfl_options.type = kOptsType;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions2(op, std::move(tfl_options));

  auto res = GetOptionsAs<CompositeOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->name, kName);
  EXPECT_EQ(res->subgraph, kSubgraph);
  EXPECT_FALSE(res->attributes_map.has_value());
}

TEST(OpOptionsTest, GetUnsupportedOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloAdd);
  ASSERT_FALSE(GetOptionsAs<CompositeOptions>(&op));
}

TEST(OpOptionsTest, CompositeOptionsGetNameVersionAndAttributes) {
  auto model = testing::LoadTestFileModel("simple_shlo_composite.tflite");
  auto subgraph = model.MainSubgraph();
  auto stablehlo_add_n_op = subgraph->Ops().front().Get();
  auto info = GetOptionsAs<CompositeOptions>(stablehlo_add_n_op);
  ASSERT_TRUE(info);

  EXPECT_EQ(info->name, "stablehlo.add_n");
  EXPECT_EQ(info->version, 3);
  EXPECT_TRUE(info->attributes_map.has_value());
  EXPECT_STREQ(info->attributes_map.value()["an_attribute"].AsString().c_str(),
               "foo");
  EXPECT_EQ(info->attributes_map.value()["meaning_of_life"].AsInt32(), 42);
}

TEST(OpOptionsTest, GetRmsNormEpsilon) {
  auto model = testing::LoadTestFileModel("rms_norm_composite.tflite");
  auto subgraph = model.MainSubgraph();
  auto rms_norm_composite_op = subgraph->Ops().front().Get();
  auto info = GetOptionsAs<RmsNormOpts>(rms_norm_composite_op);

  ASSERT_TRUE(info);
  float epsilon = info->epsilon;
  ASSERT_TRUE(epsilon);
  EXPECT_FLOAT_EQ(epsilon, 9.99999997E-7);
}

TEST(OpOptionsTest, GetRmsNormEpsilonFromSimpleComposite) {
  auto model = testing::LoadTestFileModel("simple_shlo_composite.tflite");
  auto subgraph = model.MainSubgraph();
  auto rms_norm_composite_op = subgraph->Ops().front().Get();
  litert::Expected<RmsNormOpts> info =
      GetOptionsAs<RmsNormOpts>(rms_norm_composite_op);

  EXPECT_FALSE(info);
  EXPECT_EQ(info.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(OpOptionsTest, GetAddOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  tflite::AddOptionsT options;
  options.fused_activation_function = tflite::ActivationFunctionType_NONE;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_AddOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<AddOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeNone);
  EXPECT_NE(res->fused_activation_function, kActivationFunctionTypeRelu);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetBatchMatmulOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  tflite::BatchMatMulOptionsT options;
  options.adj_x = false;
  options.adj_y = false;
  options.asymmetric_quantize_inputs = true;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<BatchMatmulOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->adj_x, false);
  EXPECT_EQ(res->adj_y, false);
  EXPECT_EQ(res->asymmetric_quantize_input, true);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetConcatenationOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  tflite::ConcatenationOptionsT options;
  options.axis = 1;
  options.fused_activation_function = tflite::ActivationFunctionType_NONE;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<ConcatenationOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->axis, 1);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeNone);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetDivOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflDiv);
  tflite::DivOptionsT options;
  options.fused_activation_function = tflite::ActivationFunctionType_RELU;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_DivOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<DivOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeRelu);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetFullyConnectedOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflFullyConnected);
  tflite::FullyConnectedOptionsT options;
  options.fused_activation_function = tflite::ActivationFunctionType_RELU;
  options.keep_num_dims = true;
  options.quantized_bias_type = tflite::TensorType_FLOAT32;
  options.asymmetric_quantize_inputs = false;
  options.weights_format =
      tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8;

  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_FullyConnectedOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<FullyConnectedOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeRelu);
  EXPECT_EQ(res->keep_num_dims, true);
  EXPECT_EQ(res->quantized_bias_type, kLiteRtElementTypeFloat32);
  EXPECT_EQ(res->asymmetric_quantize_input, false);
  EXPECT_EQ(res->weights_format,
            kFullyConnectedOptionsWeightsFormatShuffled4x16Int8);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetMulOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflMul);
  tflite::MulOptionsT options;
  options.fused_activation_function =
      tflite::ActivationFunctionType_RELU_N1_TO_1;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_MulOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<MulOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeReluN1To1);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetSoftmaxOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflSoftmax);
  tflite::SoftmaxOptionsT options;
  options.beta = 1.0;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_SoftmaxOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<SoftmaxOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->beta, 1.0);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetStridedSliceOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflStridedSlice);
  tflite::StridedSliceOptionsT options;
  options.begin_mask = 1;
  options.end_mask = 2;
  options.ellipsis_mask = 3;
  options.new_axis_mask = 4;
  options.shrink_axis_mask = 5;
  options.offset = true;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_StridedSliceOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<StridedSliceOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->begin_mask, 1);
  EXPECT_EQ(res->end_mask, 2);
  EXPECT_EQ(res->ellipsis_mask, 3);
  EXPECT_EQ(res->new_axis_mask, 4);
  EXPECT_EQ(res->shrink_axis_mask, 5);
  EXPECT_EQ(res->offset, true);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetSubOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflSub);
  tflite::SubOptionsT options;
  options.fused_activation_function = tflite::ActivationFunctionType_TANH;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_SubOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<SubOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeTanh);
  EXPECT_EQ(&op, res->op);
}

TEST(GetOpOptionTest, TestGetReshapeOptions2x3To3x2) {
  LiteRtModelT model_t;
  auto& subgraph = model_t.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflReshape);

  LiteRtTensorT tensor;
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 3}));
  op.Inputs().push_back(&tensor);

  int32_t kTensorData[] = {3, 2};
  LiteRtTensorT tensor2;
  tensor2.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));
  auto& weights = tensor2.Weights();
  weights.SetBufferManager(model_t.Buffers());

  litert::BufferRef<uint8_t> buffer(kTensorData, sizeof(kTensorData));
  litert::internal::BufferContext context;
  context.should_append = true;
  SetWeightsFromUnownedBuffer(weights, std::move(buffer), context);

  op.Inputs().push_back(&tensor2);
  LiteRtTensorT tensor3;
  tensor3.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3, 2}));
  op.Outputs().push_back(&tensor3);

  auto res = GetOptionsAs<ReshapeOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->new_shape[0], 3);
  EXPECT_EQ(res->new_shape[1], 2);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetSumOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflSum);
  tflite::ReducerOptionsT options;
  options.keep_dims = true;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_ReducerOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<SumOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->keep_dims, true);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetReduceMaxOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflReduceMax);
  tflite::ReducerOptionsT options;
  options.keep_dims = false;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_ReducerOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<ReduceMaxOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->keep_dims, false);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetPackOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflPack);
  tflite::PackOptionsT options;
  options.axis = 1;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_PackOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<PackOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->axis, 1);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetGatherOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflGather);
  tflite::GatherOptionsT options;
  options.axis = 1;
  options.batch_dims = 2;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_GatherOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<GatherOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->axis, 1);
  EXPECT_EQ(res->batch_dims, 2);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetMeanOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflMean);
  tflite::ReducerOptionsT options;
  options.keep_dims = true;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_ReducerOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<MeanOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->keep_dims, true);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetSplitOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflSplit);
  tflite::SplitOptionsT options;
  options.num_splits = 3;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_SplitOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<SplitOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->num_splits, 3);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetConv2dOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflConv2d);
  tflite::Conv2DOptionsT options;
  options.padding = tflite::Padding_SAME;
  options.stride_w = 1;
  options.stride_h = 2;
  options.dilation_w_factor = 3;
  options.dilation_h_factor = 4;
  options.fused_activation_function = tflite::ActivationFunctionType_SIGN_BIT;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<Conv2dOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->padding, kPaddingSame);
  EXPECT_EQ(res->stride_w, 1);
  EXPECT_EQ(res->stride_h, 2);
  EXPECT_EQ(res->dilation_w_factor, 3);
  EXPECT_EQ(res->dilation_h_factor, 4);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeSignBit);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetConv3dOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflConv3d);
  tflite::Conv3DOptionsT options;
  options.padding = tflite::Padding_SAME;
  options.stride_w = 1;
  options.stride_h = 2;
  options.stride_d = 3;
  options.dilation_w_factor = 3;
  options.dilation_h_factor = 4;
  options.dilation_d_factor = 5;
  options.fused_activation_function = tflite::ActivationFunctionType_NONE;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_Conv3DOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<Conv3dOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->padding, kPaddingSame);
  EXPECT_EQ(res->stride_w, 1);
  EXPECT_EQ(res->stride_h, 2);
  EXPECT_EQ(res->stride_d, 3);
  EXPECT_EQ(res->dilation_w_factor, 3);
  EXPECT_EQ(res->dilation_h_factor, 4);
  EXPECT_EQ(res->dilation_d_factor, 5);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeNone);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetAveragePool2dOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflAveragePool2d);
  tflite::Pool2DOptionsT options;
  options.padding = tflite::Padding_VALID;
  options.stride_w = 1;
  options.stride_h = 2;
  options.filter_width = 3;
  options.filter_height = 4;
  options.fused_activation_function = tflite::ActivationFunctionType_RELU;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<AveragePool2dOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->padding, kPaddingValid);
  EXPECT_EQ(res->stride_w, 1);
  EXPECT_EQ(res->stride_h, 2);
  EXPECT_EQ(res->filter_width, 3);
  EXPECT_EQ(res->filter_height, 4);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeRelu);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetMaxPool2dOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflMaxPool2d);
  tflite::Pool2DOptionsT options;
  options.padding = tflite::Padding_VALID;
  options.stride_w = 1;
  options.stride_h = 2;
  options.filter_width = 3;
  options.filter_height = 4;
  options.fused_activation_function = tflite::ActivationFunctionType_RELU;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<MaxPool2dOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->padding, kPaddingValid);
  EXPECT_EQ(res->stride_w, 1);
  EXPECT_EQ(res->stride_h, 2);
  EXPECT_EQ(res->filter_width, 3);
  EXPECT_EQ(res->filter_height, 4);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeRelu);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetResizeBilinearOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflResizeBilinear);
  tflite::ResizeBilinearOptionsT options;
  options.align_corners = true;
  options.half_pixel_centers = false;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_ResizeBilinearOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<ResizeBilinearOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->align_corners, true);
  EXPECT_EQ(res->half_pixel_centers, false);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetLeakyReluOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflLeakyRelu);
  tflite::LeakyReluOptionsT options;
  options.alpha = 0.1;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_LeakyReluOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<LeakyReluOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_FLOAT_EQ(res->alpha, 0.1);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetSpaceToDepthOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflSpaceToDepth);
  tflite::SpaceToDepthOptionsT options;
  options.block_size = 1;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_SpaceToDepthOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<SpaceToDepthOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->block_size, 1);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetDepthToSpaceOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflDepthToSpace);
  tflite::DepthToSpaceOptionsT options;
  options.block_size = 1;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_DepthToSpaceOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<DepthToSpaceOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->block_size, 1);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetResizeNearestNeighborOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflResizeNearestNeighbor);
  tflite::ResizeNearestNeighborOptionsT options;
  options.align_corners = true;
  options.half_pixel_centers = false;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_ResizeNearestNeighborOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<ResizeNearestNeighborOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->align_corners, true);
  EXPECT_EQ(res->half_pixel_centers, false);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetCumSumOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflCumsum);
  tflite::CumsumOptionsT options;
  options.exclusive = true;
  options.reverse = false;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_CumsumOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<CumSumOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->exclusive, true);
  EXPECT_EQ(res->reverse, false);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetGeluOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflGelu);
  tflite::GeluOptionsT options;
  options.approximate = true;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_GeluOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<GeluOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->approximate, true);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, GetMirrorPadOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflMirrorPad);
  tflite::MirrorPadOptionsT options;
  options.mode = tflite::MirrorPadMode_REFLECT;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_MirrorPadOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<MirrorPadOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->mode, kMirrorPadModeReflect);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, TestGetOptionsAsInvalidOpOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloComposite);
  ASSERT_FALSE(GetOptionsAs<AddOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<BatchMatmulOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<ConcatenationOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<DivOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<FullyConnectedOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<MulOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<SoftmaxOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<StridedSliceOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<SubOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<ReshapeOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<SumOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<ReduceMaxOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<PackOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<GatherOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<MeanOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<SplitOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<Conv2dOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<Conv3dOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<AveragePool2dOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<MaxPool2dOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<ResizeBilinearOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<LeakyReluOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<SpaceToDepthOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<DepthToSpaceOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<ResizeNearestNeighborOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<CumSumOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<GeluOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<MirrorPadOptions>(&op));
}

TEST(OpOptionsTest, TestSetAddOpOptionsSuccess) {
  LiteRtBuilderT builder;
  auto& op = builder.BuildOp(kLiteRtOpCodeTflAdd, {}, {});
  {
    AddOptions options = {};
    options.fused_activation_function = kActivationFunctionTypeRelu;
    options.op = &op;
    options.SetOpOptions(&builder);
  }
  auto res = GetOptionsAs<AddOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeRelu);
}

}  // namespace
}  // namespace litert
