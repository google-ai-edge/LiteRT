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

#include "litert/cc/litert_op_options.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
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
}

}  // namespace
}  // namespace litert
