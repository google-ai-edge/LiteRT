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

#include "litert/vendors/mediatek/compiler/transformations/rms_norm_quant_transformation.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/core/model/model.h"
#include "litert/test/load_test_model.h"

namespace litert::mediatek {
namespace {

using ::litert::ElementType;
using ::litert::compiler::Builder;
using ::litert::compiler::CompositeOptions;
using ::litert::compiler::GetOptionsAs;
using ::litert::compiler::Op;
using ::litert::compiler::RankedTensorSpecBuilder;
using ::litert::compiler::Tensor;

TEST(RmsNormQuantTransformationTest, QuantizedInt16ModelMatched) {
  auto model_wrap =
      litert::testing::LoadTestFileModel("rms_norm_composite_quantized.tflite");
  auto subgraph = model_wrap.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());

  LiteRtBuilderT builder;
  auto* compiler_ctx = LrtGetCompilerContext();

  auto ops = subgraph->Get()->Ops();
  ASSERT_FALSE(ops.empty());

  LiteRtOp rms_norm_op = nullptr;
  for (auto* op : ops) {
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      rms_norm_op = op;
      break;
    }
  }
  ASSERT_NE(rms_norm_op, nullptr);

  // Verify that the original composite op is INT16 quantized
  Op orig_comp_op(compiler_ctx, rms_norm_op);
  auto orig_inputs = orig_comp_op.Inputs();
  ASSERT_EQ(orig_inputs.size(), 2);
  EXPECT_EQ(orig_inputs[0].ElementType(), litert::ElementType::Int16);
  EXPECT_FALSE(orig_inputs[0].HasWeights());
  EXPECT_EQ(orig_inputs[1].ElementType(), litert::ElementType::Int16);
  EXPECT_TRUE(orig_inputs[1].HasWeights());

  auto orig_outputs = orig_comp_op.Outputs();
  ASSERT_EQ(orig_outputs.size(), 1);
  EXPECT_EQ(orig_outputs[0].ElementType(), litert::ElementType::Int16);

  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &builder, rms_norm_op),
            kLiteRtStatusOk);

  builder.ApplyChanges(subgraph->Get());

  int dequant_count = 0;
  int composite_count = 0;
  int quant_count = 0;
  LiteRtOp new_composite_op = nullptr;

  for (auto* op : subgraph->Get()->Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflDequantize) dequant_count++;
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      composite_count++;
      new_composite_op = op;
    }
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) quant_count++;
  }

  EXPECT_EQ(dequant_count, 1);
  EXPECT_EQ(composite_count, 1);
  EXPECT_EQ(quant_count, 1);
  ASSERT_NE(new_composite_op, nullptr);

  // Verify the new composite op has float32 inputs and output
  auto opts = GetOptionsAs<CompositeOptions>(compiler_ctx, new_composite_op);
  ASSERT_TRUE(opts.HasValue());
  EXPECT_EQ(opts->name, "odml.rms_norm");

  Op comp_op(compiler_ctx, new_composite_op);
  auto inputs = comp_op.Inputs();
  ASSERT_EQ(inputs.size(), 2);
  EXPECT_EQ(inputs[0].ElementType(), litert::ElementType::Float32);
  EXPECT_FALSE(inputs[0].HasWeights());

  EXPECT_EQ(inputs[1].ElementType(), litert::ElementType::Float32);
  EXPECT_TRUE(inputs[1].HasWeights());
  auto weights_span = inputs[1].WeightsData<float>();
  ASSERT_TRUE(weights_span.HasValue());
  EXPECT_EQ(weights_span->size(), 2304);
  for (size_t i = 0; i < 2304; ++i) {
    EXPECT_NEAR((*weights_span)[i], 1.0f, 1e-4f);
  }

  auto outputs = comp_op.Outputs();
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].ElementType(), litert::ElementType::Float32);
}

TEST(RmsNormQuantTransformationTest, QuantizedInt8ModelMatched) {
  auto model_wrap = litert::testing::LoadTestFileModel(
      "rms_norm_composite_quantized_int8.tflite");
  auto subgraph = model_wrap.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());

  LiteRtBuilderT builder;
  auto* compiler_ctx = LrtGetCompilerContext();

  auto ops = subgraph->Get()->Ops();
  ASSERT_FALSE(ops.empty());

  LiteRtOp rms_norm_op = nullptr;
  for (auto* op : ops) {
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      rms_norm_op = op;
      break;
    }
  }
  ASSERT_NE(rms_norm_op, nullptr);

  // Verify that the original composite op is INT8 quantized
  Op orig_comp_op(compiler_ctx, rms_norm_op);
  auto orig_inputs = orig_comp_op.Inputs();
  ASSERT_EQ(orig_inputs.size(), 2);
  EXPECT_EQ(orig_inputs[0].ElementType(), litert::ElementType::Int8);
  EXPECT_FALSE(orig_inputs[0].HasWeights());
  EXPECT_EQ(orig_inputs[1].ElementType(), litert::ElementType::Int8);
  EXPECT_TRUE(orig_inputs[1].HasWeights());

  auto orig_outputs = orig_comp_op.Outputs();
  ASSERT_EQ(orig_outputs.size(), 1);
  EXPECT_EQ(orig_outputs[0].ElementType(), litert::ElementType::Int8);

  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &builder, rms_norm_op),
            kLiteRtStatusOk);

  builder.ApplyChanges(subgraph->Get());

  int dequant_count = 0;
  int composite_count = 0;
  int quant_count = 0;
  LiteRtOp new_composite_op = nullptr;

  for (auto* op : subgraph->Get()->Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflDequantize) dequant_count++;
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      composite_count++;
      new_composite_op = op;
    }
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) quant_count++;
  }

  EXPECT_EQ(dequant_count, 1);
  EXPECT_EQ(composite_count, 1);
  EXPECT_EQ(quant_count, 1);
  ASSERT_NE(new_composite_op, nullptr);

  auto opts_int8 =
      GetOptionsAs<CompositeOptions>(compiler_ctx, new_composite_op);
  ASSERT_TRUE(opts_int8.HasValue());
  EXPECT_EQ(opts_int8->name, "odml.rms_norm");

  Op comp_op_int8(compiler_ctx, new_composite_op);
  auto inputs = comp_op_int8.Inputs();
  ASSERT_EQ(inputs.size(), 2);
  EXPECT_EQ(inputs[0].ElementType(), litert::ElementType::Float32);
  EXPECT_FALSE(inputs[0].HasWeights());

  EXPECT_EQ(inputs[1].ElementType(), litert::ElementType::Float32);
  EXPECT_TRUE(inputs[1].HasWeights());
  auto weights_span = inputs[1].WeightsData<float>();
  ASSERT_TRUE(weights_span.HasValue());
  EXPECT_EQ(weights_span->size(), 2304);
  for (size_t i = 0; i < 2304; ++i) {
    EXPECT_NEAR((*weights_span)[i], 1.0f, 1e-4f);
  }

  auto outputs = comp_op_int8.Outputs();
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].ElementType(), litert::ElementType::Float32);
}

TEST(RmsNormQuantTransformationTest, Float32ModelNonMatched) {
  auto model_wrap =
      litert::testing::LoadTestFileModel("rms_norm_composite.tflite");
  auto subgraph = model_wrap.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());

  LiteRtBuilderT builder;
  auto* compiler_ctx = LrtGetCompilerContext();

  auto ops = subgraph->Get()->Ops();
  ASSERT_FALSE(ops.empty());

  LiteRtOp rms_norm_op = nullptr;
  for (auto* op : ops) {
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      rms_norm_op = op;
      break;
    }
  }
  ASSERT_NE(rms_norm_op, nullptr);

  // Pure Float32 rms_norm should not match the quantization transformation.
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &builder, rms_norm_op),
            kLiteRtStatusPatternNoMatch);
}

TEST(RmsNormQuantTransformationTest, ConstantGammaFoldedToFloat32) {
  const LiteRtCompilerContext* compiler_ctx = LrtGetCompilerContext();
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();

  LiteRtBuilderT setup_builder;
  Builder cc_setup(compiler_ctx, &setup_builder);

  // 1. Quantized input tensor (activation)
  int32_t in_dims[] = {1, 128, 4};
  auto in_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int16,
              litert::Layout(litert::BuildLayout(in_dims, in_dims + 3))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.05f, 0})
          .Build();
  auto in_tensor = cc_setup.BuildTensor(in_spec);
  ASSERT_TRUE(in_tensor.HasValue());

  // 2. Quantized constant gamma tensor with weights
  int32_t gamma_dims[] = {4};
  auto gamma_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int16,
              litert::Layout(litert::BuildLayout(gamma_dims, gamma_dims + 1))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.002f, 0})
          .Build();
  auto gamma_tensor = cc_setup.BuildTensor(gamma_spec);
  ASSERT_TRUE(gamma_tensor.HasValue());

  const int16_t kGammaRaw[] = {1000, 2000, 3000, 4000};
  auto gamma_weights = cc_setup.BuildWeights<int16_t>(
      absl::MakeConstSpan(kGammaRaw), *gamma_tensor);
  ASSERT_TRUE(gamma_weights.HasValue());

  // 3. Quantized output tensor
  int32_t out_dims[] = {1, 128, 4};
  auto out_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int16,
              litert::Layout(litert::BuildLayout(out_dims, out_dims + 3))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.05f, 0})
          .Build();
  auto out_tensor = cc_setup.BuildTensor(out_spec);
  ASSERT_TRUE(out_tensor.HasValue());

  // 4. Composite op
  auto composite_op = cc_setup.BuildOp(
      kLiteRtOpCodeShloComposite, {*in_tensor, *gamma_tensor}, {*out_tensor});
  ASSERT_TRUE(composite_op.HasValue());

  CompositeOptions options;
  options.name = "odml.rms_norm";
  options.subgraph = 0;
  options.version = 0;
  auto set_opts_res = cc_setup.SetOpOptions<CompositeOptions>(
      *composite_op, std::move(options));
  ASSERT_TRUE(set_opts_res.HasValue());

  // Transfer created ops/tensors to model subgraph
  setup_builder.ApplyChanges(&subgraph);

  // Now run RmsNormQuantTransformation!
  LiteRtBuilderT trans_builder;
  auto root_op = subgraph.Ops()[0];
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &trans_builder, root_op),
            kLiteRtStatusOk);
  trans_builder.ApplyChanges(&subgraph);

  // Verify:
  // 1. Only ONE dequantize op was added (for input). Gamma was folded offline!
  int dequant_count = 0;
  int quant_count = 0;
  int composite_count = 0;
  LiteRtOp new_composite_op = nullptr;

  for (auto* op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflDequantize) dequant_count++;
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) quant_count++;
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      composite_count++;
      new_composite_op = op;
    }
  }

  EXPECT_EQ(dequant_count, 1);
  EXPECT_EQ(quant_count, 1);
  EXPECT_EQ(composite_count, 1);
  ASSERT_NE(new_composite_op, nullptr);

  Op new_op(compiler_ctx, new_composite_op);
  auto inputs = new_op.Inputs();
  ASSERT_EQ(inputs.size(), 2);

  // Input 0: Float32 activation from dequantize (no static weights)
  EXPECT_EQ(inputs[0].ElementType(), litert::ElementType::Float32);
  EXPECT_FALSE(inputs[0].HasWeights());

  // Input 1: Float32 constant tensor with folded weights!
  EXPECT_EQ(inputs[1].ElementType(), litert::ElementType::Float32);
  EXPECT_TRUE(inputs[1].HasWeights());

  auto weights_span = inputs[1].WeightsData<float>();
  ASSERT_TRUE(weights_span.HasValue());
  ASSERT_EQ(weights_span->size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR((*weights_span)[i], kGammaRaw[i] * 0.002f, 1e-5f);
  }
}

TEST(RmsNormQuantTransformationTest, DynamicActivationGammaMatched) {
  const LiteRtCompilerContext* compiler_ctx = LrtGetCompilerContext();
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();

  LiteRtBuilderT setup_builder;
  Builder cc_setup(compiler_ctx, &setup_builder);

  int32_t in_dims[] = {1, 128, 4};
  auto in_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(in_dims, in_dims + 3))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.05f, 0})
          .Build();
  auto in_tensor = cc_setup.BuildTensor(in_spec);
  ASSERT_TRUE(in_tensor.HasValue());

  int32_t gamma_dims[] = {4};
  auto gamma_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(gamma_dims, gamma_dims + 1))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.01f, 0})
          .Build();
  // Dynamic gamma: no weights attached
  auto gamma_tensor = cc_setup.BuildTensor(gamma_spec);
  ASSERT_TRUE(gamma_tensor.HasValue());

  int32_t out_dims[] = {1, 128, 4};
  auto out_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(out_dims, out_dims + 3))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.05f, 0})
          .Build();
  auto out_tensor = cc_setup.BuildTensor(out_spec);
  ASSERT_TRUE(out_tensor.HasValue());

  auto composite_op = cc_setup.BuildOp(
      kLiteRtOpCodeShloComposite, {*in_tensor, *gamma_tensor}, {*out_tensor});
  ASSERT_TRUE(composite_op.HasValue());

  CompositeOptions options;
  options.name = "odml.rms_norm";
  options.subgraph = 0;
  options.version = 0;
  auto set_opts_res = cc_setup.SetOpOptions<CompositeOptions>(
      *composite_op, std::move(options));
  ASSERT_TRUE(set_opts_res.HasValue());

  setup_builder.ApplyChanges(&subgraph);

  LiteRtBuilderT trans_builder;
  auto root_op = subgraph.Ops()[0];
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &trans_builder, root_op),
            kLiteRtStatusOk);
  trans_builder.ApplyChanges(&subgraph);

  int dequant_count = 0;
  int quant_count = 0;
  int composite_count = 0;
  LiteRtOp new_composite_op = nullptr;

  for (auto* op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflDequantize) dequant_count++;
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) quant_count++;
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      composite_count++;
      new_composite_op = op;
    }
  }

  // Since gamma is dynamic (no weights), both input and gamma get dequantized.
  EXPECT_EQ(dequant_count, 2);
  EXPECT_EQ(quant_count, 1);
  EXPECT_EQ(composite_count, 1);
  ASSERT_NE(new_composite_op, nullptr);

  Op new_op(compiler_ctx, new_composite_op);
  auto inputs = new_op.Inputs();
  ASSERT_EQ(inputs.size(), 2);
  EXPECT_EQ(inputs[0].ElementType(), litert::ElementType::Float32);
  EXPECT_FALSE(inputs[0].HasWeights());
  EXPECT_EQ(inputs[1].ElementType(), litert::ElementType::Float32);
  EXPECT_FALSE(inputs[1].HasWeights());
}

TEST(RmsNormQuantTransformationTest, ArityMismatchIgnored) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto* compiler_ctx = LrtGetCompilerContext();

  LiteRtBuilderT setup_builder;
  Builder cc_setup(compiler_ctx, &setup_builder);

  int32_t dims[] = {1, 16};
  auto in_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(dims, dims + 2))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.1f, 0})
          .Build();
  auto in_tensor = cc_setup.BuildTensor(in_spec);
  ASSERT_TRUE(in_tensor.HasValue());

  int32_t out_dims[] = {1, 16};
  auto out_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(out_dims, out_dims + 2))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.1f, 0})
          .Build();
  auto out_tensor = cc_setup.BuildTensor(out_spec);
  ASSERT_TRUE(out_tensor.HasValue());

  // Only 1 input instead of 2
  auto composite_op =
      cc_setup.BuildOp(kLiteRtOpCodeShloComposite, {*in_tensor}, {*out_tensor});
  ASSERT_TRUE(composite_op.HasValue());

  CompositeOptions options;
  options.name = "odml.rms_norm";
  options.subgraph = 0;
  options.version = 0;
  auto set_opts_res = cc_setup.SetOpOptions<CompositeOptions>(
      *composite_op, std::move(options));
  ASSERT_TRUE(set_opts_res.HasValue());

  setup_builder.ApplyChanges(&subgraph);

  LiteRtBuilderT trans_builder;
  auto root_op = subgraph.Ops()[0];
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &trans_builder, root_op),
            kLiteRtStatusPatternNoMatch);
}

TEST(RmsNormQuantTransformationTest, CorruptedWeightsBufferSizeReturnsError) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto* compiler_ctx = LrtGetCompilerContext();

  LiteRtBuilderT setup_builder;
  Builder cc_setup(compiler_ctx, &setup_builder);

  int32_t in_dims[] = {1, 4};
  auto in_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int16,
              litert::Layout(litert::BuildLayout(in_dims, in_dims + 2))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.1f, 0})
          .Build();
  auto in_tensor = cc_setup.BuildTensor(in_spec);
  ASSERT_TRUE(in_tensor.HasValue());

  // Gamma specifies 4 Int16 elements (requires 8 bytes), but we only provide 2
  // bytes
  int32_t gamma_dims[] = {4};
  auto gamma_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int16,
              litert::Layout(litert::BuildLayout(gamma_dims, gamma_dims + 1))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.002f, 0})
          .Build();
  auto gamma_tensor = cc_setup.BuildTensor(gamma_spec);
  ASSERT_TRUE(gamma_tensor.HasValue());

  const uint8_t corrupted_bytes[] = {0x01, 0x02};
  auto gamma_weights = cc_setup.BuildWeights<uint8_t>(
      absl::MakeConstSpan(corrupted_bytes), *gamma_tensor);
  ASSERT_TRUE(gamma_weights.HasValue());

  int32_t out_dims[] = {1, 4};
  auto out_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int16,
              litert::Layout(litert::BuildLayout(out_dims, out_dims + 2))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.05f, 0})
          .Build();
  auto out_tensor = cc_setup.BuildTensor(out_spec);
  ASSERT_TRUE(out_tensor.HasValue());

  auto composite_op = cc_setup.BuildOp(
      kLiteRtOpCodeShloComposite, {*in_tensor, *gamma_tensor}, {*out_tensor});
  ASSERT_TRUE(composite_op.HasValue());

  CompositeOptions options;
  options.name = "odml.rms_norm";
  options.subgraph = 0;
  options.version = 0;
  ASSERT_TRUE(
      cc_setup.SetOpOptions<CompositeOptions>(*composite_op, std::move(options))
          .HasValue());

  setup_builder.ApplyChanges(&subgraph);

  LiteRtBuilderT trans_builder;
  auto root_op = subgraph.Ops()[0];
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &trans_builder, root_op),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(RmsNormQuantTransformationTest, PerChannelSymmetricQuantizationSucceeds) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto* compiler_ctx = LrtGetCompilerContext();

  LiteRtBuilderT setup_builder;
  Builder cc_setup(compiler_ctx, &setup_builder);

  int32_t in_dims[] = {1, 4};
  auto in_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(in_dims, in_dims + 2))))
          .Build();
  auto in_tensor = cc_setup.BuildTensor(in_spec);
  ASSERT_TRUE(in_tensor.HasValue());

  int32_t gamma_dims[] = {4};
  float scales[] = {0.1f, 0.2f, 0.3f, 0.4f};
  LiteRtQuantizationPerChannel per_channel_q{};
  per_channel_q.num_channels = 4;
  per_channel_q.quantized_dimension = 0;
  per_channel_q.scales = scales;
  per_channel_q.zero_points = nullptr;  // Symmetric per-channel

  auto gamma_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(gamma_dims, gamma_dims + 1))))
          .WithPerChannelQuantization(per_channel_q)
          .Build();
  auto gamma_tensor = cc_setup.BuildTensor(gamma_spec);
  ASSERT_TRUE(gamma_tensor.HasValue());

  const int8_t raw_gamma[] = {10, -20, 30, -40};
  auto gamma_weights = cc_setup.BuildWeights<int8_t>(
      absl::MakeConstSpan(raw_gamma), *gamma_tensor);
  ASSERT_TRUE(gamma_weights.HasValue());

  int32_t out_dims[] = {1, 4};
  auto out_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(out_dims, out_dims + 2))))
          .Build();
  auto out_tensor = cc_setup.BuildTensor(out_spec);
  ASSERT_TRUE(out_tensor.HasValue());

  auto composite_op = cc_setup.BuildOp(
      kLiteRtOpCodeShloComposite, {*in_tensor, *gamma_tensor}, {*out_tensor});
  ASSERT_TRUE(composite_op.HasValue());

  CompositeOptions options;
  options.name = "odml.rms_norm";
  options.subgraph = 0;
  options.version = 0;
  ASSERT_TRUE(
      cc_setup.SetOpOptions<CompositeOptions>(*composite_op, std::move(options))
          .HasValue());

  setup_builder.ApplyChanges(&subgraph);

  LiteRtBuilderT trans_builder;
  auto root_op = subgraph.Ops()[0];
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &trans_builder, root_op),
            kLiteRtStatusOk);
  trans_builder.ApplyChanges(&subgraph);

  LiteRtOp new_composite = nullptr;
  for (auto* op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      new_composite = op;
      break;
    }
  }
  ASSERT_NE(new_composite, nullptr);
  Op comp_op(compiler_ctx, new_composite);
  auto inputs = comp_op.Inputs();
  ASSERT_EQ(inputs.size(), 2);
  EXPECT_TRUE(inputs[1].HasWeights());
  auto f32_weights = inputs[1].WeightsData<float>();
  ASSERT_TRUE(f32_weights.HasValue());
  ASSERT_EQ(f32_weights->size(), 4);
  EXPECT_NEAR((*f32_weights)[0], 10 * 0.1f, 1e-5f);
  EXPECT_NEAR((*f32_weights)[1], -20 * 0.2f, 1e-5f);
  EXPECT_NEAR((*f32_weights)[2], 30 * 0.3f, 1e-5f);
  EXPECT_NEAR((*f32_weights)[3], -40 * 0.4f, 1e-5f);
}

TEST(RmsNormQuantTransformationTest,
     DynamicDimensionsWeightsReturnsUnsupported) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto* compiler_ctx = LrtGetCompilerContext();

  LiteRtBuilderT setup_builder;
  Builder cc_setup(compiler_ctx, &setup_builder);

  int32_t in_dims[] = {1, 4};
  auto in_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(in_dims, in_dims + 2))))
          .Build();
  auto in_tensor = cc_setup.BuildTensor(in_spec);
  ASSERT_TRUE(in_tensor.HasValue());

  int32_t gamma_dims[] = {-1};
  auto gamma_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(gamma_dims, gamma_dims + 1))))
          .WithPerTensorQuantization(LiteRtQuantizationPerTensor{0.1f, 0})
          .Build();
  auto gamma_tensor = cc_setup.BuildTensor(gamma_spec);
  ASSERT_TRUE(gamma_tensor.HasValue());

  const int8_t raw_gamma[] = {1, 2, 3, 4};
  auto gamma_weights = cc_setup.BuildWeights<int8_t>(
      absl::MakeConstSpan(raw_gamma), *gamma_tensor);
  ASSERT_TRUE(gamma_weights.HasValue());

  int32_t out_dims[] = {1, 4};
  auto out_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(out_dims, out_dims + 2))))
          .Build();
  auto out_tensor = cc_setup.BuildTensor(out_spec);
  ASSERT_TRUE(out_tensor.HasValue());

  auto composite_op = cc_setup.BuildOp(
      kLiteRtOpCodeShloComposite, {*in_tensor, *gamma_tensor}, {*out_tensor});
  ASSERT_TRUE(composite_op.HasValue());

  CompositeOptions options;
  options.name = "odml.rms_norm";
  options.subgraph = 0;
  options.version = 0;
  ASSERT_TRUE(
      cc_setup.SetOpOptions<CompositeOptions>(*composite_op, std::move(options))
          .HasValue());

  setup_builder.ApplyChanges(&subgraph);

  LiteRtBuilderT trans_builder;
  auto root_op = subgraph.Ops()[0];
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &trans_builder, root_op),
            kLiteRtStatusErrorUnsupported);
}

TEST(RmsNormQuantTransformationTest, BlockWiseQuantizedInputGracefullySkipped) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto* compiler_ctx = LrtGetCompilerContext();

  LiteRtBuilderT setup_builder;
  Builder cc_setup(compiler_ctx, &setup_builder);

  int32_t in_dims[] = {1, 32};
  LiteRtQuantizationBlockWise bw_quant{nullptr, nullptr, 32};
  auto in_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(in_dims, in_dims + 2))))
          .WithBlockWiseQuantization(bw_quant)
          .Build();
  auto in_tensor = cc_setup.BuildTensor(in_spec);
  ASSERT_TRUE(in_tensor.HasValue());

  int32_t gamma_dims[] = {32};
  auto gamma_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(gamma_dims, gamma_dims + 1))))
          .Build();
  auto gamma_tensor = cc_setup.BuildTensor(gamma_spec);
  ASSERT_TRUE(gamma_tensor.HasValue());

  int32_t out_dims[] = {1, 32};
  auto out_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(out_dims, out_dims + 2))))
          .Build();
  auto out_tensor = cc_setup.BuildTensor(out_spec);
  ASSERT_TRUE(out_tensor.HasValue());

  auto composite_op = cc_setup.BuildOp(
      kLiteRtOpCodeShloComposite, {*in_tensor, *gamma_tensor}, {*out_tensor});
  ASSERT_TRUE(composite_op.HasValue());

  CompositeOptions options;
  options.name = "odml.rms_norm";
  options.subgraph = 0;
  options.version = 0;
  ASSERT_TRUE(
      cc_setup.SetOpOptions<CompositeOptions>(*composite_op, std::move(options))
          .HasValue());

  setup_builder.ApplyChanges(&subgraph);

  LiteRtBuilderT trans_builder;
  auto root_op = subgraph.Ops()[0];
  // Block-wise quantization should be gracefully skipped (PatternNoMatch).
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &trans_builder, root_op),
            kLiteRtStatusPatternNoMatch);
}

TEST(RmsNormQuantTransformationTest, BlockWiseQuantizedGammaGracefullySkipped) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto* compiler_ctx = LrtGetCompilerContext();

  LiteRtBuilderT setup_builder;
  Builder cc_setup(compiler_ctx, &setup_builder);

  int32_t in_dims[] = {1, 32};
  auto in_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(in_dims, in_dims + 2))))
          .Build();
  auto in_tensor = cc_setup.BuildTensor(in_spec);
  ASSERT_TRUE(in_tensor.HasValue());

  int32_t gamma_dims[] = {32};
  LiteRtQuantizationBlockWise bw_quant{nullptr, nullptr, 32};
  auto gamma_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(gamma_dims, gamma_dims + 1))))
          .WithBlockWiseQuantization(bw_quant)
          .Build();
  auto gamma_tensor = cc_setup.BuildTensor(gamma_spec);
  ASSERT_TRUE(gamma_tensor.HasValue());

  int32_t out_dims[] = {1, 32};
  auto out_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(out_dims, out_dims + 2))))
          .Build();
  auto out_tensor = cc_setup.BuildTensor(out_spec);
  ASSERT_TRUE(out_tensor.HasValue());

  auto composite_op = cc_setup.BuildOp(
      kLiteRtOpCodeShloComposite, {*in_tensor, *gamma_tensor}, {*out_tensor});
  ASSERT_TRUE(composite_op.HasValue());

  CompositeOptions options;
  options.name = "odml.rms_norm";
  options.subgraph = 0;
  options.version = 0;
  ASSERT_TRUE(
      cc_setup.SetOpOptions<CompositeOptions>(*composite_op, std::move(options))
          .HasValue());

  setup_builder.ApplyChanges(&subgraph);

  LiteRtBuilderT trans_builder;
  auto root_op = subgraph.Ops()[0];
  // Block-wise quantization should be gracefully skipped (PatternNoMatch).
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &trans_builder, root_op),
            kLiteRtStatusPatternNoMatch);
}

TEST(RmsNormQuantTransformationTest,
     BlockWiseQuantizedOutputGracefullySkipped) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto* compiler_ctx = LrtGetCompilerContext();

  LiteRtBuilderT setup_builder;
  Builder cc_setup(compiler_ctx, &setup_builder);

  int32_t in_dims[] = {1, 32};
  auto in_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(in_dims, in_dims + 2))))
          .Build();
  auto in_tensor = cc_setup.BuildTensor(in_spec);
  ASSERT_TRUE(in_tensor.HasValue());

  int32_t gamma_dims[] = {32};
  auto gamma_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(litert::BuildLayout(gamma_dims, gamma_dims + 1))))
          .Build();
  auto gamma_tensor = cc_setup.BuildTensor(gamma_spec);
  ASSERT_TRUE(gamma_tensor.HasValue());

  int32_t out_dims[] = {1, 32};
  LiteRtQuantizationBlockWise bw_quant{nullptr, nullptr, 32};
  auto out_spec =
      RankedTensorSpecBuilder(
          litert::RankedTensorType(
              litert::ElementType::Int8,
              litert::Layout(litert::BuildLayout(out_dims, out_dims + 2))))
          .WithBlockWiseQuantization(bw_quant)
          .Build();
  auto out_tensor = cc_setup.BuildTensor(out_spec);
  ASSERT_TRUE(out_tensor.HasValue());

  auto composite_op = cc_setup.BuildOp(
      kLiteRtOpCodeShloComposite, {*in_tensor, *gamma_tensor}, {*out_tensor});
  ASSERT_TRUE(composite_op.HasValue());

  CompositeOptions options;
  options.name = "odml.rms_norm";
  options.subgraph = 0;
  options.version = 0;
  ASSERT_TRUE(
      cc_setup.SetOpOptions<CompositeOptions>(*composite_op, std::move(options))
          .HasValue());

  setup_builder.ApplyChanges(&subgraph);

  LiteRtBuilderT trans_builder;
  auto root_op = subgraph.Ops()[0];
  // Block-wise quantization should be gracefully skipped (PatternNoMatch).
  EXPECT_EQ(RmsNormQuantTransformation(compiler_ctx, &trans_builder, root_op),
            kLiteRtStatusPatternNoMatch);
}

}  // namespace
}  // namespace litert::mediatek
