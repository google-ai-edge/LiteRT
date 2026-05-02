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

#include "litert/test/generators/fully_connected.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/core/model/model.h"
#include "litert/test/matchers.h"

#include "tflite/converter/schema/schema_generated.h"

namespace litert::testing {
namespace {

using ::litert::internal::GetTflOptions;

template <typename GraphT>
size_t MakeInputCount(DefaultDevice& params_rng, DefaultDevice& data_rng) {
  auto graph_result = GraphT::Create(params_rng);
  LITERT_EXPECT_OK(graph_result);
  if (!graph_result.HasValue()) {
    return 0;
  }
  auto graph = std::move(graph_result.Value());
  RandomTensorDataBuilder data_builder;
  data_builder.SetFloatDummy();
  data_builder.SetIntDummy();
  auto inputs_result = graph->MakeInputs(data_rng, data_builder);
  LITERT_EXPECT_OK(inputs_result);
  if (!inputs_result.HasValue()) {
    return 0;
  }
  return inputs_result.Value().size();
}

template <typename GraphT>
size_t OpCount(DefaultDevice& rng) {
  auto graph_result = GraphT::Create(rng);
  LITERT_EXPECT_OK(graph_result);
  if (!graph_result.HasValue()) {
    return 0;
  }
  return graph_result.Value()->Graph().Subgraph(0).Ops().size();
}

template <typename GraphT>
void ExpectShapeContract(DefaultDevice& rng, size_t expected_input_rank,
                         size_t expected_output_rank,
                         bool expected_keep_num_dims) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
  const auto& subgraph = graph->Graph().Subgraph(0);
  const auto& input_tensor = *subgraph.Inputs().front();
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_type, input_tensor.Ranked());
  EXPECT_EQ(input_type.layout.rank, expected_input_rank);

  const auto& output_tensor = *subgraph.Outputs().front();
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_type, output_tensor.Ranked());
  EXPECT_EQ(output_type.layout.rank, expected_output_rank);

  const auto& op = subgraph.Op(0);
  const auto& tfl_opts = GetTflOptions(op);
  ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
  const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
  ASSERT_NE(fc_opts, nullptr);
  EXPECT_EQ(fc_opts->keep_num_dims, expected_keep_num_dims);
}

TEST(FullyConnectedGeneratorTest,
     FloatPresetRuntimeInputsMatchDynamicContract) {
  DefaultDevice params_rng(1234);
  DefaultDevice data_rng(5678);

  EXPECT_EQ((MakeInputCount<FullyConnected<
                 FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic>>>(
                params_rng, data_rng)),
            1u);
  EXPECT_EQ(
      (MakeInputCount<FullyConnected<
           FullyConnectedPresetC<FullyConnectedPreset::kFloatStaticNoBias>>>(
          params_rng, data_rng)),
      1u);
  EXPECT_EQ((MakeInputCount<FullyConnected<
                 FullyConnectedPresetC<FullyConnectedPreset::kFloatDynamic>>>(
                params_rng, data_rng)),
            3u);
  EXPECT_EQ(
      (MakeInputCount<FullyConnected<
           FullyConnectedPresetC<FullyConnectedPreset::kFloatDynamicFilter>>>(
          params_rng, data_rng)),
      2u);
  EXPECT_EQ((MakeInputCount<FullyConnected<FullyConnectedPresetC<
                 FullyConnectedPreset::kFloatDynamicFilterNoBias>>>(params_rng,
                                                                    data_rng)),
            2u);
  EXPECT_EQ(
      (MakeInputCount<FullyConnected<
           FullyConnectedPresetC<FullyConnectedPreset::kFloatDynamicBias>>>(
          params_rng, data_rng)),
      2u);
}

TEST(FullyConnectedGeneratorTest,
     QuantizedNoBiasPresetsHaveSingleRuntimeInput) {
  DefaultDevice params_rng(1234);
  DefaultDevice data_rng(5678);

  EXPECT_EQ(
      (MakeInputCount<FullyConnected<
           FullyConnectedPresetC<FullyConnectedPreset::kUint8StaticNoBias>>>(
          params_rng, data_rng)),
      1u);
  EXPECT_EQ(
      (MakeInputCount<FullyConnected<
           FullyConnectedPresetC<FullyConnectedPreset::kInt8StaticNoBias>>>(
          params_rng, data_rng)),
      1u);
  EXPECT_EQ((MakeInputCount<FullyConnected<FullyConnectedPresetC<
                 FullyConnectedPreset::kInt8PerChannelStaticNoBias>>>(
                params_rng, data_rng)),
            1u);
}

TEST(FullyConnectedGeneratorTest,
     ActivationSpecificFloatPresetsSetExpectedOption) {
  DefaultDevice rng(1234);

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kFloatStaticRelu>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
    const auto& op = graph->Graph().Subgraph(0).Op(0);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
    const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
    ASSERT_NE(fc_opts, nullptr);
    EXPECT_EQ(fc_opts->fused_activation_function,
              tflite::ActivationFunctionType_RELU);
  }

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kFloatStaticRelu6>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
    const auto& op = graph->Graph().Subgraph(0).Op(0);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
    const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
    ASSERT_NE(fc_opts, nullptr);
    EXPECT_EQ(fc_opts->fused_activation_function,
              tflite::ActivationFunctionType_RELU6);
  }

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kFloatStaticReluN1To1>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
    const auto& op = graph->Graph().Subgraph(0).Op(0);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
    const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
    ASSERT_NE(fc_opts, nullptr);
    EXPECT_EQ(fc_opts->fused_activation_function,
              tflite::ActivationFunctionType_RELU_N1_TO_1);
  }
}

TEST(FullyConnectedGeneratorTest,
     ActivationSpecificQuantizedPresetsSetExpectedOption) {
  DefaultDevice rng(1234);

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kUint8StaticRelu>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
    const auto& op = graph->Graph().Subgraph(0).Op(0);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
    const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
    ASSERT_NE(fc_opts, nullptr);
    EXPECT_EQ(fc_opts->fused_activation_function,
              tflite::ActivationFunctionType_RELU);
  }

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kUint8StaticRelu6>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
    const auto& op = graph->Graph().Subgraph(0).Op(0);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
    const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
    ASSERT_NE(fc_opts, nullptr);
    EXPECT_EQ(fc_opts->fused_activation_function,
              tflite::ActivationFunctionType_RELU6);
  }

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kUint8StaticReluN1To1>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
    const auto& op = graph->Graph().Subgraph(0).Op(0);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
    const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
    ASSERT_NE(fc_opts, nullptr);
    EXPECT_EQ(fc_opts->fused_activation_function,
              tflite::ActivationFunctionType_RELU_N1_TO_1);
  }

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kInt8StaticRelu>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
    const auto& op = graph->Graph().Subgraph(0).Op(0);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
    const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
    ASSERT_NE(fc_opts, nullptr);
    EXPECT_EQ(fc_opts->fused_activation_function,
              tflite::ActivationFunctionType_RELU);
  }

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kInt8StaticRelu6>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
    const auto& op = graph->Graph().Subgraph(0).Op(0);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
    const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
    ASSERT_NE(fc_opts, nullptr);
    EXPECT_EQ(fc_opts->fused_activation_function,
              tflite::ActivationFunctionType_RELU6);
  }

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kInt8StaticReluN1To1>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));
    const auto& op = graph->Graph().Subgraph(0).Op(0);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_FullyConnectedOptions);
    const auto* fc_opts = tfl_opts.AsFullyConnectedOptions();
    ASSERT_NE(fc_opts, nullptr);
    EXPECT_EQ(fc_opts->fused_activation_function,
              tflite::ActivationFunctionType_RELU_N1_TO_1);
  }
}

TEST(FullyConnectedGeneratorTest, Float3dReshapePresetFlattensNonBatchDims) {
  DefaultDevice rng(1234);
  using GraphT = FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic3dReshape>>;
  LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));

  const auto& sg = graph->Graph().Subgraph(0);
  const auto& input_tensor = *sg.Inputs().front();
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_type, input_tensor.Ranked());
  ASSERT_EQ(input_type.layout.rank, 3u);

  const auto& filter_tensor = sg.Tensor(1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto filter_type, filter_tensor.Ranked());
  ASSERT_EQ(filter_type.layout.rank, 2u);
  EXPECT_EQ(filter_type.layout.dimensions[1],
            input_type.layout.dimensions[1] * input_type.layout.dimensions[2]);

  const auto& output_tensor = *sg.Outputs().front();
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_type, output_tensor.Ranked());
  ASSERT_EQ(output_type.layout.rank, 2u);
  EXPECT_EQ(output_type.layout.dimensions[0], input_type.layout.dimensions[0]);
}

TEST(FullyConnectedGeneratorTest,
     RankSpecificFloatPresetsUseExpectedShapeContract) {
  DefaultDevice rng(1234);

  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic1d>>>(rng, 1u, 2u,
                                                                    false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic1dKeepDims>>>(
      rng, 1u, 1u, true);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic2d>>>(rng, 2u, 2u,
                                                                    false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic2dKeepDims>>>(
      rng, 2u, 2u, true);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic3d>>>(rng, 3u, 2u,
                                                                    false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic3dKeepDims>>>(
      rng, 3u, 3u, true);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic4d>>>(rng, 4u, 2u,
                                                                    false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kFloatStatic4dKeepDims>>>(
      rng, 4u, 4u, true);
}

TEST(FullyConnectedGeneratorTest,
     RankSpecificQuantizedPresetsUseExpectedShapeContract) {
  DefaultDevice rng(1234);

  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kUint8Static1d>>>(rng, 1u, 2u,
                                                                    false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kUint8Static1dKeepDims>>>(
      rng, 1u, 1u, true);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kUint8Static2d>>>(rng, 2u, 2u,
                                                                    false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kUint8Static2dKeepDims>>>(
      rng, 2u, 2u, true);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kUint8Static3d>>>(rng, 3u, 2u,
                                                                    false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kUint8Static3dKeepDims>>>(
      rng, 3u, 3u, true);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kUint8Static4d>>>(rng, 4u, 2u,
                                                                    false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kUint8Static4dKeepDims>>>(
      rng, 4u, 4u, true);

  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kInt8Static1d>>>(rng, 1u, 2u,
                                                                   false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kInt8Static1dKeepDims>>>(
      rng, 1u, 1u, true);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kInt8Static2d>>>(rng, 2u, 2u,
                                                                   false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kInt8Static2dKeepDims>>>(
      rng, 2u, 2u, true);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kInt8Static3d>>>(rng, 3u, 2u,
                                                                   false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kInt8Static3dKeepDims>>>(
      rng, 3u, 3u, true);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kInt8Static4d>>>(rng, 4u, 2u,
                                                                   false);
  ExpectShapeContract<FullyConnected<
      FullyConnectedPresetC<FullyConnectedPreset::kInt8Static4dKeepDims>>>(
      rng, 4u, 4u, true);
}

TEST(FullyConnectedGeneratorTest,
     Quantized3dReshapePresetsFlattenNonBatchDims) {
  DefaultDevice rng(1234);

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kUint8Static3dReshape>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));

    const auto& sg = graph->Graph().Subgraph(0);
    const auto& input_tensor = *sg.Inputs().front();
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_type, input_tensor.Ranked());
    ASSERT_EQ(input_type.layout.rank, 3u);

    const auto& filter_tensor = sg.Tensor(1);
    LITERT_ASSERT_OK_AND_ASSIGN(auto filter_type, filter_tensor.Ranked());
    ASSERT_EQ(filter_type.layout.rank, 2u);
    EXPECT_EQ(
        filter_type.layout.dimensions[1],
        input_type.layout.dimensions[1] * input_type.layout.dimensions[2]);
  }

  {
    using GraphT = FullyConnected<
        FullyConnectedPresetC<FullyConnectedPreset::kInt8Static3dReshape>>;
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, GraphT::Create(rng));

    const auto& sg = graph->Graph().Subgraph(0);
    const auto& input_tensor = *sg.Inputs().front();
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_type, input_tensor.Ranked());
    ASSERT_EQ(input_type.layout.rank, 3u);

    const auto& filter_tensor = sg.Tensor(1);
    LITERT_ASSERT_OK_AND_ASSIGN(auto filter_type, filter_tensor.Ranked());
    ASSERT_EQ(filter_type.layout.rank, 2u);
    EXPECT_EQ(
        filter_type.layout.dimensions[1],
        input_type.layout.dimensions[1] * input_type.layout.dimensions[2]);
  }
}

TEST(FullyConnectedGeneratorTest, Fp16WeightsPresetsUseDequantizeGraph) {
  DefaultDevice rng(1234);

  EXPECT_EQ((OpCount<FullyConnected<FullyConnectedPresetC<
                 FullyConnectedPreset::kFloatFp16WeightsStatic>>>(rng)),
            3u);
  EXPECT_EQ((OpCount<FullyConnected<FullyConnectedPresetC<
                 FullyConnectedPreset::kFloatFp16WeightsStaticF32Bias>>>(rng)),
            2u);
  EXPECT_EQ((OpCount<FullyConnected<FullyConnectedPresetC<
                 FullyConnectedPreset::kFloatFp16WeightsStaticNoBias>>>(rng)),
            2u);
}

TEST(FullyConnectedGeneratorTest, Fp16WeightsF32BiasPresetKeepsBiasFloat32) {
  DefaultDevice rng(1234);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto graph, (FullyConnected<FullyConnectedPresetC<
                       FullyConnectedPreset::kFloatFp16WeightsStaticF32Bias>>::
                       Create(rng)));

  const auto& subgraph = graph->Graph().Subgraph(0);
  ASSERT_EQ(subgraph.Ops().size(), 2u);
  EXPECT_EQ(subgraph.Op(0).OpCode(), kLiteRtOpCodeTflDequantize);
  EXPECT_EQ(subgraph.Op(1).OpCode(), kLiteRtOpCodeTflFullyConnected);
  EXPECT_EQ(subgraph.Op(1).NumInputs(), 3u);

  const auto& bias_tensor = subgraph.Op(1).Input(2);
  LITERT_ASSERT_OK_AND_ASSIGN(auto bias_type, bias_tensor.Ranked());
  EXPECT_EQ(bias_type.element_type, kLiteRtElementTypeFloat32);
}

}  // namespace
}  // namespace litert::testing
