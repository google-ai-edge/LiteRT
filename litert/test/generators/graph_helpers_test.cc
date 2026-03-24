// Copyright 2025 Google LLC.
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

#include "litert/test/generators/graph_helpers.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/test/matchers.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {
namespace {

using ::litert::internal::GetTflOpCodes;
using ::litert::internal::GetTflOpCodeInd;
using ::litert::internal::GetTflOptions;

TEST(GraphHelpersTest, TestSingleOpModel) {
  TensorDetails lhs = {{1, 2, 3}, kLiteRtElementTypeInt32, "lhs"};

  TensorDetails rhs = {
      {}, kLiteRtElementTypeInt32, "cst", MakeBufferRef<int32_t>({1})};

  TensorDetails output = {{1, 2, 3}, kLiteRtElementTypeInt32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflAdd>(
                      {std::move(lhs), std::move(rhs)}, {std::move(output)},
                      tflite::ActivationFunctionType_NONE, false));

  auto& sg = model->Subgraph(0);
  {
    EXPECT_EQ(sg.Inputs().size(), 1);
    EXPECT_EQ(sg.Inputs()[0]->Name(), "lhs");

    EXPECT_EQ(sg.Outputs().size(), 1);
    EXPECT_EQ(sg.Outputs()[0]->Name(), "output");

    ASSERT_EQ(sg.Tensors().size(), 3);
  }

  {
    ASSERT_EQ(sg.Ops().size(), 1);
    const auto& op = sg.Op(0);
    ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
    EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 1);
    EXPECT_EQ(op.OpCode(), kLiteRtOpCodeTflAdd);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_AddOptions);
    const auto* add_opts = tfl_opts.AsAddOptions();
    ASSERT_NE(add_opts, nullptr);
    EXPECT_EQ(add_opts->fused_activation_function,
              tflite::ActivationFunctionType_NONE);
    EXPECT_EQ(add_opts->pot_scale_int16, false);
    EXPECT_EQ(op.NumInputs(), 2);
    EXPECT_EQ(op.NumOutputs(), 1);
  }

  {
    const auto& lhs_tensor = sg.Tensor(0);
    EXPECT_EQ(lhs_tensor.Name(), "lhs");
  }

  {
    const auto& rhs_tensor = sg.Tensor(1);
    EXPECT_EQ(rhs_tensor.Name(), "cst");
    EXPECT_EQ(rhs_tensor.Weights().Buffer().Size(), sizeof(int32_t));
  }

  {
    const auto& output_tensor = sg.Tensor(2);
    EXPECT_EQ(output_tensor.Name(), "output");
  }
}

TEST(GraphHelpersTest, TestSingleFullyConnectedModelVersionInt16Int16) {
  TensorDetails input = {{2, 4}, kLiteRtElementTypeInt16, "input"};
  TensorDetails filter = {
      {3, 4}, kLiteRtElementTypeInt16, "filter",
      MakeBufferRef<int16_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})};
  TensorDetails bias = {
      {3}, kLiteRtElementTypeInt64, "bias", MakeBufferRef<int64_t>({0, 0, 0})};
  TensorDetails output = {{2, 3}, kLiteRtElementTypeInt16, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflFullyConnected>(
                      {std::move(input), std::move(filter), std::move(bias)},
                      {std::move(output)},
                      tflite::ActivationFunctionType_NONE,
                      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                      /*keep_num_dims=*/false,
                      /*asymmetric_quantize_inputs=*/false,
                      tflite::TensorType_FLOAT32));

  const auto& op = model->Subgraph(0).Op(0);
  ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
  EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 7);
}

TEST(GraphHelpersTest, TestSingleFullyConnectedModelVersionInt16Int8) {
  TensorDetails input = {{2, 4}, kLiteRtElementTypeInt16, "input"};
  TensorDetails filter = {
      {3, 4}, kLiteRtElementTypeInt8, "filter",
      MakeBufferRef<int8_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})};
  TensorDetails bias = {
      {3}, kLiteRtElementTypeInt32, "bias", MakeBufferRef<int32_t>({0, 0, 0})};
  TensorDetails output = {{2, 3}, kLiteRtElementTypeInt16, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflFullyConnected>(
                      {std::move(input), std::move(filter), std::move(bias)},
                      {std::move(output)},
                      tflite::ActivationFunctionType_NONE,
                      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                      /*keep_num_dims=*/false,
                      /*asymmetric_quantize_inputs=*/false,
                      tflite::TensorType_INT32));

  const auto& op = model->Subgraph(0).Op(0);
  ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
  EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 11);
}

TEST(GraphHelpersTest, TestSingleFullyConnectedModelVersionNoBias) {
  TensorDetails input = {{2, 4}, kLiteRtElementTypeFloat32, "input"};
  TensorDetails filter = {
      {3, 4}, kLiteRtElementTypeFloat32, "filter",
      MakeBufferRef<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f})};
  TensorDetails output = {{2, 3}, kLiteRtElementTypeFloat32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflFullyConnected>(
                      {std::move(input), std::move(filter)},
                      {std::move(output)},
                      tflite::ActivationFunctionType_NONE,
                      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                      /*keep_num_dims=*/false,
                      /*asymmetric_quantize_inputs=*/false,
                      tflite::TensorType_FLOAT32));

  const auto& op = model->Subgraph(0).Op(0);
  ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
  EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 6);
}

TEST(GraphHelpersTest, TestSingleFullyConnectedModelVersionHybridAsymmetric) {
  TensorDetails input = {{2, 4}, kLiteRtElementTypeFloat32, "input"};
  TensorDetails filter = {
      {3, 4}, kLiteRtElementTypeInt8, "filter",
      MakeBufferRef<int8_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})};
  TensorDetails bias = {
      {3}, kLiteRtElementTypeFloat32, "bias",
      MakeBufferRef<float>({0.0f, 0.0f, 0.0f})};
  TensorDetails output = {{2, 3}, kLiteRtElementTypeFloat32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflFullyConnected>(
                      {std::move(input), std::move(filter), std::move(bias)},
                      {std::move(output)},
                      tflite::ActivationFunctionType_NONE,
                      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                      /*keep_num_dims=*/false,
                      /*asymmetric_quantize_inputs=*/true,
                      tflite::TensorType_FLOAT32));

  const auto& op = model->Subgraph(0).Op(0);
  ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
  EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 9);
}

TEST(GraphHelpersTest, TestSingleFullyConnectedModelVersionHybridPerChannel) {
  TensorDetails input = {{2, 4}, kLiteRtElementTypeFloat32, "input"};
  TensorDetails filter = {
      {3, 4}, kLiteRtElementTypeInt8, "filter",
      MakeBufferRef<int8_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})};
  filter.quantization = TensorDetails::QuantizationDetails::PerChannel(
      /*quantized_dimension=*/0, {0.5f, 0.25f, 0.125f}, {0, 0, 0});
  TensorDetails bias = {
      {3}, kLiteRtElementTypeFloat32, "bias",
      MakeBufferRef<float>({0.0f, 0.0f, 0.0f})};
  TensorDetails output = {{2, 3}, kLiteRtElementTypeFloat32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflFullyConnected>(
                      {std::move(input), std::move(filter), std::move(bias)},
                      {std::move(output)},
                      tflite::ActivationFunctionType_NONE,
                      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                      /*keep_num_dims=*/false,
                      /*asymmetric_quantize_inputs=*/false,
                      tflite::TensorType_FLOAT32));

  const auto& op = model->Subgraph(0).Op(0);
  ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
  EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 12);
}

TEST(GraphHelpersTest, TestSingleFullyConnectedModelVersionInt8Int4) {
  TensorDetails input = {{2, 4}, kLiteRtElementTypeInt8, "input"};
  TensorDetails filter = {{3, 4}, kLiteRtElementTypeInt4, "filter",
                          MakeBufferRef<uint8_t>({0x11, 0x11, 0x11, 0x11,
                                                  0x11, 0x11})};
  TensorDetails bias = {
      {3}, kLiteRtElementTypeInt32, "bias", MakeBufferRef<int32_t>({0, 0, 0})};
  TensorDetails output = {{2, 3}, kLiteRtElementTypeInt8, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflFullyConnected>(
                      {std::move(input), std::move(filter), std::move(bias)},
                      {std::move(output)},
                      tflite::ActivationFunctionType_NONE,
                      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                      /*keep_num_dims=*/false,
                      /*asymmetric_quantize_inputs=*/false,
                      tflite::TensorType_INT32));

  const auto& op = model->Subgraph(0).Op(0);
  ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
  EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 10);
}

TEST(GraphHelpersTest, TestSingleFullyConnectedModelVersionInt16Int4) {
  TensorDetails input = {{2, 4}, kLiteRtElementTypeInt16, "input"};
  TensorDetails filter = {{3, 4}, kLiteRtElementTypeInt4, "filter",
                          MakeBufferRef<uint8_t>({0x11, 0x11, 0x11, 0x11,
                                                  0x11, 0x11})};
  TensorDetails bias = {
      {3}, kLiteRtElementTypeInt32, "bias", MakeBufferRef<int32_t>({0, 0, 0})};
  TensorDetails output = {{2, 3}, kLiteRtElementTypeInt16, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflFullyConnected>(
                      {std::move(input), std::move(filter), std::move(bias)},
                      {std::move(output)},
                      tflite::ActivationFunctionType_NONE,
                      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                      /*keep_num_dims=*/false,
                      /*asymmetric_quantize_inputs=*/false,
                      tflite::TensorType_INT32));

  const auto& op = model->Subgraph(0).Op(0);
  ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
  EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 13);
}

TEST(GraphHelpersTest, TestSingleFullyConnectedModelVersionInt2) {
  TensorDetails input = {{2, 4}, kLiteRtElementTypeInt8, "input"};
  TensorDetails filter = {
      {3, 4}, kLiteRtElementTypeInt2, "filter", MakeBufferRef<uint8_t>({0, 0, 0})};
  TensorDetails bias = {
      {3}, kLiteRtElementTypeInt32, "bias", MakeBufferRef<int32_t>({0, 0, 0})};
  TensorDetails output = {{2, 3}, kLiteRtElementTypeInt8, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflFullyConnected>(
                      {std::move(input), std::move(filter), std::move(bias)},
                      {std::move(output)},
                      tflite::ActivationFunctionType_NONE,
                      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                      /*keep_num_dims=*/false,
                      /*asymmetric_quantize_inputs=*/false,
                      tflite::TensorType_INT32));

  const auto& op = model->Subgraph(0).Op(0);
  ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
  EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 14);
}

TEST(GraphHelpersTest, TestSingleFullyConnectedModelWithInternalOutputs) {
  TensorDetails input = {{1, 16}, kLiteRtElementTypeUInt8, "input"};
  TensorDetails filter = {
      {4, 16}, kLiteRtElementTypeUInt8, "filter",
      MakeBufferRef<uint8_t>({
          0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
          0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
          0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
          0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
          0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
          0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
          0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
          0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      })};
  TensorDetails bias = {
      {4}, kLiteRtElementTypeInt32, "bias", MakeBufferRef<int32_t>({0, 0, 0, 0})};
  TensorDetails output = {{1, 4}, kLiteRtElementTypeInt16, "output"};
  TensorDetails workspace = {{1}, kLiteRtElementTypeUInt8, "workspace"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      SingleOpModelWithInternalOutputs<kLiteRtOpCodeTflFullyConnected>(
          {std::move(input), std::move(filter), std::move(bias)},
          {std::move(output)}, {std::move(workspace)},
          tflite::ActivationFunctionType_NONE,
          tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8,
          /*keep_num_dims=*/false,
          /*asymmetric_quantize_inputs=*/false, tflite::TensorType_INT32));

  const auto& sg = model->Subgraph(0);
  EXPECT_EQ(sg.Outputs().size(), 1);
  ASSERT_EQ(sg.Ops().size(), 1);

  const auto& op = sg.Op(0);
  EXPECT_EQ(op.NumOutputs(), 2);
  ASSERT_LT(GetTflOpCodeInd(op), GetTflOpCodes(*model).size());
  EXPECT_EQ(GetTflOpCodes(*model)[GetTflOpCodeInd(op)]->version, 2);
}

}  // namespace
}  // namespace litert::testing
