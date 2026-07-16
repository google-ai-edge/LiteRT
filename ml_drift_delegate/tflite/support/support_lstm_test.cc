// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/lstm_shared.h"
#include "tflite/schema/schema_generated.h"

// These tests indirectly verify IsNodeSupported through GetOpsToReplace,
// which in turn uses GetSupportedNodes to leverage existing matchers.

namespace litert::ml_drift::ir {

// GetSupportedNodes is module-private (support.cc) and not public (support.h),
// prioritizing encapsulation over test convenience.
extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 3, 2, 4};
constexpr std::array<int, 4> kDefaultWeightsDims = {4, 1, 1, 4};
constexpr std::array<int, 1> kDefaultBiasDims = {4};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 3, 2, 4};
constexpr TfLiteLSTMParams kDefaultLstmParams = {
    .activation = kTfLiteActTanh,
    .cell_clip = 0.0f,
    .proj_clip = 0.0f,
    .kernel_type = kTfLiteLSTMBasicKernel,
};

struct VersionTestCase {
  TfLiteBuiltinOperator op = kTfLiteBuiltinLstm;
  int version = 0;
};

// Test suite for lstm ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(GetParam().op, GetParam().version, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    LstmOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinLstm, 1},
        {kTfLiteBuiltinLstm, 4},
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for lstm ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(GetParam().op, GetParam().version, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    LstmOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinLstm, 0},
        {kTfLiteBuiltinLstm, 5},
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for lstm ops for different number of I/O tensors.
using NumInputOutputTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(NumInputOutputTest, Supports5Input4Output) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  context_builder.SetOp(op, /*version=*/1, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Rejects4Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  context_builder.SetOp(op, /*version=*/1, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects3Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  context_builder.SetOp(op, /*version=*/1, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    LstmOps, NumInputOutputTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinLstm,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for lstm ops x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(SupportedDtypeTest, RejectsUnsupportedInputDtype) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedWeightsDtype) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kTfLiteNoType, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedBiasDtype) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kTfLiteNoType, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    LstmOps, SupportedDtypeTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinLstm,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for lstm ops with different parameters.
using ParamsTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(ParamsTest, RejectsUnsupportedActivation) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  TfLiteLSTMParams params = kDefaultLstmParams;
  params.activation = kTfLiteActRelu;
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(ParamsTest, RejectsCellClip) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  TfLiteLSTMParams params = kDefaultLstmParams;
  params.cell_clip = 1.0f;
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(ParamsTest, RejectsProjClip) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  TfLiteLSTMParams params = kDefaultLstmParams;
  params.proj_clip = 1.0f;
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(ParamsTest, SupportsFullKernel) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;

  std::vector<int> inputs(24, kTfLiteOptionalTensor);
  constexpr std::array<int, 4> kInputShape = {1, 1, 1, 4};
  constexpr std::array<int, 4> kStateShape = {1, 1, 1, 4};
  constexpr std::array<int, 2> kWeightsShape = {4, 4};
  constexpr std::array<int, 1> kBiasShape = {4};

  namespace lstm_full = ::tflite::ops::builtin::lstm::full;
  // inputs
  inputs[lstm_full::kInputTensor] =
      context_builder.AddTensor(kDefaultDtype, kInputShape);
  inputs[lstm_full::kInputToInputWeightsTensor] =
      context_builder.AddTensor(kDefaultDtype, kWeightsShape);
  // input to gate
  inputs[lstm_full::kInputToForgetWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kInputToCellWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kInputToOutputWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  // recurrent to gate
  inputs[lstm_full::kRecurrentToInputWeightsTensor] =
      context_builder.AddTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kRecurrentToForgetWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kRecurrentToCellWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kRecurrentToOutputWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  // peephole
  inputs[lstm_full::kCellToInputWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kCellToForgetWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kCellToOutputWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  // biases
  inputs[lstm_full::kInputGateBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kForgetGateBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kCellGateBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kOutputGateBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kProjectionBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kProjectionWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  // states
  inputs[lstm_full::kOutputStateTensor] =
      context_builder.AddTensor(kDefaultDtype, kStateShape);
  inputs[lstm_full::kCellStateTensor] =
      context_builder.AddTensor(kDefaultDtype, kStateShape);
  // layer norm
  inputs[lstm_full::kInputLayerNormCoefficientsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kForgetLayerNormCoefficientsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kCellLayerNormCoefficientsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kOutputLayerNormCoefficientsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);

  constexpr std::array<int, 4> kOutputShape = {1, 1, 1, 4};
  const int out0 = context_builder.AddTensor(kDefaultDtype, kOutputShape);

  TfLiteLSTMParams params = kDefaultLstmParams;
  params.kernel_type = kTfLiteLSTMFullKernel;
  context_builder.SetOp(op, /*version=*/1, &params, inputs,
                        /*outputs=*/{out0});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(ParamsTest, FullKernelRejectsInvalidOutputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;

  std::vector<int> inputs(24, kTfLiteOptionalTensor);
  constexpr std::array<int, 4> kInputShape = {1, 1, 1, 4};
  constexpr std::array<int, 4> kStateShape = {1, 1, 1, 4};
  constexpr std::array<int, 2> kWeightsShape = {4, 4};
  constexpr std::array<int, 1> kBiasShape = {4};

  namespace lstm_full = ::tflite::ops::builtin::lstm::full;
  // inputs
  inputs[lstm_full::kInputTensor] =
      context_builder.AddTensor(kDefaultDtype, kInputShape);
  inputs[lstm_full::kInputToInputWeightsTensor] =
      context_builder.AddTensor(kDefaultDtype, kWeightsShape);
  // input to gate
  inputs[lstm_full::kInputToForgetWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kInputToCellWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kInputToOutputWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  // recurrent to gate
  inputs[lstm_full::kRecurrentToInputWeightsTensor] =
      context_builder.AddTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kRecurrentToForgetWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kRecurrentToCellWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kRecurrentToOutputWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  // peephole
  inputs[lstm_full::kCellToInputWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kCellToForgetWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kCellToOutputWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  // biases
  inputs[lstm_full::kInputGateBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kForgetGateBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kCellGateBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kOutputGateBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kProjectionBiasTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kBiasShape);
  inputs[lstm_full::kProjectionWeightsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  // states
  inputs[lstm_full::kOutputStateTensor] =
      context_builder.AddTensor(kDefaultDtype, kStateShape);
  inputs[lstm_full::kCellStateTensor] =
      context_builder.AddTensor(kDefaultDtype, kStateShape);
  // layer norm
  inputs[lstm_full::kInputLayerNormCoefficientsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kForgetLayerNormCoefficientsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kCellLayerNormCoefficientsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);
  inputs[lstm_full::kOutputLayerNormCoefficientsTensor] =
      context_builder.AddConstTensor(kDefaultDtype, kWeightsShape);

  constexpr std::array<int, 4> kOutputShape = {1, 1, 1, 4};
  const int out0 = context_builder.AddTensor(kDefaultDtype, kOutputShape) + 1;

  TfLiteLSTMParams params = kDefaultLstmParams;
  params.kernel_type = kTfLiteLSTMFullKernel;
  context_builder.SetOp(op, /*version=*/1, &params, inputs,
                        /*outputs=*/{out0});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    LstmOps, ParamsTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinLstm,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<ParamsTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for lstm ops with different dims.
using DimsTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(DimsTest, RejectsDifferentInputOutputDims) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, {1, 2, 3});
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, {3, 2, 1});
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    LstmOps, DimsTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinLstm,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<DimsTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for lstm ops with constant inputs.
using ConstantInputTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(ConstantInputTest, RejectsConstantInput0) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(ConstantInputTest, RejectsConstantInput1) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int in1 =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int in3 = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int in4 = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out0 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out1 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int out3 = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultLstmParams,
                        /*inputs=*/{in0, in1, in2, in3, in4},
                        /*outputs=*/{out0, out1, out2, out3});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    LstmOps, ConstantInputTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinLstm,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<ConstantInputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

}  // namespace
}  // namespace litert::ml_drift::ir
