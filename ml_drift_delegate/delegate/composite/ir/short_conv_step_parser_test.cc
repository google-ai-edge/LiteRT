// Copyright 2026 Google LLC.
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

#include "ml_drift_delegate/delegate/composite/ir/short_conv_step_parser.h"

#include <any>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/short_conv_step_parser.h"
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;

constexpr int kSeqLen = 1;
constexpr int kHiddenSize = 64;
constexpr int kConvLCache = 3;

// Builds a single ungated `odml.short_conv_step` node with the given conv_state
// and conv_weight shapes and optional `channels_last` attribute. `builder` owns
// the custom options, so it must outlive the interpreter.
std::unique_ptr<::tflite::Interpreter> BuildShortConvStep(
    SingleOpInterpreterBuilder& builder, const std::vector<int>& state_shape,
    const std::vector<int>& weight_shape, std::optional<bool> channels_last) {
  builder.SetCustomName("odml.short_conv_step");
  builder.AddInput(kTfLiteFloat32, {1, kSeqLen, kHiddenSize});   // in_proj
  builder.AddInput(kTfLiteFloat32, state_shape);                 // conv_state
  builder.AddInput(kTfLiteFloat32, weight_shape);                // conv_weight
  builder.AddOutput(kTfLiteFloat32, {1, kSeqLen, kHiddenSize});  // out
  builder.AddOutput(kTfLiteFloat32, state_shape);                // next_state

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("conv_L_cache", kConvLCache);
    fbb.Bool("is_gated", false);
    fbb.Bool("use_silu", true);
    if (channels_last.has_value()) {
      fbb.Bool("channels_last", *channels_last);
    }
  });
  fbb.Finish();
  const auto& fbb_buf = fbb.GetBuffer();
  void* custom_data = malloc(fbb_buf.size());
  memcpy(custom_data, fbb_buf.data(), fbb_buf.size());
  builder.SetCustomData(custom_data, fbb_buf.size());
  return builder.Build();
}

class ConvertShortConvStepTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CustomIrOpMap custom_parsers;
    custom_parsers["odml.short_conv_step"] = GetShortConvStepParser();
    delegate_ = CreateStubDelegate(/*options=*/{}, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertShortConvStepTest, ConvertsChannelsLastStateAndWeight) {
  for (std::optional<bool> channels_last :
       {std::optional<bool>(), std::optional<bool>(true)}) {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
    auto interpreter = BuildShortConvStep(
        builder, /*state_shape=*/{1, kConvLCache - 1, kHiddenSize},
        /*weight_shape=*/{1, kConvLCache, kHiddenSize}, channels_last);
    ASSERT_NE(interpreter, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    ASSERT_TRUE(ir_model);
    ASSERT_THAT(ir_model->ops(), SizeIs(1));
    const auto& op = ir_model->ops()[0];
    EXPECT_THAT(op->name, Eq("short_conv_step"));
    // conv_state (input 1) is updated in place through next_state.
    EXPECT_THAT(op->inputs, SizeIs(2));
    const auto* attr =
        std::any_cast<::litert::ml_drift::ShortConvStepAttributes>(&op->attr);
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->conv_L_cache, kConvLCache);
    EXPECT_TRUE(attr->channels_last);
  }
}

// Older channels-first exports ([1, H, L - 1] state, [H, 1, L] weight) are not
// delegated, so they fall back to the CPU decomposition instead of being
// misread.
TEST_F(ConvertShortConvStepTest, RejectsChannelsFirstState) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  auto interpreter = BuildShortConvStep(
      builder, /*state_shape=*/{1, kHiddenSize, kConvLCache - 1},
      /*weight_shape=*/{1, kConvLCache, kHiddenSize}, std::nullopt);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  EXPECT_EQ(GetIrModel(delegate_), nullptr);
}

TEST_F(ConvertShortConvStepTest, RejectsChannelsFirstWeight) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  auto interpreter = BuildShortConvStep(
      builder, /*state_shape=*/{1, kConvLCache - 1, kHiddenSize},
      /*weight_shape=*/{kHiddenSize, 1, kConvLCache}, std::nullopt);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  EXPECT_EQ(GetIrModel(delegate_), nullptr);
}

TEST_F(ConvertShortConvStepTest, RejectsChannelsLastFalse) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  auto interpreter = BuildShortConvStep(
      builder, /*state_shape=*/{1, kConvLCache - 1, kHiddenSize},
      /*weight_shape=*/{1, kConvLCache, kHiddenSize},
      /*channels_last=*/false);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  EXPECT_EQ(GetIrModel(delegate_), nullptr);
}

// The GraphFloat32 parser shares the check with the IR parser.
TEST(ShortConvStepOperationParserTest, IsSupportedChecksChannelsLast) {
  struct Case {
    std::vector<int> state_shape;
    std::vector<int> weight_shape;
    std::optional<bool> channels_last;
    bool supported;
  };
  const std::vector<Case> cases = {
      {{1, kConvLCache - 1, kHiddenSize},
       {1, kConvLCache, kHiddenSize},
       std::nullopt,
       true},
      {{1, kHiddenSize, kConvLCache - 1},
       {1, kConvLCache, kHiddenSize},
       std::nullopt,
       false},
      {{1, kConvLCache - 1, kHiddenSize},
       {kHiddenSize, 1, kConvLCache},
       std::nullopt,
       false},
      {{1, kConvLCache - 1, kHiddenSize},
       {1, kConvLCache, kHiddenSize},
       false,
       false},
  };
  for (const Case& c : cases) {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
    auto interpreter = BuildShortConvStep(builder, c.state_shape,
                                          c.weight_shape, c.channels_last);
    ASSERT_NE(interpreter, nullptr);
    const auto* node_and_reg = interpreter->node_and_registration(0);
    ASSERT_NE(node_and_reg, nullptr);
    ::litert::ml_drift::ShortConvStepOperationParser parser;
    EXPECT_EQ(parser
                  .IsSupported(interpreter->primary_subgraph().context(),
                               &node_and_reg->first, &node_and_reg->second)
                  .ok(),
              c.supported);
  }
}

}  // namespace
}  // namespace litert::ml_drift::ir
