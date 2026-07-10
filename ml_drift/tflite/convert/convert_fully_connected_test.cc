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

#include <any>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertFullyConnectedTest
    : public ::testing::TestWithParam<
          std::tuple<TfLiteType, bool, bool, TfLiteFusedActivation>> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    if (!delegate_) {
      ADD_FAILURE() << "Failed to create StubDelegate";
      return;
    }
    std::tie(dtype_, const_weights_, use_bias_, activation_) = GetParam();
  }

  void TearDown() override {
    if (delegate_) {
      DeleteStubDelegate(delegate_);
    }
  }

  TfLiteDelegate* delegate_;
  TfLiteType dtype_;
  bool const_weights_;
  bool use_bias_;
  TfLiteFusedActivation activation_;
  std::vector<std::byte> weights_data_;
  std::vector<std::byte> bias_data_;
};

TEST_P(ConvertFullyConnectedTest, Parameterized) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinFullyConnected, /*version=*/1);
  const int input_channels = 4;
  const int output_channels = 8;
  model.AddInput(kTfLiteFloat32, {1, 1, 1, input_channels});  // input data
  model.AddInput(dtype_, {output_channels, input_channels});  // weights
  if (use_bias_) {
    model.AddInput(kTfLiteFloat32, {output_channels});  // bias
  }
  model.AddOutput(kTfLiteFloat32, {1, output_channels});  // output
  TfLiteFullyConnectedParams* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(
          calloc(1, sizeof(TfLiteFullyConnectedParams)));
  params->activation = activation_;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  if (const_weights_) {
    TfLiteTensor* weights = interpreter->tensor(interpreter->inputs()[1]);
    weights->allocation_type = kTfLiteMmapRo;
    weights_data_.resize(weights->bytes);
    weights->data.raw = reinterpret_cast<char*>(weights_data_.data());

    if (use_bias_) {
      TfLiteTensor* bias = interpreter->tensor(interpreter->inputs()[2]);
      bias->allocation_type = kTfLiteMmapRo;
      bias_data_.resize(bias->bytes);
      bias->data.raw = reinterpret_cast<char*>(bias_data_.data());
    }
  }

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  std::string expected_op_name;
  if (dtype_ == kTfLiteInt8 && const_weights_) {
    expected_op_name = "fully_connected_int8";
  } else if (dtype_ == kTfLiteInt4 && const_weights_) {
    expected_op_name = "fully_connected_int4";
  } else if (dtype_ == kTfLiteInt2 && const_weights_) {
    expected_op_name = "fully_connected_int2";
  } else {
    expected_op_name = "fully_connected";
  }
  const ::ml_drift::ir::IrOp* fc_op = nullptr;
  for (const auto& op : ir_model->ops()) {
    if (op->name == expected_op_name) {
      fc_op = op.get();
      break;
    }
  }
  ASSERT_NE(fc_op, nullptr)
      << "Expected op " << expected_op_name << " not found in IR model";

  if (dtype_ == kTfLiteInt4 && const_weights_) {
    EXPECT_TRUE(fc_op->attr.has_value());
    EXPECT_NO_THROW(
        (void)std::any_cast<::ml_drift::FullyConnectedInt4Attributes>(
            fc_op->attr));
  } else if (dtype_ == kTfLiteInt2 && const_weights_) {
    EXPECT_TRUE(fc_op->attr.has_value());
    EXPECT_NO_THROW(
        (void)std::any_cast<::ml_drift::FullyConnectedInt2Attributes>(
            fc_op->attr));
  } else if (dtype_ == kTfLiteInt8 && const_weights_) {
    EXPECT_TRUE(fc_op->attr.has_value());
    EXPECT_NO_THROW(
        (void)std::any_cast<::ml_drift::FullyConnectedInt8Attributes>(
            fc_op->attr));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ConvertFullyConnectedTest, ConvertFullyConnectedTest,
    ::testing::Combine(::testing::Values(kTfLiteFloat32, kTfLiteFloat16,
                                         kTfLiteInt8, kTfLiteInt4,
                                         kTfLiteInt2),  // dtype
                       ::testing::Bool(),               // const_weights
                       ::testing::Bool(),               // use_bias
                       ::testing::Values(kTfLiteActNone, kTfLiteActRelu)),
    [](const ::testing::TestParamInfo<ConvertFullyConnectedTest::ParamType>&
           info) {
      TfLiteType dtype = std::get<0>(info.param);
      std::string dtype_str;
      if (dtype == kTfLiteFloat32) {
        dtype_str = "float32";
      } else if (dtype == kTfLiteInt8) {
        dtype_str = "int8";
      } else if (dtype == kTfLiteInt4) {
        dtype_str = "int4";
      } else if (dtype == kTfLiteInt2) {
        dtype_str = "int2";
      }
      return absl::StrCat(
          dtype_str, std::get<1>(info.param) ? "_const" : "_runtime",
          std::get<2>(info.param) ? "_bias" : "_nobias",
          (std::get<3>(info.param) == kTfLiteActNone) ? "_noact" : "_relu");
    });

TEST(ConvertFullyConnectedFallbackTest, Conv2DFallback) {
  IrModelBuilderOptions options;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate(
      CreateStubDelegate(options), DeleteStubDelegate);
  ASSERT_NE(delegate, nullptr);

  SingleOpInterpreterBuilder model(kTfLiteBuiltinFullyConnected, /*version=*/1);
  model.AddInput(kTfLiteFloat32, {1, 2, 2, 4});  // input data h,w != 1
  model.AddInput(kTfLiteFloat32, {8, 16});       // weights (8 x 4*2*2)
  model.AddOutput(kTfLiteFloat32, {1, 8});
  TfLiteFullyConnectedParams* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(
          calloc(1, sizeof(TfLiteFullyConnectedParams)));
  params->activation = kTfLiteActNone;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  // Force constant weights
  TfLiteTensor* weights = interpreter->tensor(interpreter->inputs()[1]);
  weights->allocation_type = kTfLiteMmapRo;
  std::unique_ptr<void, decltype(&free)> weights_data(malloc(weights->bytes),
                                                      &free);
  weights->data.raw = static_cast<char*>(weights_data.get());

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  bool found_conv = false;
  for (const auto& op : ir_model->ops()) {
    if (op->name == "convolution_2d") {
      found_conv = true;
      break;
    }
  }
  EXPECT_TRUE(found_conv);
}

TEST(ConvertFullyConnectedTest, WeightsReshapeNeeded) {
  IrModelBuilderOptions options;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate(
      CreateStubDelegate(options), DeleteStubDelegate);
  ASSERT_NE(delegate, nullptr);

  SingleOpInterpreterBuilder model(kTfLiteBuiltinFullyConnected, /*version=*/1);
  model.AddInput(kTfLiteFloat32, {1, 4});
  model.AddInput(kTfLiteFloat32, {1, 2, 2, 4});
  model.AddOutput(kTfLiteFloat32, {1, 4});

  TfLiteFullyConnectedParams* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(
          calloc(1, sizeof(TfLiteFullyConnectedParams)));
  params->activation = kTfLiteActNone;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  const ::ml_drift::ir::IrOp* reshape_op = nullptr;
  for (const auto& op : ir_model->ops()) {
    if (op->name == "reshape") {
      reshape_op = op.get();
      break;
    }
  }
  ASSERT_NE(reshape_op, nullptr);

  auto reshape_attr =
      std::any_cast<::ml_drift::ReshapeAttributes>(reshape_op->attr);
  EXPECT_EQ(reshape_attr.new_shape.b, 4);
  EXPECT_EQ(reshape_attr.new_shape.h, 1);
  EXPECT_EQ(reshape_attr.new_shape.w, 1);
  EXPECT_EQ(reshape_attr.new_shape.c, 4);
}

TEST(ConvertFullyConnectedTest, ReshapeNeeded) {
  IrModelBuilderOptions options;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate(
      CreateStubDelegate(options), DeleteStubDelegate);
  ASSERT_NE(delegate, nullptr);

  SingleOpInterpreterBuilder model(kTfLiteBuiltinFullyConnected, /*version=*/1);
  // input_shape [1, 2, 2, 4] -> 16 elements
  model.AddInput(kTfLiteFloat32, {1, 2, 2, 4});
  // weights [8, 16]
  model.AddInput(kTfLiteFloat32, {8, 16});
  // output [1, 8]
  model.AddOutput(kTfLiteFloat32, {1, 8});

  TfLiteFullyConnectedParams* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(
          calloc(1, sizeof(TfLiteFullyConnectedParams)));
  params->activation = kTfLiteActNone;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  // Force constant weights
  TfLiteTensor* weights = interpreter->tensor(interpreter->inputs()[1]);
  weights->allocation_type = kTfLiteMmapRo;
  std::unique_ptr<void, decltype(&free)> weights_data(malloc(weights->bytes),
                                                      &free);
  weights->data.raw = static_cast<char*>(weights_data.get());

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  bool found_fc_or_conv = false;
  const ::ml_drift::ir::IrOp* reshape_op = nullptr;
  for (const auto& op : ir_model->ops()) {
    if (op->name == "fully_connected" || op->name == "convolution_2d") {
      found_fc_or_conv = true;
    }
    if (op->name == "reshape") {
      reshape_op = op.get();
    }
  }
  EXPECT_TRUE(found_fc_or_conv);
  ASSERT_NE(reshape_op, nullptr);

  auto reshape_attr =
      std::any_cast<::ml_drift::ReshapeAttributes>(reshape_op->attr);
  // output [1, 8] -> BHWC(1, 1, 8, 1)
  EXPECT_EQ(reshape_attr.new_shape.b, 1);
  EXPECT_EQ(reshape_attr.new_shape.h, 1);
  EXPECT_EQ(reshape_attr.new_shape.w, 1);
  EXPECT_EQ(reshape_attr.new_shape.c, 8);
}
TEST(ConvertFullyConnectedTest, SetFullyConnectedOutputShapeTest) {
  IrModelBuilderOptions options;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate(
      CreateStubDelegate(options), DeleteStubDelegate);
  ASSERT_NE(delegate, nullptr);

  SingleOpInterpreterBuilder model(kTfLiteBuiltinFullyConnected, /*version=*/1);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddInput(kTfLiteFloat32, {8, 24});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 8});

  TfLiteFullyConnectedParams* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(
          calloc(1, sizeof(TfLiteFullyConnectedParams)));
  params->activation = kTfLiteActNone;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  const ::ml_drift::ir::IrOp* fc_op = nullptr;
  for (const auto& op : ir_model->ops()) {
    if (op->name == "fully_connected") {
      fc_op = op.get();
      break;
    }
  }
  ASSERT_NE(fc_op, nullptr);

  const auto* out_tensor = ir_model->tensor(fc_op->outputs[0]);
  auto out_shape = out_tensor->desc.GetBHWCShape();
  EXPECT_EQ(out_shape.b, 1);
  EXPECT_EQ(out_shape.h, 2);
  EXPECT_EQ(out_shape.w, 3);
  EXPECT_EQ(out_shape.c, 8);
}

}  // namespace
}  // namespace litert::ml_drift::ir
