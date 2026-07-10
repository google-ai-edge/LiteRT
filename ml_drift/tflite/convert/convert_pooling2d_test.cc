// Copyright 2026 The ML Drift Authors.
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
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

// Test fixture for builtin pooling operations (AveragePool2D, MaxPool2D).
class ConvertPooling2dTest
    : public ::testing::TestWithParam<std::tuple<
          TfLiteBuiltinOperator, TfLitePadding, TfLiteFusedActivation>> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
    std::tie(op_code_, padding_, activation_) = GetParam();
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
  TfLiteBuiltinOperator op_code_;
  TfLitePadding padding_;
  TfLiteFusedActivation activation_;
};

TEST_P(ConvertPooling2dTest, Parameterized) {
  SingleOpInterpreterBuilder model(op_code_, /*version=*/1);

  model.AddInput(kTfLiteFloat32, {1, 4, 4, 4});   // input
  model.AddOutput(kTfLiteFloat32, {1, 2, 2, 4});  // output

  TfLitePoolParams* params =
      reinterpret_cast<TfLitePoolParams*>(calloc(1, sizeof(TfLitePoolParams)));
  params->padding = padding_;
  params->stride_height = 2;
  params->stride_width = 2;
  params->filter_height = 2;
  params->filter_width = 2;
  params->activation = activation_;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // Expect 1 POOLING_2D op [+ 1 Activation op if activation != None]
  const int expected_ops = (activation_ == kTfLiteActNone) ? 1 : 2;
  ASSERT_EQ(ir_model->ops().size(), expected_ops);

  const ::ml_drift::ir::IrOp* pool_op = ir_model->op(0);
  EXPECT_EQ(pool_op->name, "pooling_2d");
  const auto* attr =
      std::any_cast<::ml_drift::Pooling2DAttributes>(&pool_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->type, (op_code_ == kTfLiteBuiltinAveragePool2d
                             ? ::ml_drift::PoolingType::AVERAGE
                             : ::ml_drift::PoolingType::MAX));
  EXPECT_EQ(attr->kernel.h, 2);
  EXPECT_EQ(attr->kernel.w, 2);
  EXPECT_EQ(attr->strides.h, 2);
  EXPECT_EQ(attr->strides.w, 2);
  EXPECT_FALSE(attr->output_indices);
  EXPECT_EQ(pool_op->outputs.size(), 1);
}

TEST_P(ConvertPooling2dTest, GlobalAveragePoolingToMean) {
  if (op_code_ != kTfLiteBuiltinAveragePool2d) {
    GTEST_SKIP() << "Test only applies to AveragePool2D.";
  }

  SingleOpInterpreterBuilder model(op_code_, /*version=*/1);

  model.AddInput(kTfLiteFloat32, {1, 4, 4, 4});   // input
  model.AddOutput(kTfLiteFloat32, {1, 1, 1, 4});  // output

  TfLitePoolParams* params =
      reinterpret_cast<TfLitePoolParams*>(calloc(1, sizeof(TfLitePoolParams)));
  params->padding = kTfLitePaddingValid;
  params->stride_height = 1;
  params->stride_width = 1;
  params->filter_height = 4;
  params->filter_width = 4;
  params->activation = activation_;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // Expect 1 MEAN op [+ 1 Activation op if activation != None]
  const int expected_ops = (activation_ == kTfLiteActNone) ? 1 : 2;
  ASSERT_EQ(ir_model->ops().size(), expected_ops);

  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "mean");
  const auto* attr = std::any_cast<::ml_drift::ReduceAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->dims.size(), 2);
  EXPECT_TRUE(attr->dims.contains(::ml_drift::Axis::WIDTH));
  EXPECT_TRUE(attr->dims.contains(::ml_drift::Axis::HEIGHT));
  EXPECT_EQ(op->outputs.size(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    ConvertPooling2dTest, ConvertPooling2dTest,
    ::testing::Combine(
        ::testing::Values(kTfLiteBuiltinAveragePool2d, kTfLiteBuiltinMaxPool2d),
        ::testing::Values(kTfLitePaddingSame, kTfLitePaddingValid),
        ::testing::Values(kTfLiteActNone, kTfLiteActRelu)),
    [](const ::testing::TestParamInfo<ConvertPooling2dTest::ParamType>& info) {
      std::string name;
      name += (std::get<0>(info.param) == kTfLiteBuiltinAveragePool2d ? "avg"
                                                                      : "max");
      name +=
          (std::get<1>(info.param) == kTfLitePaddingSame ? "_same" : "_valid");
      name += (std::get<2>(info.param) == kTfLiteActNone ? "_noact" : "_relu");
      return name;
    });

// Test fixture for custom MaxPoolingWithArgmax2D operation.
class ConvertCustomMaxPooling2dTest
    : public ::testing::TestWithParam<
          std::tuple<TfLitePadding, TfLiteFusedActivation>> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
    std::tie(padding_, activation_) = GetParam();
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
  TfLitePadding padding_;
  TfLiteFusedActivation activation_;
};

TEST_P(ConvertCustomMaxPooling2dTest, Parameterized) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCustom, /*version=*/1);
  model.SetCustomName("MaxPoolingWithArgmax2D");

  model.AddInput(kTfLiteFloat32, {1, 4, 4, 4});   // input
  model.AddOutput(kTfLiteFloat32, {1, 2, 2, 4});  // output
  model.AddOutput(kTfLiteFloat32, {1, 2, 2, 4});  // argmax indices

  TfLitePoolParams* params =
      reinterpret_cast<TfLitePoolParams*>(calloc(1, sizeof(TfLitePoolParams)));
  params->padding = padding_;
  params->stride_height = 2;
  params->stride_width = 2;
  params->filter_height = 2;
  params->filter_width = 2;
  params->activation = activation_;
  model.SetCustomData(params, sizeof(TfLitePoolParams));

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // Expect 1 POOLING_2D op [+ 1 Activation op if activation != None]
  const int expected_ops = (activation_ == kTfLiteActNone) ? 1 : 2;
  ASSERT_EQ(ir_model->ops().size(), expected_ops);

  const ::ml_drift::ir::IrOp* pool_op = ir_model->op(0);
  EXPECT_EQ(pool_op->name, "pooling_2d");
  const auto* attr =
      std::any_cast<::ml_drift::Pooling2DAttributes>(&pool_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->type, ::ml_drift::PoolingType::MAX);
  EXPECT_EQ(attr->kernel.h, 2);
  EXPECT_EQ(attr->kernel.w, 2);
  EXPECT_EQ(attr->strides.h, 2);
  EXPECT_EQ(attr->strides.w, 2);
  EXPECT_TRUE(attr->output_indices);
  ASSERT_EQ(pool_op->outputs.size(), 2);
  const ::ml_drift::ir::IrTensor* indices_tensor =
      ir_model->tensor(pool_op->outputs[1]);
  ASSERT_NE(indices_tensor, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    ConvertCustomMaxPooling2dTest, ConvertCustomMaxPooling2dTest,
    ::testing::Combine(::testing::Values(kTfLitePaddingSame,
                                         kTfLitePaddingValid),
                       ::testing::Values(kTfLiteActNone, kTfLiteActRelu)),
    [](const ::testing::TestParamInfo<ConvertCustomMaxPooling2dTest::ParamType>&
           info) {
      std::string name = "max";
      name +=
          (std::get<0>(info.param) == kTfLitePaddingSame ? "_same" : "_valid");
      name += (std::get<1>(info.param) == kTfLiteActNone ? "_noact" : "_relu");
      name += "_argmax";
      return name;
    });

}  // namespace
}  // namespace litert::ml_drift::ir
