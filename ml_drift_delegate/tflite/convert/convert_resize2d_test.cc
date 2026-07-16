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

#include <any>
#include <cstdlib>
#include <memory>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertResize2dTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertResize2dTest, ResizeBilinear) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinResizeBilinear, /*version=*/3);
  const std::vector<int> input_shape = {1, 2, 2, 1};
  const std::vector<int> output_shape = {1, 4, 4, 1};
  model.AddInput(GetParam(), input_shape);
  model.AddOutput(GetParam(), output_shape);

  TfLiteResizeBilinearParams* params = static_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = true;
  params->half_pixel_centers = false;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "resize");

  const ::ml_drift::Resize2DAttributes* attr =
      std::any_cast<::ml_drift::Resize2DAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->type, ::ml_drift::SamplingType::BILINEAR);
  EXPECT_EQ(attr->align_corners, true);
  EXPECT_EQ(attr->half_pixel_centers, false);
  EXPECT_EQ(attr->new_shape.h, 4);
  EXPECT_EQ(attr->new_shape.w, 4);

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_P(ConvertResize2dTest, ResizeNearestNeighbor) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinResizeNearestNeighbor,
                                   /*version=*/3);
  const std::vector<int> input_shape = {1, 2, 2, 3};
  const std::vector<int> output_shape = {1, 10, 10, 3};
  model.AddInput(GetParam(), input_shape);
  model.AddOutput(GetParam(), output_shape);

  TfLiteResizeNearestNeighborParams* params =
      static_cast<TfLiteResizeNearestNeighborParams*>(
          malloc(sizeof(TfLiteResizeNearestNeighborParams)));
  params->align_corners = false;
  params->half_pixel_centers = true;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "resize");

  const ::ml_drift::Resize2DAttributes* attr =
      std::any_cast<::ml_drift::Resize2DAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->type, ::ml_drift::SamplingType::NEAREST);
  EXPECT_EQ(attr->align_corners, false);
  EXPECT_EQ(attr->half_pixel_centers, true);
  EXPECT_EQ(attr->new_shape.h, 10);
  EXPECT_EQ(attr->new_shape.w, 10);

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

INSTANTIATE_TEST_SUITE_P(ConvertResize2dTest, ConvertResize2dTest,
                         ::testing::Values(kTfLiteFloat32, kTfLiteFloat16));

}  // namespace
}  // namespace litert::ml_drift::ir
