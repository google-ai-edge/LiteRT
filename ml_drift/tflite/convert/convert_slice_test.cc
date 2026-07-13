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
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertSliceTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        CreateStubDelegate(), DeleteStubDelegate);
    ASSERT_TRUE(delegate_);
  }

  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_ = {
      nullptr, [](TfLiteDelegate*) {}};
};

std::vector<uint8_t> ToBytes(const std::vector<int32_t>& data) {
  std::vector<uint8_t> bytes(data.size() * sizeof(int32_t));
  std::memcpy(bytes.data(), data.data(), bytes.size());
  return bytes;
}

TEST_P(ConvertSliceTest, Slice1D) {
  std::vector<int32_t> begins_data = {1};
  std::vector<int32_t> sizes_data = {2};

  std::vector<uint8_t> begins_bytes = ToBytes(begins_data);
  std::vector<uint8_t> sizes_bytes = ToBytes(sizes_data);

  SingleOpInterpreterBuilder model(kTfLiteBuiltinSlice, /*version=*/1);
  model.AddInput(GetParam(), {4});
  model.AddConstInput(kTfLiteInt32, {1}, begins_bytes);
  model.AddConstInput(kTfLiteInt32, {1}, sizes_bytes);
  model.AddOutput(GetParam(), {2});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);
  const ::ml_drift::ir::IrOp* slice_op = ir_model->op(ir_model->ops()[0]->id);
  const ::ml_drift::SliceAttributes* attr =
      std::any_cast<::ml_drift::SliceAttributes>(&slice_op->attr);
  ASSERT_TRUE(attr);
  // MapToBHWDC for size 1: (start_val, start_val, start_val, start_val,
  // values[0]) starts: (0, 0, 0, 0, 1), sizes: (1, 1, 1, 1, 2)
  EXPECT_EQ(attr->starts, ::ml_drift::BHWC(1, 0, 0, 0));
  EXPECT_EQ(attr->ends, ::ml_drift::BHWC(3, 1, 1, 1));
}

TEST_P(ConvertSliceTest, Slice2D) {
  std::vector<int32_t> begins_data = {1, 1};
  std::vector<int32_t> sizes_data = {2, 2};

  std::vector<uint8_t> begins_bytes = ToBytes(begins_data);
  std::vector<uint8_t> sizes_bytes = ToBytes(sizes_data);

  SingleOpInterpreterBuilder model(kTfLiteBuiltinSlice, /*version=*/1);
  model.AddInput(GetParam(), {4, 4});
  model.AddConstInput(kTfLiteInt32, {2}, begins_bytes);
  model.AddConstInput(kTfLiteInt32, {2}, sizes_bytes);
  model.AddOutput(GetParam(), {2, 2});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);
  const ::ml_drift::ir::IrOp* slice_op = ir_model->op(ir_model->ops()[0]->id);
  const ::ml_drift::SliceAttributes* attr =
      std::any_cast<::ml_drift::SliceAttributes>(&slice_op->attr);
  ASSERT_TRUE(attr);
  // MapToBHWDC for size 2: (start_val, values[0], values[1], start_val,
  // start_val) starts: (0, 1, 1, 0, 0), sizes: (1, 2, 2, 1, 1)
  EXPECT_EQ(attr->starts, ::ml_drift::BHWC(1, 0, 0, 1));
  EXPECT_EQ(attr->ends, ::ml_drift::BHWC(3, 1, 1, 3));
}

TEST_P(ConvertSliceTest, Slice3D) {
  std::vector<int32_t> begins_data = {1, 1, 1};
  std::vector<int32_t> sizes_data = {2, 2, 2};

  std::vector<uint8_t> begins_bytes = ToBytes(begins_data);
  std::vector<uint8_t> sizes_bytes = ToBytes(sizes_data);

  SingleOpInterpreterBuilder model(kTfLiteBuiltinSlice, /*version=*/1);
  model.AddInput(GetParam(), {4, 4, 4});
  model.AddConstInput(kTfLiteInt32, {3}, begins_bytes);
  model.AddConstInput(kTfLiteInt32, {3}, sizes_bytes);
  model.AddOutput(GetParam(), {2, 2, 2});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);
  const ::ml_drift::ir::IrOp* slice_op = ir_model->op(ir_model->ops()[0]->id);
  const ::ml_drift::SliceAttributes* attr =
      std::any_cast<::ml_drift::SliceAttributes>(&slice_op->attr);
  ASSERT_TRUE(attr);
  // MapToBHWDC for size 3: (start_val, values[0], values[1], start_val,
  // values[2]) starts: (0, 1, 1, 0, 1), sizes: (1, 2, 2, 1, 2)
  EXPECT_EQ(attr->starts, ::ml_drift::BHWC(1, 0, 1, 1));
  EXPECT_EQ(attr->ends, ::ml_drift::BHWC(3, 1, 3, 3));
}

TEST_P(ConvertSliceTest, Slice4D) {
  std::vector<int32_t> begins_data = {0, 1, 1, 0};
  std::vector<int32_t> sizes_data = {1, 2, 2, 3};

  std::vector<uint8_t> begins_bytes = ToBytes(begins_data);
  std::vector<uint8_t> sizes_bytes = ToBytes(sizes_data);

  // Generate a model with a slice op.
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSlice, /*version=*/1);
  model.AddInput(GetParam(), {1, 4, 4, 3});  // input data
  model.AddConstInput(kTfLiteInt32, {4}, begins_bytes);
  model.AddConstInput(kTfLiteInt32, {4}, sizes_bytes);
  model.AddOutput(GetParam(), {1, 2, 2, 3});

  // Build the interpreter and delegate it.
  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  // Inspect the IrModel generated by the delegate.
  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);
  EXPECT_EQ(ir_model->inputs().size(), 1);
  EXPECT_EQ(ir_model->outputs().size(), 1);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* slice_op = ir_model->op(ir_model->ops()[0]->id);
  EXPECT_EQ(slice_op->inputs.size(), 1);
  EXPECT_EQ(slice_op->outputs.size(), 1);
  EXPECT_EQ(slice_op->name, "slice");
  const ::ml_drift::SliceAttributes* attr =
      std::any_cast<::ml_drift::SliceAttributes>(&slice_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->starts, ::ml_drift::BHWC(0, 1, 1, 0));
  EXPECT_EQ(attr->ends, ::ml_drift::BHWC(1, 3, 3, 3));
  EXPECT_EQ(attr->strides, ::ml_drift::BHWC(1, 1, 1, 1));

  // Sanity check inference.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_P(ConvertSliceTest, Slice5D) {
  std::vector<int32_t> begins_data = {1, 2, 3, 4, 5};
  std::vector<int32_t> sizes_data = {6, 7, 8, 9, 10};

  std::vector<uint8_t> begins_bytes = ToBytes(begins_data);
  std::vector<uint8_t> sizes_bytes = ToBytes(sizes_data);

  SingleOpInterpreterBuilder model(kTfLiteBuiltinSlice, /*version=*/1);
  model.AddInput(GetParam(), {10, 20, 30, 40, 50});
  model.AddConstInput(kTfLiteInt32, {5}, begins_bytes);
  model.AddConstInput(kTfLiteInt32, {5}, sizes_bytes);
  model.AddOutput(GetParam(), {6, 7, 8, 9, 10});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);
  const ::ml_drift::ir::IrOp* slice_op = ir_model->op(ir_model->ops()[0]->id);
  const ::ml_drift::Slice3DAttributes* attr =
      std::any_cast<::ml_drift::Slice3DAttributes>(&slice_op->attr);
  ASSERT_TRUE(attr);
  // MapToBHWDC for size 5: (values[0], values[1], values[2], values[3],
  // values[4]) starts: (1, 2, 3, 4, 5), sizes: (6, 7, 8, 9, 10) attr maps to
  // BHWC(b, h, w, c) from starts(b, h, w, d, c)
  EXPECT_EQ(attr->starts, ::ml_drift::BHWDC(1, 2, 3, 4, 5));
  EXPECT_EQ(attr->ends, ::ml_drift::BHWDC(7, 9, 11, 13, 15));
}

TEST_P(ConvertSliceTest, SliceWithNegativeSize) {
  std::vector<int32_t> begins_data = {0, 1, 1, 0};
  std::vector<int32_t> sizes_data = {-1, -1, -1, -1};

  std::vector<uint8_t> begins_bytes = ToBytes(begins_data);
  std::vector<uint8_t> sizes_bytes = ToBytes(sizes_data);

  // Generate a model with a slice op.
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSlice, /*version=*/1);
  model.AddInput(GetParam(), {1, 4, 4, 3});  // input data
  model.AddConstInput(kTfLiteInt32, {4}, begins_bytes);
  model.AddConstInput(kTfLiteInt32, {4}, sizes_bytes);
  model.AddOutput(GetParam(), {1, 3, 3, 3});

  // Build the interpreter and delegate it.
  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  // Inspect the IrModel generated by the delegate.
  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);
  EXPECT_EQ(ir_model->inputs().size(), 1);
  EXPECT_EQ(ir_model->outputs().size(), 1);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* slice_op = ir_model->op(ir_model->ops()[0]->id);
  EXPECT_EQ(slice_op->inputs.size(), 1);
  EXPECT_EQ(slice_op->outputs.size(), 1);
  EXPECT_EQ(slice_op->name, "slice");
  const ::ml_drift::SliceAttributes* attr =
      std::any_cast<::ml_drift::SliceAttributes>(&slice_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->starts, ::ml_drift::BHWC(0, 1, 1, 0));
  EXPECT_EQ(attr->ends, ::ml_drift::BHWC(1, 4, 4, 3));
  EXPECT_EQ(attr->strides, ::ml_drift::BHWC(1, 1, 1, 1));

  // Sanity check inference.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

INSTANTIATE_TEST_SUITE_P(ConvertSliceTest, ConvertSliceTest,
                         ::testing::Values(    // clang-format off
                             // go/keep-sorted start numeric=yes
                             kTfLiteBool,
                             kTfLiteFloat16,
                             kTfLiteFloat32,
                             kTfLiteInt8,
                             kTfLiteInt16,
                             kTfLiteInt32,
                             kTfLiteUInt8,
                             kTfLiteUInt16,
                             kTfLiteUInt32
                             // go/keep-sorted end
                             // clang-format on
                             ));

}  // namespace
}  // namespace litert::ml_drift::ir
