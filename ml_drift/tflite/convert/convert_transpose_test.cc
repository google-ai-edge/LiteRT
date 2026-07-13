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
#include <string>
#include <tuple>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::SizeIs;

struct TransposeTestParamShape {
  std::vector<int> input_shape;
  std::vector<int32_t> perm;
  std::vector<int> output_shape;
};

using TransposeTestParam = std::tuple<TransposeTestParamShape, TfLiteType>;

class ConvertTransposeTest
    : public ::testing::TestWithParam<TransposeTestParam> {
 protected:
  void SetUp() override {
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        CreateStubDelegate(), DeleteStubDelegate);
  }

  const ::ml_drift::ir::IrModel* GetIrModelFromBuilder(
      SingleOpInterpreterBuilder& model_builder) {
    interpreter_ = model_builder.Build();
    if (!interpreter_) return nullptr;
    if (interpreter_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk) {
      return nullptr;
    }
    return GetIrModel(delegate_.get());
  }

  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_ = {
      nullptr, [](TfLiteDelegate*) {}};
  std::unique_ptr<::tflite::Interpreter> interpreter_;
};

TEST_P(ConvertTransposeTest, ConvertTranspose) {
  const auto& [shape_param, data_type] = GetParam();
  SingleOpInterpreterBuilder model(kTfLiteBuiltinTranspose);
  model.AddInput(data_type, shape_param.input_shape);

  std::vector<uint8_t> perm_bytes(shape_param.perm.size() * sizeof(int32_t));
  std::memcpy(perm_bytes.data(), shape_param.perm.data(), perm_bytes.size());
  model.AddConstInput(kTfLiteInt32, {static_cast<int>(shape_param.perm.size())},
                      perm_bytes);

  model.AddOutput(data_type, shape_param.output_shape);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "transpose");
  EXPECT_THAT(op->inputs, SizeIs(1));
  EXPECT_THAT(op->outputs, SizeIs(1));

  const ::ml_drift::TransposeAttributes* attr =
      std::any_cast<::ml_drift::TransposeAttributes>(&op->attr);
  ASSERT_TRUE(attr);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(ConvertTransposeTest, Convert5DTranspose) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinTranspose);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4, 5});

  std::vector<int32_t> perm = {0, 4, 1, 2, 3};
  std::vector<uint8_t> perm_bytes(perm.size() * sizeof(int32_t));
  std::memcpy(perm_bytes.data(), perm.data(), perm_bytes.size());
  model.AddConstInput(kTfLiteInt32, {5}, perm_bytes);

  model.AddOutput(kTfLiteFloat32, {1, 5, 2, 3, 4});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "transpose");

  const ::ml_drift::Transpose3DAttributes* attr =
      std::any_cast<::ml_drift::Transpose3DAttributes>(&op->attr);
  ASSERT_TRUE(attr);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeOps, ConvertTransposeTest,
    ::testing::Combine(
        ::testing::Values(
            TransposeTestParamShape{{1, 2, 3, 4}, {0, 3, 1, 2}, {1, 4, 2, 3}},
            TransposeTestParamShape{{1, 2, 3}, {0, 2, 1}, {1, 3, 2}},
            TransposeTestParamShape{{1, 2}, {1, 0}, {2, 1}}),
        ::testing::Values(  // clang-format off
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
            )));

}  // namespace
}  // namespace litert::ml_drift::ir

