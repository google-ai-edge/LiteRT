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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
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

struct ReduceTestParam {
  TfLiteBuiltinOperator builtin_code;
  TfLiteType type;
  std::string expected_op_name;
  bool keep_dims;
  std::vector<int> input_shape;
  std::vector<int32_t> axes;
  std::vector<int> output_shape;
};

class ConvertReduceTest : public ::testing::TestWithParam<ReduceTestParam> {
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

TEST_P(ConvertReduceTest, ConvertReduce) {
  const auto& param = GetParam();
  SingleOpInterpreterBuilder model(param.builtin_code);
  model.AddInput(param.type, param.input_shape);

  std::vector<uint8_t> axes_bytes(param.axes.size() * sizeof(int32_t));
  std::memcpy(axes_bytes.data(), param.axes.data(), axes_bytes.size());
  model.AddConstInput(kTfLiteInt32, {static_cast<int>(param.axes.size())},
                      axes_bytes);

  model.AddOutput(param.type, param.output_shape);

  TfLiteReducerParams* params = reinterpret_cast<TfLiteReducerParams*>(
      calloc(1, sizeof(TfLiteReducerParams)));
  params->keep_dims = param.keep_dims;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  if (param.keep_dims) {
    // Expect 1 op.
    ASSERT_EQ(ir_model->ops().size(), 1);
    const ::ml_drift::ir::IrOp* op = ir_model->op(0);
    EXPECT_EQ(op->name, param.expected_op_name);

    const ::ml_drift::ReduceAttributes* attr =
        std::any_cast<::ml_drift::ReduceAttributes>(&op->attr);
    ASSERT_TRUE(attr);
    EXPECT_EQ(attr->dims.size(), param.axes.size());
  } else {
    // Expect 2 ops: reduce and reshape.
    ASSERT_EQ(ir_model->ops().size(), 2);
    EXPECT_EQ(ir_model->op(0)->name, param.expected_op_name);
    EXPECT_EQ(ir_model->op(1)->name, "reshape");

    const ::ml_drift::ReduceAttributes* reduce_attr =
        std::any_cast<::ml_drift::ReduceAttributes>(&ir_model->op(0)->attr);
    ASSERT_TRUE(reduce_attr);
    EXPECT_EQ(reduce_attr->dims.size(), param.axes.size());

    if (param.output_shape.size() <= 4) {
      const ::ml_drift::ReshapeAttributes* reshape_attr =
          std::any_cast<::ml_drift::ReshapeAttributes>(&ir_model->op(1)->attr);
      ASSERT_TRUE(reshape_attr);
    } else {
      const ::ml_drift::Reshape3DAttributes* reshape_attr =
          std::any_cast<::ml_drift::Reshape3DAttributes>(
              &ir_model->op(1)->attr);
      ASSERT_TRUE(reshape_attr);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReduceOps, ConvertReduceTest,
    ::testing::Values(ReduceTestParam{kTfLiteBuiltinMean,
                                      kTfLiteFloat32,
                                      "mean",
                                      true,
                                      {1, 2, 3, 4},
                                      {1, 2},
                                      {1, 1, 1, 4}},
                      ReduceTestParam{kTfLiteBuiltinSum,
                                      kTfLiteFloat32,
                                      "reduce_sum",
                                      false,
                                      {1, 2, 3, 4},
                                      {1, 2},
                                      {1, 4}},
                      ReduceTestParam{kTfLiteBuiltinReduceMax,
                                      kTfLiteFloat32,
                                      "reduce_maximum",
                                      true,
                                      {1, 2, 3, 4},
                                      {-1},
                                      {1, 2, 3, 1}},
                      ReduceTestParam{kTfLiteBuiltinReduceMin,
                                      kTfLiteInt32,
                                      "reduce_minimum",
                                      true,
                                      {1, 5, 5, 3},
                                      {1, 2},
                                      {1, 1, 1, 3}},
                      ReduceTestParam{kTfLiteBuiltinReduceProd,
                                      kTfLiteFloat32,
                                      "reduce_product",
                                      false,
                                      {1, 10},
                                      {1},
                                      {1}},
                      ReduceTestParam{kTfLiteBuiltinReduceAll,
                                      kTfLiteBool,
                                      "reduce_all",
                                      true,
                                      {1, 4, 4, 1},
                                      {1, 2},
                                      {1, 1, 1, 1}},
                      ReduceTestParam{kTfLiteBuiltinReduceAny,
                                      kTfLiteBool,
                                      "reduce_any",
                                      false,
                                      {1, 4, 4, 1},
                                      {1, 2},
                                      {1, 1}},
                      ReduceTestParam{kTfLiteBuiltinMean,
                                      kTfLiteFloat32,
                                      "mean",
                                      false,
                                      {1, 64, 2, 49, 49},
                                      {1, 3, 4},
                                      {1, 2}}),
    [](const ::testing::TestParamInfo<ReduceTestParam>& info) {
      return absl::StrCat(info.param.expected_op_name,
                          info.param.keep_dims ? "KeepDims" : "NoKeepDims");
    });

}  // namespace
}  // namespace litert::ml_drift::ir
