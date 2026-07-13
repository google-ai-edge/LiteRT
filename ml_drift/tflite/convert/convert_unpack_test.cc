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
#include <cstdlib>
#include <memory>
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

using ::testing::Eq;
using ::testing::SizeIs;

class ConvertUnpackTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertUnpackTest, SingleOutput) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinUnpack);
  builder.AddInput(GetParam(), {1, 2, 3});
  builder.AddOutput(GetParam(), {2, 3});

  TfLiteUnpackParams* params =
      static_cast<TfLiteUnpackParams*>(calloc(1, sizeof(TfLiteUnpackParams)));
  params->axis = 0;
  params->num = 1;
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // Should be a single RESHAPE op.
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  EXPECT_THAT(ir_model->ops()[0]->name,
              Eq(ToString(::ml_drift::OperationType::RESHAPE)));
}

TEST_P(ConvertUnpackTest, MultiOutputAxis0) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinUnpack);
  builder.AddInput(GetParam(), {2, 2, 3});
  builder.AddOutput(GetParam(), {2, 3});
  builder.AddOutput(GetParam(), {2, 3});

  TfLiteUnpackParams* params =
      static_cast<TfLiteUnpackParams*>(calloc(1, sizeof(TfLiteUnpackParams)));
  params->axis = 0;
  params->num = 2;
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // Should have 1 SPLIT op followed by 2 RESHAPE ops.
  ASSERT_THAT(ir_model->ops(), SizeIs(3));
  EXPECT_THAT(ir_model->ops()[0]->name,
              Eq(ToString(::ml_drift::OperationType::SPLIT)));
  EXPECT_THAT(ir_model->ops()[1]->name,
              Eq(ToString(::ml_drift::OperationType::RESHAPE)));
  EXPECT_THAT(ir_model->ops()[2]->name,
              Eq(ToString(::ml_drift::OperationType::RESHAPE)));

  const auto* split_attr =
      std::any_cast<::ml_drift::SplitAttributes>(&ir_model->ops()[0]->attr);
  ASSERT_NE(split_attr, nullptr);
}

TEST_P(ConvertUnpackTest, MultiOutputAxis2) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinUnpack);
  builder.AddInput(GetParam(), {2, 2, 3});
  builder.AddOutput(GetParam(), {2, 2});
  builder.AddOutput(GetParam(), {2, 2});
  builder.AddOutput(GetParam(), {2, 2});

  TfLiteUnpackParams* params =
      static_cast<TfLiteUnpackParams*>(calloc(1, sizeof(TfLiteUnpackParams)));
  params->axis = 2;
  params->num = 3;
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // 1 SPLIT + 3 RESHAPEs
  ASSERT_THAT(ir_model->ops(), SizeIs(4));
  EXPECT_THAT(ir_model->ops()[0]->name,
              Eq(ToString(::ml_drift::OperationType::SPLIT)));
}

TEST_P(ConvertUnpackTest, NegativeAxis) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinUnpack);
  builder.AddInput(GetParam(), {2, 2, 2});
  builder.AddOutput(GetParam(), {2, 2});
  builder.AddOutput(GetParam(), {2, 2});

  TfLiteUnpackParams* params =
      static_cast<TfLiteUnpackParams*>(calloc(1, sizeof(TfLiteUnpackParams)));
  params->axis = -1;
  params->num = 2;
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(3));
  EXPECT_THAT(ir_model->ops()[0]->name,
              Eq(ToString(::ml_drift::OperationType::SPLIT)));
}

INSTANTIATE_TEST_SUITE_P(ConvertUnpackTest, ConvertUnpackTest,
                         ::testing::Values(kTfLiteFloat32, kTfLiteFloat16,
                                           kTfLiteInt32, kTfLiteInt8));

}  // namespace
}  // namespace litert::ml_drift::ir
