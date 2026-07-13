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
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;

class ConvertPackTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertPackTest, SingleInput) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinPack);
  builder.AddInput(GetParam(), {2, 3});
  builder.AddOutput(GetParam(), {1, 2, 3});

  TfLitePackParams* params =
      static_cast<TfLitePackParams*>(calloc(1, sizeof(TfLitePackParams)));
  params->axis = 0;
  params->values_count = 1;
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

TEST_P(ConvertPackTest, MultiInputAxis0) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinPack);
  builder.AddInput(GetParam(), {2, 3});
  builder.AddInput(GetParam(), {2, 3});
  builder.AddOutput(GetParam(), {2, 2, 3});

  TfLitePackParams* params =
      static_cast<TfLitePackParams*>(calloc(1, sizeof(TfLitePackParams)));
  params->axis = 0;
  params->values_count = 2;
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // Should have 2 RESHAPE ops followed by 1 CONCAT op.
  ASSERT_THAT(ir_model->ops(), SizeIs(3));
  EXPECT_THAT(ir_model->ops()[0]->name,
              Eq(ToString(::ml_drift::OperationType::RESHAPE)));
  EXPECT_THAT(ir_model->ops()[1]->name,
              Eq(ToString(::ml_drift::OperationType::RESHAPE)));
  EXPECT_THAT(ir_model->ops()[2]->name,
              Eq(ToString(::ml_drift::OperationType::CONCAT)));

  const auto* concat_attr =
      std::any_cast<::ml_drift::ConcatAttributes>(&ir_model->ops()[2]->attr);
  ASSERT_NE(concat_attr, nullptr);
}

TEST_P(ConvertPackTest, MultiInputAxis2) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinPack);
  builder.AddInput(GetParam(), {2, 2});
  builder.AddInput(GetParam(), {2, 2});
  builder.AddInput(GetParam(), {2, 2});
  builder.AddOutput(GetParam(), {2, 2, 3});

  TfLitePackParams* params =
      static_cast<TfLitePackParams*>(calloc(1, sizeof(TfLitePackParams)));
  params->axis = 2;
  params->values_count = 3;
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // 3 RESHAPEs + 1 CONCAT
  ASSERT_THAT(ir_model->ops(), SizeIs(4));
  EXPECT_THAT(ir_model->ops()[3]->name,
              Eq(ToString(::ml_drift::OperationType::CONCAT)));
}

TEST_P(ConvertPackTest, NegativeAxis) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinPack);
  builder.AddInput(GetParam(), {2, 2});
  builder.AddInput(GetParam(), {2, 2});
  builder.AddOutput(GetParam(), {2, 2, 2});

  TfLitePackParams* params =
      static_cast<TfLitePackParams*>(calloc(1, sizeof(TfLitePackParams)));
  params->axis = -1;
  params->values_count = 2;
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(3));
  EXPECT_THAT(ir_model->ops()[2]->name,
              Eq(ToString(::ml_drift::OperationType::CONCAT)));
}

INSTANTIATE_TEST_SUITE_P(ConvertPackTest, ConvertPackTest,
                         ::testing::Values(kTfLiteBFloat16, kTfLiteBool,
                                           kTfLiteFloat16, kTfLiteFloat32,
                                           kTfLiteInt8, kTfLiteInt16,
                                           kTfLiteInt32, kTfLiteUInt8,
                                           kTfLiteUInt16, kTfLiteUInt32));

}  // namespace
}  // namespace litert::ml_drift::ir
