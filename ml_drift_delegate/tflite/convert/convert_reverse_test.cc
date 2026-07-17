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
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;

class ConvertReverseTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertReverseTest, SingleAxis) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinReverseV2);
  builder.AddInput(GetParam(), {1, 2, 2, 1});
  // Axis 1 (HEIGHT).
  std::vector<int32_t> axes = {1};
  std::vector<uint8_t> axes_data(
      reinterpret_cast<const uint8_t*>(axes.data()),
      reinterpret_cast<const uint8_t*>(axes.data() + axes.size()));
  builder.AddConstInput(kTfLiteInt32, {static_cast<int>(axes.size())},
                        axes_data);
  builder.AddOutput(GetParam(), {1, 2, 2, 1});

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  EXPECT_THAT(ir_model->ops()[0]->name,
              Eq(ToString(::ml_drift::OperationType::REVERSE)));

  const auto* attr =
      std::any_cast<::ml_drift::ReverseAttributes>(&ir_model->ops()[0]->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_THAT(attr->axes, SizeIs(1));
  EXPECT_TRUE(attr->axes.count(::ml_drift::Axis::HEIGHT));
}

TEST_P(ConvertReverseTest, MultipleAxes) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinReverseV2);
  builder.AddInput(GetParam(), {1, 2, 3, 4});
  // Axes 1 (HEIGHT) and 2 (WIDTH).
  std::vector<int32_t> axes = {1, 2};
  std::vector<uint8_t> axes_data(
      reinterpret_cast<const uint8_t*>(axes.data()),
      reinterpret_cast<const uint8_t*>(axes.data() + axes.size()));
  builder.AddConstInput(kTfLiteInt32, {static_cast<int>(axes.size())},
                        axes_data);
  builder.AddOutput(GetParam(), {1, 2, 3, 4});

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  EXPECT_THAT(ir_model->ops()[0]->name,
              Eq(ToString(::ml_drift::OperationType::REVERSE)));

  const auto* attr =
      std::any_cast<::ml_drift::ReverseAttributes>(&ir_model->ops()[0]->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_THAT(attr->axes, SizeIs(2));
  EXPECT_TRUE(attr->axes.count(::ml_drift::Axis::HEIGHT));
  EXPECT_TRUE(attr->axes.count(::ml_drift::Axis::WIDTH));
}

INSTANTIATE_TEST_SUITE_P(ConvertReverseTest, ConvertReverseTest,
                         ::testing::Values(kTfLiteBFloat16, kTfLiteBool,
                                           kTfLiteFloat16, kTfLiteFloat32,
                                           kTfLiteInt8, kTfLiteInt16,
                                           kTfLiteInt32, kTfLiteUInt8,
                                           kTfLiteUInt16, kTfLiteUInt32));

}  // namespace
}  // namespace litert::ml_drift::ir
