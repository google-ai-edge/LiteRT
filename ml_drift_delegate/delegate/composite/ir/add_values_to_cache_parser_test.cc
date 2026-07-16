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

#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/ir/add_values_to_cache_parser.h"
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;

TfLiteStablehloCompositeParams* CreateAddValuesToCacheParams(
    int kv_cache_batch_size, int cache_size, int head_size,
    std::optional<float> scale_k = std::nullopt,
    std::optional<float> scale_v = std::nullopt) {
  size_t total_size = sizeof(TfLiteStablehloCompositeParams);
  std::vector<uint8_t> buffer;

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("kv_cache_batch_size", kv_cache_batch_size);
    fbb.Int("cache_size", cache_size);
    fbb.Int("head_size", head_size);
    if (scale_k.has_value()) {
      fbb.Float("scale_k", *scale_k);
    }
    if (scale_v.has_value()) {
      fbb.Float("scale_v", *scale_v);
    }
  });
  fbb.Finish();
  buffer = fbb.GetBuffer();
  total_size += buffer.size();

  void* block = calloc(1, total_size);
  TfLiteStablehloCompositeParams* params =
      reinterpret_cast<TfLiteStablehloCompositeParams*>(block);
  params->name = "odml.cache_update";

  uint8_t* attr_data = reinterpret_cast<uint8_t*>(params + 1);
  params->attributes = attr_data;
  params->attributes_size = buffer.size();
  memcpy(attr_data, buffer.data(), buffer.size());

  return params;
}

class ConvertAddValuesToCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CustomIrOpMap custom_parsers;
    custom_parsers["odml.cache_update"] = GetAddValuesToCacheParser();
    delegate_ = CreateStubDelegate(/*options=*/{}, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertAddValuesToCacheTest, Basic) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 64});   // src_k
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 64});   // src_v
  builder.AddInput(kTfLiteInt32, {2});               // params
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 64});  // cache_k
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 64});  // cache_v

  TfLiteStablehloCompositeParams* params =
      CreateAddValuesToCacheParams(2, 128, 64, 0.5f, 2.0f);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("add_values_to_cache"));
  EXPECT_THAT(op->inputs, SizeIs(3));
  EXPECT_THAT(op->outputs, SizeIs(2));

  const auto* attr =
      std::any_cast<::litert::ml_drift::AddValuesToCacheAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->kv_cache_batch_size, 2);
  EXPECT_EQ(attr->cache_size, 128);
  EXPECT_EQ(attr->head_size, 64);
  EXPECT_TRUE(attr->scale_k.has_value());
  EXPECT_THAT(*attr->scale_k, Eq(0.5f));
  EXPECT_TRUE(attr->scale_v.has_value());
  EXPECT_THAT(*attr->scale_v, Eq(2.0f));
}

}  // namespace
}  // namespace litert::ml_drift::ir
