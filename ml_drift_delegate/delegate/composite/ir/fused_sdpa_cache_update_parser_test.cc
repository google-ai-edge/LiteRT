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

#include "ml_drift_delegate/delegate/composite/ir/fused_sdpa_cache_update_parser.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"
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

TfLiteStablehloCompositeParams* CreateFusedSdpaCacheUpdateParams(
    std::optional<float> softcap = std::nullopt) {
  size_t total_size = sizeof(TfLiteStablehloCompositeParams);
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    if (softcap.has_value()) {
      fbb.Float("softcap", *softcap);
    }
  });
  fbb.Finish();
  std::vector<uint8_t> buffer = fbb.GetBuffer();
  total_size += buffer.size();

  void* block = calloc(1, total_size);
  auto* params = reinterpret_cast<TfLiteStablehloCompositeParams*>(block);
  params->name = kFusedSdpaCacheUpdateCompositeName;

  uint8_t* attr_data = reinterpret_cast<uint8_t*>(params + 1);
  params->attributes = attr_data;
  params->attributes_size = buffer.size();
  memcpy(attr_data, buffer.data(), buffer.size());
  return params;
}

class ConvertFusedSdpaCacheUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CustomIrOpMap custom_parsers;
    custom_parsers[kFusedSdpaCacheUpdateCompositeName] =
        GetFusedSdpaCacheUpdateParser();
    delegate_ = CreateStubDelegate(/*options=*/{}, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertFusedSdpaCacheUpdateTest, ThreeOutputsWithSoftcap) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  const int Hkv = 2, G = 2, T = 4, W = 16, D = 8;
  builder.AddInput(kTfLiteFloat32, {1, Hkv, G * T, D});  // query
  builder.AddInput(kTfLiteFloat32, {1, Hkv, W, D});      // key_cache
  builder.AddInput(kTfLiteFloat32, {1, Hkv, D, W});      // value_cache
  builder.AddInput(kTfLiteFloat32, {1, Hkv, T, D});      // key_new
  builder.AddInput(kTfLiteFloat32, {1, Hkv, D, T});      // value_new
  builder.AddInput(kTfLiteBool, {1, 1, T, W + T});       // mask
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});          // param
  builder.AddOutput(kTfLiteFloat32, {1, Hkv, G * T, D});
  builder.AddOutput(kTfLiteFloat32, {1, Hkv, W, D});
  builder.AddOutput(kTfLiteFloat32, {1, Hkv, D, W});
  builder.SetParameters(CreateFusedSdpaCacheUpdateParams(/*softcap=*/30.0f));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("fused_sdpa_cache_update"));
  EXPECT_THAT(op->inputs, SizeIs(7));
  EXPECT_THAT(op->outputs, SizeIs(3));

  const auto* attr =
      std::any_cast<::litert::ml_drift::FusedSdpaCacheUpdateAttributes>(
          &op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_TRUE(attr->update_cache);
  ASSERT_TRUE(attr->softcap.has_value());
  EXPECT_EQ(*attr->softcap, 30.0f);
}

TEST_F(ConvertFusedSdpaCacheUpdateTest, SingleOutputSharedKvLayer) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  const int Hkv = 2, G = 1, T = 1, W = 16, D = 8;
  builder.AddInput(kTfLiteFloat32, {1, Hkv, G * T, D});  // query
  builder.AddInput(kTfLiteFloat32, {1, Hkv, W, D});      // key_cache
  builder.AddInput(kTfLiteFloat32, {1, Hkv, D, W});      // value_cache
  builder.AddInput(kTfLiteFloat32, {1, Hkv, T, D});      // key_new
  builder.AddInput(kTfLiteFloat32, {1, Hkv, D, T});      // value_new
  builder.AddInput(kTfLiteFloat32, {1, 1, T, W + T});    // float mask
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});          // param
  builder.AddOutput(kTfLiteFloat32, {1, Hkv, G * T, D});
  builder.SetParameters(CreateFusedSdpaCacheUpdateParams());

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("fused_sdpa_cache_update"));
  EXPECT_THAT(op->inputs, SizeIs(7));
  EXPECT_THAT(op->outputs, SizeIs(1));

  const auto* attr =
      std::any_cast<::litert::ml_drift::FusedSdpaCacheUpdateAttributes>(
          &op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_FALSE(attr->update_cache);
  EXPECT_FALSE(attr->softcap.has_value());
}

TEST_F(ConvertFusedSdpaCacheUpdateTest, RejectsNonMultipleOf4CacheSize) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  const int Hkv = 2, G = 1, T = 1, W = 14, D = 8;
  builder.AddInput(kTfLiteFloat32, {1, Hkv, G * T, D});
  builder.AddInput(kTfLiteFloat32, {1, Hkv, W, D});
  builder.AddInput(kTfLiteFloat32, {1, Hkv, D, W});
  builder.AddInput(kTfLiteFloat32, {1, Hkv, T, D});
  builder.AddInput(kTfLiteFloat32, {1, Hkv, D, T});
  builder.AddInput(kTfLiteBool, {1, 1, T, W + T});
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});
  builder.AddOutput(kTfLiteFloat32, {1, Hkv, G * T, D});
  builder.SetParameters(CreateFusedSdpaCacheUpdateParams());

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  EXPECT_EQ(GetIrModel(delegate_), nullptr);
}

}  // namespace
}  // namespace litert::ml_drift::ir
