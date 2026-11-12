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
#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
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

void SetQuantizationScale(TfLiteTensor* tensor, float scale) {
  ASSERT_EQ(tensor->quantization.type, kTfLiteAffineQuantization);
  auto* quantization =
      static_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  ASSERT_EQ(quantization->scale->size, 1);
  quantization->scale->data[0] = scale;
}

void SetQuantizationZeroPoint(TfLiteTensor* tensor, int32_t zero_point) {
  ASSERT_EQ(tensor->quantization.type, kTfLiteAffineQuantization);
  auto* quantization =
      static_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  ASSERT_EQ(quantization->zero_point->size, 1);
  quantization->zero_point->data[0] = zero_point;
  tensor->params.zero_point = zero_point;
}

TEST_F(ConvertFusedSdpaCacheUpdateTest, Gemma5SlidingInt8CacheWithFloatSlices) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  const int Hkv = 2, G = 4, T = 128, W = 512, D = 256;
  builder.AddInput(kTfLiteFloat32, {1, Hkv, G * T, D});  // query
  builder.AddInput(kTfLiteInt8, {1, Hkv, W, D});         // key_cache
  builder.AddInput(kTfLiteInt8, {1, Hkv, D, W});         // value_cache
  builder.AddInput(kTfLiteFloat32, {1, Hkv, T, D});      // key_new
  builder.AddInput(kTfLiteFloat32, {1, Hkv, D, T});      // value_new
  builder.AddInput(kTfLiteBool, {1, 1, T, W + T});       // mask
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});          // param
  builder.AddOutput(kTfLiteFloat32, {1, Hkv, G * T, D});
  builder.AddOutput(kTfLiteInt8, {1, Hkv, W, D});
  builder.AddOutput(kTfLiteInt8, {1, Hkv, D, W});
  builder.SetParameters(CreateFusedSdpaCacheUpdateParams());

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_NO_FATAL_FAILURE(
      SetQuantizationScale(interpreter->tensor(1), 0.03893793f));
  ASSERT_NO_FATAL_FAILURE(
      SetQuantizationScale(interpreter->tensor(2), 0.06902508f));
  ASSERT_NO_FATAL_FAILURE(
      SetQuantizationScale(interpreter->tensor(8), 0.03893793f));
  ASSERT_NO_FATAL_FAILURE(
      SetQuantizationScale(interpreter->tensor(9), 0.06902508f));
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_EQ(ir_model->tensor(op->inputs[1])->desc.GetDataType(),
            ::ml_drift::DataType::kUint8);
  EXPECT_EQ(ir_model->tensor(op->inputs[2])->desc.GetDataType(),
            ::ml_drift::DataType::kUint8);
  EXPECT_EQ(ir_model->tensor(op->outputs[1])->desc.GetDataType(),
            ::ml_drift::DataType::kUint8);
  EXPECT_EQ(ir_model->tensor(op->outputs[2])->desc.GetDataType(),
            ::ml_drift::DataType::kUint8);

  const auto* attr =
      std::any_cast<::litert::ml_drift::FusedSdpaCacheUpdateAttributes>(
          &op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->kv_quant_bits, 8);
  ASSERT_TRUE(attr->scale_k.has_value());
  EXPECT_FLOAT_EQ(*attr->scale_k, 0.03893793f);
  ASSERT_TRUE(attr->scale_v.has_value());
  EXPECT_FLOAT_EQ(*attr->scale_v, 0.06902508f);
}

TEST_F(ConvertFusedSdpaCacheUpdateTest,
       Int8QueryAndInt8SlicesWithPackedInt4KvCache) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  const int Hkv = 2, G = 2, T = 4, W = 16, D = 16;
  builder.AddInput(kTfLiteInt8, {1, Hkv, G * T, D});  // query
  builder.AddInput(kTfLiteInt4, {1, Hkv, W, D});      // key_cache
  builder.AddInput(kTfLiteInt4, {1, Hkv, D, W});      // value_cache
  builder.AddInput(kTfLiteInt8, {1, Hkv, T, D});      // key_new
  builder.AddInput(kTfLiteInt8, {1, Hkv, D, T});      // value_new
  builder.AddInput(kTfLiteBool, {1, 1, T, W + T});    // mask
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param
  builder.AddOutput(kTfLiteFloat32, {1, Hkv, G * T, D});
  builder.AddOutput(kTfLiteInt4, {1, Hkv, W, D});
  builder.AddOutput(kTfLiteInt4, {1, Hkv, D, W});
  builder.SetParameters(CreateFusedSdpaCacheUpdateParams());

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_NO_FATAL_FAILURE(
      SetQuantizationScale(interpreter->tensor(0), 0.015625f));
  ASSERT_NO_FATAL_FAILURE(SetQuantizationScale(interpreter->tensor(1), 0.25f));
  ASSERT_NO_FATAL_FAILURE(SetQuantizationScale(interpreter->tensor(2), 0.5f));
  ASSERT_NO_FATAL_FAILURE(
      SetQuantizationScale(interpreter->tensor(3), 0.0625f));
  ASSERT_NO_FATAL_FAILURE(SetQuantizationScale(interpreter->tensor(4), 0.125f));
  ASSERT_NO_FATAL_FAILURE(SetQuantizationScale(interpreter->tensor(8), 0.25f));
  ASSERT_NO_FATAL_FAILURE(SetQuantizationScale(interpreter->tensor(9), 0.5f));
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_EQ(ir_model->tensor(op->inputs[0])->desc.GetDataType(),
            ::ml_drift::DataType::kInt8);
  EXPECT_EQ(ir_model->tensor(op->inputs[3])->desc.GetDataType(),
            ::ml_drift::DataType::kInt8);
  EXPECT_EQ(ir_model->tensor(op->inputs[4])->desc.GetDataType(),
            ::ml_drift::DataType::kInt8);
  EXPECT_EQ(ir_model->tensor(op->inputs[1])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, Hkv, W, D / 2));
  EXPECT_EQ(ir_model->tensor(op->inputs[2])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, Hkv, D / 2, W));
  EXPECT_EQ(ir_model->tensor(op->outputs[1])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, Hkv, W, D / 2));
  EXPECT_EQ(ir_model->tensor(op->outputs[2])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, Hkv, D / 2, W));

  const auto* attr =
      std::any_cast<::litert::ml_drift::FusedSdpaCacheUpdateAttributes>(
          &op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->kv_quant_bits, 4);
  ASSERT_TRUE(attr->scale_q.has_value());
  EXPECT_FLOAT_EQ(*attr->scale_q, 0.015625f);
  ASSERT_TRUE(attr->scale_key_new.has_value());
  EXPECT_FLOAT_EQ(*attr->scale_key_new, 0.0625f);
  ASSERT_TRUE(attr->scale_value_new.has_value());
  EXPECT_FLOAT_EQ(*attr->scale_value_new, 0.125f);
}

TEST_F(ConvertFusedSdpaCacheUpdateTest, RejectsInvalidQuantizedConfigurations) {
  // INT4 requires head_dim divisible by 8 (D = 12 fails).
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 12});
    builder.AddInput(kTfLiteInt4, {1, 2, 16, 12});
    builder.AddInput(kTfLiteInt4, {1, 2, 12, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 12});
    builder.AddInput(kTfLiteFloat32, {1, 2, 12, 4});
    builder.AddInput(kTfLiteBool, {1, 1, 4, 20});
    builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});
    builder.AddOutput(kTfLiteFloat32, {1, 2, 4, 12});
    builder.SetParameters(CreateFusedSdpaCacheUpdateParams());
    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
    EXPECT_EQ(GetIrModel(delegate_), nullptr);
  }
  // Non-zero zero_point on query is rejected.
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteInt8, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 16, 4});
    builder.AddInput(kTfLiteBool, {1, 1, 4, 20});
    builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});
    builder.AddOutput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.SetParameters(CreateFusedSdpaCacheUpdateParams());
    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    ASSERT_NO_FATAL_FAILURE(
        SetQuantizationZeroPoint(interpreter->tensor(0), 3));
    const auto* pair = interpreter->node_and_registration(0);
    const CustomIrOpParser parser = GetFusedSdpaCacheUpdateParser();
    EXPECT_EQ(parser
                  .is_supported(interpreter->primary_subgraph().context(),
                                &pair->first, &pair->second)
                  .code(),
              absl::StatusCode::kUnimplemented);
  }
  // Mismatched K and V cache types (INT8 vs FLOAT32) are rejected.
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 16, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 16, 4});
    builder.AddInput(kTfLiteBool, {1, 1, 4, 20});
    builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});
    builder.AddOutput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.SetParameters(CreateFusedSdpaCacheUpdateParams());
    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    const auto* pair = interpreter->node_and_registration(0);
    const CustomIrOpParser parser = GetFusedSdpaCacheUpdateParser();
    EXPECT_EQ(parser
                  .is_supported(interpreter->primary_subgraph().context(),
                                &pair->first, &pair->second)
                  .code(),
              absl::StatusCode::kUnimplemented);
  }
  // Float KV caches paired with INT8 key_new/value_new slices are rejected.
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 16, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 16, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 4});
    builder.AddInput(kTfLiteBool, {1, 1, 4, 20});
    builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});
    builder.AddOutput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.SetParameters(CreateFusedSdpaCacheUpdateParams());
    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    const auto* pair = interpreter->node_and_registration(0);
    const CustomIrOpParser parser = GetFusedSdpaCacheUpdateParser();
    EXPECT_EQ(parser
                  .is_supported(interpreter->primary_subgraph().context(),
                                &pair->first, &pair->second)
                  .code(),
              absl::StatusCode::kUnimplemented);
  }
  // Mismatched output cache scale vs input cache scale is rejected.
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddInput(kTfLiteFloat32, {1, 2, 16, 4});
    builder.AddInput(kTfLiteBool, {1, 1, 4, 20});
    builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});
    builder.AddOutput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddOutput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddOutput(kTfLiteInt8, {1, 2, 16, 16});
    builder.SetParameters(CreateFusedSdpaCacheUpdateParams());
    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    ASSERT_NO_FATAL_FAILURE(
        SetQuantizationScale(interpreter->tensor(1), 0.125f));
    ASSERT_NO_FATAL_FAILURE(
        SetQuantizationScale(interpreter->tensor(2), 0.125f));
    ASSERT_NO_FATAL_FAILURE(
        SetQuantizationScale(interpreter->tensor(8), 0.25f));
    ASSERT_NO_FATAL_FAILURE(
        SetQuantizationScale(interpreter->tensor(9), 0.125f));
    const auto* pair = interpreter->node_and_registration(0);
    const CustomIrOpParser parser = GetFusedSdpaCacheUpdateParser();
    EXPECT_EQ(parser
                  .is_supported(interpreter->primary_subgraph().context(),
                                &pair->first, &pair->second)
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  // Non-zero zero_point on key_new/value_new slices is rejected.
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 4});
    builder.AddInput(kTfLiteBool, {1, 1, 4, 20});
    builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});
    builder.AddOutput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.SetParameters(CreateFusedSdpaCacheUpdateParams());
    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    ASSERT_NO_FATAL_FAILURE(
        SetQuantizationZeroPoint(interpreter->tensor(3), 5));
    const auto* pair = interpreter->node_and_registration(0);
    const CustomIrOpParser parser = GetFusedSdpaCacheUpdateParser();
    EXPECT_EQ(parser
                  .is_supported(interpreter->primary_subgraph().context(),
                                &pair->first, &pair->second)
                  .code(),
              absl::StatusCode::kUnimplemented);
  }
  // Symmetric INT4 key_new/value_new slices are accepted for INT4 KV caches and
  // rejected for INT8 KV caches.
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt4, {1, 2, 16, 16});
    builder.AddInput(kTfLiteInt4, {1, 2, 16, 16});
    builder.AddInput(kTfLiteInt4, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt4, {1, 2, 16, 4});
    builder.AddInput(kTfLiteBool, {1, 1, 4, 20});
    builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});
    builder.AddOutput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.SetParameters(CreateFusedSdpaCacheUpdateParams());
    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    for (int idx : {1, 2, 3, 4}) {
      ASSERT_NO_FATAL_FAILURE(
          SetQuantizationScale(interpreter->tensor(idx), 0.25f));
    }
    const auto* pair = interpreter->node_and_registration(0);
    const CustomIrOpParser parser = GetFusedSdpaCacheUpdateParser();
    EXPECT_TRUE(parser
                    .is_supported(interpreter->primary_subgraph().context(),
                                  &pair->first, &pair->second)
                    .ok());
  }
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddInput(kTfLiteInt8, {1, 2, 16, 16});
    builder.AddInput(kTfLiteInt4, {1, 2, 4, 16});
    builder.AddInput(kTfLiteInt4, {1, 2, 16, 4});
    builder.AddInput(kTfLiteBool, {1, 1, 4, 20});
    builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});
    builder.AddOutput(kTfLiteFloat32, {1, 2, 4, 16});
    builder.SetParameters(CreateFusedSdpaCacheUpdateParams());
    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    for (int idx : {1, 2, 3, 4}) {
      ASSERT_NO_FATAL_FAILURE(
          SetQuantizationScale(interpreter->tensor(idx), 0.25f));
    }
    const auto* pair = interpreter->node_and_registration(0);
    const CustomIrOpParser parser = GetFusedSdpaCacheUpdateParser();
    EXPECT_EQ(parser
                  .is_supported(interpreter->primary_subgraph().context(),
                                &pair->first, &pair->second)
                  .code(),
              absl::StatusCode::kUnimplemented);
  }
}

}  // namespace
}  // namespace litert::ml_drift::ir
