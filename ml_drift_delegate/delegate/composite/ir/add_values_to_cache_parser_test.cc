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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
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
    std::optional<float> scale_v = std::nullopt,
    std::optional<bool> is_ring_buffer = std::nullopt,
    const std::vector<std::pair<std::string, int>>& layout = {}) {
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
    if (is_ring_buffer.has_value()) {
      fbb.Bool("is_ring_buffer", *is_ring_buffer);
    }
    for (const auto& [name, value] : layout) {
      fbb.Int(name.c_str(), value);
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

TEST_F(ConvertAddValuesToCacheTest, RingBuffer) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 64});   // src_k
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 64});   // src_v
  builder.AddInput(kTfLiteInt32, {2});               // params
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 64});  // cache_k
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 64});  // cache_v

  TfLiteStablehloCompositeParams* params = CreateAddValuesToCacheParams(
      2, 128, 64, /*scale_k=*/std::nullopt, /*scale_v=*/std::nullopt,
      /*is_ring_buffer=*/true);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("add_values_to_cache"));

  const auto* attr =
      std::any_cast<::litert::ml_drift::AddValuesToCacheAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_TRUE(attr->is_ring_buffer);
  // Absent layout attributes keep the historical layouts.
  EXPECT_EQ(attr->k_cache_ts_idx, 2);
  EXPECT_EQ(attr->v_cache_ts_idx, 3);
  EXPECT_EQ(attr->k_update_ts_idx, 2);
  EXPECT_EQ(attr->v_update_ts_idx, 2);
}

TEST_F(ConvertAddValuesToCacheTest, TransposedUpdates) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 64, 1});   // src_k, [D, T]
  builder.AddInput(kTfLiteFloat32, {1, 1, 64, 1});   // src_v, [D, T]
  builder.AddInput(kTfLiteInt32, {2});               // params
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 64});  // cache_k
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 64});  // cache_v

  builder.SetParameters(CreateAddValuesToCacheParams(
      2, 128, 64, std::nullopt, std::nullopt, /*is_ring_buffer=*/true,
      {{"k_cache_ts_idx", 2},
       {"v_cache_ts_idx", 3},
       {"k_update_ts_idx", 3},
       {"v_update_ts_idx", 3}}));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto* attr =
      std::any_cast<::litert::ml_drift::AddValuesToCacheAttributes>(
          &ir_model->ops()[0]->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->k_update_ts_idx, 3);
  EXPECT_EQ(attr->v_update_ts_idx, 3);
}

TEST_F(ConvertAddValuesToCacheTest, UnsupportedLayoutIsNotDelegated) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 64});   // src_k
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 64});   // src_v
  builder.AddInput(kTfLiteInt32, {2});               // params
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 64});  // cache_k
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 64});  // cache_v

  // A [B, H, D, S] key cache is not implemented; the op must stay on the CPU
  // decomposition rather than be written with the wrong layout.
  builder.SetParameters(CreateAddValuesToCacheParams(
      2, 128, 64, std::nullopt, std::nullopt, /*is_ring_buffer=*/true,
      {{"k_cache_ts_idx", 3}}));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  if (ir_model != nullptr) {
    EXPECT_THAT(ir_model->ops(), SizeIs(0));
  }
}

void SetQuantizationScale(TfLiteTensor* tensor, float scale,
                          int32_t zero_point = 0) {
  ASSERT_EQ(tensor->quantization.type, kTfLiteAffineQuantization);
  auto* quant =
      static_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  ASSERT_NE(quant, nullptr);
  ASSERT_EQ(quant->scale->size, 1);
  ASSERT_EQ(quant->zero_point->size, 1);
  quant->scale->data[0] = scale;
  quant->zero_point->data[0] = zero_point;
  tensor->params.scale = scale;
  tensor->params.zero_point = zero_point;
}

TEST_F(ConvertAddValuesToCacheTest, Int4CachePacksHalfShapeAndSetsDtype) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});   // src_k
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});   // src_v
  builder.AddInput(kTfLiteInt32, {4});               // params
  builder.AddOutput(kTfLiteInt4, {1, 2, 128, 64});   // cache_k [B, K, S, H]
  builder.AddOutput(kTfLiteInt4, {1, 2, 64, 128});   // cache_v [B, K, H, S]

  builder.SetParameters(CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  const auto* attr =
      std::any_cast<::litert::ml_drift::AddValuesToCacheAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->kv_cache_dtype, ::ml_drift::DataType::kInt4);
  EXPECT_EQ(attr->scale_k, 0.25f);
  EXPECT_EQ(attr->scale_v, 0.5f);

  const auto* out_k = ir_model->tensor(op->outputs[0]);
  const auto* out_v = ir_model->tensor(op->outputs[1]);
  ASSERT_NE(out_k, nullptr);
  ASSERT_NE(out_v, nullptr);
  EXPECT_EQ(out_k->desc.GetDataType(), ::ml_drift::DataType::kUint8);
  EXPECT_EQ(out_v->desc.GetDataType(), ::ml_drift::DataType::kUint8);
  EXPECT_EQ(out_k->desc.GetBHWCShape(), ::ml_drift::BHWC(1, 2, 128, 32));
  EXPECT_EQ(out_v->desc.GetBHWCShape(), ::ml_drift::BHWC(1, 2, 32, 128));
  EXPECT_FALSE(out_k->quant_params.has_value());
  EXPECT_FALSE(out_v->quant_params.has_value());
}

TEST_F(ConvertAddValuesToCacheTest, Int4CacheReadsScaleFromTensorQuantParams) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteInt32, {4});
  builder.AddOutput(kTfLiteInt4, {1, 2, 128, 64});
  builder.AddOutput(kTfLiteInt4, {1, 2, 64, 128});

  builder.SetParameters(CreateAddValuesToCacheParams(2, 128, 64));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  SetQuantizationScale(interpreter->tensor(interpreter->outputs()[0]), 0.125f);
  SetQuantizationScale(interpreter->tensor(interpreter->outputs()[1]), 0.375f);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto* attr =
      std::any_cast<::litert::ml_drift::AddValuesToCacheAttributes>(
          &ir_model->ops()[0]->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->kv_cache_dtype, ::ml_drift::DataType::kInt4);
  EXPECT_EQ(attr->scale_k, 0.125f);
  EXPECT_EQ(attr->scale_v, 0.375f);
}

TEST_F(ConvertAddValuesToCacheTest, Int8CacheKeepsFullByteShape) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteInt32, {4});
  builder.AddOutput(kTfLiteInt8, {1, 2, 128, 64});
  builder.AddOutput(kTfLiteInt8, {1, 2, 64, 128});

  builder.SetParameters(CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  const auto* attr =
      std::any_cast<::litert::ml_drift::AddValuesToCacheAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->kv_cache_dtype, ::ml_drift::DataType::kInt8);
  EXPECT_EQ(ir_model->tensor(op->outputs[0])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 2, 128, 64));
  EXPECT_EQ(ir_model->tensor(op->outputs[1])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 2, 64, 128));
}

TEST_F(ConvertAddValuesToCacheTest, Int4CacheUnalignedHeadSizeIsNotDelegated) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 12});
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 12});
  builder.AddInput(kTfLiteInt32, {4});
  builder.AddOutput(kTfLiteInt4, {1, 2, 128, 12});
  builder.AddOutput(kTfLiteInt4, {1, 2, 12, 128});

  builder.SetParameters(CreateAddValuesToCacheParams(2, 128, 12, 0.25f, 0.5f));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  if (ir_model != nullptr) {
    EXPECT_THAT(ir_model->ops(), SizeIs(0));
  }
}

TEST_F(ConvertAddValuesToCacheTest, Int4CacheMissingScalesIsNotDelegated) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteInt32, {4});
  builder.AddOutput(kTfLiteInt4, {1, 2, 128, 64});
  builder.AddOutput(kTfLiteInt4, {1, 2, 64, 128});

  builder.SetParameters(CreateAddValuesToCacheParams(2, 128, 64));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  // SingleOpInterpreterBuilder defaults affine scale to 1.0f; clear it to 0.0f
  // to verify non-positive / missing scales are rejected.
  SetQuantizationScale(interpreter->tensor(interpreter->outputs()[0]), 0.0f);
  SetQuantizationScale(interpreter->tensor(interpreter->outputs()[1]), 0.0f);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  if (ir_model != nullptr) {
    EXPECT_THAT(ir_model->ops(), SizeIs(0));
  }
}

TEST_F(ConvertAddValuesToCacheTest,
       Int4CacheAsymmetricZeroPointIsNotDelegated) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteInt32, {4});
  builder.AddOutput(kTfLiteInt4, {1, 2, 128, 64});
  builder.AddOutput(kTfLiteInt4, {1, 2, 64, 128});

  builder.SetParameters(CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  SetQuantizationScale(interpreter->tensor(interpreter->outputs()[0]), 0.25f,
                       /*zero_point=*/1);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  if (ir_model != nullptr) {
    EXPECT_THAT(ir_model->ops(), SizeIs(0));
  }
}

TEST_F(ConvertAddValuesToCacheTest,
       Int4CacheMismatchedAttrAndTensorDimsIsNotDelegated) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
  builder.AddInput(kTfLiteInt32, {4});
  builder.AddOutput(kTfLiteInt4, {1, 2, 64, 64});
  builder.AddOutput(kTfLiteInt4, {1, 2, 64, 64});

  // attr.cache_size (128) exceeds out_k/out_v time dim (64).
  builder.SetParameters(CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  if (ir_model != nullptr) {
    EXPECT_THAT(ir_model->ops(), SizeIs(0));
  }
}

TEST_F(ConvertAddValuesToCacheTest, Int4CacheNonFiniteScaleIsNotDelegated) {
  const float bad_scales[] = {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity()};
  for (const float bad_scale : bad_scales) {
    SCOPED_TRACE(bad_scale);
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
    builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
    builder.AddInput(kTfLiteInt32, {4});
    builder.AddOutput(kTfLiteInt4, {1, 2, 128, 64});
    builder.AddOutput(kTfLiteInt4, {1, 2, 64, 128});

    builder.SetParameters(
        CreateAddValuesToCacheParams(2, 128, 64, bad_scale, 0.5f));

    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    if (ir_model != nullptr) {
      EXPECT_THAT(ir_model->ops(), SizeIs(0));
    }
  }
}

// Quantized K/V updates must be produced in the graph (e.g., by QUANTIZE);
// quantized graph inputs would need host-side dequantization.
TEST_F(ConvertAddValuesToCacheTest,
       QuantizedCacheQuantizedGraphInputSourcesAreNotDelegated) {
  const TfLiteType src_types[] = {kTfLiteInt8, kTfLiteInt4};
  for (const TfLiteType src_type : src_types) {
    SCOPED_TRACE(src_type);
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(src_type, {1, 2, 1, 64});
    builder.AddInput(src_type, {1, 2, 1, 64});
    builder.AddInput(kTfLiteInt32, {4});
    builder.AddOutput(kTfLiteInt4, {1, 2, 128, 64});
    builder.AddOutput(kTfLiteInt4, {1, 2, 64, 128});

    builder.SetParameters(
        CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f));

    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    SetQuantizationScale(interpreter->tensor(interpreter->inputs()[0]), 0.25f);
    SetQuantizationScale(interpreter->tensor(interpreter->inputs()[1]), 0.5f);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    if (ir_model != nullptr) {
      EXPECT_THAT(ir_model->ops(), SizeIs(0));
    }
  }
}

// The kernel reads `head_size` channels per cache row, so multiple heads packed
// in the update channels are rejected.
TEST_F(ConvertAddValuesToCacheTest,
       QuantizedCacheMultiHeadUpdateIsNotDelegated) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 128});
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 128});
  builder.AddInput(kTfLiteInt32, {4});
  builder.AddOutput(kTfLiteInt8, {1, 2, 128, 64});
  builder.AddOutput(kTfLiteInt8, {1, 2, 64, 128});

  builder.SetParameters(CreateAddValuesToCacheParams(1, 128, 64, 0.25f, 0.5f));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  if (ir_model != nullptr) {
    EXPECT_THAT(ir_model->ops(), SizeIs(0));
  }
}

// UINT8 caches hold INT8 values offset by +128, so they need zero point 128.
TEST_F(ConvertAddValuesToCacheTest, UInt8CacheRequiresZeroPoint128) {
  for (const int32_t zero_point : {0, 128}) {
    SCOPED_TRACE(zero_point);
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
    builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
    builder.AddInput(kTfLiteInt32, {4});
    builder.AddOutput(kTfLiteUInt8, {1, 2, 128, 64});
    builder.AddOutput(kTfLiteUInt8, {1, 2, 64, 128});

    builder.SetParameters(CreateAddValuesToCacheParams(2, 128, 64));

    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    SetQuantizationScale(interpreter->tensor(interpreter->outputs()[0]), 0.25f,
                         zero_point);
    SetQuantizationScale(interpreter->tensor(interpreter->outputs()[1]), 0.5f,
                         zero_point);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    if (zero_point == 128) {
      ASSERT_NE(ir_model, nullptr);
      ASSERT_THAT(ir_model->ops(), SizeIs(1));
      const auto* attr =
          std::any_cast<::litert::ml_drift::AddValuesToCacheAttributes>(
              &ir_model->ops()[0]->attr);
      ASSERT_NE(attr, nullptr);
      EXPECT_EQ(attr->kv_cache_dtype, ::ml_drift::DataType::kUint8);
    } else if (ir_model != nullptr) {
      EXPECT_THAT(ir_model->ops(), SizeIs(0));
    }
  }
}

TEST_F(ConvertAddValuesToCacheTest,
       QuantizedCacheNonFloatInputIsNotDelegated) {
  const TfLiteType unsupported_src_types[] = {kTfLiteInt32, kTfLiteBFloat16};
  const TfLiteType cache_types[] = {kTfLiteInt4, kTfLiteInt8};
  for (const TfLiteType bad_src_type : unsupported_src_types) {
    for (const TfLiteType cache_type : cache_types) {
      SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
      builder.AddInput(bad_src_type, {1, 2, 1, 64});
      builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
      builder.AddInput(kTfLiteInt32, {4});
      builder.AddOutput(cache_type, {1, 2, 128, 64});
      builder.AddOutput(cache_type, {1, 2, 64, 128});

      builder.SetParameters(
          CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f));

      auto interpreter = builder.Build();
      ASSERT_NE(interpreter, nullptr);
      ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

      const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
      if (ir_model != nullptr) {
        EXPECT_THAT(ir_model->ops(), SizeIs(0));
      }
    }
  }
}

TEST_F(ConvertAddValuesToCacheTest,
       QuantizedCacheMismatchedKvHeadsOrInvalidInt8ScaleIsNotDelegated) {
  // Mismatched K vs V head counts (dims[1]: 2 vs 4).
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
    builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
    builder.AddInput(kTfLiteInt32, {4});
    builder.AddOutput(kTfLiteInt4, {1, 2, 128, 64});
    builder.AddOutput(kTfLiteInt4, {1, 4, 64, 128});

    builder.SetParameters(
        CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f));

    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    if (ir_model != nullptr) {
      EXPECT_THAT(ir_model->ops(), SizeIs(0));
    }
  }

  // INT8 cache with non-finite scale_k.
  {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
    builder.AddInput(kTfLiteFloat32, {1, 2, 1, 64});
    builder.AddInput(kTfLiteInt32, {4});
    builder.AddOutput(kTfLiteInt8, {1, 2, 128, 64});
    builder.AddOutput(kTfLiteInt8, {1, 2, 64, 128});

    builder.SetParameters(CreateAddValuesToCacheParams(
        2, 128, 64, std::numeric_limits<float>::quiet_NaN(), 0.5f));

    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    if (ir_model != nullptr) {
      EXPECT_THAT(ir_model->ops(), SizeIs(0));
    }
  }
}

TfLiteQuantization MakePerTensorQuantization(float scale,
                                             int32_t zero_point = 0) {
  auto* quant_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  quant_params->scale = TfLiteFloatArrayCreate(1);
  quant_params->scale->data[0] = scale;
  quant_params->zero_point = TfLiteIntArrayCreate(1);
  quant_params->zero_point->data[0] = zero_point;
  quant_params->quantized_dimension = 0;
  return {kTfLiteAffineQuantization, quant_params};
}

TEST_F(ConvertAddValuesToCacheTest,
       QuantizedCacheWithValidQuantizeAndLayoutChainIsDelegated) {
  auto interpreter = std::make_unique<::tflite::Interpreter>();
  ASSERT_EQ(interpreter->AddTensors(10), kTfLiteOk);

  const TfLiteQuantization no_quant = {kTfLiteNoQuantization, nullptr};
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "in_k", {1, 2, 1, 64}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                1, kTfLiteFloat32, "in_v", {1, 2, 64, 1}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(2, kTfLiteInt32, "params",
                                                      {4}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                3, kTfLiteInt4, "q_k", {1, 2, 1, 64},
                MakePerTensorQuantization(0.25f)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                4, kTfLiteInt4, "reshaped_k", {1, 2, 1, 64},
                MakePerTensorQuantization(0.25f)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                5, kTfLiteInt4, "q_v", {1, 2, 64, 1},
                MakePerTensorQuantization(0.5f)),
            kTfLiteOk);
  const int32_t perm_data[4] = {0, 1, 3, 2};
  ASSERT_EQ(interpreter->SetTensorParametersReadOnly(
                6, kTfLiteInt32, "perm", {4}, no_quant,
                reinterpret_cast<const char*>(perm_data), sizeof(perm_data)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                7, kTfLiteInt4, "transposed_v", {1, 2, 1, 64},
                MakePerTensorQuantization(0.5f)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                8, kTfLiteInt4, "cache_k", {1, 2, 128, 64},
                MakePerTensorQuantization(0.25f)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                9, kTfLiteInt4, "cache_v", {1, 2, 64, 128},
                MakePerTensorQuantization(0.5f)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetInputs({0, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter->SetOutputs({8, 9}), kTfLiteOk);

  TfLiteRegistration quant_reg = {};
  quant_reg.builtin_code = kTfLiteBuiltinQuantize;
  quant_reg.version = 4;
  TfLiteRegistration reshape_reg = {};
  reshape_reg.builtin_code = kTfLiteBuiltinReshape;
  reshape_reg.version = 1;
  TfLiteRegistration transpose_reg = {};
  transpose_reg.builtin_code = kTfLiteBuiltinTranspose;
  transpose_reg.version = 1;
  TfLiteRegistration composite_reg = {};
  composite_reg.builtin_code = kTfLiteBuiltinStablehloComposite;
  composite_reg.version = 1;

  int node_idx = -1;
  ASSERT_EQ(interpreter->AddNodeWithParameters({0}, {3}, nullptr, 0, nullptr,
                                               &quant_reg, &node_idx),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AddNodeWithParameters({3}, {4}, nullptr, 0, nullptr,
                                               &reshape_reg, &node_idx),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AddNodeWithParameters({1}, {5}, nullptr, 0, nullptr,
                                               &quant_reg, &node_idx),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AddNodeWithParameters({5, 6}, {7}, nullptr, 0, nullptr,
                                               &transpose_reg, &node_idx),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AddNodeWithParameters(
                {4, 7, 2}, {8, 9}, nullptr, 0,
                CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f),
                &composite_reg, &node_idx),
            kTfLiteOk);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_NE(ir_model, nullptr);
  bool found_cache_update = false;
  for (const auto& op : ir_model->ops()) {
    if (op != nullptr && op->name == "add_values_to_cache") {
      found_cache_update = true;
    }
  }
  EXPECT_TRUE(found_cache_update);
}

TEST_F(ConvertAddValuesToCacheTest,
       QuantizedCacheWithNonDelegatedProducerChainIsNotDelegated) {
  // Non-constant perm on TRANSPOSE makes TRANSPOSE fall back to CPU;
  // CheckQuantizedInGraph must reject odml.cache_update as well rather than
  // splitting the partition on a kTfLiteInt4 tensor.
  {
    auto interpreter = std::make_unique<::tflite::Interpreter>();
    ASSERT_EQ(interpreter->AddTensors(9), kTfLiteOk);

    const TfLiteQuantization no_quant = {kTfLiteNoQuantization, nullptr};
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  0, kTfLiteFloat32, "in_k", {1, 2, 1, 64}, no_quant),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  1, kTfLiteFloat32, "in_v", {1, 2, 64, 1}, no_quant),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  2, kTfLiteInt32, "params", {4}, no_quant),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  3, kTfLiteInt4, "q_k", {1, 2, 1, 64},
                  MakePerTensorQuantization(0.25f)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  4, kTfLiteInt4, "q_v", {1, 2, 64, 1},
                  MakePerTensorQuantization(0.5f)),
              kTfLiteOk);
    // Dynamic (ReadWrite) perm tensor -> rejected by IsTransposeSupported.
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  5, kTfLiteInt32, "dyn_perm", {4}, no_quant),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  6, kTfLiteInt4, "transposed_v", {1, 2, 1, 64},
                  MakePerTensorQuantization(0.5f)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  7, kTfLiteInt4, "cache_k", {1, 2, 128, 64},
                  MakePerTensorQuantization(0.25f)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  8, kTfLiteInt4, "cache_v", {1, 2, 64, 128},
                  MakePerTensorQuantization(0.5f)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetInputs({0, 1, 2, 5}), kTfLiteOk);
    ASSERT_EQ(interpreter->SetOutputs({7, 8}), kTfLiteOk);

    TfLiteRegistration quant_reg = {};
    quant_reg.builtin_code = kTfLiteBuiltinQuantize;
    quant_reg.version = 4;
    TfLiteRegistration transpose_reg = {};
    transpose_reg.builtin_code = kTfLiteBuiltinTranspose;
    transpose_reg.version = 1;
    TfLiteRegistration composite_reg = {};
    composite_reg.builtin_code = kTfLiteBuiltinStablehloComposite;
    composite_reg.version = 1;

    int node_idx = -1;
    ASSERT_EQ(interpreter->AddNodeWithParameters({0}, {3}, nullptr, 0, nullptr,
                                                 &quant_reg, &node_idx),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AddNodeWithParameters({1}, {4}, nullptr, 0, nullptr,
                                                 &quant_reg, &node_idx),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AddNodeWithParameters({4, 5}, {6}, nullptr, 0,
                                                 nullptr, &transpose_reg,
                                                 &node_idx),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AddNodeWithParameters(
                  {3, 6, 2}, {7, 8}, nullptr, 0,
                  CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f),
                  &composite_reg, &node_idx),
              kTfLiteOk);

    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    if (ir_model != nullptr) {
      for (const auto& op : ir_model->ops()) {
        if (op != nullptr) {
          EXPECT_NE(op->name, "add_values_to_cache");
        }
      }
    }
  }

  // Unquantized intermediate INT4 tensor between two RESHAPEs must also cause
  // CheckQuantizedInGraph to reject odml.cache_update.
  {
    auto interpreter = std::make_unique<::tflite::Interpreter>();
    ASSERT_EQ(interpreter->AddTensors(9), kTfLiteOk);

    const TfLiteQuantization no_quant = {kTfLiteNoQuantization, nullptr};
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  0, kTfLiteFloat32, "in_k", {1, 2, 1, 64}, no_quant),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  1, kTfLiteFloat32, "in_v", {1, 2, 1, 64}, no_quant),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  2, kTfLiteInt32, "params", {4}, no_quant),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  3, kTfLiteInt4, "q_k", {1, 2, 1, 64},
                  MakePerTensorQuantization(0.25f)),
              kTfLiteOk);
    // Unquantized intermediate INT4 tensor!
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  4, kTfLiteInt4, "unquant_mid_k", {1, 2, 1, 64}, no_quant),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  5, kTfLiteInt4, "restored_k", {1, 2, 1, 64},
                  MakePerTensorQuantization(0.25f)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  6, kTfLiteInt4, "q_v", {1, 2, 1, 64},
                  MakePerTensorQuantization(0.5f)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  7, kTfLiteInt4, "cache_k", {1, 2, 128, 64},
                  MakePerTensorQuantization(0.25f)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                  8, kTfLiteInt4, "cache_v", {1, 2, 64, 128},
                  MakePerTensorQuantization(0.5f)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->SetInputs({0, 1, 2}), kTfLiteOk);
    ASSERT_EQ(interpreter->SetOutputs({7, 8}), kTfLiteOk);

    TfLiteRegistration quant_reg = {};
    quant_reg.builtin_code = kTfLiteBuiltinQuantize;
    quant_reg.version = 4;
    TfLiteRegistration reshape_reg = {};
    reshape_reg.builtin_code = kTfLiteBuiltinReshape;
    reshape_reg.version = 1;
    TfLiteRegistration composite_reg = {};
    composite_reg.builtin_code = kTfLiteBuiltinStablehloComposite;
    composite_reg.version = 1;

    int node_idx = -1;
    ASSERT_EQ(interpreter->AddNodeWithParameters({0}, {3}, nullptr, 0, nullptr,
                                                 &quant_reg, &node_idx),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AddNodeWithParameters({3}, {4}, nullptr, 0, nullptr,
                                                 &reshape_reg, &node_idx),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AddNodeWithParameters({4}, {5}, nullptr, 0, nullptr,
                                                 &reshape_reg, &node_idx),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AddNodeWithParameters({1}, {6}, nullptr, 0, nullptr,
                                                 &quant_reg, &node_idx),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AddNodeWithParameters(
                  {5, 6, 2}, {7, 8}, nullptr, 0,
                  CreateAddValuesToCacheParams(2, 128, 64, 0.25f, 0.5f),
                  &composite_reg, &node_idx),
              kTfLiteOk);

    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    if (ir_model != nullptr) {
      for (const auto& op : ir_model->ops()) {
        if (op != nullptr) {
          EXPECT_NE(op->name, "add_values_to_cache");
        }
      }
    }
  }
}

}  // namespace
}  // namespace litert::ml_drift::ir
