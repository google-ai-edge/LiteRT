// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
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
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/ir/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/ir/qsa_indexer_update_parser.h"
#include "ml_drift_delegate/delegate/composite/ir/sparse_sdpa_transposed_parser.h"
#include "ml_drift_delegate/delegate/composite/qsa_indexer_update_parser.h"
#include "ml_drift_delegate/delegate/composite/sparse_sdpa_transposed_parser.h"
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

std::vector<uint8_t> MakeConstFloatBytes(int count, float val = 1.0f) {
  std::vector<float> vals(count, val);
  return std::vector<uint8_t>(
      reinterpret_cast<const uint8_t*>(vals.data()),
      reinterpret_cast<const uint8_t*>(vals.data() + vals.size()));
}

TfLiteStablehloCompositeParams* CreateRawCompositeParams(
    const char* name, const std::vector<uint8_t>& buffer) {
  const size_t total_size =
      sizeof(TfLiteStablehloCompositeParams) + buffer.size();
  void* block = calloc(1, total_size);
  auto* params = reinterpret_cast<TfLiteStablehloCompositeParams*>(block);
  params->name = name;
  if (!buffer.empty()) {
    uint8_t* attr_data = reinterpret_cast<uint8_t*>(params + 1);
    params->attributes = attr_data;
    params->attributes_size = buffer.size();
    memcpy(attr_data, buffer.data(), buffer.size());
  }
  return params;
}

TfLiteStablehloCompositeParams* CreateQsaIndexerUpdateParams(
    int compress_ratio = 4, int block_topk = 2) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("compress_ratio", compress_ratio);
    fbb.Int("block_topk", block_topk);
    fbb.Float("rms_norm_eps", 1e-6f);
    fbb.Float("scale", 0.088388f);
  });
  fbb.Finish();
  return CreateRawCompositeParams(kQsaIndexerUpdateCompositeName,
                                  fbb.GetBuffer());
}

TfLiteStablehloCompositeParams* CreateSparseSdpaTransposedParams(
    int compress_ratio = 4, int block_topk = 2) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("compress_ratio", compress_ratio);
    fbb.Int("block_topk", block_topk);
    fbb.Float("scale", 0.088388f);
    fbb.Int("k_ts_idx", 2);
    fbb.Int("v_ts_idx", 3);
  });
  fbb.Finish();
  return CreateRawCompositeParams(kSparseSdpaTransposedCompositeName,
                                  fbb.GetBuffer());
}

TfLiteStablehloCompositeParams* CreateCacheUpdateParams(int num_kv_heads,
                                                        int cache_len,
                                                        int head_dim) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("kv_cache_batch_size", num_kv_heads);
    fbb.Int("cache_size", cache_len);
    fbb.Int("head_size", head_dim);
    fbb.Int("k_cache_ts_idx", 2);
    fbb.Int("v_cache_ts_idx", 3);
    fbb.Int("k_update_ts_idx", 2);
    fbb.Int("v_update_ts_idx", 2);
  });
  fbb.Finish();
  return CreateRawCompositeParams("odml.cache_update", fbb.GetBuffer());
}

void PopulateDefaultSparseSdpaBuilder(SingleOpInterpreterBuilder* builder,
                                      TfLiteStablehloCompositeParams* params) {
  builder->AddInput(kTfLiteFloat32, {1, 4, 1, 16});
  builder->AddInput(kTfLiteFloat32, {1, 2, 32, 16});
  builder->AddInput(kTfLiteFloat32, {1, 2, 16, 32});
  builder->AddInput(kTfLiteInt32, {1, 1, 2});
  builder->AddInput(kTfLiteBool, {1, 1, 1, 32});
  builder->AddOutput(kTfLiteFloat32, {1, 1, 4, 16});
  builder->SetParameters(params);
}

void PopulateDefaultQsaIndexerBuilder(
    SingleOpInterpreterBuilder* builder,
    const std::vector<int>& start_pos_shape = {1},
    bool const_ring_k = false,
    TfLiteType norm_dtype = kTfLiteFloat32,
    int norm_byte_elems = 16) {
  const int S = 1, H_idx = 4, D_idx = 16, R = 4, max_blocks = 8, cache_len = 32;
  const int rotary_dim = 8, block_topk = 2;
  builder->AddInput(kTfLiteFloat32, {1, S, H_idx, D_idx});
  builder->AddInput(kTfLiteFloat32, {1, S, D_idx});
  if (const_ring_k) {
    builder->AddConstInput(kTfLiteFloat32, {1, R, D_idx},
                           MakeConstFloatBytes(R * D_idx, 0.0f));
  } else {
    builder->AddInput(kTfLiteFloat32, {1, R, D_idx});
  }
  builder->AddInput(kTfLiteFloat32, {1, max_blocks, D_idx});
  if (norm_dtype == kTfLiteInt8) {
    builder->AddConstInput(kTfLiteInt8, {D_idx},
                           std::vector<uint8_t>(D_idx, 1));
  } else {
    builder->AddConstInput(norm_dtype, {D_idx},
                           MakeConstFloatBytes(norm_byte_elems, 1.0f));
  }
  builder->AddConstInput(kTfLiteFloat32, {1, max_blocks, rotary_dim},
                         MakeConstFloatBytes(max_blocks * rotary_dim, 1.0f));
  builder->AddConstInput(kTfLiteFloat32, {1, max_blocks, rotary_dim},
                         MakeConstFloatBytes(max_blocks * rotary_dim, 0.0f));
  builder->AddInput(kTfLiteInt32, start_pos_shape);
  builder->AddInput(kTfLiteBool, {1, 1, S, cache_len});
  builder->AddOutput(kTfLiteInt32, {1, S, block_topk});
  builder->AddOutput(kTfLiteFloat32, {1, R, D_idx});
  builder->AddOutput(kTfLiteFloat32, {1, max_blocks, D_idx});
  builder->SetParameters(CreateQsaIndexerUpdateParams(R, block_topk));
}

class ConvertQsaCompositesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CustomIrOpMap custom_parsers;
    custom_parsers["odml.cache_update"] = GetAddValuesToCacheParser();
    custom_parsers[kQsaIndexerUpdateCompositeName] =
        GetQsaIndexerUpdateParser();
    custom_parsers[kSparseSdpaTransposedCompositeName] =
        GetSparseSdpaTransposedParser();
    delegate_ = CreateStubDelegate(/*options=*/{}, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  void ExpectRejected(std::unique_ptr<::tflite::Interpreter> interpreter) {
    ASSERT_NE(interpreter, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    EXPECT_TRUE(ir_model == nullptr || ir_model->ops().empty());
  }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertQsaCompositesTest,
       QsaIndexerUpdateValidWithConstNormWeightAndRopeTables) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultQsaIndexerBuilder(&builder);

  std::unique_ptr<::tflite::Interpreter> interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  // 1 `qsa_indexer_update` op + 3 `const` ops for k_norm_weight/cos/sin.
  // `ring_k` and `block_k_pool` are omitted on GPU (in-place on outputs 1 & 2).
  ASSERT_THAT(ir_model->ops(), SizeIs(4));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("qsa_indexer_update"));
  EXPECT_THAT(op->inputs, SizeIs(7));
  EXPECT_THAT(op->outputs, SizeIs(3));
  const auto* attr =
      std::any_cast<::litert::ml_drift::QsaIndexerUpdateAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->compress_ratio, 4);
  EXPECT_EQ(attr->block_topk, 2);
}

TEST_F(ConvertQsaCompositesTest,
       QsaIndexerUpdateAcceptsZeroRankScalarStartPos) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultQsaIndexerBuilder(&builder, /*start_pos_shape=*/{});

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(4));
  EXPECT_THAT(ir_model->ops()[0]->inputs, SizeIs(7));
}

TEST_F(ConvertQsaCompositesTest, SparseSdpaTransposedValid) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultSparseSdpaBuilder(&builder,
                                   CreateSparseSdpaTransposedParams(4, 2));

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sparse_sdpa_transposed"));
  EXPECT_THAT(op->inputs, SizeIs(5));
  EXPECT_THAT(op->outputs, SizeIs(1));
  const auto* attr =
      std::any_cast<::litert::ml_drift::SparseSdpaTransposedAttributes>(
          &op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->compress_ratio, 4);
  EXPECT_EQ(attr->block_topk, 2);
  EXPECT_EQ(attr->k_ts_idx, 2);
  EXPECT_EQ(attr->v_ts_idx, 3);
  EXPECT_FALSE(attr->from_cache_update);
}

TEST_F(ConvertQsaCompositesTest,
       SparseSdpaTransposedSetsFromCacheUpdateWhenFedByCacheUpdate) {
  auto interpreter = std::make_unique<::tflite::Interpreter>();
  const int H_q = 4, H_kv = 2, S = 1, D = 16, cache_len = 32, block_topk = 2;
  int base_idx = 0;
  ASSERT_EQ(interpreter->AddTensors(9, &base_idx), kTfLiteOk);
  TfLiteQuantizationParams no_quant{};
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "k_up", {1, H_kv, S, D}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                1, kTfLiteFloat32, "v_up", {1, H_kv, S, D}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(2, kTfLiteInt32, "params",
                                                      {1, 1, 1, 4}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                3, kTfLiteFloat32, "kc", {1, H_kv, cache_len, D}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                4, kTfLiteFloat32, "vc", {1, H_kv, D, cache_len}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                5, kTfLiteFloat32, "q", {1, H_q, S, D}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                6, kTfLiteInt32, "topk", {1, S, block_topk}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                7, kTfLiteBool, "mask", {1, 1, S, cache_len}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetTensorParametersReadWrite(
                8, kTfLiteFloat32, "out", {1, S, H_q, D}, no_quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter->SetInputs({0, 1, 2, 5, 6, 7}), kTfLiteOk);
  ASSERT_EQ(interpreter->SetOutputs({8}), kTfLiteOk);

  TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinStablehloComposite;
  reg.version = 1;
  reg.invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; };
  int node0 = -1, node1 = -1;
  ASSERT_EQ(interpreter->AddNodeWithParameters(
                {0, 1, 2}, {3, 4}, nullptr, 0,
                CreateCacheUpdateParams(H_kv, cache_len, D), &reg, &node0),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AddNodeWithParameters(
                {5, 3, 4, 6, 7}, {8}, nullptr, 0,
                CreateSparseSdpaTransposedParams(4, block_topk), &reg, &node1),
            kTfLiteOk);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(2));
  const auto& sdpa_op = ir_model->ops()[1];
  EXPECT_THAT(sdpa_op->name, Eq("sparse_sdpa_transposed"));
  const auto* attr =
      std::any_cast<::litert::ml_drift::SparseSdpaTransposedAttributes>(
          &sdpa_op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_TRUE(attr->from_cache_update);
}

TEST_F(ConvertQsaCompositesTest, RejectsUnsupportedCompressRatio) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultSparseSdpaBuilder(
      &builder, CreateSparseSdpaTransposedParams(/*compress_ratio=*/8, 2));
  ExpectRejected(builder.Build());
}

TEST_F(ConvertQsaCompositesTest, RejectsTruncatedFlexBuffersAttributes) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultSparseSdpaBuilder(
      &builder,
      CreateRawCompositeParams(kSparseSdpaTransposedCompositeName, {4, 1}));
  ExpectRejected(builder.Build());
}

TEST_F(ConvertQsaCompositesTest, RejectsNonMapFlexBuffersAttributes) {
  flexbuffers::Builder fbb;
  fbb.Int(42);
  fbb.Finish();
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultSparseSdpaBuilder(
      &builder, CreateRawCompositeParams(kSparseSdpaTransposedCompositeName,
                                         fbb.GetBuffer()));
  ExpectRejected(builder.Build());
}

TEST_F(ConvertQsaCompositesTest, RejectsConstantTensorWithMismatchedByteSize) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultQsaIndexerBuilder(&builder, /*start_pos_shape=*/{1},
                                   /*const_ring_k=*/false,
                                   /*norm_dtype=*/kTfLiteFloat32,
                                   /*norm_byte_elems=*/18);
  ExpectRejected(builder.Build());
}

TEST_F(ConvertQsaCompositesTest,
       QsaIndexerUpdateAcceptsFloat16ConstantNormWeight) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  // 16 FP16 elements == 32 bytes == 8 float32 elements of byte payload.
  PopulateDefaultQsaIndexerBuilder(&builder, /*start_pos_shape=*/{1},
                                   /*const_ring_k=*/false,
                                   /*norm_dtype=*/kTfLiteFloat16,
                                   /*norm_byte_elems=*/8);
  std::unique_ptr<::tflite::Interpreter> interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(4));
  EXPECT_THAT(ir_model->ops()[0]->inputs, SizeIs(7));
}

TEST_F(ConvertQsaCompositesTest, SparseSdpaTransposedAcceptsConstantBoolMask) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 4, 1, 16});
  builder.AddInput(kTfLiteFloat32, {1, 2, 32, 16});
  builder.AddInput(kTfLiteFloat32, {1, 2, 16, 32});
  builder.AddInput(kTfLiteInt32, {1, 1, 2});
  builder.AddConstInput(kTfLiteBool, {1, 1, 1, 32},
                        std::vector<uint8_t>(32, 1));
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 16});
  builder.SetParameters(CreateSparseSdpaTransposedParams(4, 2));
  std::unique_ptr<::tflite::Interpreter> interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  // 1 `sparse_sdpa_transposed` op + 1 `const` op for the bool mask.
  ASSERT_THAT(ir_model->ops(), SizeIs(2));
}

TEST_F(ConvertQsaCompositesTest, RejectsUnsupportedConstantTensorDtype) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultQsaIndexerBuilder(&builder, /*start_pos_shape=*/{1},
                                   /*const_ring_k=*/false,
                                   /*norm_dtype=*/kTfLiteInt8);
  ExpectRejected(builder.Build());
}

TEST_F(ConvertQsaCompositesTest, RejectsConstantOnQsaIndexerRuntimeInput) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultQsaIndexerBuilder(&builder, /*start_pos_shape=*/{1},
                                   /*const_ring_k=*/true);
  ExpectRejected(builder.Build());
}

TEST_F(ConvertQsaCompositesTest, RejectsConstantOnSparseSdpaRuntimeInput) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddConstInput(kTfLiteFloat32, {1, 4, 1, 16},
                        MakeConstFloatBytes(4 * 16, 0.1f));
  builder.AddInput(kTfLiteFloat32, {1, 2, 32, 16});
  builder.AddInput(kTfLiteFloat32, {1, 2, 16, 32});
  builder.AddInput(kTfLiteInt32, {1, 1, 2});
  builder.AddInput(kTfLiteBool, {1, 1, 1, 32});
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 16});
  builder.SetParameters(CreateSparseSdpaTransposedParams(4, 2));
  ExpectRejected(builder.Build());
}

TEST_F(ConvertQsaCompositesTest, RejectsNonUnitStartPosShape) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultQsaIndexerBuilder(&builder, /*start_pos_shape=*/{2});
  ExpectRejected(builder.Build());
}

TEST_F(ConvertQsaCompositesTest, RejectsRankFiveStartPosShape) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultQsaIndexerBuilder(&builder,
                                   /*start_pos_shape=*/{1, 1, 1, 1, 1});
  ExpectRejected(builder.Build());
}

TEST_F(ConvertQsaCompositesTest, RejectsSparseConstantTensor) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultQsaIndexerBuilder(&builder);
  TfLiteSparsity dummy_sparsity{};
  std::unique_ptr<::tflite::Interpreter> interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  interpreter->tensor(4)->sparsity = &dummy_sparsity;
  const TfLiteStatus status = interpreter->ModifyGraphWithDelegate(delegate_);
  interpreter->tensor(4)->sparsity = nullptr;
  ASSERT_EQ(status, kTfLiteOk);
  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  EXPECT_TRUE(ir_model == nullptr || ir_model->ops().empty());
}

TEST_F(ConvertQsaCompositesTest, RejectsNullRuntimeTensorDims) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultQsaIndexerBuilder(&builder);
  std::unique_ptr<::tflite::Interpreter> interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  TfLiteIntArray* saved_dims = interpreter->tensor(0)->dims;
  interpreter->tensor(0)->dims = nullptr;
  const TfLiteStatus status = interpreter->ModifyGraphWithDelegate(delegate_);
  interpreter->tensor(0)->dims = saved_dims;
  ASSERT_EQ(status, kTfLiteOk);
  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  EXPECT_TRUE(ir_model == nullptr || ir_model->ops().empty());
}

TEST_F(ConvertQsaCompositesTest, RejectsNullOutputTensorDims) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  PopulateDefaultSparseSdpaBuilder(&builder,
                                   CreateSparseSdpaTransposedParams(4, 2));
  std::unique_ptr<::tflite::Interpreter> interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const int out_idx = interpreter->outputs()[0];
  TfLiteIntArray* saved_dims = interpreter->tensor(out_idx)->dims;
  interpreter->tensor(out_idx)->dims = nullptr;
  const TfLiteStatus status = interpreter->ModifyGraphWithDelegate(delegate_);
  interpreter->tensor(out_idx)->dims = saved_dims;
  ASSERT_EQ(status, kTfLiteOk);
  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  EXPECT_TRUE(ir_model == nullptr || ir_model->ops().empty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
