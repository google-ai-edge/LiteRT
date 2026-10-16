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

#include "litert/vendors/qualcomm/compiler/transformations/fold_const_dequantize.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"
#include "tensor/internal/fp16.h"

namespace litert::compiler {
namespace {

using ::litert::OwningBufferRef;
using ::litert::internal::AttachInput;
using ::litert::internal::AttachOutput;
using ::litert::internal::BufferManager;
using ::litert::tensor::fp16_ieee_from_fp32_value;
using ::testing::FloatNear;
using ::testing::SizeIs;

// Returns the fp32 value at index i in the tensor's weights buffer.
float GetFp32Weight(const LiteRtTensorT& tensor, int i) {
  const uint8_t* bytes = tensor.Weights().Buffer().Data();
  return reinterpret_cast<const float*>(bytes)[i];
}

// True when the tensor has a non-empty weights buffer.
bool TensorHasWeights(const LiteRtTensorT& tensor) {
  return tensor.Weights().Buffer().Size() > 0;
}

//===----------------------------------------------------------------------===//
// Negative (no-match) cases
// These tests only check the return status; no ApplyChanges is called so
// they do not require a BufferManager on the subgraph.
//===----------------------------------------------------------------------===//

TEST(FoldConstDequantizeTest, RejectsNonDequantizeOp) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in = subgraph.EmplaceTensor();
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));

  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflRelu);
  AttachInput(&in, op);
  AttachOutput(&out, op);

  EXPECT_EQ(FoldConstDequantizeTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(FoldConstDequantizeTest, RejectsNonConstantInput) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  // Input has no weights — treated as a runtime (non-constant) tensor.
  auto& in = subgraph.EmplaceTensor();
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {4}));

  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflDequantize);
  AttachInput(&in, op);
  AttachOutput(&out, op);

  EXPECT_EQ(FoldConstDequantizeTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(FoldConstDequantizeTest, RejectsNonFloat32Output) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in = subgraph.EmplaceTensor();
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {4}));
  std::vector<uint16_t> fp16_data = {fp16_ieee_from_fp32_value(1.0f), fp16_ieee_from_fp32_value(2.0f),
                                     fp16_ieee_from_fp32_value(3.0f), fp16_ieee_from_fp32_value(4.0f)};
  SetWeightsFromOwnedBuffer(
      in.Weights(),
      OwningBufferRef<uint8_t>(
          reinterpret_cast<const uint8_t*>(fp16_data.data()),
          fp16_data.size() * sizeof(uint16_t)));

  // Output is fp16, not fp32 — should be rejected.
  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {4}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflDequantize);
  AttachInput(&in, op);
  AttachOutput(&out, op);

  EXPECT_EQ(FoldConstDequantizeTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(FoldConstDequantizeTest, RejectsQuantizedInputWithoutQuantizationParams) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  // int8 input but no quantization params — Case B requires them.
  auto& in = subgraph.EmplaceTensor();
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {4}));
  std::vector<int8_t> int8_data = {1, 2, 3, 4};
  SetWeightsFromOwnedBuffer(
      in.Weights(),
      OwningBufferRef<uint8_t>(
          reinterpret_cast<const uint8_t*>(int8_data.data()),
          int8_data.size() * sizeof(int8_t)));

  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflDequantize);
  AttachInput(&in, op);
  AttachOutput(&out, op);

  EXPECT_EQ(FoldConstDequantizeTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

//===----------------------------------------------------------------------===//
// Case A — float16 constant folding
//===----------------------------------------------------------------------===//

TEST(FoldConstDequantizeTest, CaseA_Fp16FoldsToFp32) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  // Build: fp16_const -> dequantize -> fp32_out -> consumer
  auto& fp16_in = subgraph.EmplaceTensor();
  fp16_in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {4}));
  std::vector<uint16_t> fp16_data = {fp16_ieee_from_fp32_value(1.0f), fp16_ieee_from_fp32_value(2.0f),
                                     fp16_ieee_from_fp32_value(3.0f), fp16_ieee_from_fp32_value(4.0f)};
  SetWeightsFromOwnedBuffer(
      fp16_in.Weights(),
      OwningBufferRef<uint8_t>(
          reinterpret_cast<const uint8_t*>(fp16_data.data()),
          fp16_data.size() * sizeof(uint16_t)));

  auto& fp32_out = subgraph.EmplaceTensor();
  fp32_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));

  auto& dequant = subgraph.EmplaceOp();
  dequant.SetOpCode(kLiteRtOpCodeTflDequantize);
  AttachInput(&fp16_in, dequant);
  AttachOutput(&fp32_out, dequant);

  // Consumer op that takes fp32_out as input.
  auto& consumer_out = subgraph.EmplaceTensor();
  consumer_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));
  auto& consumer = subgraph.EmplaceOp();
  consumer.SetOpCode(kLiteRtOpCodeTflRelu);
  AttachInput(&fp32_out, consumer);
  AttachOutput(&consumer_out, consumer);

  ASSERT_EQ(FoldConstDequantizeTransformation(ctx, &builder_impl, &dequant),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  // The dequantize op must be gone; the consumer survives.
  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflRelu);

  // The consumer's input must now be a new fp32 constant tensor.
  ASSERT_THAT(subgraph.Ops().front()->Inputs(), SizeIs(1));
  const LiteRtTensorT* new_tensor = subgraph.Ops().front()->Inputs()[0];
  EXPECT_TRUE(TensorHasWeights(*new_tensor));
  EXPECT_EQ(new_tensor->Type().second.ranked_tensor_type.element_type,
            kLiteRtElementTypeFloat32);

  // Verify the folded fp32 values match the original fp16 inputs.
  EXPECT_THAT(GetFp32Weight(*new_tensor, 0), FloatNear(1.0f, 1e-3f));
  EXPECT_THAT(GetFp32Weight(*new_tensor, 1), FloatNear(2.0f, 1e-3f));
  EXPECT_THAT(GetFp32Weight(*new_tensor, 2), FloatNear(3.0f, 1e-3f));
  EXPECT_THAT(GetFp32Weight(*new_tensor, 3), FloatNear(4.0f, 1e-3f));
}

TEST(FoldConstDequantizeTest, CaseA_Fp16MultipleConsumers) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  auto& fp16_in = subgraph.EmplaceTensor();
  fp16_in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {2}));
  std::vector<uint16_t> fp16_data = {fp16_ieee_from_fp32_value(1.0f), fp16_ieee_from_fp32_value(-1.0f)};
  SetWeightsFromOwnedBuffer(
      fp16_in.Weights(),
      OwningBufferRef<uint8_t>(
          reinterpret_cast<const uint8_t*>(fp16_data.data()),
          fp16_data.size() * sizeof(uint16_t)));

  auto& fp32_out = subgraph.EmplaceTensor();
  fp32_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2}));

  auto& dequant = subgraph.EmplaceOp();
  dequant.SetOpCode(kLiteRtOpCodeTflDequantize);
  AttachInput(&fp16_in, dequant);
  AttachOutput(&fp32_out, dequant);

  // Two consumers both reading fp32_out.
  auto& out_a = subgraph.EmplaceTensor();
  auto& consumer_a = subgraph.EmplaceOp();
  consumer_a.SetOpCode(kLiteRtOpCodeTflRelu);
  AttachInput(&fp32_out, consumer_a);
  AttachOutput(&out_a, consumer_a);

  auto& out_b = subgraph.EmplaceTensor();
  auto& consumer_b = subgraph.EmplaceOp();
  consumer_b.SetOpCode(kLiteRtOpCodeTflAbs);
  AttachInput(&fp32_out, consumer_b);
  AttachOutput(&out_b, consumer_b);

  ASSERT_EQ(FoldConstDequantizeTransformation(ctx, &builder_impl, &dequant),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  // Dequantize gone; two consumers remain.
  EXPECT_EQ(subgraph.Ops().size(), 2);
  for (const auto* op : subgraph.Ops()) {
    ASSERT_THAT(op->Inputs(), SizeIs(1));
    const LiteRtTensorT* inp = op->Inputs()[0];
    EXPECT_TRUE(TensorHasWeights(*inp));
    EXPECT_EQ(inp->Type().second.ranked_tensor_type.element_type,
              kLiteRtElementTypeFloat32);
  }
  // Both consumers were rewired to the same new tensor.
  EXPECT_EQ(subgraph.Ops()[0]->Inputs()[0], subgraph.Ops()[1]->Inputs()[0]);
}

//===----------------------------------------------------------------------===//
// Case B — quantized integer constant folding
//===----------------------------------------------------------------------===//

TEST(FoldConstDequantizeTest, CaseB_Int8PerTensorFoldsToFp32) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  // int8 input with per-tensor quantization: scale=0.5, zero_point=0
  // raw values {2, 4, -2, -4} → dequantized {1.0, 2.0, -1.0, -2.0}
  auto& int8_in = subgraph.EmplaceTensor();
  int8_in.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {4}));
  int8_in.SetQarams(MakePerTensorQuantization(/*scale=*/0.5f, /*zp=*/0));
  std::vector<int8_t> int8_data = {2, 4, -2, -4};
  SetWeightsFromOwnedBuffer(
      int8_in.Weights(),
      OwningBufferRef<uint8_t>(
          reinterpret_cast<const uint8_t*>(int8_data.data()),
          int8_data.size() * sizeof(int8_t)));

  auto& fp32_out = subgraph.EmplaceTensor();
  fp32_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));

  auto& dequant = subgraph.EmplaceOp();
  dequant.SetOpCode(kLiteRtOpCodeTflDequantize);
  AttachInput(&int8_in, dequant);
  AttachOutput(&fp32_out, dequant);

  auto& consumer_out = subgraph.EmplaceTensor();
  auto& consumer = subgraph.EmplaceOp();
  consumer.SetOpCode(kLiteRtOpCodeTflRelu);
  AttachInput(&fp32_out, consumer);
  AttachOutput(&consumer_out, consumer);

  ASSERT_EQ(FoldConstDequantizeTransformation(ctx, &builder_impl, &dequant),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflRelu);

  const LiteRtTensorT* new_tensor = subgraph.Ops().front()->Inputs()[0];
  EXPECT_TRUE(TensorHasWeights(*new_tensor));
  EXPECT_EQ(new_tensor->Type().second.ranked_tensor_type.element_type,
            kLiteRtElementTypeFloat32);

  EXPECT_THAT(GetFp32Weight(*new_tensor, 0), FloatNear(1.0f, 1e-5f));
  EXPECT_THAT(GetFp32Weight(*new_tensor, 1), FloatNear(2.0f, 1e-5f));
  EXPECT_THAT(GetFp32Weight(*new_tensor, 2), FloatNear(-1.0f, 1e-5f));
  EXPECT_THAT(GetFp32Weight(*new_tensor, 3), FloatNear(-2.0f, 1e-5f));
}

TEST(FoldConstDequantizeTest, CaseB_Uint8PerTensorFoldsToFp32) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  // uint8 input: scale=1.0, zero_point=128
  // raw values {128, 129, 127} → dequantized {0.0, 1.0, -1.0}
  auto& uint8_in = subgraph.EmplaceTensor();
  uint8_in.SetType(MakeRankedTensorType(kLiteRtElementTypeUInt8, {3}));
  uint8_in.SetQarams(MakePerTensorQuantization(/*scale=*/1.0f, /*zp=*/128));
  std::vector<uint8_t> raw = {128, 129, 127};
  SetWeightsFromOwnedBuffer(
      uint8_in.Weights(),
      OwningBufferRef<uint8_t>(static_cast<const uint8_t*>(raw.data()),
                               raw.size()));

  auto& fp32_out = subgraph.EmplaceTensor();
  fp32_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {3}));

  auto& dequant = subgraph.EmplaceOp();
  dequant.SetOpCode(kLiteRtOpCodeTflDequantize);
  AttachInput(&uint8_in, dequant);
  AttachOutput(&fp32_out, dequant);

  auto& consumer_out = subgraph.EmplaceTensor();
  auto& consumer = subgraph.EmplaceOp();
  consumer.SetOpCode(kLiteRtOpCodeTflRelu);
  AttachInput(&fp32_out, consumer);
  AttachOutput(&consumer_out, consumer);

  ASSERT_EQ(FoldConstDequantizeTransformation(ctx, &builder_impl, &dequant),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 1);
  const LiteRtTensorT* new_tensor = subgraph.Ops().front()->Inputs()[0];
  EXPECT_THAT(GetFp32Weight(*new_tensor, 0), FloatNear(0.0f, 1e-5f));
  EXPECT_THAT(GetFp32Weight(*new_tensor, 1), FloatNear(1.0f, 1e-5f));
  EXPECT_THAT(GetFp32Weight(*new_tensor, 2), FloatNear(-1.0f, 1e-5f));
}

}  // namespace
}  // namespace litert::compiler
