// Copyright 2025 Google LLC.
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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_builder.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/vendors/examples/example_transformations.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert {

template <typename T>
RankedTensorSpec MakeRankedTensorSpec(absl::Span<const int32_t> dims) {
  return RankedTensorSpecBuilder(
             RankedTensorType(
                 GetElementType<T>(),
                 Layout(BuildLayout(dims.data(), dims.data() + dims.size()))))
      .Build();
}

namespace {

TEST(MatMulRequantTest, FuseMatMulRequantSuccessTest) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;

  // Tensors
  auto& input0 = subgraph.EmplaceTensor();
  auto& input1 = subgraph.EmplaceTensor();
  auto& matmul_to_convert = subgraph.EmplaceTensor();
  auto& output0 = subgraph.EmplaceTensor();

  // Set element types to be the same (requantization)
  matmul_to_convert.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  output0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));

  // Set specific quantization params for output
  output0.SetQarams(MakePerTensorQuantization(0.5f, 10));

  // MatMul0: In0, In1 -> matmul_to_convert
  auto& matmul0 = subgraph.EmplaceOp();
  matmul0.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  // Set non-default option to verify preservation
  litert::internal::TflOptions matmul0_opts;
  matmul0_opts.type = tflite::BuiltinOptions_BatchMatMulOptions;
  auto options = std::make_unique<tflite::BatchMatMulOptionsT>();
  options->adj_x = true;
  matmul0_opts.value = options.release();
  litert::internal::SetTflOptions(matmul0, std::move(matmul0_opts));

  internal::AttachInput(&input0, matmul0);
  internal::AttachInput(&input1, matmul0);
  internal::AttachOutput(&matmul_to_convert, matmul0);

  // Convert (Quantize): matmul_to_convert -> Output0
  auto& convert = subgraph.EmplaceOp();
  convert.SetOpCode(kLiteRtOpCodeTflQuantize);
  internal::AttachInput(&matmul_to_convert, convert);
  internal::AttachOutput(&output0, convert);

  // Call the transformation.
  FuseMatMulRequantTransformation(&builder, &convert);

  // Apply the changes.
  builder.ApplyChanges(&subgraph);

  // Verify the changes.
  int matmul_count = 0;
  int quantize_count = 0;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) matmul_count++;
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) quantize_count++;
  }

  EXPECT_EQ(matmul_count, 1);
  EXPECT_EQ(quantize_count, 0);
  EXPECT_EQ(subgraph.Ops().size(), 1);

  // Check that MatMul outputs to output0
  const auto& matmul_final = *subgraph.Ops().front();
  EXPECT_EQ(matmul_final.OpCode(), kLiteRtOpCodeTflBatchMatmul);
  ASSERT_EQ(matmul_final.Outputs().size(), 1);
  EXPECT_EQ(matmul_final.Outputs()[0], &output0);

  // Verify quantization parameters preserved on the final output
  EXPECT_EQ(output0.Qparams().first, kLiteRtQuantizationPerTensor);
  EXPECT_FLOAT_EQ(output0.Qparams().second.per_tensor.scale, 0.5f);
  EXPECT_EQ(output0.Qparams().second.per_tensor.zero_point, 10);

  // Verify options preserved
  const auto& opts = litert::internal::GetTflOptions(matmul_final);
  ASSERT_TRUE(opts.value != nullptr);
  EXPECT_TRUE(opts.AsBatchMatMulOptions()->adj_x);
}

TEST(MatMulRequantTest, FuseMatMulRequantNoMatchTest) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;

  // Tensors
  auto& input0 = subgraph.EmplaceTensor();
  auto& input1 = subgraph.EmplaceTensor();
  auto& matmul_to_convert = subgraph.EmplaceTensor();
  auto& output0 = subgraph.EmplaceTensor();

  // Set different element types (NOT a requantization)
  matmul_to_convert.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 10}));
  output0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));

  // MatMul0: In0, In1 -> matmul_to_convert
  auto& matmul0 = subgraph.EmplaceOp();
  matmul0.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&input0, matmul0);
  internal::AttachInput(&input1, matmul0);
  internal::AttachOutput(&matmul_to_convert, matmul0);

  // Convert (Quantize): matmul_to_convert -> Output0
  auto& convert = subgraph.EmplaceOp();
  convert.SetOpCode(kLiteRtOpCodeTflQuantize);
  internal::AttachInput(&matmul_to_convert, convert);
  internal::AttachOutput(&output0, convert);

  // Call the transformation.
  auto status = FuseMatMulRequantTransformation(&builder, &convert);
  EXPECT_EQ(status, kLiteRtStatusPatternNoMatch);

  // Apply the changes (none expected).
  builder.ApplyChanges(&subgraph);

  // Verify no changes were made.
  EXPECT_EQ(subgraph.Ops().size(), 2);
  EXPECT_EQ(subgraph.Ops()[0]->OpCode(), kLiteRtOpCodeTflBatchMatmul);
  EXPECT_EQ(subgraph.Ops()[1]->OpCode(), kLiteRtOpCodeTflQuantize);
}

TEST(MatMulRequantTest, FuseMatMulRequantComplexDagTest) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;

  // Tensors
  auto& input0 = subgraph.EmplaceTensor();
  auto& input1 = subgraph.EmplaceTensor();
  auto& input2 = subgraph.EmplaceTensor();
  auto& input3 = subgraph.EmplaceTensor();
  auto& inter0 = subgraph.EmplaceTensor();  // matmul0 output
  auto& inter1 = subgraph.EmplaceTensor();  // quant output / concat input 0
  auto& inter2 = subgraph.EmplaceTensor();  // matmul1 output / concat input 1
  auto& out = subgraph.EmplaceTensor();     // concat output

  // Set types for requantization
  inter0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  inter1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  inter2.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 20}));

  // MatMul0: In0, In1 -> inter0
  auto& matmul0 = subgraph.EmplaceOp();
  matmul0.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&input0, matmul0);
  internal::AttachInput(&input1, matmul0);
  internal::AttachOutput(&inter0, matmul0);

  // Quant: inter0 -> inter1
  auto& quant = subgraph.EmplaceOp();
  quant.SetOpCode(kLiteRtOpCodeTflQuantize);
  internal::AttachInput(&inter0, quant);
  internal::AttachOutput(&inter1, quant);

  // MatMul1: In2, In3 -> inter2
  auto& matmul1 = subgraph.EmplaceOp();
  matmul1.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&input2, matmul1);
  internal::AttachInput(&input3, matmul1);
  internal::AttachOutput(&inter2, matmul1);

  // Concat: inter1, inter2 -> out
  auto& concat = subgraph.EmplaceOp();
  concat.SetOpCode(kLiteRtOpCodeTflConcatenation);
  internal::AttachInput(&inter1, concat);
  internal::AttachInput(&inter2, concat);
  internal::AttachOutput(&out, concat);

  // Call the transformation on the Quant op.
  FuseMatMulRequantTransformation(&builder, &quant);

  // Apply the changes.
  builder.ApplyChanges(&subgraph);

  // Verify the changes.
  int matmul_count = 0;
  int quantize_count = 0;
  int concat_count = 0;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) matmul_count++;
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) quantize_count++;
    if (op->OpCode() == kLiteRtOpCodeTflConcatenation) concat_count++;
  }

  EXPECT_EQ(matmul_count, 2);
  EXPECT_EQ(quantize_count, 0);
  EXPECT_EQ(concat_count, 1);

  // Find the concat op and check its inputs
  LiteRtOpT* final_concat = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflConcatenation) {
      final_concat = op;
      break;
    }
  }
  ASSERT_NE(final_concat, nullptr) << "Concat op is MISSING from the subgraph!";
  ASSERT_EQ(final_concat->Inputs().size(), 2)
      << "Concat op lost an input! Current size: "
      << final_concat->Inputs().size();
  EXPECT_EQ(final_concat->Inputs()[0], &inter1);
  EXPECT_EQ(final_concat->Inputs()[1], &inter2);

  // Check the matmuls
  LiteRtOpT* new_matmul0 = nullptr;
  LiteRtOpT* matmul1_ptr = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) {
      if (!op->Outputs().empty()) {
        if (op->Outputs()[0] == &inter1) new_matmul0 = op;
        if (op->Outputs()[0] == &inter2) matmul1_ptr = op;
      }
    }
  }
  ASSERT_NE(new_matmul0, nullptr) << "The rewritten MatMul op is MISSING!";
  ASSERT_NE(matmul1_ptr, nullptr) << "The original MatMul1 op is MISSING!";

  // Verify that new_matmul0 is indeed the defining op of inter1
  EXPECT_EQ(inter1.DefiningOp(), new_matmul0);

  // Verify that inter1 still has its user (concat)
  ASSERT_EQ(inter1.NumUses(), 1);
  EXPECT_EQ(inter1.Users()[0], final_concat);
  EXPECT_EQ(inter1.UserArgInds()[0], 0);

  // Verify that inter2 still has its user (concat)
  ASSERT_EQ(inter2.NumUses(), 1);
  EXPECT_EQ(inter2.Users()[0], final_concat);
  EXPECT_EQ(inter2.UserArgInds()[0], 1);

  // Check shared inputs if any.
  ASSERT_EQ(input0.NumUses(), 1);
  EXPECT_EQ(input0.Users()[0], new_matmul0);
}

TEST(MatMulRequantTest, FuseMatMulRequantSharedInputTest) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;

  // Tensors
  auto& input_shared = subgraph.EmplaceTensor();
  auto& input1 = subgraph.EmplaceTensor();
  auto& input2 = subgraph.EmplaceTensor();
  auto& inter0 = subgraph.EmplaceTensor();  // matmul0 output
  auto& inter1 = subgraph.EmplaceTensor();  // quant output
  auto& out1 = subgraph.EmplaceTensor();    // matmul1 output

  // Set types for requantization
  inter0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  inter1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));
  out1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 10}));

  // MatMul0: input_shared, input1 -> inter0
  auto& matmul0 = subgraph.EmplaceOp();
  matmul0.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&input_shared, matmul0);
  internal::AttachInput(&input1, matmul0);
  internal::AttachOutput(&inter0, matmul0);

  // Quant: inter0 -> inter1
  auto& quant = subgraph.EmplaceOp();
  quant.SetOpCode(kLiteRtOpCodeTflQuantize);
  internal::AttachInput(&inter0, quant);
  internal::AttachOutput(&inter1, quant);

  // MatMul1: input_shared, input2 -> out1
  auto& matmul1 = subgraph.EmplaceOp();
  matmul1.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  internal::AttachInput(&input_shared, matmul1);
  internal::AttachInput(&input2, matmul1);
  internal::AttachOutput(&out1, matmul1);

  // Verification before transformation
  ASSERT_EQ(input_shared.NumUses(), 2);

  // Call the transformation on the Quant op.
  FuseMatMulRequantTransformation(&builder, &quant);

  // Apply the changes.
  builder.ApplyChanges(&subgraph);

  // Verify the changes.
  LiteRtOpT* new_matmul0 = nullptr;
  LiteRtOpT* matmul1_ptr = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) {
      if (!op->Outputs().empty()) {
        if (op->Outputs()[0] == &inter1) new_matmul0 = op;
        if (op->Outputs()[0] == &out1) matmul1_ptr = op;
      }
    }
  }
  ASSERT_NE(new_matmul0, nullptr);
  ASSERT_NE(matmul1_ptr, nullptr);

  // Check shared input uses.
  ASSERT_EQ(input_shared.NumUses(), 2)
      << "Shared input lost a user! Current size: " << input_shared.NumUses();

  bool found_new_matmul = false;
  bool found_matmul1 = false;
  for (size_t i = 0; i < input_shared.NumUses(); ++i) {
    if (input_shared.Users()[i] == new_matmul0) found_new_matmul = true;
    if (input_shared.Users()[i] == matmul1_ptr) found_matmul1 = true;
  }
  EXPECT_TRUE(found_new_matmul);
  EXPECT_TRUE(found_matmul1);
}

TEST(MatMulRequantTest, FuseMatMulRequantRealModelTest) {
  // 1. Load model from file
  auto model_wrap = testing::LoadTestFileModel("matmul_quant.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto* subgraph = model.MainSubgraph();

  LiteRtBuilderT builder;

  // 2. Apply transformation
  // Identify the Quantize op that follows a BatchMatMul.
  LiteRtOp quant_op_ptr = nullptr;
  for (auto* op : subgraph->Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) {
      if (FuseMatMulRequantTransformation(&builder, op) == kLiteRtStatusOk) {
        quant_op_ptr = op;
        break;
      }
    }
  }
  ASSERT_NE(quant_op_ptr, nullptr) << "Failed to find the pattern to fuse";

  // 3. Apply changes
  builder.ApplyChanges(subgraph);

  // 4. Serialize model to byte.
  auto serialized = internal::SerializeModel(std::move(model));
  ASSERT_TRUE(serialized);

  // 5. Check serialization
  EXPECT_TRUE(internal::VerifyFlatbuffer(serialized->Span()));

  // 6. Check transformed model content (directly on subgraph)
  int matmul_count = 0;
  int quantize_count = 0;
  int concat_count = 0;
  LiteRtOp final_concat = nullptr;

  for (auto* op : subgraph->Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) matmul_count++;
    if (op->OpCode() == kLiteRtOpCodeTflQuantize) quantize_count++;
    if (op->OpCode() == kLiteRtOpCodeTflConcatenation) {
      concat_count++;
      final_concat = op;
    }
  }

  EXPECT_EQ(matmul_count, 2);
  EXPECT_EQ(quantize_count, 0);
  EXPECT_EQ(concat_count, 1);

  ASSERT_NE(final_concat, nullptr);
  // Check that Concatenation inputs are now directly from BatchMatMul ops.
  ASSERT_EQ(final_concat->Inputs().size(), 2);
}

}  // namespace
}  // namespace litert
