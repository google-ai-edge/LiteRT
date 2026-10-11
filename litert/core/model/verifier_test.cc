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

#include "litert/core/model/verifier.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tflite/schema/schema_generated.h"
#include "tflite/stderr_reporter.h"
#include "tflite/tools/verifier.h"
#include "tflite/version.h"

namespace litert {
namespace {

template <typename T>
flatbuffers::Offset<tflite::Buffer> CreateTypedBuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return tflite::CreateBuffer(builder, builder.CreateVector(bytes));
}

void BuildSliceModel(flatbuffers::FlatBufferBuilder& builder,
                     const std::vector<int32_t>& input_shape,
                     tflite::TensorType begin_type,
                     const std::vector<int64_t>& begin_vals,
                     tflite::TensorType size_type,
                     const std::vector<int64_t>& size_vals) {
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));  // 0: empty

  if (begin_type == tflite::TensorType_INT64) {
    buffers.push_back(CreateTypedBuffer<int64_t>(builder, begin_vals));
  } else {
    std::vector<int32_t> vals32(begin_vals.begin(), begin_vals.end());
    buffers.push_back(CreateTypedBuffer<int32_t>(builder, vals32));
  }

  if (size_type == tflite::TensorType_INT64) {
    buffers.push_back(CreateTypedBuffer<int64_t>(builder, size_vals));
  } else {
    std::vector<int32_t> vals32(size_vals.begin(), size_vals.end());
    buffers.push_back(CreateTypedBuffer<int32_t>(builder, vals32));
  }

  buffers.push_back(tflite::CreateBuffer(builder));  // 3: output

  const int32_t index_len = static_cast<int32_t>(input_shape.size());
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      tflite::CreateTensor(builder, builder.CreateVector(input_shape),
                           tflite::TensorType_FLOAT32, /*buffer=*/0,
                           builder.CreateString("input")),
      tflite::CreateTensor(
          builder, builder.CreateVector(std::vector<int32_t>{index_len}),
          begin_type, /*buffer=*/1, builder.CreateString("begin")),
      tflite::CreateTensor(
          builder, builder.CreateVector(std::vector<int32_t>{index_len}),
          size_type, /*buffer=*/2, builder.CreateString("size")),
      tflite::CreateTensor(builder,
                           builder.CreateVector(std::vector<int32_t>{}),
                           tflite::TensorType_FLOAT32, /*buffer=*/3,
                           builder.CreateString("output")),
  };

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_SLICE);
  std::vector<int32_t> op_inputs = {0, 1, 2};
  std::vector<int32_t> op_outputs = {3};
  auto op = tflite::CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector(op_inputs),
      builder.CreateVector(op_outputs), tflite::BuiltinOptions_SliceOptions,
      tflite::CreateSliceOptions(builder).Union());

  std::vector<int32_t> model_inputs = {0};
  std::vector<int32_t> model_outputs = {3};
  auto subgraph = tflite::CreateSubGraph(builder, builder.CreateVector(tensors),
                                         builder.CreateVector(model_inputs),
                                         builder.CreateVector(model_outputs),
                                         builder.CreateVector(&op, 1));

  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&opcode, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("slice_test"),
      builder.CreateVector(buffers));
  tflite::FinishModelBuffer(builder, model);
}

TEST(VerifierTest, ValidSliceModelPasses) {
  flatbuffers::FlatBufferBuilder builder;
  BuildSliceModel(builder, /*input_shape=*/{4, 6}, tflite::TensorType_INT32,
                  /*begin_vals=*/{1, 2}, tflite::TensorType_INT32,
                  /*size_vals=*/{2, -1});

  EXPECT_TRUE(Verify(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(VerifierTest, RejectsSliceWithMismatchedBeginAndSizeTypes) {
  flatbuffers::FlatBufferBuilder builder;
  BuildSliceModel(builder, /*input_shape=*/{4, 6}, tflite::TensorType_INT32,
                  /*begin_vals=*/{1, 2}, tflite::TensorType_INT64,
                  /*size_vals=*/{2, 3});

  // FlatBuffer schema verifier passes this model, but LiteRT verifier rejects
  // it because begin and size have different data types.
  EXPECT_TRUE(tflite::Verify(builder.GetBufferPointer(), builder.GetSize(),
                             tflite::DefaultErrorReporter()));
  EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(VerifierTest, RejectsSliceWithOutOfBoundsIndices) {
  flatbuffers::FlatBufferBuilder builder;
  BuildSliceModel(builder, /*input_shape=*/{4, 6}, tflite::TensorType_INT32,
                  /*begin_vals=*/{2, 2}, tflite::TensorType_INT32,
                  /*size_vals=*/{3, 2});

  EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(VerifierTest, ConfigurableMaxRankRestrictsTensorRank) {
  flatbuffers::FlatBufferBuilder builder;
  // Rank-5 input tensor: {1, 2, 3, 4, 5}.
  BuildSliceModel(builder, /*input_shape=*/{1, 2, 3, 4, 5},
                  tflite::TensorType_INT32, /*begin_vals=*/{0, 0, 0, 0, 0},
                  tflite::TensorType_INT32, /*size_vals=*/{1, 1, 1, 1, 1});

  // Passes with default max_rank (8).
  EXPECT_TRUE(Verify(builder.GetBufferPointer(), builder.GetSize()));

  // Rejected when max_rank is restricted to 4 at runtime.
  VerifyOptions strict_rank_options;
  strict_rank_options.max_rank = 4;
  EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(),
                      strict_rank_options));

  LiteRtVerifier verifier(strict_rank_options);
  EXPECT_FALSE(verifier.Verify(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      static_cast<int>(builder.GetSize()), tflite::DefaultErrorReporter()));
}

TEST(VerifierTest, RequireAllTensorShapeRanksKnownRejectsUnrankedTensor) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
  };
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      // Unranked tensor: shape offset is 0 and has_rank is false.
      tflite::CreateTensor(builder, /*shape=*/0, tflite::TensorType_FLOAT32,
                           /*buffer=*/0, builder.CreateString("unranked_in")),
      tflite::CreateTensor(builder,
                           builder.CreateVector(std::vector<int32_t>{2}),
                           tflite::TensorType_FLOAT32, /*buffer=*/1,
                           builder.CreateString("out")),
  };
  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_RELU);
  std::vector<int32_t> op_inputs = {0};
  std::vector<int32_t> op_outputs = {1};
  auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                   builder.CreateVector(op_inputs),
                                   builder.CreateVector(op_outputs));
  std::vector<int32_t> model_inputs = {0};
  std::vector<int32_t> model_outputs = {1};
  auto subgraph = tflite::CreateSubGraph(builder, builder.CreateVector(tensors),
                                         builder.CreateVector(model_inputs),
                                         builder.CreateVector(model_outputs),
                                         builder.CreateVector(&op, 1));
  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&opcode, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("unranked_test"),
      builder.CreateVector(buffers));
  tflite::FinishModelBuffer(builder, model);

  VerifyOptions options;
  options.require_all_tensor_shape_ranks_known = true;
  EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(), options));
}

TEST(VerifierTest, HandlesModelWithUnnamedMetadataWithoutCrash) {
  const std::string fuzzed(
      "$\000\000\000TFL3\000\000\000\000\000\000\000\000\024\000\030\000\004"
      "\000\010\000\014\000\000\000\014\000\000\000\010\000\000\000\024\000\000"
      "\000\003\000\000\000\344\001\000\000\230\000\000\000\200\000\000\000\004"
      "\000\000\000\377\377\377\037\020\000\000\000\000\000\n\000\020\000\004"
      "\000\010\000\014\000\n\000\000\000<\000\000\000\034\000\000\000\004\000"
      "\000\000\017\000\000\000serving_default\000\001\000\000\000\004\000\000"
      "\000\344\377\377d\014<\010\004\002\000\000\000\001\000\000\000x\000\000"
      "\000\001\000\000\000\014\000\000\000\010\000\014\000\004\000\000\000\010"
      "\000\000\000\010\000\000\000\001\000\000\000\001\000\000\000a\000\000"
      "\000\001\000\000\000\004\000\000\000\244\376\377\377\000\000\000\000\000"
      "\000\000\000\001\000\000\000\020\000\000\000\014\000\024\010\000\000\000"
      "\000\000\000\004\000\014\000\000\000\224\000\000\000\210\000\000\000|"
      "\000\000\000\004\000\000\000\002\000\000\000D\000\000\000\004\000\000"
      "\000\322\377\377\377\000\000\000\n\010\002\001\0036Z\251Z\000\033 "
      "\373\367\253\377\377\000\000\000\000\002\000\000\000\002\000\000\000\000"
      "\000\000\000\001\000\000\000\000\000\016\000\024\000\000\000\010\000\014"
      "\000\007\000\020\000\016\000\000\000\000\000\000\013\030\000\000\000\014"
      "\000\000\000\004\000\000\0004\377\377\377\001\000\000\000\000\000\000"
      "\000\002\000\000\000\001\000\000\000\001\000\\\000\001\000\000\000\002"
      "\000\000\000\001\000 \002\000\000\000\000\000\000\000\000p\000\000\0004"
      "\000\000\000\004\000\000\000\250\377\377\377\024\000\000\000\004\000\000"
      "\000\006\000\000\000output\000\000\004\000\000\264\000\000\000\000\000"
      "\000\000\000\010\000\000\000\003\000\000\000\324\377\377\377\024\000\000"
      "\0020\000\000\000\005\000\000\000input\000\000\000\004\000\000\000\001"
      "\000\000\000\010\000\000\000\010\000\000\000\003\000\000\000\014\000\014"
      "\000\004\000\000\000\000\000\010\000\014\000\000\000\020\000\000\000\004"
      "\000\000\000\003\000\000\000add\000\004\000\000\000\001\000\000\000\010"
      "\000\000\000\010\000\000\000\003N\004@\001\000\000\000\010\000\000\000"
      "\004\000\004\000\004\000\000\000",
      544);
  EXPECT_TRUE(Verify(fuzzed.data(), fuzzed.size()));
}

void BuildQuantizedReluModel(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<int32_t>& input_shape, tflite::TensorType input_type,
    flatbuffers::Offset<tflite::QuantizationParameters> quant_params,
    bool add_scale_tensor = false,
    tflite::TensorType scale_type = tflite::TensorType_FLOAT16) {
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
  };
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      tflite::CreateTensor(builder, builder.CreateVector(input_shape),
                           input_type, /*buffer=*/0, builder.CreateString("in"),
                           quant_params),
      tflite::CreateTensor(builder, builder.CreateVector(input_shape),
                           tflite::TensorType_FLOAT32, /*buffer=*/1,
                           builder.CreateString("out")),
  };
  if (add_scale_tensor) {
    buffers.push_back(tflite::CreateBuffer(builder));
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(std::vector<int32_t>{2, 1}), scale_type,
        /*buffer=*/2, builder.CreateString("scales")));
  }
  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_RELU);
  std::vector<int32_t> op_inputs = {0};
  std::vector<int32_t> op_outputs = {1};
  auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                   builder.CreateVector(op_inputs),
                                   builder.CreateVector(op_outputs));
  std::vector<int32_t> model_inputs = {0};
  std::vector<int32_t> model_outputs = {1};
  auto subgraph = tflite::CreateSubGraph(builder, builder.CreateVector(tensors),
                                         builder.CreateVector(model_inputs),
                                         builder.CreateVector(model_outputs),
                                         builder.CreateVector(&op, 1));
  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&opcode, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("quant_test"),
      builder.CreateVector(buffers));
  tflite::FinishModelBuffer(builder, model);
}

absl::Status ValidateFirstTensorQuantization(
    const flatbuffers::FlatBufferBuilder& builder) {
  const tflite::Model* model = tflite::GetModel(builder.GetBufferPointer());
  const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
  const tflite::Tensor* tensor = subgraph->tensors()->Get(0);
  return internal::ValidateFlatBufferTensorQuantization(*subgraph, *tensor);
}

TEST(VerifierTest, ValidateFlatBufferTensorQuantizationAcceptsValidAffine) {
  // Unquantized tensor (quantization == nullptr).
  {
    flatbuffers::FlatBufferBuilder builder;
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_FLOAT32,
                            /*quant_params=*/0);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
  }

  // Empty scale and zero_point vectors with QuantizationDetails_NONE.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(builder);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
  }

  // Valid per-tensor quantization (1 scale, 1 zero_point).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{0.5f}),
        builder.CreateVector(std::vector<int64_t>{0}),
        tflite::QuantizationDetails_NONE, /*details=*/0,
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
  }

  // Valid per-axis quantization matching static dimension size (2).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{0.5f, 0.25f}),
        builder.CreateVector(std::vector<int64_t>{0, 1}),
        tflite::QuantizationDetails_NONE, /*details=*/0,
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
  }

  // Valid per-axis quantization on a dynamic dimension (-1).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{0.5f, 0.25f, 0.125f}),
        builder.CreateVector(std::vector<int64_t>{0, 0, 0}),
        tflite::QuantizationDetails_NONE, /*details=*/0,
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {-1, 4}, tflite::TensorType_INT8, quant);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
  }
}

TEST(VerifierTest, RejectsInvalidAffineQuantizationParameters) {
  // Single scale with empty zero_point vector.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{0.5f}),
        builder.CreateVector(std::vector<int64_t>{}));
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
  }

  // Empty scale vector with non-empty zero_point vector.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{}),
        builder.CreateVector(std::vector<int64_t>{0}));
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Mismatched non-empty scale (2) and zero_point (1) sizes.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{0.5f, 0.25f}),
        builder.CreateVector(std::vector<int64_t>{0}));
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Negative quantized_dimension (-1).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{0.5f}),
        builder.CreateVector(std::vector<int64_t>{0}),
        tflite::QuantizationDetails_NONE, /*details=*/0,
        /*quantized_dimension=*/-1);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Per-axis scale count (3) does not match quantized_dimension (0) size (2).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{0.5f, 0.25f, 0.125f}),
        builder.CreateVector(std::vector<int64_t>{0, 0, 0}),
        tflite::QuantizationDetails_NONE, /*details=*/0,
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
  }

  // Out-of-range quantized_dimension (2 for rank-2 tensor).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{0.5f}),
        builder.CreateVector(std::vector<int64_t>{0}),
        tflite::QuantizationDetails_NONE, /*details=*/0,
        /*quantized_dimension=*/2);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
  }
}

TEST(VerifierTest, ValidateFlatBufferTensorQuantizationAcceptsValidBlockwise) {
  // Valid INT4 block_size (32) with no zero_points (-1).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto bw = tflite::CreateBlockwiseQuantization(builder, /*scales=*/2,
                                                  /*zero_points=*/-1,
                                                  /*block_size=*/32);
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
  }

  // Valid INT2 block_size (32, divisible by 4) with valid zero_points tensor.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto bw = tflite::CreateBlockwiseQuantization(builder, /*scales=*/2,
                                                  /*zero_points=*/1,
                                                  /*block_size=*/32);
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT2, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
  }

  // Valid block_shape ({1, 16}) matching rank-2 tensor ({2, 32}).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto bw = tflite::CreateBlockwiseQuantization(
        builder, /*scales=*/2, /*zero_points=*/-1, /*block_size=*/0,
        builder.CreateVector(std::vector<int32_t>{1, 16}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
  }
}

TEST(VerifierTest, RejectsInvalidBlockwiseQuantizationParameters) {
  // Missing BlockwiseQuantization details union table.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, /*details=*/0,
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Zero, negative, or unaligned block_size for INT4 and UINT4.
  for (tflite::TensorType type :
       {tflite::TensorType_INT4, tflite::TensorType_UINT4}) {
    for (int32_t bad_block_size : {0, -32, 3}) {
      flatbuffers::FlatBufferBuilder builder;
      auto bw = tflite::CreateBlockwiseQuantization(
          builder, /*scales=*/2, /*zero_points=*/-1,
          /*block_size=*/bad_block_size);
      auto quant = tflite::CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
          tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
          /*quantized_dimension=*/1);
      BuildQuantizedReluModel(builder, {2, 32}, type, quant,
                              /*add_scale_tensor=*/true);
      EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
                absl::StatusCode::kInvalidArgument);
      EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
    }
  }

  // Unaligned block_size (6, not divisible by 4) for INT2.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto bw = tflite::CreateBlockwiseQuantization(builder, /*scales=*/2,
                                                  /*zero_points=*/-1,
                                                  /*block_size=*/6);
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 36}, tflite::TensorType_INT2, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Negative or out-of-bounds scales tensor index (-1, 99).
  for (int32_t bad_scales : {-1, 99}) {
    flatbuffers::FlatBufferBuilder builder;
    auto bw =
        tflite::CreateBlockwiseQuantization(builder, /*scales=*/bad_scales,
                                            /*zero_points=*/-1,
                                            /*block_size=*/32);
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
  }

  // Invalid zero_points tensor index (-2 or out-of-bounds 99).
  for (int32_t bad_zp : {-2, 99}) {
    flatbuffers::FlatBufferBuilder builder;
    auto bw = tflite::CreateBlockwiseQuantization(builder, /*scales=*/2,
                                                  /*zero_points=*/bad_zp,
                                                  /*block_size=*/32);
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Last dimension (34) not divisible by block_size (32).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto bw = tflite::CreateBlockwiseQuantization(builder, /*scales=*/2,
                                                  /*zero_points=*/-1,
                                                  /*block_size=*/32);
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 34}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
  }

  // block_shape rank (3) does not match tensor rank (2).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto bw = tflite::CreateBlockwiseQuantization(
        builder, /*scales=*/2, /*zero_points=*/-1, /*block_size=*/0,
        builder.CreateVector(std::vector<int32_t>{1, 16, 1}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Non-positive block_shape entry (0 or -4).
  for (int32_t bad_dim : {0, -4}) {
    flatbuffers::FlatBufferBuilder builder;
    auto bw = tflite::CreateBlockwiseQuantization(
        builder, /*scales=*/2, /*zero_points=*/-1, /*block_size=*/0,
        builder.CreateVector(std::vector<int32_t>{1, bad_dim}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
        /*quantized_dimension=*/1);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(VerifierTest, ValidateFlatBufferTensorQuantizationAcceptsValidMultiAxis) {
  // Per-channel multi-axis quantization (block_size = 0, no zero_points).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto multi_axis_q = tflite::CreateMultiAxisQuantization(
        builder, /*scales=*/2, /*zero_points=*/-1, /*block_size=*/0,
        builder.CreateVector(std::vector<int32_t>{0, 1}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_MultiAxisQuantization, multi_axis_q.Union(),
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
    EXPECT_TRUE(Verify(builder.GetBufferPointer(), builder.GetSize()));
  }

  // Block-wise multi-axis quantization (block_size = 32 for INT4, valid
  // zero_points tensor index = 1).
  {
    flatbuffers::FlatBufferBuilder builder;
    auto multi_axis_q = tflite::CreateMultiAxisQuantization(
        builder, /*scales=*/2, /*zero_points=*/1, /*block_size=*/32,
        builder.CreateVector(std::vector<int32_t>{0, 1}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_MultiAxisQuantization, multi_axis_q.Union(),
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_TRUE(ValidateFirstTensorQuantization(builder).ok());
  }
}

TEST(VerifierTest, RejectsInvalidMultiAxisQuantizationParameters) {
  // Missing MultiAxisQuantization details or empty quantized_dimensions.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_MultiAxisQuantization, /*details=*/0,
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
  }
  {
    flatbuffers::FlatBufferBuilder builder;
    auto multi_axis_q = tflite::CreateMultiAxisQuantization(
        builder, /*scales=*/2, /*zero_points=*/-1, /*block_size=*/0,
        builder.CreateVector(std::vector<int32_t>{}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_MultiAxisQuantization, multi_axis_q.Union(),
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Invalid scales (-1, 99) or zero_points (-2, 99) tensor indices.
  for (int32_t bad_scales : {-1, 99}) {
    flatbuffers::FlatBufferBuilder builder;
    auto multi_axis_q = tflite::CreateMultiAxisQuantization(
        builder, /*scales=*/bad_scales, /*zero_points=*/-1, /*block_size=*/0,
        builder.CreateVector(std::vector<int32_t>{0}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_MultiAxisQuantization, multi_axis_q.Union(),
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }
  for (int32_t bad_zp : {-2, 99}) {
    flatbuffers::FlatBufferBuilder builder;
    auto multi_axis_q = tflite::CreateMultiAxisQuantization(
        builder, /*scales=*/2, /*zero_points=*/bad_zp, /*block_size=*/0,
        builder.CreateVector(std::vector<int32_t>{0}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_MultiAxisQuantization, multi_axis_q.Union(),
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Negative block_size (-1) or odd block_size (3) for INT4.
  for (int32_t bad_block_size : {-1, 3}) {
    flatbuffers::FlatBufferBuilder builder;
    auto multi_axis_q = tflite::CreateMultiAxisQuantization(
        builder, /*scales=*/2, /*zero_points=*/-1,
        /*block_size=*/bad_block_size,
        builder.CreateVector(std::vector<int32_t>{0, 1}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_MultiAxisQuantization, multi_axis_q.Union(),
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 32}, tflite::TensorType_INT4, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Negative, out-of-range, or duplicate quantized_dimensions entry.
  for (int32_t bad_qdim : {-1, 2, 0}) {
    flatbuffers::FlatBufferBuilder builder;
    auto multi_axis_q = tflite::CreateMultiAxisQuantization(
        builder, /*scales=*/2, /*zero_points=*/-1, /*block_size=*/0,
        builder.CreateVector(std::vector<int32_t>{0, bad_qdim}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_MultiAxisQuantization, multi_axis_q.Union(),
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant,
                            /*add_scale_tensor=*/true);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(VerifierTest, RejectsUnsupportedQuantizationDetails) {
  // CustomQuantization is unsupported and must not be silently ignored.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto custom_q = tflite::CreateCustomQuantization(
        builder, builder.CreateVector(std::vector<uint8_t>{1, 2, 3, 4}));
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        tflite::QuantizationDetails_CustomQuantization, custom_q.Union(),
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kUnimplemented);
    EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
  }

  // Out-of-range raw QuantizationDetails enum value.
  {
    flatbuffers::FlatBufferBuilder builder;
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
        static_cast<tflite::QuantizationDetails>(99), /*details=*/0,
        /*quantized_dimension=*/0);
    BuildQuantizedReluModel(builder, {2, 4}, tflite::TensorType_INT8, quant);
    EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(VerifierTest, VerifyModelAndValidateFlatBufferTensorsReturnStatus) {
  // Valid model passes both ValidateFlatBufferTensors and VerifyModel.
  {
    flatbuffers::FlatBufferBuilder builder;
    BuildSliceModel(builder, /*input_shape=*/{4, 6}, tflite::TensorType_INT32,
                    /*begin_vals=*/{1, 2}, tflite::TensorType_INT32,
                    /*size_vals=*/{2, -1});
    const tflite::Model* tfl_model =
        tflite::GetModel(builder.GetBufferPointer());
    EXPECT_TRUE(internal::ValidateFlatBufferTensors(tfl_model).ok());
    EXPECT_TRUE(
        VerifyModel(builder.GetBufferPointer(), builder.GetSize()).ok());
  }

  // Null/empty buffer returns InvalidArgumentError.
  EXPECT_EQ(VerifyModel(nullptr, 0).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(internal::ValidateFlatBufferTensors(nullptr).code(),
            absl::StatusCode::kInvalidArgument);

  // Rank violation returns InvalidArgumentError from both functions.
  {
    flatbuffers::FlatBufferBuilder builder;
    BuildSliceModel(builder, /*input_shape=*/{1, 2, 3, 4, 5},
                    tflite::TensorType_INT32, /*begin_vals=*/{0, 0, 0, 0, 0},
                    tflite::TensorType_INT32, /*size_vals=*/{1, 1, 1, 1, 1});
    VerifyOptions options;
    options.max_rank = 4;
    const tflite::Model* tfl_model =
        tflite::GetModel(builder.GetBufferPointer());
    EXPECT_EQ(internal::ValidateFlatBufferTensors(tfl_model, options).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(
        VerifyModel(builder.GetBufferPointer(), builder.GetSize(), options)
            .code(),
        absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace litert
