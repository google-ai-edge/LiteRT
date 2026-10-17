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

struct AffineQuantizationTestCase {
  std::string test_name;
  std::vector<int32_t> input_shape;
  tflite::TensorType input_type;
  bool omit_quantization = false;
  bool null_vectors = false;
  std::vector<float> scale = {};
  std::vector<int64_t> zero_point = {};
  int32_t quantized_dimension = 0;
  absl::StatusCode expected_status = absl::StatusCode::kOk;
  bool check_full_verify = false;
};

using AffineQuantizationVerifierTest =
    ::testing::TestWithParam<AffineQuantizationTestCase>;

TEST_P(AffineQuantizationVerifierTest, ValidatesAffineQuantization) {
  const auto& param = GetParam();
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::QuantizationParameters> quant = 0;
  if (!param.omit_quantization) {
    if (param.null_vectors) {
      quant = tflite::CreateQuantizationParameters(builder);
    } else {
      quant = tflite::CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0, builder.CreateVector(param.scale),
          builder.CreateVector(param.zero_point),
          tflite::QuantizationDetails_NONE, /*details=*/0,
          param.quantized_dimension);
    }
  }
  BuildQuantizedReluModel(builder, param.input_shape, param.input_type, quant);
  EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
            param.expected_status);
  if (param.check_full_verify) {
    EXPECT_EQ(Verify(builder.GetBufferPointer(), builder.GetSize()),
              param.expected_status == absl::StatusCode::kOk);
  }
}

INSTANTIATE_TEST_SUITE_P(
    AffineQuantizationVerifierTests, AffineQuantizationVerifierTest,
    ::testing::Values(
        AffineQuantizationTestCase{
            "Unquantized",
            {2, 4},
            tflite::TensorType_FLOAT32,
            /*omit_quantization=*/true,
        },
        AffineQuantizationTestCase{
            "NullScaleAndZeroPointVectors",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/true,
        },
        AffineQuantizationTestCase{
            "ValidPerTensor",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/false,
            {0.5f},
            {0},
            /*quantized_dimension=*/0,
        },
        AffineQuantizationTestCase{
            "ValidPerAxisStaticDim",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/false,
            {0.5f, 0.25f},
            {0, 1},
            /*quantized_dimension=*/0,
        },
        AffineQuantizationTestCase{
            "ValidPerAxisDynamicDim",
            {-1, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/false,
            {0.5f, 0.25f, 0.125f},
            {0, 0, 0},
            /*quantized_dimension=*/0,
        },
        AffineQuantizationTestCase{
            "SingleScaleEmptyZeroPoint",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/false,
            {0.5f},
            {},
            /*quantized_dimension=*/0,
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        AffineQuantizationTestCase{
            "EmptyScaleNonEmptyZeroPoint",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/false,
            {},
            {0},
            /*quantized_dimension=*/0,
            absl::StatusCode::kInvalidArgument,
        },
        AffineQuantizationTestCase{
            "MismatchedScaleAndZeroPointCount",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/false,
            {0.5f, 0.25f},
            {0},
            /*quantized_dimension=*/0,
            absl::StatusCode::kInvalidArgument,
        },
        AffineQuantizationTestCase{
            "NegativeQuantizedDimension",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/false,
            {0.5f},
            {0},
            /*quantized_dimension=*/-1,
            absl::StatusCode::kInvalidArgument,
        },
        AffineQuantizationTestCase{
            "PerAxisCountMismatchStaticDim",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/false,
            {0.5f, 0.25f, 0.125f},
            {0, 0, 0},
            /*quantized_dimension=*/0,
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        AffineQuantizationTestCase{
            "OutOfRangeQuantizedDimension",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_quantization=*/false,
            /*null_vectors=*/false,
            {0.5f},
            {0},
            /*quantized_dimension=*/2,
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        }),
    [](const ::testing::TestParamInfo<AffineQuantizationTestCase>& info) {
      return info.param.test_name;
    });

struct BlockwiseQuantizationTestCase {
  std::string test_name;
  std::vector<int32_t> input_shape;
  tflite::TensorType input_type;
  bool omit_details = false;
  int32_t scales = 2;
  int32_t zero_points = -1;
  int32_t block_size = 32;
  std::vector<int32_t> block_shape = {};
  absl::StatusCode expected_status = absl::StatusCode::kOk;
  bool check_full_verify = false;
};

using BlockwiseQuantizationVerifierTest =
    ::testing::TestWithParam<BlockwiseQuantizationTestCase>;

TEST_P(BlockwiseQuantizationVerifierTest, ValidatesBlockwiseQuantization) {
  const auto& param = GetParam();
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<void> details = 0;
  if (!param.omit_details) {
    auto block_shape_vec =
        param.block_shape.empty() ? 0 : builder.CreateVector(param.block_shape);
    details = tflite::CreateBlockwiseQuantization(
                  builder, param.scales, param.zero_points, param.block_size,
                  block_shape_vec)
                  .Union();
  }
  auto quant = tflite::CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      tflite::QuantizationDetails_BlockwiseQuantization, details,
      /*quantized_dimension=*/1);
  BuildQuantizedReluModel(builder, param.input_shape, param.input_type, quant,
                          /*add_scale_tensor=*/true);
  EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
            param.expected_status);
  if (param.check_full_verify) {
    EXPECT_EQ(Verify(builder.GetBufferPointer(), builder.GetSize()),
              param.expected_status == absl::StatusCode::kOk);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BlockwiseQuantizationVerifierTests, BlockwiseQuantizationVerifierTest,
    ::testing::Values(
        BlockwiseQuantizationTestCase{
            "ValidInt4BlockSize",
            {2, 32},
            tflite::TensorType_INT4,
        },
        BlockwiseQuantizationTestCase{
            "ValidInt2BlockSizeWithZeroPoints",
            {2, 32},
            tflite::TensorType_INT2,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/1,
            /*block_size=*/32,
        },
        BlockwiseQuantizationTestCase{
            "ValidBlockShape",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*block_shape=*/{1, 16},
        },
        BlockwiseQuantizationTestCase{
            "MissingDetailsTable",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/true,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/32,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
        },
        BlockwiseQuantizationTestCase{
            "Int4ZeroBlockSize",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        BlockwiseQuantizationTestCase{
            "Int4NegativeBlockSize",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/-32,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        BlockwiseQuantizationTestCase{
            "Int4OddBlockSize",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/3,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        BlockwiseQuantizationTestCase{
            "UInt4ZeroBlockSize",
            {2, 32},
            tflite::TensorType_UINT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        BlockwiseQuantizationTestCase{
            "UInt4NegativeBlockSize",
            {2, 32},
            tflite::TensorType_UINT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/-32,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        BlockwiseQuantizationTestCase{
            "UInt4OddBlockSize",
            {2, 32},
            tflite::TensorType_UINT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/3,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        BlockwiseQuantizationTestCase{
            "Int2UnalignedBlockSize",
            {2, 36},
            tflite::TensorType_INT2,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/6,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
        },
        BlockwiseQuantizationTestCase{
            "NegativeScalesTensorIndex",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/-1,
            /*zero_points=*/-1,
            /*block_size=*/32,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        BlockwiseQuantizationTestCase{
            "OutOfBoundsScalesTensorIndex",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/99,
            /*zero_points=*/-1,
            /*block_size=*/32,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        BlockwiseQuantizationTestCase{
            "InvalidNegativeZeroPointsTensorIndex",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-2,
            /*block_size=*/32,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
        },
        BlockwiseQuantizationTestCase{
            "OutOfBoundsZeroPointsTensorIndex",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/99,
            /*block_size=*/32,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
        },
        BlockwiseQuantizationTestCase{
            "LastDimNotDivisibleByBlockSize",
            {2, 34},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/32,
            /*block_shape=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        BlockwiseQuantizationTestCase{
            "BlockShapeRankMismatch",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*block_shape=*/{1, 16, 1},
            absl::StatusCode::kInvalidArgument,
        },
        BlockwiseQuantizationTestCase{
            "ZeroBlockShapeDim",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*block_shape=*/{1, 0},
            absl::StatusCode::kInvalidArgument,
        },
        BlockwiseQuantizationTestCase{
            "NegativeBlockShapeDim",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*block_shape=*/{1, -4},
            absl::StatusCode::kInvalidArgument,
        }),
    [](const ::testing::TestParamInfo<BlockwiseQuantizationTestCase>& info) {
      return info.param.test_name;
    });

void BuildBlockwiseEmbeddingLookupModel(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<int32_t>& params_shape,
    const std::vector<int32_t>& params_shape_signature,
    const std::vector<int32_t>& scales_shape,
    const std::vector<int32_t>& scales_shape_signature, int32_t block_size) {
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
  };
  auto bw = tflite::CreateBlockwiseQuantization(builder, /*scales=*/2,
                                                /*zero_points=*/-1, block_size);
  auto quant = tflite::CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      tflite::QuantizationDetails_BlockwiseQuantization, bw.Union(),
      /*quantized_dimension=*/0);
  auto params_sig = params_shape_signature.empty()
                        ? 0
                        : builder.CreateVector(params_shape_signature);
  auto scales_sig = scales_shape_signature.empty()
                        ? 0
                        : builder.CreateVector(scales_shape_signature);
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      tflite::CreateTensor(builder,
                           builder.CreateVector(std::vector<int32_t>{1}),
                           tflite::TensorType_INT32, /*buffer=*/1,
                           builder.CreateString("ids")),
      tflite::CreateTensor(
          builder, builder.CreateVector(params_shape), tflite::TensorType_INT4,
          /*buffer=*/2, builder.CreateString("params"), quant,
          /*is_variable=*/false, /*sparsity=*/0, params_sig),
      tflite::CreateTensor(
          builder, builder.CreateVector(scales_shape),
          tflite::TensorType_FLOAT16, /*buffer=*/3,
          builder.CreateString("scales"), /*quantization=*/0,
          /*is_variable=*/false, /*sparsity=*/0, scales_sig),
      tflite::CreateTensor(builder,
                           builder.CreateVector(
                               std::vector<int32_t>{1, params_shape[1]}),
                           tflite::TensorType_FLOAT32, /*buffer=*/4,
                           builder.CreateString("out")),
  };
  auto opcode = tflite::CreateOperatorCode(
      builder, tflite::BuiltinOperator_EMBEDDING_LOOKUP);
  std::vector<int32_t> op_inputs = {0, 1};
  std::vector<int32_t> op_outputs = {3};
  auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                   builder.CreateVector(op_inputs),
                                   builder.CreateVector(op_outputs));
  std::vector<int32_t> model_inputs = {0, 1};
  std::vector<int32_t> model_outputs = {3};
  auto subgraph = tflite::CreateSubGraph(builder, builder.CreateVector(tensors),
                                         builder.CreateVector(model_inputs),
                                         builder.CreateVector(model_outputs),
                                         builder.CreateVector(&op, 1));
  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&opcode, 1),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("bw_embedding_lookup"),
      builder.CreateVector(buffers));
  tflite::FinishModelBuffer(builder, model);
}

struct BlockwiseEmbeddingLookupTestCase {
  std::string test_name;
  std::vector<int32_t> params_shape;
  std::vector<int32_t> params_shape_signature;
  std::vector<int32_t> scales_shape;
  std::vector<int32_t> scales_shape_signature;
  int32_t block_size;
  absl::StatusCode expected_status;
};

using BlockwiseEmbeddingLookupVerifierTest =
    ::testing::TestWithParam<BlockwiseEmbeddingLookupTestCase>;

TEST_P(BlockwiseEmbeddingLookupVerifierTest,
       ValidatesScalarScaleAndStaticParameterContract) {
  const auto& param = GetParam();
  flatbuffers::FlatBufferBuilder builder;
  BuildBlockwiseEmbeddingLookupModel(
      builder, param.params_shape, param.params_shape_signature,
      param.scales_shape, param.scales_shape_signature, param.block_size);
  EXPECT_EQ(VerifyModel(builder.GetBufferPointer(), builder.GetSize()).code(),
            param.expected_status);
  EXPECT_EQ(Verify(builder.GetBufferPointer(), builder.GetSize()),
            param.expected_status == absl::StatusCode::kOk);
}

INSTANTIATE_TEST_SUITE_P(
    BlockwiseEmbeddingLookupVerifierTests, BlockwiseEmbeddingLookupVerifierTest,
    ::testing::Values(
        BlockwiseEmbeddingLookupTestCase{
            "ScalarScaleSingleBlockAccepted",
            /*params_shape=*/{1, 32},
            /*params_shape_signature=*/{},
            /*scales_shape=*/{},
            /*scales_shape_signature=*/{},
            /*block_size=*/32,
            absl::StatusCode::kOk,
        },
        BlockwiseEmbeddingLookupTestCase{
            "ScalarScaleInsufficientElementsRejected",
            /*params_shape=*/{2, 32},
            /*params_shape_signature=*/{},
            /*scales_shape=*/{},
            /*scales_shape_signature=*/{},
            /*block_size=*/32,
            absl::StatusCode::kInvalidArgument,
        },
        BlockwiseEmbeddingLookupTestCase{
            "DynamicScaleDimensionRejected",
            /*params_shape=*/{2, 32},
            /*params_shape_signature=*/{},
            /*scales_shape=*/{2, 1},
            /*scales_shape_signature=*/{-1, 1},
            /*block_size=*/32,
            absl::StatusCode::kInvalidArgument,
        }),
    [](const ::testing::TestParamInfo<BlockwiseEmbeddingLookupTestCase>& info) {
      return info.param.test_name;
    });

struct MultiAxisQuantizationTestCase {
  std::string test_name;
  std::vector<int32_t> input_shape;
  tflite::TensorType input_type;
  bool omit_details = false;
  int32_t scales = 2;
  int32_t zero_points = -1;
  int32_t block_size = 0;
  std::vector<int32_t> quantized_dimensions = {};
  absl::StatusCode expected_status = absl::StatusCode::kOk;
  bool check_full_verify = false;
};

using MultiAxisQuantizationVerifierTest =
    ::testing::TestWithParam<MultiAxisQuantizationTestCase>;

TEST_P(MultiAxisQuantizationVerifierTest, ValidatesMultiAxisQuantization) {
  const auto& param = GetParam();
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<void> details = 0;
  if (!param.omit_details) {
    details = tflite::CreateMultiAxisQuantization(
                  builder, param.scales, param.zero_points, param.block_size,
                  builder.CreateVector(param.quantized_dimensions))
                  .Union();
  }
  auto quant = tflite::CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      tflite::QuantizationDetails_MultiAxisQuantization, details,
      /*quantized_dimension=*/0);
  BuildQuantizedReluModel(builder, param.input_shape, param.input_type, quant,
                          /*add_scale_tensor=*/true);
  EXPECT_EQ(ValidateFirstTensorQuantization(builder).code(),
            param.expected_status);
  if (param.check_full_verify) {
    EXPECT_EQ(Verify(builder.GetBufferPointer(), builder.GetSize()),
              param.expected_status == absl::StatusCode::kOk);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MultiAxisQuantizationVerifierTests, MultiAxisQuantizationVerifierTest,
    ::testing::Values(
        MultiAxisQuantizationTestCase{
            "ValidPerChannel",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*quantized_dimensions=*/{0, 1},
            absl::StatusCode::kOk,
            /*check_full_verify=*/true,
        },
        MultiAxisQuantizationTestCase{
            "ValidBlockwiseWithZeroPoints",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/1,
            /*block_size=*/32,
            /*quantized_dimensions=*/{0, 1},
        },
        MultiAxisQuantizationTestCase{
            "MissingDetailsTable",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/true,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*quantized_dimensions=*/{},
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        MultiAxisQuantizationTestCase{
            "EmptyQuantizedDimensions",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*quantized_dimensions=*/{},
            absl::StatusCode::kInvalidArgument,
        },
        MultiAxisQuantizationTestCase{
            "NegativeScalesTensorIndex",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/false,
            /*scales=*/-1,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*quantized_dimensions=*/{0},
            absl::StatusCode::kInvalidArgument,
        },
        MultiAxisQuantizationTestCase{
            "OutOfBoundsScalesTensorIndex",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/false,
            /*scales=*/99,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*quantized_dimensions=*/{0},
            absl::StatusCode::kInvalidArgument,
        },
        MultiAxisQuantizationTestCase{
            "InvalidNegativeZeroPointsTensorIndex",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-2,
            /*block_size=*/0,
            /*quantized_dimensions=*/{0},
            absl::StatusCode::kInvalidArgument,
        },
        MultiAxisQuantizationTestCase{
            "OutOfBoundsZeroPointsTensorIndex",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/99,
            /*block_size=*/0,
            /*quantized_dimensions=*/{0},
            absl::StatusCode::kInvalidArgument,
        },
        MultiAxisQuantizationTestCase{
            "NegativeBlockSize",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/-1,
            /*quantized_dimensions=*/{0, 1},
            absl::StatusCode::kInvalidArgument,
        },
        MultiAxisQuantizationTestCase{
            "UnalignedInt4BlockSize",
            {2, 32},
            tflite::TensorType_INT4,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/3,
            /*quantized_dimensions=*/{0, 1},
            absl::StatusCode::kInvalidArgument,
        },
        MultiAxisQuantizationTestCase{
            "NegativeQuantizedDimensionEntry",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*quantized_dimensions=*/{0, -1},
            absl::StatusCode::kInvalidArgument,
        },
        MultiAxisQuantizationTestCase{
            "OutOfRangeQuantizedDimensionEntry",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*quantized_dimensions=*/{0, 2},
            absl::StatusCode::kInvalidArgument,
        },
        MultiAxisQuantizationTestCase{
            "DuplicateQuantizedDimensionEntry",
            {2, 4},
            tflite::TensorType_INT8,
            /*omit_details=*/false,
            /*scales=*/2,
            /*zero_points=*/-1,
            /*block_size=*/0,
            /*quantized_dimensions=*/{0, 0},
            absl::StatusCode::kInvalidArgument,
        }),
    [](const ::testing::TestParamInfo<MultiAxisQuantizationTestCase>& info) {
      return info.param.test_name;
    });

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

struct TensorElementCountOverflowCase {
  std::string test_name;
  std::vector<int32_t> input_shape;
  std::vector<int64_t> begin_vals;
  std::vector<int64_t> size_vals;
};

using TensorElementCountOverflowTest =
    ::testing::TestWithParam<TensorElementCountOverflowCase>;

TEST_P(TensorElementCountOverflowTest,
       RejectsTensorWithElementCountExceedingInt32Max) {
  const auto& param = GetParam();
  flatbuffers::FlatBufferBuilder builder;
  BuildSliceModel(builder, param.input_shape, tflite::TensorType_INT32,
                  param.begin_vals, tflite::TensorType_INT32, param.size_vals);
  const tflite::Model* tfl_model = tflite::GetModel(builder.GetBufferPointer());
  EXPECT_EQ(internal::ValidateFlatBufferTensors(tfl_model).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(VerifyModel(builder.GetBufferPointer(), builder.GetSize()).code(),
            absl::StatusCode::kInvalidArgument);
}

constexpr int32_t kHugeDim = 46341;  // 46341 * 46341 > INT32_MAX

INSTANTIATE_TEST_SUITE_P(
    TensorElementCountOverflowTests, TensorElementCountOverflowTest,
    ::testing::Values(
        TensorElementCountOverflowCase{
            "StaticShapeExceedingInt32Max",
            {kHugeDim, kHugeDim},
            {0, 0},
            {1, 1},
        },
        TensorElementCountOverflowCase{
            "PartiallyDynamicShapeExceedingInt32Max",
            {-1, kHugeDim, kHugeDim},
            {0, 0, 0},
            {1, 1, 1},
        }),
    [](const ::testing::TestParamInfo<TensorElementCountOverflowCase>& info) {
      return info.param.test_name;
    });

void BuildOperatorCodeModel(flatbuffers::FlatBufferBuilder& builder,
                            int8_t deprecated_builtin_code,
                            tflite::BuiltinOperator builtin_code) {
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
  };
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      tflite::CreateTensor(
          builder, builder.CreateVector(std::vector<int32_t>{2}),
          tflite::TensorType_FLOAT32, /*buffer=*/0, builder.CreateString("in")),
      tflite::CreateTensor(builder,
                           builder.CreateVector(std::vector<int32_t>{2}),
                           tflite::TensorType_FLOAT32, /*buffer=*/1,
                           builder.CreateString("out")),
  };
  auto opcode = tflite::CreateOperatorCode(builder, deprecated_builtin_code,
                                           /*custom_code=*/0, /*version=*/1,
                                           builtin_code);
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
      builder.CreateVector(&subgraph, 1), builder.CreateString("opcode_test"),
      builder.CreateVector(buffers));
  tflite::FinishModelBuffer(builder, model);
}

struct OperatorCodeVerifierCase {
  std::string test_name;
  int8_t deprecated_builtin_code;
  tflite::BuiltinOperator builtin_code;
  absl::StatusCode expected_status;
  bool check_full_verify = false;
};

using OperatorCodeVerifierTest =
    ::testing::TestWithParam<OperatorCodeVerifierCase>;

TEST_P(OperatorCodeVerifierTest, ValidatesFlatBufferOperatorCode) {
  const auto& param = GetParam();
  flatbuffers::FlatBufferBuilder builder;
  BuildOperatorCodeModel(builder, param.deprecated_builtin_code,
                         param.builtin_code);
  const tflite::Model* tfl_model = tflite::GetModel(builder.GetBufferPointer());
  EXPECT_EQ(internal::ValidateFlatBufferOperatorCodes(tfl_model).code(),
            param.expected_status);
  if (param.check_full_verify) {
    EXPECT_EQ(VerifyModel(builder.GetBufferPointer(), builder.GetSize()).code(),
              param.expected_status);
    EXPECT_EQ(Verify(builder.GetBufferPointer(), builder.GetSize()),
              param.expected_status == absl::StatusCode::kOk);
  }
}

INSTANTIATE_TEST_SUITE_P(
    OperatorCodeVerifierTests, OperatorCodeVerifierTest,
    ::testing::Values(
        OperatorCodeVerifierCase{
            "NegativeDeprecatedBuiltinCode",
            /*deprecated_builtin_code=*/-1,
            static_cast<tflite::BuiltinOperator>(0),
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        OperatorCodeVerifierCase{
            "NegativeBuiltinCode",
            /*deprecated_builtin_code=*/0,
            static_cast<tflite::BuiltinOperator>(-5),
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        OperatorCodeVerifierCase{
            "BuiltinCodeExceedingMax",
            /*deprecated_builtin_code=*/0,
            static_cast<tflite::BuiltinOperator>(300),
            absl::StatusCode::kInvalidArgument,
            /*check_full_verify=*/true,
        },
        OperatorCodeVerifierCase{
            "ValidWrappedInt8DeprecatedCode",
            static_cast<int8_t>(tflite::BuiltinOperator_BROADCAST_TO),
            tflite::BuiltinOperator_BROADCAST_TO,
            absl::StatusCode::kOk,
        },
        OperatorCodeVerifierCase{
            "ValidPlaceholderDeprecatedCode",
            static_cast<int8_t>(
                tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES),
            tflite::BuiltinOperator_BROADCAST_TO,
            absl::StatusCode::kOk,
        }),
    [](const ::testing::TestParamInfo<OperatorCodeVerifierCase>& info) {
      return info.param.test_name;
    });

TEST(VerifierTest, ValidateFlatBufferOperatorCodesHandlesNullAndEmptyModels) {
  EXPECT_EQ(internal::ValidateFlatBufferOperatorCodes(nullptr).code(),
            absl::StatusCode::kInvalidArgument);

  flatbuffers::FlatBufferBuilder builder;
  auto model = tflite::CreateModel(builder, TFLITE_SCHEMA_VERSION);
  tflite::FinishModelBuffer(builder, model);
  const tflite::Model* tfl_model = tflite::GetModel(builder.GetBufferPointer());
  EXPECT_TRUE(internal::ValidateFlatBufferOperatorCodes(tfl_model).ok());
}

using ZeroOutputGatherVerifierTest =
    ::testing::TestWithParam<tflite::BuiltinOperator>;

TEST_P(ZeroOutputGatherVerifierTest, RejectsZeroOutputWithoutCrash) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
  };
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      tflite::CreateTensor(builder,
                           builder.CreateVector(std::vector<int32_t>{2, 2}),
                           tflite::TensorType_FLOAT32, /*buffer=*/0,
                           builder.CreateString("params")),
      tflite::CreateTensor(builder,
                           builder.CreateVector(std::vector<int32_t>{2, 1}),
                           tflite::TensorType_INT32, /*buffer=*/1,
                           builder.CreateString("indices")),
  };
  auto opcode = tflite::CreateOperatorCode(builder, GetParam());
  std::vector<int32_t> op_inputs = {0, 1};
  std::vector<int32_t> op_outputs = {};
  auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                   builder.CreateVector(op_inputs),
                                   builder.CreateVector(op_outputs));
  std::vector<int32_t> model_inputs = {0, 1};
  std::vector<int32_t> model_outputs = {};
  auto subgraph = tflite::CreateSubGraph(builder, builder.CreateVector(tensors),
                                         builder.CreateVector(model_inputs),
                                         builder.CreateVector(model_outputs),
                                         builder.CreateVector(&op, 1));
  auto model = tflite::CreateModel(builder, TFLITE_SCHEMA_VERSION,
                                   builder.CreateVector(&opcode, 1),
                                   builder.CreateVector(&subgraph, 1),
                                   builder.CreateString("zero_output_gather"),
                                   builder.CreateVector(buffers));
  tflite::FinishModelBuffer(builder, model);

  EXPECT_EQ(VerifyModel(builder.GetBufferPointer(), builder.GetSize()).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
}

INSTANTIATE_TEST_SUITE_P(
    ZeroOutputGatherVerifierTests, ZeroOutputGatherVerifierTest,
    ::testing::Values(tflite::BuiltinOperator_GATHER,
                      tflite::BuiltinOperator_GATHER_ND),
    [](const ::testing::TestParamInfo<tflite::BuiltinOperator>& info) {
      return tflite::EnumNameBuiltinOperator(info.param);
    });

TEST(VerifierTest, RejectsDepthToSpaceWithOverflowingBlockSize) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
  };
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      tflite::CreateTensor(
          builder, builder.CreateVector(std::vector<int32_t>{1, 4, 4, 16}),
          tflite::TensorType_FLOAT32, /*buffer=*/0, builder.CreateString("in")),
      tflite::CreateTensor(builder,
                           builder.CreateVector(std::vector<int32_t>{}),
                           tflite::TensorType_FLOAT32, /*buffer=*/1,
                           builder.CreateString("out")),
  };
  auto opcode = tflite::CreateOperatorCode(
      builder, tflite::BuiltinOperator_DEPTH_TO_SPACE);
  std::vector<int32_t> op_inputs = {0};
  std::vector<int32_t> op_outputs = {1};
  // 225542 * 225542 overflows signed int32_t.
  auto d2s_opts =
      tflite::CreateDepthToSpaceOptions(builder, /*block_size=*/225542);
  auto op = tflite::CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector(op_inputs),
      builder.CreateVector(op_outputs),
      tflite::BuiltinOptions_DepthToSpaceOptions, d2s_opts.Union());
  std::vector<int32_t> model_inputs = {0};
  std::vector<int32_t> model_outputs = {1};
  auto subgraph = tflite::CreateSubGraph(builder, builder.CreateVector(tensors),
                                         builder.CreateVector(model_inputs),
                                         builder.CreateVector(model_outputs),
                                         builder.CreateVector(&op, 1));
  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&opcode, 1),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("d2s_overflow_test"), builder.CreateVector(buffers));
  tflite::FinishModelBuffer(builder, model);

  EXPECT_EQ(VerifyModel(builder.GetBufferPointer(), builder.GetSize()).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
}

}  // namespace
}  // namespace litert
