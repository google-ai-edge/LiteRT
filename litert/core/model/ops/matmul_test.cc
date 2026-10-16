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

#include "litert/core/model/ops/matmul.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

// Helper to create BatchMatMul options
std::unique_ptr<tflite::BatchMatMulOptionsT> CreateBatchMatMulOptions(
    bool adj_x, bool adj_y, bool asymmetric_quantize_inputs = false) {
  auto options = std::make_unique<tflite::BatchMatMulOptionsT>();
  options->adj_x = adj_x;
  options->adj_y = adj_y;
  options->asymmetric_quantize_inputs = asymmetric_quantize_inputs;
  return options;
}

LiteRtOpT MakeBatchMatMulOp(bool adj_x, bool adj_y,
                            bool asymmetric_quantize_inputs = false) {
  LiteRtOpT op;
  auto options =
      CreateBatchMatMulOptions(adj_x, adj_y, asymmetric_quantize_inputs);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  return op;
}

// Helper to create FullyConnected options
std::unique_ptr<tflite::FullyConnectedOptionsT> CreateFullyConnectedOptions(
    bool keep_num_dims,
    tflite::FullyConnectedOptionsWeightsFormat weights_format =
        tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
    tflite::ActivationFunctionType fused_activation_function =
        tflite::ActivationFunctionType_NONE) {
  auto options = std::make_unique<tflite::FullyConnectedOptionsT>();
  options->keep_num_dims = keep_num_dims;
  options->weights_format = weights_format;
  options->fused_activation_function = fused_activation_function;
  return options;
}

LiteRtOpT MakeFullyConnectedOp(
    bool keep_num_dims,
    tflite::FullyConnectedOptionsWeightsFormat weights_format =
        tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
    tflite::ActivationFunctionType fused_activation_function =
        tflite::ActivationFunctionType_NONE) {
  LiteRtOpT op;
  auto options = CreateFullyConnectedOptions(keep_num_dims, weights_format,
                                             fused_activation_function);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_FullyConnectedOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  return op;
}

//
// ComputeStaticElementCount Tests
//

struct ComputeStaticElementCountCase {
  std::string test_name;
  Dims dims;
  bool expected_ok;
  int32_t expected_num_elements = 0;
  bool expected_has_dynamic_dim = false;
};

using ComputeStaticElementCountTest =
    ::testing::TestWithParam<ComputeStaticElementCountCase>;

TEST_P(ComputeStaticElementCountTest, ComputesExpectedCountAndFlags) {
  const ComputeStaticElementCountCase& tc = GetParam();
  int32_t num_elements = 0;
  bool has_dynamic_dim = false;
  const bool ok =
      ComputeStaticElementCount(tc.dims, num_elements, has_dynamic_dim);
  ASSERT_EQ(ok, tc.expected_ok);
  if (tc.expected_ok) {
    EXPECT_EQ(num_elements, tc.expected_num_elements);
    EXPECT_EQ(has_dynamic_dim, tc.expected_has_dynamic_dim);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ComputeStaticElementCountTests, ComputeStaticElementCountTest,
    ::testing::Values(
        ComputeStaticElementCountCase{
            "ScalarEmptyDims",
            /*dims=*/{},
            /*expected_ok=*/true,
            /*expected_num_elements=*/1,
            /*expected_has_dynamic_dim=*/false,
        },
        ComputeStaticElementCountCase{
            "StaticPositiveDims",
            /*dims=*/{2, 3, 4},
            /*expected_ok=*/true,
            /*expected_num_elements=*/24,
            /*expected_has_dynamic_dim=*/false,
        },
        ComputeStaticElementCountCase{
            "ZeroDim",
            /*dims=*/{2, 0, 4},
            /*expected_ok=*/true,
            /*expected_num_elements=*/0,
            /*expected_has_dynamic_dim=*/false,
        },
        ComputeStaticElementCountCase{
            "DynamicDim",
            /*dims=*/{2, -1, 4},
            /*expected_ok=*/true,
            /*expected_num_elements=*/8,
            /*expected_has_dynamic_dim=*/true,
        },
        ComputeStaticElementCountCase{
            "InvalidNegativeDim",
            /*dims=*/{2, -2, 4},
            /*expected_ok=*/false,
        },
        ComputeStaticElementCountCase{
            "StaticOverflow",
            /*dims=*/{65536, 32768},
            /*expected_ok=*/false,
        },
        ComputeStaticElementCountCase{
            "DynamicWithStaticOverflow",
            /*dims=*/{-1, 65536, 32768},
            /*expected_ok=*/false,
        }),
    [](const ::testing::TestParamInfo<ComputeStaticElementCountCase>& info) {
      return info.param.test_name;
    });

//
// BatchMatMul Tests
//

TEST(BatchMatMulOpTest, SimpleMatMul) {
  LiteRtOpT op = MakeBatchMatMulOp(false, false);
  // LHS [1, 2, 3], RHS [1, 3, 4] -> [1, 2, 4]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {1, 3, 4}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

TEST(BatchMatMulOpTest, MatMulBatchSizeTwo) {
  LiteRtOpT op = MakeBatchMatMulOp(false, false);
  // LHS [2, 2, 3], RHS [2, 3, 4] -> [2, 2, 4]
  std::vector<Dims> input_shapes = {{2, 2, 3}, {2, 3, 4}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2, 4));
}

TEST(BatchMatMulOpTest, BroadcastRankDiff) {
  LiteRtOpT op = MakeBatchMatMulOp(false, false);
  // LHS [2, 2, 3], RHS [3, 4] -> [1, 3, 4] (RHS broadcast to [1, 3, 4])
  // Batch: [2] vs [1] -> [2]
  // MatMul: [2, 3] x [3, 4] -> [2, 4]
  // Output: [2, 2, 4]
  std::vector<Dims> input_shapes = {{2, 2, 3}, {3, 4}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2, 4));
}

TEST(BatchMatMulOpTest, BroadcastRank4) {
  LiteRtOpT op = MakeBatchMatMulOp(false, false);
  // LHS [2, 1, 3, 2], RHS [3, 2, 4] -> [1, 3, 2, 4] (RHS broadcast)
  // Batch: [2, 1] vs [1, 3] -> [2, 3]
  // MatMul: [3, 2] x [2, 4] -> [3, 4]
  // Output: [2, 3, 3, 4]
  std::vector<Dims> input_shapes = {{2, 1, 3, 2}, {3, 2, 4}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 3, 3, 4));
}

TEST(BatchMatMulOpTest, AdjointLHS) {
  LiteRtOpT op = MakeBatchMatMulOp(true, false);
  // LHS [1, 3, 2] (adj) -> [1, 2, 3], RHS [1, 3, 4]
  // Output: [1, 2, 4]
  std::vector<Dims> input_shapes = {{1, 3, 2}, {1, 3, 4}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

TEST(BatchMatMulOpTest, AdjointRHS) {
  LiteRtOpT op = MakeBatchMatMulOp(false, true);
  // LHS [1, 2, 3], RHS [1, 4, 3] (adj) -> [1, 3, 4]
  // Output: [1, 2, 4]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {1, 4, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

TEST(BatchMatMulOpTest, AdjointBoth) {
  LiteRtOpT op = MakeBatchMatMulOp(true, true);
  // LHS [1, 3, 2] (adj) -> [1, 2, 3]
  // RHS [1, 4, 3] (adj) -> [1, 3, 4]
  // Output: [1, 2, 4]
  std::vector<Dims> input_shapes = {{1, 3, 2}, {1, 4, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

TEST(BatchMatMulOpTest, AsymmetricQuantizeInputs) {
  LiteRtOpT op = MakeBatchMatMulOp(false, false, true);
  // LHS [1, 2, 3], RHS [1, 3, 4] -> [1, 2, 4]
  // Option asymmetric_quantize_inputs should not affect shape.
  std::vector<Dims> input_shapes = {{1, 2, 3}, {1, 3, 4}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

//
// FullyConnected Tests
//

TEST(FullyConnectedOpTest, Simple) {
  LiteRtOpT op = MakeFullyConnectedOp(false);
  // Input [2, 10], Weights [20, 10] -> [2, 20]
  std::vector<Dims> input_shapes = {{2, 10}, {20, 10}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 20));
}

TEST(FullyConnectedOpTest, ScalarInputKeepNumDimsFalse) {
  LiteRtOpT op = MakeFullyConnectedOp(/*keep_num_dims=*/false);
  // Scalar input [], Weights [2, 1] -> [1, 2]
  std::vector<Dims> input_shapes = {{}, {2, 1}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2));
}

TEST(FullyConnectedOpTest, RejectsScalarInputWithKeepNumDimsTrue) {
  LiteRtOpT op = MakeFullyConnectedOp(/*keep_num_dims=*/true);
  std::vector<Dims> input_shapes = {{}, {2, 1}};
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusErrorInvalidArgument);
}

TEST(FullyConnectedOpTest, KeepNumDims) {
  LiteRtOpT op = MakeFullyConnectedOp(true);
  // Input [1, 2, 1, 10], Weights [3, 10]
  // Output [1, 2, 1, 3] (Input dims preserved)
  std::vector<Dims> input_shapes = {{1, 2, 1, 10}, {3, 10}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 1, 3));
}

TEST(FullyConnectedOpTest, FlattenRank3) {
  LiteRtOpT op = MakeFullyConnectedOp(false);
  // Input [2, 3, 10], Weights [5, 10]
  // keep_num_dims=false -> Flattens to [Batch, InputDim]
  // Batch = 2*3 = 6.
  // Output: [6, 5]
  std::vector<Dims> input_shapes = {{2, 3, 10}, {5, 10}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(6, 5));
}

TEST(FullyConnectedOpTest, FlattenRank4) {
  LiteRtOpT op = MakeFullyConnectedOp(false);
  // Input [2, 1, 3, 10], Weights [5, 10]
  // Batch = 2*1*3 = 6.
  // Output: [6, 5]
  std::vector<Dims> input_shapes = {{2, 1, 3, 10}, {5, 10}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(6, 5));
}

TEST(FullyConnectedOpTest, ShuffledWeightsFormat) {
  // Input [2, 16], Weights [4, 16].
  // weights_format = SHUFFLED4x16INT8
  // This format requires input_depth % 16 == 0 and output_depth % 4 == 0.
  // Our inference logic treats shape as standard [Out, In] for now.
  // Testing that inference still works with this option set.
  LiteRtOpT op = MakeFullyConnectedOp(
      false, tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8);
  std::vector<Dims> input_shapes = {{2, 16}, {4, 16}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 4));
}

TEST(FullyConnectedOpTest, WithFusedActivation) {
  // Input [2, 10], Weights [20, 10]
  // fused_activation_function should not affect shape.
  LiteRtOpT op = MakeFullyConnectedOp(
      false, tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
      tflite::ActivationFunctionType_RELU);
  std::vector<Dims> input_shapes = {{2, 10}, {20, 10}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 20));
}

struct FullyConnectedErrorCase {
  std::string name;
  std::vector<Dims> input_shapes;
  size_t num_outputs = 1;
};

class FullyConnectedErrorTest
    : public ::testing::TestWithParam<FullyConnectedErrorCase> {};

TEST_P(FullyConnectedErrorTest, RejectsInvalidInputAndBatchShapes) {
  const auto& tc = GetParam();
  LiteRtOpT op = MakeFullyConnectedOp(false);
  std::vector<Dims> output_shapes(tc.num_outputs);
  EXPECT_EQ(
      InferFullyConnected(op, absl::MakeSpan(tc.input_shapes), output_shapes),
      kLiteRtStatusErrorInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    FullyConnectedErrorCases, FullyConnectedErrorTest,
    ::testing::Values(
        FullyConnectedErrorCase{"EmptyOutputShapes", {{2, 10}, {20, 10}}, 0},
        FullyConnectedErrorCase{"EmptyInputRank0", {{}, {20, 10}}},
        FullyConnectedErrorCase{"NegativeBatchDim", {{-2, 3, 10}, {20, 10}}},
        FullyConnectedErrorCase{"NegativeWeightDim", {{2, 10}, {-2, 10}}},
        FullyConnectedErrorCase{"InputElementsNotDivisibleByFilterCols",
                                {{2, 7}, {4, 10}}},
        FullyConnectedErrorCase{"BiasSizeMismatch", {{2, 10}, {4, 10}, {3}}},
        FullyConnectedErrorCase{"NegativeBiasDim", {{2, 10}, {4, 10}, {-2}}},
        FullyConnectedErrorCase{"FlattenedBatchOverflow",
                                {{65536, 32768, 10}, {20, 10}}}),
    [](const ::testing::TestParamInfo<FullyConnectedErrorCase>& info) {
      return info.param.name;
    });

TEST(FullyConnectedOpTest, RejectsKeepNumDimsLastDimMismatch) {
  LiteRtOpT op = MakeFullyConnectedOp(/*keep_num_dims=*/true);
  std::vector<Dims> input_shapes = {{2, 5, 4}, {8, 10}};
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusErrorInvalidArgument);
}

struct PerChannelWeightsQuantizationCase {
  std::string test_name;
  int32_t quantized_dimension;
  size_t num_channels;
  LiteRtStatus expected_status;
};

using FullyConnectedPerChannelQuantizationTest =
    ::testing::TestWithParam<PerChannelWeightsQuantizationCase>;

TEST_P(FullyConnectedPerChannelQuantizationTest,
       ValidatesPerChannelWeightsQuantization) {
  const PerChannelWeightsQuantizationCase& param = GetParam();
  LiteRtOpT op = MakeFullyConnectedOp(/*keep_num_dims=*/false);
  LiteRtTensorT input_tensor;
  LiteRtTensorT weights_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&weights_tensor);

  Quantization per_channel_qparams;
  per_channel_qparams.first = kLiteRtQuantizationPerChannel;
  per_channel_qparams.second.per_channel.quantized_dimension =
      param.quantized_dimension;
  per_channel_qparams.second.per_channel.num_channels = param.num_channels;
  weights_tensor.SetQarams(per_channel_qparams);

  std::vector<Dims> input_shapes = {{2, 10}, {4, 10}};
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      param.expected_status);
}

INSTANTIATE_TEST_SUITE_P(
    FullyConnectedPerChannelQuantizationTests,
    FullyConnectedPerChannelQuantizationTest,
    ::testing::Values(
        PerChannelWeightsQuantizationCase{
            "InvalidQuantizedDimension",
            /*quantized_dimension=*/1,
            /*num_channels=*/4,
            kLiteRtStatusErrorInvalidArgument,
        },
        PerChannelWeightsQuantizationCase{
            "InvalidNumChannels",
            /*quantized_dimension=*/0,
            /*num_channels=*/2,
            kLiteRtStatusErrorInvalidArgument,
        },
        PerChannelWeightsQuantizationCase{
            "ValidPerChannelQuantization",
            /*quantized_dimension=*/0,
            /*num_channels=*/4,
            kLiteRtStatusOk,
        }),
    [](const ::testing::TestParamInfo<PerChannelWeightsQuantizationCase>&
           info) { return info.param.test_name; });

TEST(BatchMatMulOpTest, RejectsEmptyOutputShapes) {
  LiteRtOpT op = MakeBatchMatMulOp(false, false);
  std::vector<Dims> empty_outputs;
  std::vector<Dims> valid_inputs = {{2, 10}, {20, 10}};
  EXPECT_EQ(InferBatchMatmul(op, absl::MakeSpan(valid_inputs), empty_outputs),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(BatchMatMulOpTest, RejectsRank1Input) {
  LiteRtOpT op = MakeBatchMatMulOp(/*adj_x=*/false, /*adj_y=*/false);
  std::vector<Dims> output_shapes(1);
  std::vector<Dims> rank1_inputs = {{3}, {3, 4}};
  EXPECT_EQ(InferBatchMatmul(op, absl::MakeSpan(rank1_inputs), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(BatchMatMulOpTest, SupportsRank6Input) {
  LiteRtOpT op = MakeBatchMatMulOp(/*adj_x=*/false, /*adj_y=*/false);
  std::vector<Dims> output_shapes(1);
  std::vector<Dims> rank6_inputs = {{1, 1, 1, 1, 2, 3}, {1, 1, 1, 1, 3, 4}};
  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(rank6_inputs), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 1, 1, 2, 4));
}

struct BatchMatMulPerChannelRhsCase {
  std::string test_name;
  bool adj_y;
  Dims rhs_shape;
  int32_t quantized_dimension;
  size_t num_channels;
  LiteRtStatus expected_status;
};

using BatchMatMulPerChannelRhsTest =
    ::testing::TestWithParam<BatchMatMulPerChannelRhsCase>;

TEST_P(BatchMatMulPerChannelRhsTest, ValidatesPerChannelRhsQuantization) {
  const BatchMatMulPerChannelRhsCase& param = GetParam();
  LiteRtOpT op = MakeBatchMatMulOp(/*adj_x=*/false, param.adj_y);
  LiteRtTensorT lhs_tensor;
  LiteRtTensorT rhs_tensor;
  op.Inputs().push_back(&lhs_tensor);
  op.Inputs().push_back(&rhs_tensor);

  Quantization per_channel_qparams;
  per_channel_qparams.first = kLiteRtQuantizationPerChannel;
  per_channel_qparams.second.per_channel.quantized_dimension =
      param.quantized_dimension;
  per_channel_qparams.second.per_channel.num_channels = param.num_channels;
  rhs_tensor.SetQarams(per_channel_qparams);

  std::vector<Dims> input_shapes = {{2, 3}, param.rhs_shape};
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            param.expected_status);
}

INSTANTIATE_TEST_SUITE_P(
    BatchMatMulPerChannelRhsTests, BatchMatMulPerChannelRhsTest,
    ::testing::Values(
        BatchMatMulPerChannelRhsCase{
            "NonAdjointInvalidQuantizedDimension",
            /*adj_y=*/false,
            /*rhs_shape=*/{3, 4},
            /*quantized_dimension=*/0,
            /*num_channels=*/4,
            kLiteRtStatusErrorInvalidArgument,
        },
        BatchMatMulPerChannelRhsCase{
            "NonAdjointInvalidNumChannels",
            /*adj_y=*/false,
            /*rhs_shape=*/{3, 4},
            /*quantized_dimension=*/1,
            /*num_channels=*/3,
            kLiteRtStatusErrorInvalidArgument,
        },
        BatchMatMulPerChannelRhsCase{
            "NonAdjointValidPerChannel",
            /*adj_y=*/false,
            /*rhs_shape=*/{3, 4},
            /*quantized_dimension=*/1,
            /*num_channels=*/4,
            kLiteRtStatusOk,
        },
        BatchMatMulPerChannelRhsCase{
            "AdjointInvalidQuantizedDimension",
            /*adj_y=*/true,
            /*rhs_shape=*/{4, 3},
            /*quantized_dimension=*/1,
            /*num_channels=*/4,
            kLiteRtStatusErrorInvalidArgument,
        },
        BatchMatMulPerChannelRhsCase{
            "AdjointValidPerChannel",
            /*adj_y=*/true,
            /*rhs_shape=*/{4, 3},
            /*quantized_dimension=*/0,
            /*num_channels=*/4,
            kLiteRtStatusOk,
        }),
    [](const ::testing::TestParamInfo<BatchMatMulPerChannelRhsCase>& info) {
      return info.param.test_name;
    });

// Runs InferFullyConnected for `input` x `weights` with `keep_num_dims`.
LiteRtStatus InferFc(const Dims& input, const Dims& weights,
                     bool keep_num_dims, Dims& output) {
  LiteRtOpT op = MakeFullyConnectedOp(keep_num_dims);
  std::vector<Dims> input_shapes = {input, weights};
  std::vector<Dims> output_shapes(1);
  const LiteRtStatus status =
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes);
  output = output_shapes[0];
  return status;
}

TEST(FullyConnectedOpTest, FlattensByWeightsDepthNotLastInputDim) {
  // TFLite: batch = num_elements(input) / weights.dims[1] = 12 / 12 = 1. The
  // last input dim (3) is not K.
  Dims output;
  ASSERT_EQ(InferFc({1, 2, 2, 3}, {5, 12}, /*keep_num_dims=*/false, output),
            kLiteRtStatusOk);
  EXPECT_THAT(output, ElementsAre(1, 5));
}

TEST(FullyConnectedOpTest, FlattenRegroupsRows) {
  // 4 * 6 = 24 elements regrouped into rows of K = 8: batch 3.
  Dims output;
  ASSERT_EQ(InferFc({4, 6}, {2, 8}, /*keep_num_dims=*/false, output),
            kLiteRtStatusOk);
  EXPECT_THAT(output, ElementsAre(3, 2));
}

TEST(FullyConnectedOpTest, RejectsBatchWhenInputNotDivisibleByDepth) {
  // TFLite's Prepare requires input_size % filter->dims->data[1] == 0.
  Dims output;
  EXPECT_EQ(InferFc({1, 2, 2, 3}, {4, 5}, /*keep_num_dims=*/false, output),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(FullyConnectedOpTest, RejectsElementCountAboveInt32Max) {
  // TFLite's Prepare accumulates the input element count in an `int`.
  Dims output;
  EXPECT_EQ(InferFc({65536, 65536}, {3, 2}, /*keep_num_dims=*/false, output),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(FullyConnectedOpTest, KeepNumDimsWithDynamicDims) {
  Dims output;
  ASSERT_EQ(InferFc({-1, 7, 3}, {5, 3}, /*keep_num_dims=*/true, output),
            kLiteRtStatusOk);
  EXPECT_THAT(output, ElementsAre(-1, 7, 5));
}

TEST(FullyConnectedOpTest, FlattenWithDynamicDimHasDynamicBatch) {
  Dims output;
  ASSERT_EQ(InferFc({-1, 2, 3}, {5, 6}, /*keep_num_dims=*/false, output),
            kLiteRtStatusOk);
  EXPECT_THAT(output, ElementsAre(-1, 5));
}

TEST(FullyConnectedOpTest, ZeroDepthUsesLeadingDims) {
  // TFLite special-cases weights.dims[1] == 0 to avoid dividing by zero.
  Dims output;
  ASSERT_EQ(InferFc({2, 3, 4}, {5, 0}, /*keep_num_dims=*/false, output),
            kLiteRtStatusOk);
  EXPECT_THAT(output, ElementsAre(6, 5));
  ASSERT_EQ(InferFc({2, 3, 4}, {5, 0}, /*keep_num_dims=*/true, output),
            kLiteRtStatusOk);
  EXPECT_THAT(output, ElementsAre(2, 3, 5));
}
}  // namespace
}  // namespace litert::internal
