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

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

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

//
// BatchMatMul Tests
//

TEST(BatchMatMulOpTest, SimpleMatMul) {
  LiteRtOpT op;
  // LHS [1, 2, 3], RHS [1, 3, 4] -> [1, 2, 4]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {1, 3, 4}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateBatchMatMulOptions(false, false);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

TEST(BatchMatMulOpTest, MatMulBatchSizeTwo) {
  LiteRtOpT op;
  // LHS [2, 2, 3], RHS [2, 3, 4] -> [2, 2, 4]
  std::vector<Dims> input_shapes = {{2, 2, 3}, {2, 3, 4}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateBatchMatMulOptions(false, false);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2, 4));
}

TEST(BatchMatMulOpTest, BroadcastRankDiff) {
  LiteRtOpT op;
  // LHS [2, 2, 3], RHS [3, 4] -> [1, 3, 4] (RHS broadcast to [1, 3, 4])
  // Batch: [2] vs [1] -> [2]
  // MatMul: [2, 3] x [3, 4] -> [2, 4]
  // Output: [2, 2, 4]
  std::vector<Dims> input_shapes = {{2, 2, 3}, {3, 4}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateBatchMatMulOptions(false, false);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2, 4));
}

TEST(BatchMatMulOpTest, BroadcastRank4) {
  LiteRtOpT op;
  // LHS [2, 1, 3, 2], RHS [3, 2, 4] -> [1, 3, 2, 4] (RHS broadcast)
  // Batch: [2, 1] vs [1, 3] -> [2, 3]
  // MatMul: [3, 2] x [2, 4] -> [3, 4]
  // Output: [2, 3, 3, 4]
  std::vector<Dims> input_shapes = {{2, 1, 3, 2}, {3, 2, 4}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateBatchMatMulOptions(false, false);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 3, 3, 4));
}

TEST(BatchMatMulOpTest, AdjointLHS) {
  LiteRtOpT op;
  // LHS [1, 3, 2] (adj) -> [1, 2, 3], RHS [1, 3, 4]
  // Output: [1, 2, 4]
  std::vector<Dims> input_shapes = {{1, 3, 2}, {1, 3, 4}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateBatchMatMulOptions(true, false);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

TEST(BatchMatMulOpTest, AdjointRHS) {
  LiteRtOpT op;
  // LHS [1, 2, 3], RHS [1, 4, 3] (adj) -> [1, 3, 4]
  // Output: [1, 2, 4]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {1, 4, 3}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateBatchMatMulOptions(false, true);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

TEST(BatchMatMulOpTest, AdjointBoth) {
  LiteRtOpT op;
  // LHS [1, 3, 2] (adj) -> [1, 2, 3]
  // RHS [1, 4, 3] (adj) -> [1, 3, 4]
  // Output: [1, 2, 4]
  std::vector<Dims> input_shapes = {{1, 3, 2}, {1, 4, 3}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateBatchMatMulOptions(true, true);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

TEST(BatchMatMulOpTest, AsymmetricQuantizeInputs) {
  LiteRtOpT op;
  // LHS [1, 2, 3], RHS [1, 3, 4] -> [1, 2, 4]
  // Option asymmetric_quantize_inputs should not affect shape.
  std::vector<Dims> input_shapes = {{1, 2, 3}, {1, 3, 4}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateBatchMatMulOptions(false, false, true);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferBatchMatmul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4));
}

//
// FullyConnected Tests
//

TEST(FullyConnectedOpTest, Simple) {
  LiteRtOpT op;
  // Input [2, 10], Weights [20, 10] -> [2, 20]
  std::vector<Dims> input_shapes = {{2, 10}, {20, 10}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateFullyConnectedOptions(false);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_FullyConnectedOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 20));
}

TEST(FullyConnectedOpTest, KeepNumDims) {
  LiteRtOpT op;
  // Input [1, 2, 1, 10], Weights [3, 10]
  // Output [1, 2, 1, 3] (Input dims preserved)
  std::vector<Dims> input_shapes = {{1, 2, 1, 10}, {3, 10}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateFullyConnectedOptions(true);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_FullyConnectedOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 1, 3));
}

TEST(FullyConnectedOpTest, FlattenRank3) {
  LiteRtOpT op;
  // Input [2, 3, 10], Weights [5, 10]
  // keep_num_dims=false -> Flattens to [Batch, InputDim]
  // Batch = 2*3 = 6.
  // Output: [6, 5]
  std::vector<Dims> input_shapes = {{2, 3, 10}, {5, 10}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateFullyConnectedOptions(false);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_FullyConnectedOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(6, 5));
}

TEST(FullyConnectedOpTest, FlattenRank4) {
  LiteRtOpT op;
  // Input [2, 1, 3, 10], Weights [5, 10]
  // Batch = 2*1*3 = 6.
  // Output: [6, 5]
  std::vector<Dims> input_shapes = {{2, 1, 3, 10}, {5, 10}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateFullyConnectedOptions(false);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_FullyConnectedOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(6, 5));
}

TEST(FullyConnectedOpTest, ShuffledWeightsFormat) {
  LiteRtOpT op;
  // Input [2, 16], Weights [4, 16].
  // weights_format = SHUFFLED4x16INT8
  // This format requires input_depth % 16 == 0 and output_depth % 4 == 0.
  // Our inference logic treats shape as standard [Out, In] for now.
  // Testing that inference still works with this option set.
  std::vector<Dims> input_shapes = {{2, 16}, {4, 16}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateFullyConnectedOptions(
      false, tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_FullyConnectedOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 4));
}

TEST(FullyConnectedOpTest, WithFusedActivation) {
  LiteRtOpT op;
  // Input [2, 10], Weights [20, 10]
  // fused_activation_function should not affect shape.
  std::vector<Dims> input_shapes = {{2, 10}, {20, 10}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateFullyConnectedOptions(
      false, tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
      tflite::ActivationFunctionType_RELU);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_FullyConnectedOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferFullyConnected(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 20));
}

}  // namespace
}  // namespace litert::internal
