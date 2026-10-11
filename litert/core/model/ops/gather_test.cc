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

#include "litert/core/model/ops/gather.h"

#include <memory>
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

TEST(GatherOpTest, SimpleGatherAxis0) {
  LiteRtOpT op;
  // Input [2, 2], Indices [2]
  std::vector<Dims> input_shapes = {{2, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = 0;
  options->batch_dims = 0;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: [2] + [2] = [2, 2]
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));
}

TEST(GatherOpTest, GatherAxis1) {
  LiteRtOpT op;
  // Input [1, 2, 3], Indices [2]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = 1;
  options->batch_dims = 0;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: Input[:1] + Indices + Input[1+1:] -> [1] + [2] + [3] = [1, 2, 3]
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(GatherOpTest, GatherScalarIndex) {
  LiteRtOpT op;
  // Input [2, 2], Indices [] (scalar)
  std::vector<Dims> input_shapes = {{2, 2}, {}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = 0;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: Input[:0] + [] + Input[1:] -> [] + [] + [2] = [2]
  EXPECT_THAT(output_shapes[0], ElementsAre(2));
}

TEST(GatherOpTest, GatherLastAxis) {
  LiteRtOpT op;
  // Input [1, 2, 3], Indices [2]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = 2;  // -1 or 2

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: Input[:2] + Indices + Input[3:] -> [1, 2] + [2] + [] = [1, 2, 2]
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2));
}

TEST(GatherOpTest, GatherNegativeAxis) {
  LiteRtOpT op;
  // Input [1, 2, 3], Indices [2]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = -1;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: [1, 2, 2] same as above
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2));
}

TEST(GatherOpTest, EmbeddingLookup) {
  LiteRtOpT op;
  // Ids [2], Params [10, 5] (Lookup 2 vectors of size 5)
  std::vector<Dims> input_shapes = {{2}, {10, 5}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferEmbeddingLookup(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 5));
}

TEST(GatherOpTest, EmbeddingLookupEmptyIdsAnd3DParams) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{0}, {10, 4, 8}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferEmbeddingLookup(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(0, 4, 8));
}

TEST(GatherOpTest, EmbeddingLookupRejectsInvalidRanksAndTypes) {
  LiteRtOpT op;
  std::vector<Dims> output_shapes(1);

  // Ids must be 1D.
  std::vector<Dims> rank2_ids = {{2, 1}, {10, 5}};
  EXPECT_EQ(InferEmbeddingLookup(op, absl::MakeSpan(rank2_ids), output_shapes),
            kLiteRtStatusErrorInvalidArgument);

  // Params must have rank >= 2.
  std::vector<Dims> rank1_params = {{2}, {10}};
  EXPECT_EQ(
      InferEmbeddingLookup(op, absl::MakeSpan(rank1_params), output_shapes),
      kLiteRtStatusErrorInvalidArgument);

  // Ids tensor element type must be Int32 when op inputs are attached.
  LiteRtTensorT ids_tensor;
  ids_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));
  LiteRtTensorT params_tensor;
  params_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {10, 5}));
  op.Inputs().push_back(&ids_tensor);
  op.Inputs().push_back(&params_tensor);
  std::vector<Dims> valid_shapes = {{2}, {10, 5}};
  ids_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2}));
  EXPECT_EQ(
      InferEmbeddingLookup(op, absl::MakeSpan(valid_shapes), output_shapes),
      kLiteRtStatusErrorInvalidArgument);
}

TEST(GatherOpTest, EmbeddingLookupRejectsBlockwiseScaleOverflowAndZeroScales) {
  LiteRtOpT op;
  LiteRtTensorT ids_tensor;
  ids_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));
  LiteRtTensorT params_tensor;
  params_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt4, {4, 32}));
  LiteRtTensorT scales_tensor;
  // Rank-4 scales tensor whose dimension product overflows 64-bit size_t.
  scales_tensor.SetType(MakeRankedTensorType(
      kLiteRtElementTypeFloat16, {1073741824, 1073741824, 1073741824, 4}));

  params_tensor.SetQTypeId(kLiteRtQuantizationBlockWise);
  params_tensor.Qparams().second.block_wise.scales = &scales_tensor;
  params_tensor.Qparams().second.block_wise.zero_points = nullptr;
  params_tensor.Qparams().second.block_wise.block_size = 32;

  op.Inputs() = {&ids_tensor, &params_tensor};
  std::vector<Dims> input_shapes = {{2}, {4, 32}};
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(
      InferEmbeddingLookup(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusErrorInvalidArgument);

  // Zero-sized scales tensor for non-empty params must also be rejected.
  scales_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, {0, 1}));
  EXPECT_EQ(
      InferEmbeddingLookup(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusErrorInvalidArgument);
}

TEST(GatherOpTest, GatherNd) {
  LiteRtOpT op;
  // Input [2, 2], Indices [2, 1]
  std::vector<Dims> input_shapes = {{2, 2}, {2, 1}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferGatherNd(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));
}

TEST(GatherOpTest, GatherNdSlice) {
  LiteRtOpT op;
  // Input [2, 2], Indices [1, 2]
  std::vector<Dims> input_shapes = {{2, 2}, {1, 2}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferGatherNd(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: [1] (indices batch dim) + [] (input suffix) = [1]
  EXPECT_THAT(output_shapes[0], ElementsAre(1));
}

TEST(GatherOpTest, HashtableLookupInfersShapesAndValidatesContracts) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{4}, {3}, {3, 2}};
  std::vector<Dims> output_shapes(2);

  ASSERT_EQ(
      InferHashtableLookup(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(4, 2));
  EXPECT_THAT(output_shapes[1], ElementsAre(4));

  // Mismatched key[0] and value[0] is rejected.
  std::vector<Dims> mismatched_rows = {{4}, {3}, {5, 2}};
  EXPECT_EQ(
      InferHashtableLookup(op, absl::MakeSpan(mismatched_rows), output_shapes),
      kLiteRtStatusErrorInvalidArgument);

  // Rank-2 lookup or key is rejected.
  std::vector<Dims> bad_lookup_rank = {{4, 1}, {3}, {3, 2}};
  EXPECT_EQ(
      InferHashtableLookup(op, absl::MakeSpan(bad_lookup_rank), output_shapes),
      kLiteRtStatusErrorInvalidArgument);

  // 2D string value is rejected when tensors are attached.
  LiteRtTensorT lookup_tensor;
  lookup_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
  LiteRtTensorT key_tensor;
  key_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));
  LiteRtTensorT value_tensor;
  value_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeTfString, {3, 2}));
  LiteRtTensorT output_tensor;
  output_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeTfString, {4, 2}));
  LiteRtTensorT hits_tensor;
  hits_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeUInt8, {4}));
  op.Inputs() = {&lookup_tensor, &key_tensor, &value_tensor};
  op.Outputs() = {&output_tensor, &hits_tensor};
  EXPECT_EQ(
      InferHashtableLookup(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusErrorInvalidArgument);

  // 1D string value succeeds.
  std::vector<Dims> string_shapes = {{4}, {3}, {3}};
  value_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeTfString, {3}));
  output_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeTfString, {4}));
  ASSERT_EQ(
      InferHashtableLookup(op, absl::MakeSpan(string_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(4));
  EXPECT_THAT(output_shapes[1], ElementsAre(4));
}

}  // namespace
}  // namespace litert::internal
