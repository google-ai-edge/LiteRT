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

#include <cstddef>
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

struct EmbeddingLookupErrorCase {
  std::string test_name;
  std::vector<Dims> input_shapes;
  bool attach_tensors;
  LiteRtElementType ids_type;
};

using EmbeddingLookupErrorTest =
    ::testing::TestWithParam<EmbeddingLookupErrorCase>;

TEST_P(EmbeddingLookupErrorTest, RejectsInvalidRanksAndTypes) {
  const auto& param = GetParam();
  LiteRtOpT op;
  LiteRtTensorT ids_tensor;
  LiteRtTensorT params_tensor;
  if (param.attach_tensors) {
    ids_tensor.SetType(
        MakeRankedTensorType(param.ids_type, param.input_shapes[0]));
    params_tensor.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat32, param.input_shapes[1]));
    op.Inputs() = {&ids_tensor, &params_tensor};
  }
  std::vector<Dims> input_shapes = param.input_shapes;
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(
      InferEmbeddingLookup(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusErrorInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    EmbeddingLookupErrorTests, EmbeddingLookupErrorTest,
    ::testing::Values(EmbeddingLookupErrorCase{"Rank2Ids",
                                               {{2, 1}, {10, 5}},
                                               /*attach_tensors=*/false,
                                               kLiteRtElementTypeInt32},
                      EmbeddingLookupErrorCase{"Rank1Params",
                                               {{2}, {10}},
                                               /*attach_tensors=*/false,
                                               kLiteRtElementTypeInt32},
                      EmbeddingLookupErrorCase{"Float32IdsTensor",
                                               {{2}, {10, 5}},
                                               /*attach_tensors=*/true,
                                               kLiteRtElementTypeFloat32}),
    [](const ::testing::TestParamInfo<EmbeddingLookupErrorCase>& info) {
      return info.param.test_name;
    });

struct BlockwiseEmbeddingLookupShapeCase {
  std::string test_name;
  Dims params_shape;
  Dims scales_shape;
  LiteRtStatus expected_status;
  Dims expected_output_shape = {};
};

using BlockwiseEmbeddingLookupShapeTest =
    ::testing::TestWithParam<BlockwiseEmbeddingLookupShapeCase>;

TEST_P(BlockwiseEmbeddingLookupShapeTest,
       ValidatesScaleShapeAndStaticParameterContract) {
  const auto& param = GetParam();
  LiteRtOpT op;
  LiteRtTensorT ids_tensor;
  ids_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));
  LiteRtTensorT params_tensor;
  params_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt4, param.params_shape));
  LiteRtTensorT scales_tensor;
  scales_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, param.scales_shape));

  params_tensor.SetQTypeId(kLiteRtQuantizationBlockWise);
  params_tensor.Qparams().second.block_wise.scales = &scales_tensor;
  params_tensor.Qparams().second.block_wise.zero_points = nullptr;
  params_tensor.Qparams().second.block_wise.block_size = 32;
  op.Inputs() = {&ids_tensor, &params_tensor};

  std::vector<Dims> input_shapes = {{2}, param.params_shape};
  std::vector<Dims> output_shapes(1);
  ASSERT_EQ(
      InferEmbeddingLookup(op, absl::MakeSpan(input_shapes), output_shapes),
      param.expected_status);
  if (param.expected_status == kLiteRtStatusOk) {
    EXPECT_EQ(output_shapes[0], param.expected_output_shape);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BlockwiseEmbeddingLookupShapeTests, BlockwiseEmbeddingLookupShapeTest,
    ::testing::Values(
        BlockwiseEmbeddingLookupShapeCase{
            "ScalesDimensionProductOverflow",
            /*params_shape=*/{4, 32},
            /*scales_shape=*/{1073741824, 1073741824, 1073741824, 4},
            kLiteRtStatusErrorInvalidArgument,
        },
        BlockwiseEmbeddingLookupShapeCase{
            "ZeroSizedScalesForNonEmptyParams",
            /*params_shape=*/{4, 32},
            /*scales_shape=*/{0, 1},
            kLiteRtStatusErrorInvalidArgument,
        },
        BlockwiseEmbeddingLookupShapeCase{
            "ScalarScaleSingleBlockAccepted",
            /*params_shape=*/{1, 32},
            /*scales_shape=*/{},
            kLiteRtStatusOk,
            {2, 32},
        },
        BlockwiseEmbeddingLookupShapeCase{
            "ScalarScaleTwoBlocksRejected",
            /*params_shape=*/{2, 32},
            /*scales_shape=*/{},
            kLiteRtStatusErrorInvalidArgument,
        },
        BlockwiseEmbeddingLookupShapeCase{
            "DynamicScaleDimensionRejected",
            /*params_shape=*/{2, 32},
            /*scales_shape=*/{-1},
            kLiteRtStatusErrorInvalidArgument,
        }),
    [](const ::testing::TestParamInfo<BlockwiseEmbeddingLookupShapeCase>&
           info) { return info.param.test_name; });

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

struct HashtableLookupCase {
  std::string test_name;
  std::vector<Dims> input_shapes;
  bool attach_string_tensors = false;
  LiteRtStatus expected_status = kLiteRtStatusOk;
  Dims expected_output0 = {};
  Dims expected_output1 = {};
};

using HashtableLookupTest = ::testing::TestWithParam<HashtableLookupCase>;

TEST_P(HashtableLookupTest, InfersShapesAndValidatesContracts) {
  const auto& param = GetParam();
  LiteRtOpT op;
  LiteRtTensorT lookup_tensor;
  LiteRtTensorT key_tensor;
  LiteRtTensorT value_tensor;
  LiteRtTensorT output_tensor;
  LiteRtTensorT hits_tensor;
  if (param.attach_string_tensors) {
    lookup_tensor.SetType(
        MakeRankedTensorType(kLiteRtElementTypeInt32, param.input_shapes[0]));
    key_tensor.SetType(
        MakeRankedTensorType(kLiteRtElementTypeInt32, param.input_shapes[1]));
    value_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeTfString,
                                              param.input_shapes[2]));
    output_tensor.SetType(
        MakeRankedTensorType(kLiteRtElementTypeTfString, {4}));
    hits_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeUInt8, {4}));
    op.Inputs() = {&lookup_tensor, &key_tensor, &value_tensor};
    op.Outputs() = {&output_tensor, &hits_tensor};
  }

  std::vector<Dims> output_shapes(2);
  ASSERT_EQ(InferHashtableLookup(op, absl::MakeSpan(param.input_shapes),
                                 output_shapes),
            param.expected_status);
  if (param.expected_status == kLiteRtStatusOk) {
    EXPECT_EQ(output_shapes[0], param.expected_output0);
    EXPECT_EQ(output_shapes[1], param.expected_output1);
  }
}

INSTANTIATE_TEST_SUITE_P(
    HashtableLookupTests, HashtableLookupTest,
    ::testing::Values(HashtableLookupCase{"Valid2DValue",
                                          {{4}, {3}, {3, 2}},
                                          /*attach_string_tensors=*/false,
                                          kLiteRtStatusOk,
                                          {4, 2},
                                          {4}},
                      HashtableLookupCase{"MismatchedKeyAndValueRowsRejected",
                                          {{4}, {3}, {5, 2}},
                                          /*attach_string_tensors=*/false,
                                          kLiteRtStatusErrorInvalidArgument},
                      HashtableLookupCase{"Rank2LookupRejected",
                                          {{4, 1}, {3}, {3, 2}},
                                          /*attach_string_tensors=*/false,
                                          kLiteRtStatusErrorInvalidArgument},
                      HashtableLookupCase{"2DStringValueRejected",
                                          {{4}, {3}, {3, 2}},
                                          /*attach_string_tensors=*/true,
                                          kLiteRtStatusErrorInvalidArgument},
                      HashtableLookupCase{"1DStringValueAccepted",
                                          {{4}, {3}, {3}},
                                          /*attach_string_tensors=*/true,
                                          kLiteRtStatusOk,
                                          {4},
                                          {4}}),
    [](const ::testing::TestParamInfo<HashtableLookupCase>& info) {
      return info.param.test_name;
    });

struct GatherOutputCountErrorCase {
  std::string test_name;
  bool is_gather_nd;
  size_t num_outputs;
};

using GatherOutputCountErrorTest =
    ::testing::TestWithParam<GatherOutputCountErrorCase>;

TEST_P(GatherOutputCountErrorTest, RejectsInvalidOutputShapesCount) {
  const auto& param = GetParam();
  LiteRtOpT op;
  std::vector<Dims> output_shapes(param.num_outputs);
  if (param.is_gather_nd) {
    std::vector<Dims> inputs = {{2, 2}, {2, 1}};
    EXPECT_EQ(InferGatherNd(op, absl::MakeSpan(inputs), output_shapes),
              kLiteRtStatusErrorInvalidArgument);
  } else {
    std::vector<Dims> inputs = {{2, 2}, {2}};
    EXPECT_EQ(InferGather(op, absl::MakeSpan(inputs), output_shapes),
              kLiteRtStatusErrorInvalidArgument);
  }
}

INSTANTIATE_TEST_SUITE_P(
    GatherOutputCountErrorTests, GatherOutputCountErrorTest,
    ::testing::Values(
        GatherOutputCountErrorCase{"GatherEmptyOutputs", false, 0},
        GatherOutputCountErrorCase{"GatherTwoOutputs", false, 2},
        GatherOutputCountErrorCase{"GatherNdEmptyOutputs", true, 0},
        GatherOutputCountErrorCase{"GatherNdTwoOutputs", true, 2}),
    [](const ::testing::TestParamInfo<GatherOutputCountErrorCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace litert::internal
