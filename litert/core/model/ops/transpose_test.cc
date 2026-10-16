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

#include "litert/core/model/ops/transpose.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(TransposeOpTest, BatchedTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 3, 4}, {3}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT perm_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&perm_tensor);

  std::vector<int32_t> perm_data = {2, 1, 0};
  SetWeightsFromUnownedBuffer(
      perm_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(perm_data.data()),
                                 perm_data.size() * sizeof(int32_t)));

  auto options = std::make_unique<tflite::TransposeOptionsT>();
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_TransposeOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferTranspose(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(4, 3, 2));
}

TEST(TransposeOpTest, IdentityTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 3, 4}, {3}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT perm_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&perm_tensor);

  std::vector<int32_t> perm_data = {0, 1, 2};
  SetWeightsFromUnownedBuffer(
      perm_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(perm_data.data()),
                                 perm_data.size() * sizeof(int32_t)));

  auto options = std::make_unique<tflite::TransposeOptionsT>();
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_TransposeOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferTranspose(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 3, 4));
}

TEST(TransposeOpTest, SDPA_Transpose) {
  LiteRtOpT op;
  // Input [8, 100, 32, 4]
  std::vector<Dims> input_shapes = {{8, 100, 32, 4}, {4}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT perm_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&perm_tensor);

  // Permutation [0, 2, 1, 3] -> [8, 32, 100, 4]
  std::vector<int32_t> perm_data = {0, 2, 1, 3};
  SetWeightsFromUnownedBuffer(
      perm_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(perm_data.data()),
                                 perm_data.size() * sizeof(int32_t)));

  ASSERT_EQ(InferTranspose(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(8, 32, 100, 4));
}

TEST(TransposeOpTest, RejectsMissingInputTensors) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 3, 4}, {3}};
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferTranspose(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(TransposeOpTest, NegativePermEntries) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 3, 4}, {3}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT perm_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&perm_tensor);

  // TFLite normalizes negative entries by adding the rank: {-1, 0, -2} is
  // {2, 0, 1}.
  std::vector<int32_t> perm_data = {-1, 0, -2};
  SetWeightsFromUnownedBuffer(
      perm_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(perm_data.data()),
                                 perm_data.size() * sizeof(int32_t)));

  ASSERT_EQ(InferTranspose(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(4, 2, 3));
}

TEST(TransposeOpTest, MinusRankPermEntryIsAccepted) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 3, 4}, {3}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT perm_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&perm_tensor);

  // -rank is the smallest valid entry: {1, -3, 2} is {1, 0, 2}.
  std::vector<int32_t> perm_data = {1, -3, 2};
  SetWeightsFromUnownedBuffer(
      perm_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(perm_data.data()),
                                 perm_data.size() * sizeof(int32_t)));

  ASSERT_EQ(InferTranspose(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(3, 2, 4));
}

TEST(TransposeOpTest, ReferenceTransposeNormalizesNegativePerm) {
  const std::vector<float> input = {0, 1, 2, 3, 4, 5};  // Shape {2, 3}.
  const int32_t dims[] = {2, 3};
  const int32_t perm[] = {-1, 0};  // {1, 0}.
  std::vector<float> output(input.size());
  ReferenceTranspose(input.data(), dims, perm, /*rank=*/2, output.data());
  EXPECT_THAT(output, ElementsAre(0, 3, 1, 4, 2, 5));
}

TEST(TransposeOpTest, RejectsOutOfRangePermEntries) {
  // TFLite accepts entries in [-rank, rank) only.
  for (int32_t bad : {3, -4, std::numeric_limits<int32_t>::min(),
                      std::numeric_limits<int32_t>::max()}) {
    LiteRtOpT op;
    std::vector<Dims> input_shapes = {{2, 3, 4}, {3}};
    std::vector<Dims> output_shapes(1);

    LiteRtTensorT input_tensor;
    LiteRtTensorT perm_tensor;
    op.Inputs().push_back(&input_tensor);
    op.Inputs().push_back(&perm_tensor);

    std::vector<int32_t> perm_data = {0, bad, 1};
    SetWeightsFromUnownedBuffer(
        perm_tensor.Weights(),
        litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(perm_data.data()),
                                   perm_data.size() * sizeof(int32_t)));

    EXPECT_EQ(InferTranspose(op, absl::MakeSpan(input_shapes), output_shapes),
              kLiteRtStatusErrorShapeInferenceFailed)
        << "perm entry " << bad;
  }
}

}  // namespace
}  // namespace litert::internal
