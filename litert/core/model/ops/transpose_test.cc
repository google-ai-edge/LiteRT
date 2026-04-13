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
#include "tflite/converter/schema/schema_generated.h"

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

}  // namespace
}  // namespace litert::internal
