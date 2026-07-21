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

#include "litert/core/model/ops/pack.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

std::unique_ptr<tflite::PackOptionsT> CreatePackOptions(int axis) {
  auto options = std::make_unique<tflite::PackOptionsT>();
  options->axis = axis;
  return options;
}

TEST(PackOpTest, ThreeInputs) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2}, {2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePackOptions(0);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(3, 2));
}

TEST(PackOpTest, ThreeInputsDifferentAxis) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2}, {2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePackOptions(1);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 3));
}

TEST(PackOpTest, ThreeInputsNegativeAxis) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2}, {2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePackOptions(-1);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 3));
}

class TestShapeInferenceContext : public ShapeInferenceContext {
 public:
  explicit TestShapeInferenceContext(
      std::vector<Dims> input_shapes, TflOptions options = {},
      std::vector<std::vector<uint8_t>> input_data = {})
      : input_shapes_(std::move(input_shapes)),
        options_(std::move(options)),
        input_data_(std::move(input_data)) {}

  size_t GetNumInputs() const override { return input_shapes_.size(); }

  size_t GetNumOutputs() const override { return 1; }

  Dims GetInputShape(size_t index) const override {
    if (index >= input_shapes_.size()) return {};
    return input_shapes_[index];
  }

  absl::Span<const uint8_t> GetInputData(size_t index) const override {
    if (index >= input_data_.size()) return {};
    return absl::MakeConstSpan(input_data_[index]);
  }

  LiteRtElementType GetInputElementType(size_t index) const override {
    return kLiteRtElementTypeInt32;
  }

  const TflOptions& GetOptions() const override { return options_; }

  LiteRtOpCode GetOpCode() const override { return kLiteRtOpCodeTflPack; }

  const LiteRtOpT* GetOp() const override { return nullptr; }

 private:
  std::vector<Dims> input_shapes_;
  TflOptions options_;
  std::vector<std::vector<uint8_t>> input_data_;
};

TEST(PackOpTest, StatelessWithTransientData) {
  std::vector<int32_t> val0 = {10};
  std::vector<int32_t> val1 = {20};
  std::vector<int32_t> val2 = {30};
  std::vector<std::vector<uint8_t>> input_data = {
      std::vector<uint8_t>(reinterpret_cast<uint8_t*>(val0.data()),
                           reinterpret_cast<uint8_t*>(val0.data() + 1)),
      std::vector<uint8_t>(reinterpret_cast<uint8_t*>(val1.data()),
                           reinterpret_cast<uint8_t*>(val1.data() + 1)),
      std::vector<uint8_t>(reinterpret_cast<uint8_t*>(val2.data()),
                           reinterpret_cast<uint8_t*>(val2.data() + 1))};

  auto options = CreatePackOptions(0);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();

  TestShapeInferenceContext ctx({{1}, {1}, {1}}, std::move(tfl_options),
                                std::move(input_data));
  InferenceResult result;

  ASSERT_EQ(InferPack(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(3, 1));
  ASSERT_TRUE(result.propagated_data.contains(0));
  ASSERT_EQ(result.propagated_data[0].size(), 3 * sizeof(int32_t));
  const int32_t* prop_ptr =
      reinterpret_cast<const int32_t*>(result.propagated_data[0].data());
  EXPECT_THAT(std::vector<int32_t>(prop_ptr, prop_ptr + 3),
              ElementsAre(10, 20, 30));
}

TEST(PackOpTest, StatelessMissingDataDoesNotPropagate) {
  auto options = CreatePackOptions(0);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();

  // No static byte buffers provided
  TestShapeInferenceContext ctx({{1}, {1}, {1}}, std::move(tfl_options));
  InferenceResult result;

  ASSERT_EQ(InferPack(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(3, 1));
  EXPECT_FALSE(result.propagated_data.contains(0));
}

TEST(PackOpTest, StatelessMismatchedShapesFail) {
  auto options = CreatePackOptions(0);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();

  TestShapeInferenceContext ctx({{2}, {3}}, std::move(tfl_options));
  InferenceResult result;

  EXPECT_EQ(InferPack(ctx, result), kLiteRtStatusErrorInvalidArgument);
}

TEST(PackOpTest, AxisOutOfBoundsBaseFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2}, {2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePackOptions(5);  // Out of bounds for rank 1
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  // Must fail shape inference, but on base commit triggers std::vector::insert
  // abort (`SIGABRT`).
  EXPECT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(PackOpTest, ConflictingStaticDimsWithDynamicS0BaseFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{-1}, {128}, {256}};  // 128 != 256 conflict
  std::vector<Dims> output_shapes(1);

  auto options = CreatePackOptions(0);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  // Must fail shape inference, but on base commit returns kLiteRtStatusOk ({3,
  // -1}).
  EXPECT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(PackOpTest, InterleavingAlongAxis1BaseFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT s0, s1;
  int32_t d0[] = {10, 20};
  int32_t d1[] = {30, 40};
  SetWeightsFromOwnedBuffer(
      s0.Weights(), OwningBufferRef<uint8_t>(
                        absl::string_view(reinterpret_cast<char*>(d0), 8)));
  SetWeightsFromOwnedBuffer(
      s1.Weights(), OwningBufferRef<uint8_t>(
                        absl::string_view(reinterpret_cast<char*>(d1), 8)));
  s0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));
  s1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));

  op.Inputs().push_back(&s0);
  op.Inputs().push_back(&s1);

  auto options =
      CreatePackOptions(1);  // Pack along axis 1 -> target shape {2, 2}
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));

  auto options2 = CreatePackOptions(1);
  TflOptions tfl_options2;
  tfl_options2.type = tflite::BuiltinOptions_PackOptions;
  tfl_options2.value = options2.release();

  TestShapeInferenceContext ctx(
      {{2}, {2}}, std::move(tfl_options2),
      {std::vector<uint8_t>(reinterpret_cast<uint8_t*>(d0),
                            reinterpret_cast<uint8_t*>(d0 + 2)),
       std::vector<uint8_t>(reinterpret_cast<uint8_t*>(d1),
                            reinterpret_cast<uint8_t*>(d1 + 2))});
  InferenceResult result;
  ASSERT_EQ(InferPack(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(2, 2));
  ASSERT_TRUE(result.propagated_data.contains(0));
  const int32_t* res =
      reinterpret_cast<const int32_t*>(result.propagated_data[0].data());
  EXPECT_THAT(std::vector<int32_t>(res, res + 4), ElementsAre(10, 30, 20, 40));
}

TEST(PackOpTest, MismatchedBufferSizesBaseFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {
      {1}, {2}};  // Buffer lengths differ: 4 bytes vs 8 bytes
  std::vector<Dims> output_shapes(1);

  auto options = CreatePackOptions(0);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  // Must reject mismatched inputs, but on base commit returns kLiteRtStatusOk
  // ({2, 1}).
  EXPECT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::internal
