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

#include "litert/core/model/ops/slice.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
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

TEST(SliceOpTest, DynamicUpdateSlice) {
  LiteRtOpT op;
  // DynamicUpdateSlice takes 3 inputs: operand, update, start_indices.
  // Output shape is same as operand.
  std::vector<Dims> input_shapes = {
      {1, 10, 10, 3},  // operand
      {1, 2, 2, 3},    // update
      {4}              // start_indices
  };
  std::vector<Dims> output_shapes(1);

  // Setup inputs
  LiteRtTensorT operand;
  LiteRtTensorT update;
  LiteRtTensorT start_indices;

  op.Inputs().push_back(&operand);
  op.Inputs().push_back(&update);
  op.Inputs().push_back(&start_indices);

  ASSERT_EQ(
      InferDynamicUpdateSlice(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 10, 10, 3));
}

TEST(SliceOpTest, SliceStatic) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {
      {1, 10, 10, 3}, {4}, {4}};  // Input, Begin, Size
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT size_tensor;
  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&size_tensor);

  int32_t size_data[] = {1, 5, 5, 3};
  SetWeightsFromOwnedBuffer(
      size_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(size_data),
                               sizeof(size_data)));

  ASSERT_EQ(InferSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 5, 5, 3));
}

TEST(SliceOpTest, StridedSliceSimple) {
  LiteRtOpT op;
  // Input [2, 3, 4]
  std::vector<Dims> input_shapes = {
      {2, 3, 4}, {3}, {3}, {3}};  // Input, Begin, End, Strides
  std::vector<Dims> output_shapes(1);

  // Setup inputs for begin/end/strides (dummy rank 1 tensors)
  LiteRtTensorT begin;
  LiteRtTensorT end;
  LiteRtTensorT strides;
  // Set shapes for begin/end/strides to match '3'
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));

  op.Inputs().push_back(nullptr);  // Input
  op.Inputs().push_back(&begin);
  op.Inputs().push_back(&end);
  op.Inputs().push_back(&strides);

  auto options = std::make_unique<tflite::StridedSliceOptionsT>();
  options->begin_mask = 0;
  options->end_mask = 0;
  options->ellipsis_mask = 0;
  options->new_axis_mask = 0;
  options->shrink_axis_mask = 0;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_StridedSliceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  // The simplified logic sets output dims to -1 if not shrink/new_axis.
  ASSERT_EQ(InferStridedSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  // Expect [ -1, -1, -1 ]
  EXPECT_THAT(output_shapes[0], ElementsAre(-1, -1, -1));
}

TEST(SliceOpTest, StridedSliceShrinkAxis) {
  LiteRtOpT op;
  // Input [2, 3, 4]
  std::vector<Dims> input_shapes = {{2, 3, 4}, {3}, {3}, {3}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT begin;
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));

  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&begin);
  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(nullptr);

  auto options = std::make_unique<tflite::StridedSliceOptionsT>();
  options->shrink_axis_mask = 2;  // Shrink axis 1 (bit 1 set: 1<<1 = 2)

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_StridedSliceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  // Axis 1 removed.
  // Expect [ -1, -1 ]
  ASSERT_EQ(InferStridedSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(-1, -1));
}

TEST(SliceOpTest, Int64SizeTensorBaseFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{10}, {1}, {1}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT begin, size;
  int32_t begin_data[] = {0};
  int64_t size_data[] = {5};  // INT64 size tensor
  SetWeightsFromOwnedBuffer(begin.Weights(),
                            OwningBufferRef<uint8_t>(absl::string_view(
                                reinterpret_cast<char*>(begin_data), 4)));
  SetWeightsFromOwnedBuffer(size.Weights(),
                            OwningBufferRef<uint8_t>(absl::string_view(
                                reinterpret_cast<char*>(size_data), 8)));
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  size.SetType(MakeRankedTensorType(kLiteRtElementTypeInt64, {1}));

  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&begin);
  op.Inputs().push_back(&size);

  // Must return `{5}`, but on base commit casts INT64 to int32_t*, reading `{5,
  // 0}` (`rank = 2`).
  ASSERT_EQ(InferSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(5));
}

TEST(SliceOpTest, ZeroStrideBaseFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{10}, {1}, {1}, {1}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT begin, end, strides;
  int32_t b[] = {0}, e[] = {10}, s[] = {0};  // ZERO STRIDE
  SetWeightsFromOwnedBuffer(
      begin.Weights(), OwningBufferRef<uint8_t>(
                           absl::string_view(reinterpret_cast<char*>(b), 4)));
  SetWeightsFromOwnedBuffer(
      end.Weights(), OwningBufferRef<uint8_t>(
                         absl::string_view(reinterpret_cast<char*>(e), 4)));
  SetWeightsFromOwnedBuffer(
      strides.Weights(), OwningBufferRef<uint8_t>(
                             absl::string_view(reinterpret_cast<char*>(s), 4)));
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  end.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  strides.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&begin);
  op.Inputs().push_back(&end);
  op.Inputs().push_back(&strides);

  auto options = std::make_unique<tflite::StridedSliceOptionsT>();
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_StridedSliceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  // Must reject zero stride, but on base commit returns `kLiteRtStatusOk`
  // (`{-1}`).
  EXPECT_EQ(InferStridedSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(SliceOpTest, EndMaskOverrideBaseFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{3}, {1}, {1}, {1}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT begin, end, strides;
  int32_t b[] = {0}, e[] = {0}, s[] = {1};  // end = {0}, end_mask = 1
  SetWeightsFromOwnedBuffer(
      begin.Weights(), OwningBufferRef<uint8_t>(
                           absl::string_view(reinterpret_cast<char*>(b), 4)));
  SetWeightsFromOwnedBuffer(
      end.Weights(), OwningBufferRef<uint8_t>(
                         absl::string_view(reinterpret_cast<char*>(e), 4)));
  SetWeightsFromOwnedBuffer(
      strides.Weights(), OwningBufferRef<uint8_t>(
                             absl::string_view(reinterpret_cast<char*>(s), 4)));
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  end.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  strides.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&begin);
  op.Inputs().push_back(&end);
  op.Inputs().push_back(&strides);

  auto options = std::make_unique<tflite::StridedSliceOptionsT>();
  options->end_mask = 1;  // Override stop to dim_size (3)
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_StridedSliceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferStridedSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(3));
}

TEST(SliceOpTest, NegativeStrideBackwardBaseFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{3}, {1}, {1}, {1}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT begin, end, strides;
  int32_t b[] = {2}, e[] = {0},
          s[] = {-1};  // Slicing from index 2 down past index 0 (end_mask = 1)
  SetWeightsFromOwnedBuffer(
      begin.Weights(), OwningBufferRef<uint8_t>(
                           absl::string_view(reinterpret_cast<char*>(b), 4)));
  SetWeightsFromOwnedBuffer(
      end.Weights(), OwningBufferRef<uint8_t>(
                         absl::string_view(reinterpret_cast<char*>(e), 4)));
  SetWeightsFromOwnedBuffer(
      strides.Weights(), OwningBufferRef<uint8_t>(
                             absl::string_view(reinterpret_cast<char*>(s), 4)));
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  end.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  strides.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&begin);
  op.Inputs().push_back(&end);
  op.Inputs().push_back(&strides);

  auto options = std::make_unique<tflite::StridedSliceOptionsT>();
  options->end_mask = 1;  // TFLite requires end_mask = 1 to slice beyond index
                          // 0 when stride < 0
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_StridedSliceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferStridedSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(3));
}

class SliceTestShapeInferenceContext : public ShapeInferenceContext {
 public:
  explicit SliceTestShapeInferenceContext(
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
    if (index == 0) return kLiteRtElementTypeInt64;
    return kLiteRtElementTypeInt32;
  }
  const TflOptions& GetOptions() const override { return options_; }
  LiteRtOpCode GetOpCode() const override {
    return kLiteRtOpCodeTflStridedSlice;
  }
  const LiteRtOpT* GetOp() const override { return nullptr; }

 private:
  std::vector<Dims> input_shapes_;
  TflOptions options_;
  std::vector<std::vector<uint8_t>> input_data_;
};

TEST(SliceOpTest, Int64DataPropagationBaseFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{3}, {1}, {1}, {1}};
  std::vector<Dims> output_shapes(1);

  std::vector<int64_t> in_vec = {100, 200, 300};
  std::vector<uint8_t> in_bytes(in_vec.size() * sizeof(int64_t));
  std::memcpy(in_bytes.data(), in_vec.data(), in_bytes.size());

  std::vector<int32_t> b = {1}, e = {3}, s = {1};
  std::vector<uint8_t> bb(4), eb(4), sb(4);
  std::memcpy(bb.data(), b.data(), 4);
  std::memcpy(eb.data(), e.data(), 4);
  std::memcpy(sb.data(), s.data(), 4);

  LiteRtTensorT begin, end, strides;
  SetWeightsFromOwnedBuffer(begin.Weights(),
                            OwningBufferRef<uint8_t>(absl::string_view(
                                reinterpret_cast<char*>(bb.data()), 4)));
  SetWeightsFromOwnedBuffer(end.Weights(),
                            OwningBufferRef<uint8_t>(absl::string_view(
                                reinterpret_cast<char*>(eb.data()), 4)));
  SetWeightsFromOwnedBuffer(strides.Weights(),
                            OwningBufferRef<uint8_t>(absl::string_view(
                                reinterpret_cast<char*>(sb.data()), 4)));
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  end.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  strides.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&begin);
  op.Inputs().push_back(&end);
  op.Inputs().push_back(&strides);

  auto options = std::make_unique<tflite::StridedSliceOptionsT>();
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_StridedSliceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferStridedSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // True contract: slicing {3}[1:3:1] produces {2}.
  // On base commit without fixes (`116`), out_shape is wildcard `{-1}`, so this
  // naturally FAILS. On HEAD commit with fixes (`117`), out_shape is calculated
  // as `{2}`, so this naturally PASSES.
  EXPECT_THAT(output_shapes[0], ElementsAre(2));
}

}  // namespace
}  // namespace litert::internal
