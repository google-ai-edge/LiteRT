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

#include "litert/core/model/ops/reshape.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TflOptions MakeReshapeOptions(Dims new_shape) {
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = std::move(new_shape);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();
  return tfl_options;
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
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
    if (index == 1) {
      if (index < input_data_.size() && index < input_shapes_.size() &&
          !input_shapes_[index].empty()) {
        size_t num_elems = 1;
        for (auto d : input_shapes_[index]) num_elems *= d;
        if (num_elems > 0 &&
            input_data_[index].size() == num_elems * sizeof(int64_t)) {
          return kLiteRtElementTypeInt64;
        }
      }
      return kLiteRtElementTypeInt32;
    }
    return kLiteRtElementTypeFloat32;
  }

  const TflOptions& GetOptions() const override { return options_; }

  LiteRtOpCode GetOpCode() const override { return kLiteRtOpCodeTflReshape; }

  const LiteRtOpT* GetOp() const override { return nullptr; }

 private:
  std::vector<Dims> input_shapes_;
  TflOptions options_;
  std::vector<std::vector<uint8_t>> input_data_;
};

TEST(ReshapeOpTest, WithOptions) {
  TestShapeInferenceContext ctx({{1, 48}}, MakeReshapeOptions({1, 4, 4, 3}));
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);

  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, 4, 4, 3));
}

TEST(ReshapeOpTest, WithShapeTensor) {
  TestShapeInferenceContext ctx({{1, 48}, {4}}, {},
                                {{}, ToBytes<int32_t>({1, 4, 4, 3})});
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);

  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, 4, 4, 3));
}

TEST(ReshapeOpTest, DynamicShapeTensor) {
  TestShapeInferenceContext ctx({{1, 48}, {4}}, {}, {{}, {}});
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, InvalidShapeTensorBufferSize) {
  std::vector<uint8_t> shape_bytes = {1, 2, 3};

  TestShapeInferenceContext ctx({{1, 48}, {4}}, {}, {{}, shape_bytes});
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorInvalidArgument);
}

TEST(ReshapeOpTest, NoOptionsAndNoShapeTensor) {
  TestShapeInferenceContext ctx({{1, 48}}, {}, {{}});
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, MultipleMinusOne) {
  TestShapeInferenceContext ctx({{1, 48}}, MakeReshapeOptions({1, -1, -1, 3}));
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, MinusOneWithDynamicInput) {
  TestShapeInferenceContext ctx({{1, -1}}, MakeReshapeOptions({1, -1, 3}));
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, -1, 3));
}

TEST(ReshapeOpTest, IncompatibleVolumesWithMinusOne) {
  TestShapeInferenceContext ctx({{1, 48}}, MakeReshapeOptions({1, -1, 5}));
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, ResolveMinusOne) {
  TestShapeInferenceContext ctx({{1, 48}}, MakeReshapeOptions({1, -1, 3}));
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, 16, 3));
}

TEST(ReshapeOpTest, StaticShapeSuccess) {
  TestShapeInferenceContext ctx({{1, 48}}, MakeReshapeOptions({2, 3, 8}));
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(2, 3, 8));
}

TEST(ReshapeOpTest, StaticShapeVolumeMismatch) {
  TestShapeInferenceContext ctx({{1, 48}}, MakeReshapeOptions({2, 3, 10}));
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, StaticShapeWithShapeTensorVolumeMismatch) {
  TestShapeInferenceContext ctx({{2, 6}, {2}}, {},
                                {{}, ToBytes<int32_t>({3, 5})});
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, DivisionByZeroBaseFailure) {
  TestShapeInferenceContext ctx({{2, 3}}, MakeReshapeOptions({0, -1}));
  InferenceResult result;

  // Must return error, but on base commit crashes with hardware
  // division-by-zero (`SIGFPE`).
  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, StaticShapeVolumeMismatchBaseFailure) {
  TestShapeInferenceContext ctx({{1, 48}}, MakeReshapeOptions({2, 3, 10}));
  InferenceResult result;

  // Must return error, but on base commit skips check and returns
  // `kLiteRtStatusOk` (`{2, 3, 10}`).
  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, Int64ShapeTensorBaseFailure) {
  TestShapeInferenceContext ctx({{2, 3}, {2}}, {},
                                {{}, ToBytes<int64_t>({2, 3})});
  InferenceResult result;

  // Must output `{2, 3}`, but on base commit parses 16 bytes as `{2, 0, 3, 0}`
  // (`rank = 4`).
  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(2, 3));
}

TEST(ReshapeOpTest, ZeroElementShapeTensor) {
  TestShapeInferenceContext ctx(
      {{1}, {0}}, {}, {{}, {}});  // {0} means 1D length 0, Input volume 1
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);

  EXPECT_THAT(result.output_shapes[0], ElementsAre());  // 0D scalar
}

TEST(ReshapeOpTest, EmptyOptionsNewShape) {
  TestShapeInferenceContext ctx(
      {{1}}, MakeReshapeOptions({}));  // 1-input, Input volume 1
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);

  EXPECT_THAT(result.output_shapes[0], ElementsAre());  // 0D scalar
}

struct ReshapeErrorCase {
  std::string test_name;
  std::vector<Dims> input_shapes;
  bool use_options;
  Dims options_new_shape;
  std::vector<uint8_t> shape_tensor_bytes;
  LiteRtStatus expected_status;
};

using ReshapeErrorTest = ::testing::TestWithParam<ReshapeErrorCase>;

TEST_P(ReshapeErrorTest, RejectsInvalidShapesAndOverflows) {
  const auto& param = GetParam();
  TflOptions tfl_options = param.use_options
                               ? MakeReshapeOptions(param.options_new_shape)
                               : TflOptions{};
  std::vector<std::vector<uint8_t>> input_data;
  if (!param.shape_tensor_bytes.empty()) {
    input_data = {{}, param.shape_tensor_bytes};
  }
  TestShapeInferenceContext ctx(param.input_shapes, std::move(tfl_options),
                                std::move(input_data));
  InferenceResult result;
  EXPECT_EQ(InferReshape(ctx, result), param.expected_status);
}

INSTANTIATE_TEST_SUITE_P(
    ReshapeErrorTests, ReshapeErrorTest,
    ::testing::Values(
        ReshapeErrorCase{
            "NegativeDimInOptions",
            {{1, 8}},
            /*use_options=*/true,
            {-2, 4},
            {},
            kLiteRtStatusErrorInvalidArgument,
        },
        ReshapeErrorCase{
            "NegativeDimInInt32ShapeTensor",
            {{1, 8}, {2}},
            /*use_options=*/false,
            {},
            ToBytes<int32_t>({-2, 4}),
            kLiteRtStatusErrorInvalidArgument,
        },
        ReshapeErrorCase{
            "DimExceedingInt32MaxInInt64ShapeTensor",
            {{1, 8}, {2}},
            /*use_options=*/false,
            {},
            ToBytes<int64_t>(
                {static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
                 1}),
            kLiteRtStatusErrorInvalidArgument,
        },
        ReshapeErrorCase{
            "NegativeDimInInputShape",
            {{-2, 4}},
            /*use_options=*/true,
            {8},
            {},
            kLiteRtStatusErrorShapeInferenceFailed,
        },
        ReshapeErrorCase{
            "NewShapeProductOverflow",
            {{1, 8}},
            /*use_options=*/true,
            {1073741824, 1073741824, 1073741824, 16},
            {},
            kLiteRtStatusErrorShapeInferenceFailed,
        },
        ReshapeErrorCase{
            "InputShapeProductOverflow",
            {{1073741824, 1073741824, 1073741824, 16}},
            /*use_options=*/true,
            {-1},
            {},
            kLiteRtStatusErrorShapeInferenceFailed,
        },
        ReshapeErrorCase{
            "InferredWildcardExceedingInt32Max",
            {{65536, 65536}},
            /*use_options=*/true,
            {-1},
            {},
            kLiteRtStatusErrorShapeInferenceFailed,
        }),
    [](const ::testing::TestParamInfo<ReshapeErrorCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace litert::internal
