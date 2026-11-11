// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/vision_eval/image.h"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/tools/vision_eval/data.h"
#include "tflite/kernels/internal/reference/quantize.h"
#include "tflite/kernels/internal/runtime_shape.h"
#include "tflite/kernels/internal/types.h"

namespace litert::vision {
namespace {
template <typename T>
void CheckInputQuantization(enum xnn_datatype datatype) {
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  for (float scale : {1.0f, 0.125f, 0.1f, 0.0837f}) {
    std::vector<float> values;
    for (int k = -260; k <= 260; ++k) {
      const float value = static_cast<float>(k) * 0.5f * scale;
      values.push_back(value);
      values.push_back(
          std::nextafter(value, -std::numeric_limits<float>::infinity()));
      values.push_back(
          std::nextafter(value, std::numeric_limits<float>::infinity()));
    }
    values.insert(values.end(), {-1.0e6f, 1.0e6f});
    const size_t count = values.size();
    // XNNPACK permits vector loads beyond the final logical input element.
    values.resize(count + XNN_EXTRA_BYTES / sizeof(float));
    for (int zp = std::numeric_limits<T>::min();
         zp <= std::numeric_limits<T>::max(); ++zp) {
      SCOPED_TRACE(::testing::Message() << "scale=" << scale << " zp=" << zp);
      const xnn_quantization_params quantization{zp, scale};
      std::vector<T> expected(count);
      ASSERT_EQ(xnn_run_unary_elementwise_nc(
                    xnn_unary_convert, xnn_datatype_fp32, datatype, nullptr,
                    nullptr, &quantization, 0, 1, count, count, count, nullptr,
                    values.data(), expected.data()),
                xnn_status_success);
      const auto actual = QuantizePerTensor<T>(
          absl::MakeConstSpan(values).first(count), {scale, zp});
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(actual[count - 2], std::numeric_limits<T>::min());
      EXPECT_EQ(actual[count - 1], std::numeric_limits<T>::max());
    }
  }
}

TEST(InputQuantizationTest, MatchesXnnpackIncludingTiesAndFullRangeSaturation) {
  const int previous = std::fegetround();
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  CheckInputQuantization<int8_t>(xnn_datatype_qint8);
  CheckInputQuantization<uint8_t>(xnn_datatype_quint8);
  EXPECT_EQ(std::fesetround(previous), 0);
}

TEST(InputQuantizationTest,
     IntentionallyDisagreesWithReferenceHalfStepRounding) {
  const float input = 0.5f;
  tflite::QuantizationParams params;
  params.scale = 1.0;
  params.zero_point = 0;
  int8_t reference = 0;
  tflite::reference_ops::AffineQuantize(params, tflite::RuntimeShape({1}),
                                        &input, tflite::RuntimeShape({1}),
                                        &reference);
  EXPECT_EQ(reference, 1);
  EXPECT_EQ(QuantizePerTensor<int8_t>({input}, {1.0f, 0}),
            std::vector<int8_t>({0}));
  EXPECT_EQ(QuantizePerTensor<int8_t>({1.5f}, {1.0f, -128}),
            std::vector<int8_t>({-126}));
  EXPECT_EQ(QuantizePerTensor<int8_t>({-1.5f}, {1.0f, 127}),
            std::vector<int8_t>({125}));
}

TEST(ManifestTest, SelectionRelativePathsAndMalformedRows) {
  const auto root = Utf8ToPath(testing::TempDir());
  const auto path = root / "samples.tsv";
  auto write = [&](const std::string& text) {
    ASSERT_TRUE(WriteBytes(path, text.data(), text.size()));
  };
  write(
      "# comment\r\nid\tlabel\tpath\r\n7\t9\tsubdir/a.jpg\r\n"
      "8\t2\t/absolute/b.jpg\n");
  auto rows = ReadSamples(path, {}, 0, 1);
  ASSERT_TRUE(rows);
  ASSERT_EQ(rows->size(), 1);
  EXPECT_EQ(rows->front().id, 7);
  EXPECT_EQ(rows->front().label, 9);
  EXPECT_EQ(rows->front().path, root / "subdir/a.jpg");
  EXPECT_EQ(PathToUtf8(rows->front().path), PathToUtf8(root / "subdir/a.jpg"));
  rows = ReadSamples(path, Utf8ToPath("/override"), 0, 0);
  ASSERT_TRUE(rows);
  ASSERT_EQ(rows->size(), 2);
  EXPECT_EQ(rows->front().path, Utf8ToPath("/override/subdir/a.jpg"));
  EXPECT_EQ(rows->back().path, Utf8ToPath("/absolute/b.jpg"));
  rows = ReadSamples(path, {}, 1, 1);
  ASSERT_TRUE(rows);
  EXPECT_EQ(rows->front().id, 8);
  EXPECT_FALSE(ReadSamples(path, {}, 2, 1));
  EXPECT_FALSE(ReadSamples(path, {}, -1, 1));
  EXPECT_FALSE(ReadSamples(path, {}, 0, -1));
  for (const auto* text :
       {"", "id\tlabel\tpath\n", "bad\t2\ta.jpg\n", "1\tbad\ta.jpg\n",
        "1\t2\t\n", "1\t2\ta\textra\n", "1\t2\ta\n1\t3\tb\n"}) {
    write(text);
    EXPECT_FALSE(ReadSamples(path, {}, 0, 1)) << text;
  }
}

TEST(ComparisonTest, CodeSpaceRecoveryUsesMaximumAndMeanGates) {
  const LiteRtQuantizationPerTensor q{0.125f, -7};
  auto result = CompareQuantizedScores({0, 0, .25f, 0}, {0, 0, 0, 0}, q, 2, .5);
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->allclose);
  EXPECT_EQ(result->max_abs, 2);
  EXPECT_EQ(result->mean_abs, .5);
  result = CompareQuantizedScores({0, 0, .25f, .125f}, {0, 0, 0, 0}, q, 2, .5);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->allclose);
  EXPECT_FALSE(CompareQuantizedScores({0}, {0}, {0, 0}, 1, 1));
}

TEST(PreprocessingTest, GeometryAndPadding) {
  ImagePreprocessing config;
  config.resize = 224;
  auto dims = ResizedDimensions(400, 301, config);
  ASSERT_TRUE(dims);
  EXPECT_EQ(*dims, (std::array<int, 2>{297, 224}));
  auto offset5 = CropOffset(5);
  auto offset7 = CropOffset(7);
  ASSERT_TRUE(offset5);
  ASSERT_TRUE(offset7);
  EXPECT_EQ(*offset5, 2);
  EXPECT_EQ(*offset7, 4);
  EXPECT_FALSE(CropOffset(-1));
  config.resize = 1;
  config.crop_h = 3;
  config.crop_w = 3;
  config.mean = {0, 0, 0};
  config.stddev = {1, 1, 1};
  uint8_t rgb[] = {255, 128, 0};
  auto value = PreprocessRgb(rgb, 1, 1, config, false);
  ASSERT_TRUE(value);
  ASSERT_EQ(value->size(), 27);
  EXPECT_FLOAT_EQ((*value)[12], 1);
  EXPECT_FLOAT_EQ((*value)[13], 128 / 255.f);
  EXPECT_FLOAT_EQ((*value)[0], 0);
  EXPECT_FLOAT_EQ((*value)[26], 0);
}
TEST(PreprocessingTest, RejectsInvalidConfiguration) {
  ImagePreprocessing c;
  c.stddev[0] = 0;
  EXPECT_FALSE(ValidatePreprocessing(c));
  c = ImagePreprocessing();
  c.interpolation = "nearest";
  EXPECT_FALSE(ValidatePreprocessing(c));
  c = ImagePreprocessing();
  c.crop_h = -1;
  EXPECT_FALSE(ValidatePreprocessing(c));
}
TEST(ComparisonTest, TiesTolerancesAndNonfinite) {
  auto ranking = TopK({2, 3, 3, 0}, 3);
  ASSERT_TRUE(ranking);
  EXPECT_EQ(*ranking, (std::vector<int>{1, 2, 0}));
  auto close = CompareScores({1.0001f, 2}, {1, 2}, 0.0002, 0);
  ASSERT_TRUE(close);
  EXPECT_TRUE(close->allclose);
  auto difference = CompareScores({1, 3}, {1, 2}, 0, 0);
  ASSERT_TRUE(difference);
  EXPECT_FALSE(difference->allclose);
  EXPECT_DOUBLE_EQ(difference->max_abs, 1);
  EXPECT_DOUBLE_EQ(difference->mean_abs, 0.5);
  EXPECT_FALSE(CompareScores({1}, {1, 2}, 0, 0));
  EXPECT_FALSE(TopK({std::numeric_limits<float>::infinity()}, 1));
  EXPECT_FALSE(CompareScores({1}, {1}, -1, 0));
}
}  // namespace
}  // namespace litert::vision
