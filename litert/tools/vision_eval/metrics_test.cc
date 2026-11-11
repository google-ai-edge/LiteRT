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

#include "litert/tools/vision_eval/metrics.h"

#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <initializer_list>
#include <numeric>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/image.h"

namespace litert::vision {
namespace {
class MetricsTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto* test = testing::UnitTest::GetInstance()->current_test_info();
    root_ = Utf8ToPath(testing::TempDir()) / test->name();
    std::error_code ec;
    std::filesystem::create_directories(root_, ec);
    ASSERT_FALSE(ec);
  }
  OutputSpec Output(Task task, std::initializer_list<int> shape) {
    OutputSpec output;
    output.mutable_tensor()->set_name("output");
    for (auto dim : shape) output.mutable_tensor()->add_shape(dim);
    output.mutable_scoring()->set_task(task);
    return output;
  }
  template <class T>
  TargetFile Target(const std::vector<T>& values, ElementType type,
                    std::initializer_list<int> shape,
                    const std::string& name = "target") {
    TargetFile target;
    target.set_name("output");
    target.mutable_tensor()->set_type(type);
    for (auto dim : shape) target.mutable_tensor()->add_shape(dim);
    target.mutable_file()->set_path(name + ".raw");
    EXPECT_TRUE(WriteBytes(root_ / Utf8ToPath(target.file().path()),
                           values.data(), values.size() * sizeof(T)));
    return target;
  }
  std::filesystem::path root_;
};

TEST_F(MetricsTest, HalfPixelResizeAndClassificationTies) {
  float plane[] = {1, 3};
  auto resized = ResizePlane(plane, 1, 2, 1, 3);
  ASSERT_TRUE(resized);
  EXPECT_EQ(*resized, (std::vector<float>{1, 2, 3}));
  auto classification = Output(CLASSIFICATION, {1, 3});
  auto score = Score(classification, {1, 1, 0}, nullptr, 0, root_, root_);
  ASSERT_TRUE(score);
  EXPECT_EQ(score->classification().correct_top1(), 1);
  EXPECT_FALSE(
      Score(classification, {1, 2, std::nanf("")}, nullptr, 0, root_, root_));
}

TEST_F(MetricsTest, SegmentationResizesLogitsBeforeArgmaxAndIgnoresLabels) {
  auto output = Output(SEGMENTATION, {1, 2, 1, 2});
  output.mutable_scoring()->set_classes(2);
  auto target = Target<uint8_t>({1, 0, 255}, UINT8, {1, 3});
  auto row = Score(output, {0, 2, 1, 0}, &target, -1, root_, root_);
  ASSERT_TRUE(row);
  auto metrics = Aggregate({*row}, root_ / "scratch");
  ASSERT_TRUE(metrics);
  EXPECT_EQ(metrics->segmentation().valid_pixels(), 2);
  EXPECT_EQ(metrics->segmentation().confusion(0), 1);
  EXPECT_EQ(metrics->segmentation().confusion(3), 1);
  EXPECT_EQ(metrics->segmentation().miou(), 1);
  // Identical logits choose the lowest channel index, including NHWC layout.
  output.mutable_tensor()->mutable_shape()->Clear();
  for (int dim : {1, 1, 3, 2}) output.mutable_tensor()->add_shape(dim);
  output.mutable_tensor()->set_layout(NHWC);
  row = Score(output, {1, 1, 1, 1, 1, 1}, &target, -1, root_, root_);
  ASSERT_TRUE(row);
  EXPECT_EQ(row->segmentation().confusion(2), 1);
  EXPECT_EQ(row->segmentation().confusion(0), 1);
  target = Target<uint8_t>({255, 255, 255}, UINT8, {1, 3});
  row = Score(output, {1, 1, 1, 1, 1, 1}, &target, -1, root_, root_);
  ASSERT_TRUE(row);
  EXPECT_FALSE(row->segmentation().has_miou());
}

TEST_F(MetricsTest, DepthAveragesImagesRatherThanPoolingPixels) {
  auto output = Output(DEPTH, {1, 1, 1, 4});
  auto target1 =
      Target<float>({1, 1, 0, std::nanf("")}, FLOAT32, {1, 4}, "first");
  auto target2 = Target<float>({1, 0, 0, 0}, FLOAT32, {1, 4}, "second");
  auto first = Score(output, {2, 2, 9, 9}, &target1, -1, root_, root_);
  auto second = Score(output, {1, 9, 9, 9}, &target2, -1, root_, root_);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto metrics = Aggregate({*first, *second}, root_ / "scratch");
  ASSERT_TRUE(metrics);
  EXPECT_EQ(metrics->depth().images(), 2);
  EXPECT_EQ(metrics->depth().valid_pixels(), 3);
  EXPECT_DOUBLE_EQ(metrics->depth().abs_rel(), .5);
  EXPECT_DOUBLE_EQ(metrics->depth().rmse(), .5);
  EXPECT_DOUBLE_EQ(metrics->depth().delta1(), .5);
  auto empty = Target<float>({0, 0, 0, 0}, FLOAT32, {1, 4});
  auto row = Score(output, {1, 1, 1, 1}, &empty, -1, root_, root_);
  ASSERT_TRUE(row);
  EXPECT_EQ(row->depth().empty_images(), 1);
  EXPECT_FALSE(row->depth().has_abs_rel());
}

TEST_F(MetricsTest, NormalsAreOrientedAndZeroPredictionsScoreNinety) {
  auto output = Output(NORMALS, {1, 3, 1, 4});
  auto target =
      Target<float>({1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, FLOAT32, {3, 1, 4});
  auto score = Score(output, {1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, &target, -1,
                     root_, root_);
  ASSERT_TRUE(score);
  auto metrics = Aggregate({*score}, root_ / "scratch");
  ASSERT_TRUE(metrics);
  EXPECT_EQ(metrics->normals().valid_pixels(), 4);
  EXPECT_DOUBLE_EQ(metrics->normals().mean_degrees(), 90);
  EXPECT_DOUBLE_EQ(metrics->normals().median_degrees(), 90);
  EXPECT_DOUBLE_EQ(metrics->normals().within_11_25_degrees(), .25);
}

TEST_F(MetricsTest, ExactMedianUsesMultipleBoundedMergePasses) {
  std::vector<float> values(130);
  std::iota(values.rbegin(), values.rend(), 0);
  auto path = root_ / "errors.raw";
  ASSERT_TRUE(WriteBytes(path, values.data(), values.size() * sizeof(float)));
  auto identity = Identify(path);
  ASSERT_TRUE(identity);
  auto median = ExactMedian({*identity}, root_ / "scratch", 1);
  ASSERT_TRUE(median);
  EXPECT_DOUBLE_EQ(*median, 64.5);
  std::error_code ec;
  EXPECT_FALSE(std::filesystem::exists(root_ / "scratch", ec));
  EXPECT_FALSE(ExactMedian({*identity}, root_ / "scratch", 0));
  ASSERT_TRUE(
      WriteBytes(path, values.data(), values.size() * sizeof(float) - 1));
  identity = Identify(path);
  ASSERT_TRUE(identity);
  EXPECT_FALSE(ExactMedian({*identity}, root_ / "scratch", 1));
}
}  // namespace
}  // namespace litert::vision
