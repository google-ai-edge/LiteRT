/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <limits.h>
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "tflite/kernels/test_util.h"
#include "tflite/kernels/unsorted_segment_test.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class UnsortedSegmentMaxModel : public UnsortedSegmentModel<T> {
 public:
  UnsortedSegmentMaxModel(const TensorData& data, const TensorData& segment_ids,
                          const TensorData& num_segments)
      : UnsortedSegmentModel<T>(data, segment_ids, num_segments,
                                BuiltinOperator_UNSORTED_SEGMENT_MAX,
                                BuiltinOptions_NONE) {}

  explicit UnsortedSegmentMaxModel(
      const TensorData& data, const std::initializer_list<int>& segment_id_data,
      const std::initializer_list<int>& segment_id_shape,
      const std::initializer_list<int>& num_segments_data,
      const std::initializer_list<int>& num_segments_shape)
      : UnsortedSegmentModel<T>(data, segment_id_data, segment_id_shape,
                                num_segments_data, num_segments_shape,
                                BuiltinOperator_UNSORTED_SEGMENT_MAX,
                                BuiltinOptions_NONE) {}
};

INSTANTIATE_TEST_SUITE_P(UnsortedSegmentMaxTestP, UnsortedSegmentTest,
                         testing::Values(BuiltinOperator_UNSORTED_SEGMENT_MAX));

TEST(UnsortedSegmentMaxModelTest, Int32Test_Simple) {
  UnsortedSegmentMaxModel<int32_t> model({TensorType_INT32, {6}},
                                         {TensorType_INT32, {6}},
                                         {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {5, 1, 7, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 1, 1, 0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5, 7}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(UnsortedSegmentMaxModelTest, Int32Test_Simple2D) {
  UnsortedSegmentMaxModel<int32_t> model({TensorType_INT32, {3, 4}},
                                         {TensorType_INT32, {3}},
                                         {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(),
                                {1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 3, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(UnsortedSegmentMaxModelTest, FloatTest_Simple) {
  UnsortedSegmentMaxModel<float> model({TensorType_FLOAT32, {8}},
                                       {TensorType_INT32, {8}},
                                       {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.data(),
                              {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 0, 1, 7, 7, 7, 7, 7});
  model.PopulateTensor<int32_t>(model.num_segments(), {8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {2.0, 3.0, std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(), 4.0})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({8}));
}

TEST(UnsortedSegmentMaxModelTest, FloatTest_Simple2D) {
  UnsortedSegmentMaxModel<float> model({TensorType_FLOAT32, {3, 4}},
                                       {TensorType_INT32, {3}},
                                       {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.data(), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                             8.0, 4.0, 3.0, 2.0, 1.0});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({4.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(UnsortedSegmentMaxModelTest, SegmentsAreNegative) {
  UnsortedSegmentMaxModel<int32_t> model({TensorType_INT32, {2, 2}},
                                         {TensorType_INT32, {2}},
                                         {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {-1, -1});
  model.PopulateTensor<int32_t>(model.num_segments(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({std::numeric_limits<int32_t>::lowest(),
                                std::numeric_limits<int32_t>::lowest()}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2}));
}

TEST(UnsortedSegmentMaxModelTest, ConstParamenterTest) {
  UnsortedSegmentMaxModel<int32_t> model({TensorType_INT32, {3, 2}}, {0, 1, 0},
                                         {3}, {2}, {1});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5, 6, 3, 4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2}));
}

}  // namespace
}  // namespace tflite
