/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

// std::bit_cast is only available since c++20, provide a handrolled version
// of bit_cast here. Implementation modified from abseil bit_cast.
template <
    typename Dest, typename Source,
    typename std::enable_if<sizeof(Dest) == sizeof(Source) &&
                                std::is_trivially_copyable<Source>::value &&
                                std::is_trivially_copyable<Dest>::value &&
                                std::is_default_constructible<Dest>::value,
                            int>::type = 0>
inline Dest bit_cast(const Source& source) {
  Dest dest;
  memcpy(static_cast<void*>(std::addressof(dest)),
         static_cast<const void*>(std::addressof(source)), sizeof(dest));
  return dest;
}

class BitcastOpModel : public SingleOpModel {
 public:
  BitcastOpModel(const TensorData& input, const TensorData& output,
                 bool allocate_and_delegate = true) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_BITCAST, BuiltinOptions_BitcastOptions,
                 CreateBitcastOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true, allocate_and_delegate);
  }

  int input() const { return input_; }
  int output() const { return output_; }

 protected:
  int input_;
  int output_;
};

TEST(BitcastOpModel, BitcastInt32ToUint32) {
  BitcastOpModel m({TensorType_INT32, {2, 3}}, {TensorType_UINT32, {2, 3}});
  std::vector<int32_t> input = {INT32_MIN, -100, -1, 0, 100, INT32_MAX};
  m.PopulateTensor<int32_t>(m.input(), input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<uint32_t> output;

  std::transform(input.cbegin(), input.cend(), std::back_inserter(output),
                 [](int32_t a) { return bit_cast<std::uint32_t>(a); });
  EXPECT_THAT(m.ExtractVector<uint32_t>(m.output()), ElementsAreArray(output));
}

TEST(BitcastOpModel, BitcastUInt32ToInt32Inplace) {
  BitcastOpModel m({TensorType_UINT32, {2, 3}}, {TensorType_INT32, {2, 3}});
  std::vector<uint32_t> input = {0,
                                 1,
                                 100,
                                 bit_cast<uint32_t>(INT32_MAX),
                                 bit_cast<uint32_t>(INT32_MIN),
                                 UINT32_MAX};
  m.PopulateTensor<uint32_t>(m.input(), input);
  const int kInplaceTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<int32_t> output;

  std::transform(input.cbegin(), input.cend(), std::back_inserter(output),
                 [](uint32_t a) { return bit_cast<std::uint32_t>(a); });
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()), ElementsAreArray(output));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TEST(BitcastOpModel, BitcastUInt32ToInt32) {
  BitcastOpModel m({TensorType_UINT32, {2, 3}}, {TensorType_INT32, {2, 3}});
  std::vector<uint32_t> input = {0,
                                 1,
                                 100,
                                 bit_cast<uint32_t>(INT32_MAX),
                                 bit_cast<uint32_t>(INT32_MIN),
                                 UINT32_MAX};
  m.PopulateTensor<uint32_t>(m.input(), input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<int32_t> output;

  std::transform(input.cbegin(), input.cend(), std::back_inserter(output),
                 [](uint32_t a) { return bit_cast<std::uint32_t>(a); });
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()), ElementsAreArray(output));
}

TEST(BitcastOpModel, BitcastUInt32Toint16) {
  BitcastOpModel m({TensorType_UINT32, {2, 1}}, {TensorType_INT16, {2, 1, 2}});
  std::vector<uint32_t> input = {(uint32_t)UINT16_MAX + 1,
                                 (uint32_t)UINT16_MAX};
  m.PopulateTensor<uint32_t>(m.input(), input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // 00..01 00..00
  // 00..00 11..11
  std::vector<int16_t> output = {1, 0, 0, -1};
#else
  // 00..00 00..01
  // 11..11 00..00
  std::vector<int16_t> output = {0, 1, -1, 0};
#endif
  EXPECT_THAT(m.ExtractVector<int16_t>(m.output()), ElementsAreArray(output));
}

TEST(BitcastOpModel, BitcastInt16ToUint32) {
  BitcastOpModel m({TensorType_INT16, {2, 1, 2}}, {TensorType_UINT32, {2, 1}});
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  std::vector<int16_t> input = {1, 0, 0, -1};
#else
  std::vector<int16_t> input = {0, 1, -1, 0};
#endif
  m.PopulateTensor<int16_t>(m.input(), input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<uint32_t> output = {(uint32_t)UINT16_MAX + 1,
                                  (uint32_t)UINT16_MAX};
  EXPECT_THAT(m.ExtractVector<uint32_t>(m.output()), ElementsAreArray(output));
}

TEST(BitcastOpModel, BitcastInt16ToUint32WrongShape) {
  BitcastOpModel m({TensorType_INT16, {2, 2, 7}},
                   {TensorType_UINT32, {2, 7}},
                   /*allocate_and_delegate=*/false);
  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

TEST(BitcastOpModel, ScalarWideningRequiresFinalDimension) {
  BitcastOpModel m({TensorType_INT16, {}}, {TensorType_UINT32, {}},
                   /*allocate_and_delegate=*/false);
  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

TEST(BitcastOpModel, ScalarEqualSize) {
  BitcastOpModel m({TensorType_INT32, {}}, {TensorType_UINT32, {}});
  m.PopulateTensor<int32_t>(m.input(), {-1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetTensorShape(m.output()).empty());
  EXPECT_THAT(m.ExtractVector<uint32_t>(m.output()),
              ElementsAreArray({UINT32_MAX}));
}

TEST(BitcastOpModel, ScalarNarrowingAddsFinalDimension) {
  BitcastOpModel m({TensorType_UINT32, {}}, {TensorType_UINT16, {}});
  m.PopulateTensor<uint32_t>(m.input(), {UINT32_MAX});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({2}));
  EXPECT_THAT(m.ExtractVector<uint16_t>(m.output()),
              ElementsAreArray({UINT16_MAX, UINT16_MAX}));
}

TEST(BitcastOpModel, WideningOneDimensionProducesScalar) {
  BitcastOpModel m({TensorType_UINT16, {2}}, {TensorType_UINT32, {}});
  m.PopulateTensor<uint16_t>(m.input(), {UINT16_MAX, UINT16_MAX});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetTensorShape(m.output()).empty());
  EXPECT_THAT(m.ExtractVector<uint32_t>(m.output()),
              ElementsAreArray({UINT32_MAX}));
}

class BitcastPackedTypeTest
    : public ::testing::TestWithParam<std::pair<TensorType, TensorType>> {};

TEST_P(BitcastPackedTypeTest, PreservesEqualWidthPackedShapeAndBytes) {
  const auto types = GetParam();
  const int bit_width = types.first == TensorType_INT2 ? 2 : 4;
  for (const std::vector<int>& shape :
       {std::vector<int>{}, std::vector<int>{5}, std::vector<int>{2, 3}}) {
    BitcastOpModel m({types.first, shape}, {types.second, {}},
                     /*allocate_and_delegate=*/false);
    ASSERT_EQ(m.AllocateTensors(), kTfLiteOk);
    const TfLiteTensor* input = m.GetInputTensor(0);
    const TfLiteTensor* output = m.GetOutputTensor(0);
    size_t elements = 1;
    for (int dimension : shape) elements *= dimension;
    const size_t expected_bytes = (elements * bit_width + 7) / 8;
    ASSERT_EQ(input->bytes, expected_bytes);
    ASSERT_EQ(output->bytes, expected_bytes);
    std::vector<uint8_t> expected(expected_bytes);
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] = static_cast<uint8_t>(0xa5 + i);
    }
    memcpy(input->data.raw, expected.data(), expected.size());
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray(shape));
    EXPECT_EQ(memcmp(output->data.raw, expected.data(), expected.size()), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    EqualWidthPackedTypes, BitcastPackedTypeTest,
    ::testing::Values(std::make_pair(TensorType_INT4, TensorType_UINT4),
                      std::make_pair(TensorType_UINT4, TensorType_INT4),
                      std::make_pair(TensorType_INT4, TensorType_INT4),
                      std::make_pair(TensorType_UINT4, TensorType_UINT4),
                      std::make_pair(TensorType_INT2, TensorType_INT2)));

class BitcastCrossWidthPackedTypeTest
    : public ::testing::TestWithParam<std::pair<TensorType, TensorType>> {};

TEST_P(BitcastCrossWidthPackedTypeTest, RejectsCrossWidthPackedTypes) {
  const auto types = GetParam();
  BitcastOpModel m({types.first, {4}}, {types.second, {}},
                   /*allocate_and_delegate=*/false);
  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

INSTANTIATE_TEST_SUITE_P(
    CrossWidthPackedTypes, BitcastCrossWidthPackedTypeTest,
    ::testing::Values(std::make_pair(TensorType_INT8, TensorType_INT4),
                      std::make_pair(TensorType_INT4, TensorType_INT8),
                      std::make_pair(TensorType_INT8, TensorType_UINT4),
                      std::make_pair(TensorType_UINT4, TensorType_INT8),
                      std::make_pair(TensorType_INT8, TensorType_INT2),
                      std::make_pair(TensorType_INT2, TensorType_INT8),
                      std::make_pair(TensorType_INT4, TensorType_INT2),
                      std::make_pair(TensorType_INT2, TensorType_INT4),
                      std::make_pair(TensorType_UINT4, TensorType_INT2),
                      std::make_pair(TensorType_INT2, TensorType_UINT4)));

TEST(BitcastOpModel, RejectsSmallerOutputByteCount) {
  BitcastOpModel m({TensorType_INT32, {2}}, {TensorType_UINT32, {2}});
  m.PopulateTensor<int32_t>(m.input(), {1, 2});
  TfLiteTensor* output = m.GetOutputTensor(0);
  const size_t output_bytes = output->bytes;
  output->bytes = output_bytes - 1;
  EXPECT_EQ(m.Invoke(), kTfLiteError);
  output->bytes = output_bytes;
}

TEST(BitcastOpModel, RejectsLargerOutputByteCount) {
  BitcastOpModel m({TensorType_INT32, {2}}, {TensorType_UINT32, {2}});
  m.PopulateTensor<int32_t>(m.input(), {1, 2});
  TfLiteTensor* output = m.GetOutputTensor(0);
  const size_t output_bytes = output->bytes;
  output->bytes = output_bytes + 1;
  EXPECT_EQ(m.Invoke(), kTfLiteError);
  output->bytes = output_bytes;
}

TEST(BitcastOpModel, RejectsByteCountMismatchWithSharedBuffer) {
  BitcastOpModel m({TensorType_INT32, {2}}, {TensorType_UINT32, {2}});
  m.PopulateTensor<int32_t>(m.input(), {1, 2});
  const TfLiteTensor* input = m.GetInputTensor(0);
  TfLiteTensor* output = m.GetOutputTensor(0);
  void* output_data = output->data.data;
  const size_t output_bytes = output->bytes;
  output->data.data = input->data.data;
  output->bytes = output_bytes - 1;
  EXPECT_EQ(m.Invoke(), kTfLiteError);
  output->bytes = output_bytes;
  output->data.data = output_data;
}

}  // namespace
}  // namespace tflite
