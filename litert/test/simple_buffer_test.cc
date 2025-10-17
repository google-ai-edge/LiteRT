// Copyright 2025 Google LLC.
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

#include "litert/test/simple_buffer.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_model.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"

namespace litert {
namespace testing {
namespace {

using ::testing::ElementsAre;

TEST(TensorBufferHelperTest, CreateWithTensorType) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create(MakeRankedTensorType<int32_t>({2, 2})));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  EXPECT_EQ(buf.Size(), 4 * sizeof(int32_t));
}

TEST(TensorBufferHelperTest, CreateWithDims) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto buf, SimpleBuffer::Create<float>({2, 2}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Float32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  EXPECT_EQ(buf.Size(), 4 * sizeof(float));
}

TEST(TensorBufferHelperTest, CreateWithData) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  EXPECT_EQ(buf.Size(), 4 * sizeof(int32_t));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(1, 2, 3, 4));
}

TEST(TensorBufferHelperTest, ToAndFromTensorBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  EXPECT_EQ(buf.Size(), 4 * sizeof(int32_t));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(1, 2, 3, 4));
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buf, buf.SpawnTensorBuffer());
  LITERT_ASSERT_OK_AND_ASSIGN(auto buf2,
                              SimpleBuffer::FromTensorBuffer(tensor_buf));
  EXPECT_THAT(buf2.Span<int32_t>(), ElementsAre(1, 2, 3, 4));
}

TEST(TensorBufferHelperTest, ReadNonDividingSizedData) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<uint8_t>({5}, {'a', 'b', 'c', 'd', 'e'}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::UInt8);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 5);
  EXPECT_EQ(buf.TypedNumElements<uint8_t>(), 5);
  EXPECT_EQ(buf.TypedSize<uint8_t>(), 5);
  EXPECT_EQ(buf.TypedNumElements<uint16_t>(), 2);
  EXPECT_EQ(buf.TypedSize<uint16_t>(), 4);
  EXPECT_EQ(buf.Size(), 5);
  EXPECT_THAT(buf.Span<uint8_t>(), ElementsAre('a', 'b', 'c', 'd', 'e'));
  auto wide_span = buf.Span<uint16_t>();
  auto bytes_span = absl::Span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(wide_span.data()),
      wide_span.size() * sizeof(uint16_t));
  EXPECT_THAT(bytes_span, ElementsAre('a', 'b', 'c', 'd'));
}

TEST(TensorBufferHelperTest, Write) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  EXPECT_EQ(buf.Size(), 4 * sizeof(int32_t));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(1, 2, 3, 4));
  LITERT_ASSERT_OK(buf.Write({4, 3, 2, 1}));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(4, 3, 2, 1));
}

TEST(TensorBufferHelperTest, WritePrefix) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  EXPECT_EQ(buf.Size(), 4 * sizeof(int32_t));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(1, 2, 3, 4));
  LITERT_ASSERT_OK(buf.Write({4, 3}));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(4, 3, 3, 4));
}

TEST(TensorBufferHelperTest, WriteTruncate) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  EXPECT_EQ(buf.Size(), 4 * sizeof(int32_t));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(1, 2, 3, 4));
  LITERT_ASSERT_OK(buf.Write({4, 3, 2, 1, 0, 0}));
  EXPECT_EQ(buf.TypedNumElements<int32_t>(), 4);
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(4, 3, 2, 1));
}

TEST(TensorBufferHelperTest, WriteOffset) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  EXPECT_EQ(buf.Size(), 4 * sizeof(int32_t));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(1, 2, 3, 4));
  LITERT_ASSERT_OK(buf.Write({4, 3}, 2));
  EXPECT_EQ(buf.TypedNumElements<int32_t>(), 4);
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(1, 2, 4, 3));
}

using RngTensorBufferHelperTest = RngTest;

TEST_F(RngTensorBufferHelperTest, WriteRandom) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  auto device = this->TracedDevice();
  LITERT_ASSERT_OK((buf.WriteRandom<int32_t, DummyGenerator>(device)));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(0, 1, 2, 3));
}

TEST_F(RngTensorBufferHelperTest, WriteRandomOffset) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  auto device = this->TracedDevice();
  LITERT_ASSERT_OK((buf.WriteRandom<int32_t, DummyGenerator>(device, 2)));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(1, 2, 0, 1));
}

TEST_F(RngTensorBufferHelperTest, WriteRandomSubArray) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  auto device = this->TracedDevice();
  LITERT_ASSERT_OK((buf.WriteRandom<int32_t, DummyGenerator>(device, 1, 2)));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(1, 0, 1, 4));
}

TEST_F(RngTensorBufferHelperTest, WriteRandomWithBuilder) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buf, SimpleBuffer::Create<int32_t>({2, 2}, {1, 2, 3, 4}));
  const auto& type = buf.Type();
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_elements, type.Layout().NumElements());
  EXPECT_EQ(num_elements, 4);
  auto device = this->TracedDevice();
  RandomTensorDataBuilder b;
  b.SetIntDummy();
  LITERT_ASSERT_OK((buf.WriteRandom<int32_t>(b, device)));
  EXPECT_THAT(buf.Span<int32_t>(), ElementsAre(0, 1, 2, 3));
}

}  // namespace
}  // namespace testing
}  // namespace litert
