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

#include "litert/cc/litert_string_util.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

using ::litert::util::CreateTensorBufferFromStrings;
using ::litert::util::DeserializeStrings;
using ::litert::util::GetStringsFromTensorBuffer;
using ::litert::util::SerializeStrings;

TEST(StringUtilTest, RoundTrip) {
  std::vector<std::string> original = {"hello", "", "world", "liteRT"};
  std::vector<uint8_t> serialized = SerializeStrings(original);
  std::vector<std::string> deserialized =
      DeserializeStrings(serialized.data(), serialized.size());
  EXPECT_EQ(original, deserialized);
}

TEST(StringUtilTest, EmptyVector) {
  std::vector<std::string> original = {};
  std::vector<uint8_t> serialized = SerializeStrings(original);

  // Should be 8 bytes: 0 (num_strs) and 8 (offset[0])
  EXPECT_EQ(serialized.size(), 8);

  std::vector<std::string> deserialized =
      DeserializeStrings(serialized.data(), serialized.size());
  EXPECT_TRUE(deserialized.empty());
}

TEST(StringUtilTest, AllEmptyStrings) {
  std::vector<std::string> original = {"", "", ""};
  std::vector<uint8_t> serialized = SerializeStrings(original);

  // Should be 20 bytes: 3 (num_strs) and 4 offsets of value 20.
  EXPECT_EQ(serialized.size(), 20);

  std::vector<std::string> deserialized =
      DeserializeStrings(serialized.data(), serialized.size());
  EXPECT_EQ(original, deserialized);
}

TEST(StringUtilTest, MatchTfliteLayout) {
  std::vector<std::string> inputs = {"AB", ""};
  std::vector<uint8_t> serialized = SerializeStrings(inputs);

  // Expected layout:
  // [0, 3]   : 2 (num_strs)
  // [4, 7]   : 16 (offset of "AB")
  // [8, 11]  : 18 (offset of "")
  // [12, 15] : 18 (total size)
  // [16, 17] : 'A', 'B'

  ASSERT_EQ(serialized.size(), 18);

  int32_t num_strs;
  std::memcpy(&num_strs, serialized.data(), 4);
  EXPECT_EQ(num_strs, 2);

  const int32_t* offsets =
      reinterpret_cast<const int32_t*>(serialized.data() + 4);
  EXPECT_EQ(offsets[0], 16);
  EXPECT_EQ(offsets[1], 18);
  EXPECT_EQ(offsets[2], 18);

  EXPECT_EQ(serialized[16], 'A');
  EXPECT_EQ(serialized[17], 'B');
}

TEST(StringUtilTest, InvalidData) {
  // Too small to even hold num_strs
  uint8_t data1[] = {1, 2, 3};
  EXPECT_TRUE(DeserializeStrings(data1, sizeof(data1)).empty());

  // Too small to hold offsets
  // num_strs = 2, but we only provide 8 bytes total (need 4 + 12 = 16 bytes
  // minimum)
  union {
    int32_t val[2];
    uint8_t bytes[8];
  } data2;
  data2.val[0] = 2;
  data2.val[1] = 16;
  EXPECT_TRUE(DeserializeStrings(data2.bytes, sizeof(data2.bytes)).empty());

  // Bad offsets (negative)
  // num_strs = 1, offset[0] = -1, offset[1] = 10
  union {
    int32_t val[3];
    uint8_t bytes[12];
  } data3;
  data3.val[0] = 1;
  data3.val[1] = -1;
  data3.val[2] = 10;
  EXPECT_TRUE(DeserializeStrings(data3.bytes, sizeof(data3.bytes)).empty());

  // Bad offsets (out of bounds)
  // num_strs = 1, offset[0] = 12, offset[1] = 20, but size is 15
  union {
    struct {
      int32_t num_strs;
      int32_t offsets[2];
      char data[3];
    } val;
    uint8_t bytes[15];
  } data4;
  data4.val.num_strs = 1;
  data4.val.offsets[0] = 12;
  data4.val.offsets[1] = 20;  // Out of bounds (15)
  EXPECT_TRUE(DeserializeStrings(data4.bytes, sizeof(data4.bytes)).empty());
}

TEST(StringUtilTest, TensorBufferRoundTrip) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, Environment::Create({}));

  std::vector<std::string> original = {"hello", "world", "LiteRT"};

  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer buffer,
                              CreateTensorBufferFromStrings(env, original));

  LITERT_ASSERT_OK_AND_ASSIGN(auto type, buffer.TensorType());
  EXPECT_EQ(type.Layout().Rank(), 1);
  EXPECT_EQ(type.Layout().Dimensions()[0], 3);
  EXPECT_EQ(type.ElementType(), ElementType::TfString);

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<std::string> deserialized,
                              GetStringsFromTensorBuffer(buffer));
  EXPECT_EQ(original, deserialized);
}

}  // namespace
}  // namespace litert
