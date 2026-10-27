// Copyright 2026 Google LLC.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/imagenet_eval_data.h"

#include <limits>


#include "gtest/gtest.h"

namespace litert::imagenet {
namespace {
TEST(Sha256Test, KnownVectorsAndStreaming) {
  EXPECT_EQ(HashBytes("", 0),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(HashBytes("abc", 3),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  std::string text = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  EXPECT_EQ(HashBytes(text.data(), text.size()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  Sha256 hash;
  std::string chunk(1000, 'a');
  for (int i = 0; i < 1000; ++i) hash.Update(chunk.data(), chunk.size());
  EXPECT_EQ(hash.Hex(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  EXPECT_EQ(hash.Hex(),
            hash.Hex());  // Finalizing a snapshot must not mutate state.
}

TEST(PreprocessingTest, GeometryAndPadding) {
  Preprocessing config;
  config.resize = 224;
  auto dims = ResizedDimensions(400, 301, config);
  ASSERT_TRUE(dims);
  EXPECT_EQ(*dims,
            (std::array<int, 2>{297, 224}));
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
  Preprocessing c;
  c.stddev[0] = 0;
  EXPECT_FALSE(ValidatePreprocessing(c));
  c = Preprocessing();
  c.interpolation = "nearest";
  EXPECT_FALSE(ValidatePreprocessing(c));
  c = Preprocessing();
  c.crop_h = -1;
  EXPECT_FALSE(ValidatePreprocessing(c));
}
TEST(PreprocessingTest, RejectsMalformedJsonWithoutExceptions) {
  auto valid = DescribePreprocessing(Preprocessing{});
  EXPECT_TRUE(ParsePreprocessing(valid));
  EXPECT_FALSE(ParsePreprocessing(Json::array()));
  for (const char* field : {"resize", "crop_h", "crop_w", "mean", "std",
                            "interpolation", "antialias"}) {
    auto missing = valid;
    missing.erase(field);
    EXPECT_FALSE(ParsePreprocessing(missing)) << field;
    auto wrong_type = valid;
    wrong_type[field] = Json::object();
    EXPECT_FALSE(ParsePreprocessing(wrong_type)) << field;
  }
  valid["resize"] = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(ParsePreprocessing(valid));
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
}  // namespace litert::imagenet
