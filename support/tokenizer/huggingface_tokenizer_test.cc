// Copyright 2025 The ODML Authors.
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

#include "support/tokenizer/huggingface_tokenizer.h"

#include <fcntl.h>

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizer.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::support {
namespace {

using ::testing::status::IsOkAndHolds;

constexpr char kTestdataDir[] =
    "litert/support/tokenizer/testdata/";

std::string GetHuggingFaceModelPath() {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
          "tokenizer.json")
      .string();
}

absl::StatusOr<std::string> GetContents(const std::string& path) {
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return absl::InternalError(absl::StrCat("Could not open file: ", path));
  }

  std::string content;
  content.assign((std::istreambuf_iterator<char>(input_stream)),
                 (std::istreambuf_iterator<char>()));
  return std::move(content);
}

TEST(HuggingFaceTokenizerTtest, CreateFromFile) {
  auto tokenizer_or =
      HuggingFaceTokenizer::CreateFromFile(GetHuggingFaceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(HuggingFaceTokenizerTtest, CreateFromBuffer) {
  ASSERT_OK_AND_ASSIGN(auto json, GetContents(GetHuggingFaceModelPath()));
  auto tokenizer_or = HuggingFaceTokenizer::CreateFromJson(json);
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(HuggingFaceTokenizerTtest, Create) {
  auto tokenizer_or =
      HuggingFaceTokenizer::CreateFromFile(GetHuggingFaceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(HuggingFaceTokenizerTtest, GetTokenizerType) {
  auto tokenizer_or =
      HuggingFaceTokenizer::CreateFromFile(GetHuggingFaceModelPath());
  EXPECT_EQ(tokenizer_or.value()->GetTokenizerType(),
            TokenizerType::kHuggingFace);
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(HuggingFaceTokenizerTest, TextToTokenIds) {
  auto tokenizer_or =
      HuggingFaceTokenizer::CreateFromFile(GetHuggingFaceModelPath());
  ASSERT_OK(tokenizer_or);
  auto tokenizer = std::move(tokenizer_or.value());

  absl::string_view text = "How's it going?";
  auto ids_or = tokenizer->TextToTokenIds(text);
  ASSERT_OK(ids_or);

  EXPECT_THAT(ids_or.value(), ::testing::ElementsAre(2020, 506, 357, 2045, 47));
}

TEST(HuggingFaceTokenizerTest, TokenToId) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, HuggingFaceTokenizer::CreateFromFile(
                                           GetHuggingFaceModelPath()));
  EXPECT_THAT(tokenizer->TokenToId("X"), IsOkAndHolds(72));
}

TEST(HuggingFaceTokenizerTest, TokenIdsToText) {
  auto tokenizer_or =
      HuggingFaceTokenizer::CreateFromFile(GetHuggingFaceModelPath());
  ASSERT_OK(tokenizer_or);
  auto tokenizer = std::move(tokenizer_or.value());

  const std::vector<int> ids = {2020, 506, 357, 2045, 47};
  auto text_or = tokenizer->TokenIdsToText(ids);
  ASSERT_OK(text_or);

  EXPECT_EQ(text_or.value(), "How's it going?");
}

TEST(HuggingFaceTokenizerTest, GetTokens) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, HuggingFaceTokenizer::CreateFromFile(
                                           GetHuggingFaceModelPath()));

  std::vector<std::string> tokens = tokenizer->GetTokens();

  // Check number of tokens.
  EXPECT_EQ(tokens.size(), 49152);

  // Check a few tokens.
  EXPECT_EQ(tokens[0], "<|endoftext|>");
  EXPECT_EQ(tokens[1], "<|im_start|>");
  EXPECT_EQ(tokens[3], "<repo_name>");
  EXPECT_EQ(tokens[17], "!");
  EXPECT_EQ(tokens[47], "?");
  EXPECT_EQ(tokens[72], "X");
}

}  // namespace
}  // namespace litert::support
