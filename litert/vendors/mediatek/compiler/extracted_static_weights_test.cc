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

#include "litert/vendors/mediatek/compiler/extracted_static_weights.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"

namespace litert::mediatek {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

constexpr uint32_t kMagic = 0xbbced1ec;

// Returns extracted weights file contents with the given payload and JSON,
// followed by a footer with the given JSON offset (by default, right after the
// payload) and magic number.
std::string MakeContents(const std::string& payload, absl::string_view json,
                         std::optional<uint64_t> json_offset = std::nullopt,
                         uint32_t magic = kMagic) {
  const uint64_t offset = json_offset.value_or(payload.size());
  std::string contents = payload;
  contents.append(json);
  contents.append(reinterpret_cast<const char*>(&offset), sizeof(offset));
  contents.append(reinterpret_cast<const char*>(&magic), sizeof(magic));
  return contents;
}

// Expects parsing `contents` to fail with `kLiteRtStatusErrorInvalidArgument`.
void ExpectInvalid(absl::string_view contents) {
  auto weights = ParseExtractedStaticWeights(contents);
  ASSERT_FALSE(weights.HasValue());
  EXPECT_EQ(weights.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(ParseExtractedStaticWeightsTest, ReturnsWeightsInInputOrder) {
  // Orders need not start at zero, and entries need not be sorted.
  const std::string contents =
      MakeContents("\x01\x02\x03\x04\x05\x06\x07",
                   R"({"2": ["4", "3"], "1": ["0", "4"], "3": ["7", "0"]})");

  auto weights = ParseExtractedStaticWeights(contents);
  ASSERT_TRUE(weights.HasValue());
  EXPECT_THAT(*weights, ElementsAre("\x01\x02\x03\x04", "\x05\x06\x07", ""));
  // The weights point into `contents`.
  EXPECT_EQ((*weights)[0].data(), contents.data());
  EXPECT_EQ((*weights)[1].data(), contents.data() + 4);
}

TEST(ParseExtractedStaticWeightsTest, EmptyObjectHasNoWeights) {
  auto weights = ParseExtractedStaticWeights(MakeContents("", "{}"));
  ASSERT_TRUE(weights.HasValue());
  EXPECT_THAT(*weights, IsEmpty());
}

TEST(ParseExtractedStaticWeightsTest, EmptyContentsHaveNoWeights) {
  auto weights = ParseExtractedStaticWeights("");
  ASSERT_TRUE(weights.HasValue());
  EXPECT_THAT(*weights, IsEmpty());
}

TEST(ParseExtractedStaticWeightsTest, ContentsSmallerThanFooterAreInvalid) {
  ExpectInvalid("12345678901");
}

TEST(ParseExtractedStaticWeightsTest, InvalidFooterIsAnError) {
  const std::string json = R"({"0": ["0", "4"]})";
  ExpectInvalid(MakeContents("abcd", json, /*json_offset=*/std::nullopt,
                             /*magic=*/0x12345678));
  // The JSON offset points past the start of the footer.
  ExpectInvalid(
      MakeContents("abcd", json, /*json_offset=*/4 + json.size() + 1));
}

TEST(ParseExtractedStaticWeightsTest, MalformedJsonIsAnError) {
  for (absl::string_view json : {
           R"({"x": ["0", "4"]})",                     // Non-numeric order.
           R"({"": ["0", "4"]})",                      // Empty order.
           R"({"+1": ["0", "4"]})",                    // Signed order.
           R"({"99999999999999999999": ["0", "4"]})",  // Order overflows.
           R"({"0 ["0", "4"]})",                       // Order with a space.
           R"({"0)",                                   // Unterminated order.
           R"({"0")",                                  // Missing value.
           R"({"0": ["0"]})",                          // Missing size.
           R"({"0": ["0", "4")",                       // Missing bracket.
           R"({"0": ["0", "-4"]})",                    // Negative size.
       }) {
    SCOPED_TRACE(json);
    ExpectInvalid(MakeContents("abcd", json));
  }
}

TEST(ParseExtractedStaticWeightsTest, OutOfBoundsRangeIsAnError) {
  for (absl::string_view json : {
           R"({"0": ["5", "0"]})",                     // Offset past payload.
           R"({"0": ["2", "3"]})",                     // End past payload.
           R"({"0": ["1", "18446744073709551615"]})",  // End overflows.
       }) {
    SCOPED_TRACE(json);
    ExpectInvalid(MakeContents("abcd", json));
  }
}

TEST(ParseExtractedStaticWeightsTest, NonConsecutiveOrderIsAnError) {
  for (absl::string_view json : {
           R"({"0": ["0", "2"], "2": ["2", "2"]})",  // Gap.
           R"({"1": ["0", "2"], "1": ["2", "2"]})",  // Duplicate.
       }) {
    SCOPED_TRACE(json);
    ExpectInvalid(MakeContents("abcd", json));
  }
}

TEST(StaticWeightExtractionOptionsTest, AppendsExtractionOptions) {
  auto options = StaticWeightExtractionOptions("/data/tmp/weights.bin");
  ASSERT_TRUE(options.HasValue());
  EXPECT_EQ(
      *options,
      " --opt-static-sharing --extract-static-data=/data/tmp/weights.bin");
}

TEST(StaticWeightExtractionOptionsTest, RejectsPathsThatCannotBeOptions) {
  for (absl::string_view path : {"", "/tmp/my dir/weights.bin", "/tmp/\tw"}) {
    SCOPED_TRACE(path);
    auto options = StaticWeightExtractionOptions(path);
    ASSERT_FALSE(options.HasValue());
    EXPECT_EQ(options.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  }
}

// Sets an environment variable for the lifetime of this object and then
// restores its previous value.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old_value = std::getenv(name); old_value != nullptr) {
      old_value_ = old_value;
    }
    if (value == nullptr) {
      unsetenv(name);
    } else {
      setenv(name, value, /*overwrite=*/1);
    }
  }
  ~ScopedEnv() {
    if (old_value_.has_value()) {
      setenv(name_, old_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  std::optional<std::string> old_value_;
};

TEST(CreateExtractedStaticWeightsFileTest, CreatesUniqueFilesInDlaDir) {
  const std::string dla_dir = ::testing::TempDir() + "/dla_dir/nested";
  ScopedEnv env("MTKNN_ADAPTER_DLA_DIR", dla_dir.c_str());

  auto first = CreateExtractedStaticWeightsFile("Partition_0");
  auto second = CreateExtractedStaticWeightsFile("Partition_0");
  ASSERT_TRUE(first.HasValue());
  ASSERT_TRUE(second.HasValue());
  EXPECT_NE(*first, *second);
  for (const std::string& path : {*first, *second}) {
    SCOPED_TRACE(path);
    EXPECT_TRUE(absl::StartsWith(path, dla_dir + "/mtk_Partition_0_weights_"));
    EXPECT_TRUE(std::filesystem::is_regular_file(path));
    EXPECT_EQ(std::filesystem::file_size(path), 0);
    std::filesystem::remove(path);
  }
}

TEST(CreateExtractedStaticWeightsFileTest, UsesTempDirWithoutDlaDir) {
  const std::string temp_dir = ::testing::TempDir() + "/tmpdir";
  std::filesystem::create_directories(temp_dir);
  ScopedEnv dla_env("MTKNN_ADAPTER_DLA_DIR", "");
  ScopedEnv tmp_env("TMPDIR", temp_dir.c_str());

  auto path = CreateExtractedStaticWeightsFile("g");
  ASSERT_TRUE(path.HasValue());
  EXPECT_TRUE(absl::StartsWith(*path, temp_dir + "/mtk_g_weights_"));
  EXPECT_TRUE(std::filesystem::is_regular_file(*path));
  std::filesystem::remove(*path);
}

TEST(CreateExtractedStaticWeightsFileTest, FailsWithoutUsableTempDir) {
  ScopedEnv dla_env("MTKNN_ADAPTER_DLA_DIR", nullptr);
  ScopedEnv tmp_env("TMPDIR", "/nonexistent/litert_mtk_test_tmp");

  auto path = CreateExtractedStaticWeightsFile("g");
  ASSERT_FALSE(path.HasValue());
  EXPECT_EQ(path.Error().Status(), kLiteRtStatusErrorFileIO);
}

TEST(CreateExtractedStaticWeightsFileTest, FailsIfFileCannotBeCreated) {
  // A regular file cannot be used as a directory.
  const std::string not_a_dir = ::testing::TempDir() + "/not_a_dir";
  std::filesystem::remove_all(not_a_dir);
  ASSERT_TRUE(std::ofstream(not_a_dir).good());
  ScopedEnv env("MTKNN_ADAPTER_DLA_DIR", not_a_dir.c_str());

  auto path = CreateExtractedStaticWeightsFile("g");
  ASSERT_FALSE(path.HasValue());
  EXPECT_EQ(path.Error().Status(), kLiteRtStatusErrorFileIO);
}

}  // namespace
}  // namespace litert::mediatek
