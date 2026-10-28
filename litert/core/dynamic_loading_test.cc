// Copyright 2024 Google LLC.
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

#include "litert/core/dynamic_loading.h"

#include <cstdlib>
#include <filesystem>  // NOLINT
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/core/filesystem.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

using litert::testing::UniqueTestDirectory;
using ::testing::Contains;
using ::testing::HasSubstr;

constexpr absl::string_view kNotLiteRtSo = "notLibLiteRt.so";
constexpr absl::string_view kLiteRtSo1 = "libLiteRtCompilerPlugin_1.so";
constexpr absl::string_view kLiteRtSo2 = "libLiteRtCompilerPlugin_2.so";
constexpr absl::string_view kLiteRtSo3 = "libLiteRtDispatch_1.so";
constexpr absl::string_view kLiteRtSo4 = "libLiteRtDispatch_2.so";
constexpr absl::string_view kLiteRtDylib = "libLiteRtDispatch_3.dylib";
constexpr absl::string_view kLdLibraryPath = "LD_LIBRARY_PATH";

constexpr absl::string_view kDispatchPrefix = "libLiteRtDispatch";
constexpr absl::string_view kSoOnly[] = {".so"};
constexpr absl::string_view kSoAndDylib[] = {".so", ".dylib"};

TEST(TestIsSharedLibFileName, MatchesPrefixAndExtension) {
  EXPECT_TRUE(IsSharedLibFileName(kLiteRtSo3, kDispatchPrefix, kSoOnly));
  EXPECT_TRUE(IsSharedLibFileName(kLiteRtSo3, kLiteRtSharedLibPrefix, kSoOnly));
  // Dots in the stem are fine as long as the name ends with an extension.
  EXPECT_TRUE(
      IsSharedLibFileName("libLiteRtDispatch.v2.so", kDispatchPrefix, kSoOnly));
}

TEST(TestIsSharedLibFileName, RejectsOtherPrefixes) {
  EXPECT_FALSE(IsSharedLibFileName(kNotLiteRtSo, kDispatchPrefix, kSoOnly));
  EXPECT_FALSE(IsSharedLibFileName(kLiteRtSo1, kDispatchPrefix, kSoOnly));
  // The prefix must start the name, not merely occur in it.
  EXPECT_FALSE(
      IsSharedLibFileName("xlibLiteRtDispatch.so", kDispatchPrefix, kSoOnly));
}

TEST(TestIsSharedLibFileName, RejectsOtherExtensions) {
  EXPECT_FALSE(
      IsSharedLibFileName("libLiteRtDispatch.txt", kDispatchPrefix, kSoOnly));
  EXPECT_FALSE(
      IsSharedLibFileName("libLiteRtDispatch", kDispatchPrefix, kSoOnly));
  // Versioned names are not matched; callers ship the unversioned name.
  EXPECT_FALSE(
      IsSharedLibFileName("libLiteRtDispatch.so.1", kDispatchPrefix, kSoOnly));
  // A bare extension is not a library name.
  EXPECT_FALSE(IsSharedLibFileName(".so", "", kSoOnly));
}

TEST(TestIsSharedLibFileName, DylibDependsOnExtensionList) {
  EXPECT_FALSE(IsSharedLibFileName(kLiteRtDylib, kDispatchPrefix, kSoOnly));
  EXPECT_TRUE(IsSharedLibFileName(kLiteRtDylib, kDispatchPrefix, kSoAndDylib));
  EXPECT_TRUE(IsSharedLibFileName(kLiteRtSo3, kDispatchPrefix, kSoAndDylib));
  EXPECT_FALSE(
      IsSharedLibFileName("libOther.dylib", kDispatchPrefix, kSoAndDylib));
}

TEST(TestDynamicLoading, GlobNoMatch) {
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), kNotLiteRtSo}));

  std::vector<std::string> results;
  LITERT_ASSERT_OK(litert::internal::FindLiteRtCompilerPluginSharedLibs(
      dir->Str(), results));
  EXPECT_EQ(results.size(), 0);
  std::vector<std::string> results2;
  LITERT_ASSERT_OK(
      litert::internal::FindLiteRtDispatchSharedLibs(dir->Str(), results2));
  EXPECT_EQ(results2.size(), 0);
}

TEST(TestDynamicLoading, GlobOneMatch) {
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), kLiteRtSo1}));
  Touch(Join({dir->Str(), kLiteRtSo3}));
  Touch(Join({dir->Str(), kNotLiteRtSo}));

  std::vector<std::string> results;
  LITERT_ASSERT_OK(litert::internal::FindLiteRtCompilerPluginSharedLibs(
      dir->Str(), results));
  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(absl::string_view(results.front()).ends_with(kLiteRtSo1));

  std::vector<std::string> results2;
  LITERT_ASSERT_OK(
      litert::internal::FindLiteRtDispatchSharedLibs(dir->Str(), results2));
  ASSERT_EQ(results2.size(), 1);
  EXPECT_TRUE(absl::string_view(results2.front()).ends_with(kLiteRtSo3));
}

TEST(TestDynamicLoading, GlobMultiMatch) {
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), kLiteRtSo1}));
  Touch(Join({dir->Str(), kLiteRtSo2}));
  Touch(Join({dir->Str(), kLiteRtSo3}));
  Touch(Join({dir->Str(), kLiteRtSo4}));
  Touch(Join({dir->Str(), kNotLiteRtSo}));

  std::vector<std::string> results;
  LITERT_ASSERT_OK(litert::internal::FindLiteRtCompilerPluginSharedLibs(
      dir->Str(), results));
  ASSERT_EQ(results.size(), 2);
  EXPECT_THAT(results, Contains(HasSubstr(kLiteRtSo1)));
  EXPECT_THAT(results, Contains(HasSubstr(kLiteRtSo2)));

  std::vector<std::string> results2;
  LITERT_ASSERT_OK(
      litert::internal::FindLiteRtDispatchSharedLibs(dir->Str(), results2));
  ASSERT_EQ(results2.size(), 2);
  EXPECT_THAT(results2, Contains(HasSubstr(kLiteRtSo3)));
  EXPECT_THAT(results2, Contains(HasSubstr(kLiteRtSo4)));
}

TEST(TestDynamicLoading, GlobDylibOnlyOnApplePlatforms) {
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), kLiteRtSo3}));
  Touch(Join({dir->Str(), kLiteRtDylib}));

  std::vector<std::string> results;
  LITERT_ASSERT_OK(
      litert::internal::FindLiteRtDispatchSharedLibs(dir->Str(), results));
  EXPECT_THAT(results, Contains(HasSubstr(kLiteRtSo3)));
#if defined(__APPLE__)
  EXPECT_EQ(results.size(), 2);
  EXPECT_THAT(results, Contains(HasSubstr(kLiteRtDylib)));
#else
  EXPECT_EQ(results.size(), 1);
#endif  // defined(__APPLE__)
}

TEST(TestDynamicLoadingHelper, HelperWithFullMatch) {
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), kLiteRtSo1}));
  Touch(Join({dir->Str(), kLiteRtSo2}));
  Touch(Join({dir->Str(), kLiteRtSo3}));
  Touch(Join({dir->Str(), kLiteRtSo4}));
  Touch(Join({dir->Str(), kNotLiteRtSo}));

  std::vector<std::string> results;
  LITERT_ASSERT_OK(litert::internal::FindLiteRtSharedLibsHelper(
      std::string(dir->Str()), std::string(kLiteRtSo4), true, results));
  ASSERT_EQ(results.size(), 1);
  EXPECT_THAT(results, Contains(HasSubstr(kLiteRtSo4)));
}

TEST(TestPutLibOnLdPath, AppendToEmptyLdPath) {
  unsetenv(kLdLibraryPath.data());

  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);

  const auto dir_path = dir->Str();
  const auto lib_path = Join({dir_path, kLiteRtSo1});
  Touch(lib_path);

  LITERT_ASSERT_OK(PutLibOnLdPath(dir_path, kLiteRtSo1));
  absl::string_view ld_library_path = getenv(kLdLibraryPath.data());
  EXPECT_THAT(ld_library_path, HasSubstr(dir_path));
}

TEST(TestPutLibOnLdPath, AppendToLdPathNoMatch) {
  unsetenv(kLdLibraryPath.data());

  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);

  const auto dir_path = dir->Str();

  LITERT_ASSERT_OK(PutLibOnLdPath(dir_path, kLiteRtSo1));
  ASSERT_EQ(getenv(kLdLibraryPath.data()), nullptr);
}

TEST(TestPutLibOnLdPath, AppendToExistingLdPath) {
  static constexpr absl::string_view kExistingLdPath = "an/existing/path";

  unsetenv(kLdLibraryPath.data());
  setenv(kLdLibraryPath.data(), kExistingLdPath.data(), /*overwrite=*/1);

  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);

  const auto dir_path = dir->Str();
  const auto lib_path = Join({dir_path, kLiteRtSo1});
  Touch(lib_path);

  LITERT_ASSERT_OK(PutLibOnLdPath(dir_path, kLiteRtSo1));
  absl::string_view ld_library_path = getenv(kLdLibraryPath.data());
  EXPECT_THAT(ld_library_path, HasSubstr(dir_path));
  EXPECT_THAT(ld_library_path, HasSubstr(kExistingLdPath));
}

TEST(TestPutLibOnLdPath, AppendToLdLibraryPathNoDupePath) {
  unsetenv(kLdLibraryPath.data());

  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);

  const auto dir_path = dir->Str();
  const auto lib_path = Join({dir_path, kLiteRtSo1});
  Touch(lib_path);

  setenv(kLdLibraryPath.data(), dir_path.data(), /*overwrite=*/1);

  LITERT_ASSERT_OK(PutLibOnLdPath(dir_path, kLiteRtSo1));
  absl::string_view ld_library_path = getenv(kLdLibraryPath.data());
  EXPECT_THAT(ld_library_path, HasSubstr(dir_path));
  EXPECT_EQ(ld_library_path.size(), dir_path.size());
}

TEST(TestPutLibOnLdPath, AppendToNestedLdPath) {
  unsetenv(kLdLibraryPath.data());

  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);

  const auto dir_path = dir->Str();
  const auto nested_dir_path = Join({dir_path, "another/dir"});
  const auto lib_path = Join({nested_dir_path, kLiteRtSo1});
  ASSERT_TRUE(std::filesystem::create_directories(nested_dir_path));
  Touch(lib_path);

  LITERT_ASSERT_OK(PutLibOnLdPath(dir_path, kLiteRtSo1));
  absl::string_view ld_library_path = getenv(kLdLibraryPath.data());
  EXPECT_THAT(ld_library_path, HasSubstr(nested_dir_path));
}

}  // namespace
}  // namespace litert::internal
