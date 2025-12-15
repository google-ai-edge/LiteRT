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

#include "litert/cc/litert_options.h"

#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/internal/scoped_weight_source.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

TEST(OptionsTest, SetHardwareAccelerators) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  // Tests overload that takes litert::HwAccelerators.
  LITERT_EXPECT_OK(
      options.SetHardwareAccelerators(HwAccelerators::kCpu));

  // Tests overload that takes litert::HwAcceleratorSet.
  LITERT_EXPECT_OK(options.SetHardwareAccelerators(
      HwAccelerators::kCpu | HwAccelerators::kGpu));
}

TEST(OptionsTest, SetExternalWeightScopedFileStoresMetadata) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  const std::string path =
      absl::StrCat(::testing::TempDir(), "scoped_weights.bin");
  {
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    file << std::string(16, '\x01');
  }

  auto scoped_file = ScopedFile::Open(path);
  ASSERT_TRUE(scoped_file.ok());

  Options::ScopedWeightSectionMap sections;
  sections.emplace("weights.group",
                   ScopedWeightSection{.offset = 4, .length = 8});

  auto status =
      options.SetExternalWeightScopedFile(*scoped_file, std::move(sections));
#if defined(LITERT_WITH_EXTERNAL_WEIGHT_LOADER)
  LITERT_EXPECT_OK(status);
  auto* impl = reinterpret_cast<LiteRtOptionsT*>(options.Get());
  ASSERT_NE(impl->scoped_weight_source, nullptr);
  EXPECT_TRUE(impl->scoped_weight_source->file.IsValid());
  auto it = impl->scoped_weight_source->sections.find("weights.group");
  ASSERT_NE(it, impl->scoped_weight_source->sections.end());
  EXPECT_EQ(it->second.offset, 4);
  EXPECT_EQ(it->second.length, 8);
#else
  EXPECT_FALSE(status.HasValue());
#endif
  std::remove(path.c_str());
}
}  // namespace
}  // namespace litert
