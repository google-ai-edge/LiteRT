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

#include "litert/cc/options/litert_cpu_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {
using testing::StrEq;
using testing::litert::IsError;
using testing::litert::IsOk;
using testing::litert::IsOkAndHolds;

// A test option implementation to discriminate against CpuOptions.
struct NotCpuOptions : public OpaqueOptions {
  using OpaqueOptions::OpaqueOptions;
  static Expected<NotCpuOptions> Create() {
    LiteRtOpaqueOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
        "test-option", new int,
        [](void* i) { delete reinterpret_cast<int*>(i); }, &options));
    return NotCpuOptions(options, OwnHandle::kYes);
  }
};

TEST(CpuOptions, IdentifierIsCorrect) {
  EXPECT_THAT(CpuOptions::Identifier(), StrEq(LiteRtGetCpuOptionsIdentifier()));
}

TEST(CpuOptions, CreateAndOwnedHandle) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options, CpuOptions::Create());
  EXPECT_TRUE(options.IsOwned());
}

TEST(CpuOptions, CreateFromAnotherCpuOptionsHandleWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions original_options,
                              CpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options,
                              CpuOptions::Create(original_options));
  EXPECT_FALSE(options.IsOwned());
  EXPECT_EQ(options.Get(), original_options.Get());
}

TEST(CpuOptions, CreateFromADifferentOptionFails) {
  LITERT_ASSERT_OK_AND_ASSIGN(NotCpuOptions original_options,
                              NotCpuOptions::Create());
  OpaqueOptions& opaque_options = original_options;
  EXPECT_THAT(CpuOptions::Create(opaque_options),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(CpuOptions, CheckNumThreadsDefaultValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options, CpuOptions::Create());
  EXPECT_THAT(options.GetNumThreads(), IsOkAndHolds(0));
}

TEST(CpuOptions, SetAndGetNumThreadsWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options, CpuOptions::Create());

  LITERT_EXPECT_OK(options.SetNumThreads(5));
  EXPECT_THAT(options.GetNumThreads(), IsOkAndHolds(5));
}

TEST(CpuOptions, CheckXNNPackWeightCachePathDefaultValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options, CpuOptions::Create());
  EXPECT_THAT(options.GetXNNPackWeightCachePath(),
              IsOkAndHolds(absl::string_view()));
}

TEST(CpuOptions, SetAndGetXNNPackWeighCachePathWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options, CpuOptions::Create());

  LITERT_EXPECT_OK(options.SetXNNPackWeightCachePath("a/path"));
  EXPECT_THAT(options.GetXNNPackWeightCachePath(),
              IsOkAndHolds(StrEq("a/path")));
}

TEST(CpuOptions, CheckXNNPackWeightCacheFileDescriptorDefaultValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options, CpuOptions::Create());
  EXPECT_THAT(options.GetXNNPackWeightCacheFileDescriptor(),
              IsOkAndHolds(-1));
}

TEST(CpuOptions, SetAndGetXNNPackWeighCacheFileDescriptorWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options, CpuOptions::Create());

  LITERT_EXPECT_OK(options.SetXNNPackWeightCacheFileDescriptor(1234));
  EXPECT_THAT(options.GetXNNPackWeightCacheFileDescriptor(),
              IsOkAndHolds(1234));
}

TEST(CpuOptions, CheckXNNPackFlagsDefaultValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options, CpuOptions::Create());
  // Note: we can't check the default value for this as XNNPack compile options
  // may affect it.
  EXPECT_THAT(options.GetXNNPackFlags(), IsOk());
}

TEST(CpuOptions, SetAndGetXNNPackFlagsWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(CpuOptions options, CpuOptions::Create());

  LITERT_EXPECT_OK(options.SetXNNPackFlags(3));
  EXPECT_THAT(options.GetXNNPackFlags(), IsOkAndHolds(3));
}

TEST(CpuOptions, GetXNNPackFlagsFailsIfErroneousCast) {
  LITERT_ASSERT_OK_AND_ASSIGN(NotCpuOptions original_options,
                              NotCpuOptions::Create());
  OpaqueOptions& opaque_options = original_options;
  CpuOptions& options = static_cast<CpuOptions&>(opaque_options);
  EXPECT_THAT(options.GetXNNPackFlags(), IsError());
}

}  // namespace
}  // namespace litert
