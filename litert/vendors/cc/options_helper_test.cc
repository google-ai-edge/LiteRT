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

#include "litert/vendors/cc/options_helper.h"

#include <utility>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"

namespace litert {
class SimpleOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;
  using Payload = int;

  static Expected<SimpleOptions> Create() {
    LiteRtOpaqueOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
        Discriminator(), new int(1),
        [](void* d) { delete reinterpret_cast<int*>(d); }, &options));
    return SimpleOptions(options, OwnHandle::kYes);
  }

  static Expected<SimpleOptions> Create(OpaqueOptions& options) {
    const auto id = options.GetIdentifier();
    if (!id || *id != Discriminator()) {
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    return SimpleOptions(options.Get(), OwnHandle::kNo);
  }

  static const char* Discriminator() { return "simple"; }
};

class SimpleOptions2 : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;
  using Payload = int;

  static Expected<SimpleOptions2> Create() {
    LiteRtOpaqueOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
        Discriminator(), new int(1),
        [](void* d) { delete reinterpret_cast<int*>(d); }, &options));
    return SimpleOptions2(options, OwnHandle::kYes);
  }

  static Expected<SimpleOptions2> Create(OpaqueOptions& options) {
    const auto id = options.GetIdentifier();
    if (!id || *id != Discriminator()) {
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    return SimpleOptions2(options.Get(), OwnHandle::kNo);
  }

  static const char* Discriminator() { return "simple2"; }
};

namespace {

TEST(OptionsHelperTest, ParseOptionsEmpty) {
  auto [opts, opq, simple] = ParseOptions<SimpleOptions>(nullptr);
  EXPECT_FALSE(opts);
  EXPECT_FALSE(opq);
  EXPECT_FALSE(simple);
}

TEST(OptionsHelperTest, WithOpqOptions) {
  auto lrt_opts = Options::Create();
  ASSERT_TRUE(lrt_opts);

  auto lrt_simple = SimpleOptions::Create();
  ASSERT_TRUE(lrt_simple);

  ASSERT_TRUE(lrt_opts->AddOpaqueOptions(std::move(*lrt_simple)));

  auto [opts, opq, simple] = ParseOptions<SimpleOptions>(lrt_opts->Get());

  EXPECT_TRUE(opts);
  EXPECT_TRUE(opq);
  EXPECT_TRUE(simple);
}

TEST(OptionsHelperTest, With2OpqOptions) {
  auto lrt_opts = Options::Create();
  ASSERT_TRUE(lrt_opts);

  auto lrt_simple = SimpleOptions::Create();
  ASSERT_TRUE(lrt_simple);

  ASSERT_TRUE(lrt_opts->AddOpaqueOptions(std::move(*lrt_simple)));

  auto lrt_simple2 = SimpleOptions2::Create();
  ASSERT_TRUE(lrt_simple2);

  ASSERT_TRUE(lrt_opts->AddOpaqueOptions(std::move(*lrt_simple2)));

  auto [opts, opq, simple, simple2] =
      ParseOptions<SimpleOptions, SimpleOptions2>(lrt_opts->Get());

  EXPECT_TRUE(opts);
  EXPECT_TRUE(opq);
  EXPECT_TRUE(simple);
  EXPECT_TRUE(simple2);
}

}  // namespace
}  // namespace litert
