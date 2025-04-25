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

#include "litert/cc/litert_opaque_options.h"

#include <cstdlib>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

class SimpleOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;
  using Payload = int;

  static const char* Discriminator() { return "simple"; }

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

  int Data() const {
    auto data = GetData<Payload>();
    if (!data) {
      return -1;
    }
    return **data;
  }
};

TEST(OpaqueOptionsTest, FindData) {
  auto opts = SimpleOptions::Create();
  ASSERT_TRUE(opts);

  auto data = FindOpaqueData<SimpleOptions::Payload>(
      *opts, SimpleOptions::Discriminator());
  ASSERT_TRUE(data);
  EXPECT_EQ(**data, 1);
}

TEST(OpaqueOptionsTest, Find) {
  void* payload = malloc(8);
  auto options =
      OpaqueOptions::Create("not-simple", payload, [](void* d) { free(d); });
  ASSERT_TRUE(options);

  auto simple_options = SimpleOptions::Create();
  ASSERT_TRUE(simple_options);

  LITERT_ASSERT_OK(options->Append(std::move(*simple_options)));

  auto found =
      FindOpaqueOptions(*options, std::string(SimpleOptions::Discriminator()));
  ASSERT_TRUE(found);
  EXPECT_EQ(**found->GetData<SimpleOptions::Payload>(), 1);
}

TEST(OpaqueOptionsTest, FindType) {
  void* payload = malloc(8);
  auto options =
      OpaqueOptions::Create("not-simple", payload, [](void* d) { free(d); });
  ASSERT_TRUE(options);

  auto simple_options = SimpleOptions::Create();
  ASSERT_TRUE(simple_options);

  LITERT_ASSERT_OK(options->Append(std::move(*simple_options)));

  auto found = FindOpaqueOptions<SimpleOptions>(*options);
  ASSERT_TRUE(found);
  EXPECT_EQ(found->Data(), 1);
}

}  // namespace
}  // namespace litert
