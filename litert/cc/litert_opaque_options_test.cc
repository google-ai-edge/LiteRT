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

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {
using testing::Pointee;
using testing::litert::IsOkAndHolds;

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

  static Expected<SimpleOptions> Create(int value) {
    LiteRtOpaqueOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
        Discriminator(), new int(value),
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
  LITERT_ASSERT_OK_AND_ASSIGN(SimpleOptions opts, SimpleOptions::Create());

  LITERT_ASSERT_OK_AND_ASSIGN(SimpleOptions::Payload * data,
                              FindOpaqueData<SimpleOptions::Payload>(
                                  opts, SimpleOptions::Discriminator()));
  EXPECT_EQ(*data, 1);
}

TEST(OpaqueOptionsTest, Find) {
  void* payload = malloc(8);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      OpaqueOptions::Create("not-simple", payload, [](void* d) { free(d); }));

  LITERT_ASSERT_OK_AND_ASSIGN(SimpleOptions simple_options,
                              SimpleOptions::Create());

  LITERT_ASSERT_OK(options.Append(std::move(simple_options)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      OpaqueOptions found,
      FindOpaqueOptions(options, std::string(SimpleOptions::Discriminator())));
  EXPECT_THAT(found.GetData<SimpleOptions::Payload>(),
              IsOkAndHolds(Pointee(1)));
}

TEST(OpaqueOptionsTest, FindType) {
  void* payload = malloc(8);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      OpaqueOptions::Create("not-simple", payload, [](void* d) { free(d); }));

  LITERT_ASSERT_OK_AND_ASSIGN(SimpleOptions simple_options,
                              SimpleOptions::Create());

  LITERT_ASSERT_OK(options.Append(std::move(simple_options)));

  LITERT_ASSERT_OK_AND_ASSIGN(SimpleOptions found,
                              FindOpaqueOptions<SimpleOptions>(options));
  EXPECT_EQ(found.Data(), 1);
}

TEST(OpaqueOptionsTest, GetPayloadHashFailsIfUnset) {
  LITERT_ASSERT_OK_AND_ASSIGN(SimpleOptions opts, SimpleOptions::Create());
  auto hash_result = opts.Hash();
  EXPECT_FALSE(hash_result);
  EXPECT_EQ(hash_result.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(OpaqueOptionsTest, SetAndGetPayloadHash) {
  const int kPayloadValue = 42;
  LITERT_ASSERT_OK_AND_ASSIGN(SimpleOptions opts,
                              SimpleOptions::Create(kPayloadValue));

  auto std_hash = [](const void* payload_data) -> uint64_t {
    const int* payload = reinterpret_cast<const int*>(payload_data);
    return std::hash<int>{}(*payload);
  };

  LITERT_ASSERT_OK(opts.SetHash(std_hash));

  LITERT_ASSERT_OK_AND_ASSIGN(uint64_t hash, opts.Hash());
  EXPECT_EQ(hash, std::hash<int>{}(kPayloadValue));
}

}  // namespace
}  // namespace litert
