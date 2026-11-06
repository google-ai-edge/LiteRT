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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/internal/scoped_weight_source.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/options/litert_concrete_options_base.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/core/options.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

TEST(OptionsTest, SetHardwareAccelerators) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  // Tests overload that takes litert::HwAccelerators.
  LITERT_EXPECT_OK(
      options.SetHardwareAccelerators(HwAccelerators::kCpu));

  // Tests overload that takes litert::HwAcceleratorSet.
  LITERT_EXPECT_OK(options.SetHardwareAccelerators(
      HwAccelerators::kCpu | HwAccelerators::kGpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_handle,
      internal::LiteRtOptionsPtrBuilder::Build(options, env.GetHolder()));
}

TEST(OptionsTest, SetExternalWeightScopedFileStoresMetadata) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  const std::string path =
      absl::StrCat(::testing::TempDir(), "scoped_weights.bin");
  {
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    file << std::string(16, '\x01');
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(path));

  Options::ScopedWeightSectionMap sections;
  sections.emplace("weights.group",
                   ScopedWeightSection{.offset = 4, .length = 8});

  auto status =
      options.SetExternalWeightScopedFile(scoped_file, std::move(sections));
  LITERT_EXPECT_OK(status);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_handle,
      internal::LiteRtOptionsPtrBuilder::Build(options, env.GetHolder()));
  auto* impl = reinterpret_cast<LiteRtOptionsT*>(options_handle.get());
  ASSERT_NE(impl->scoped_weight_source, nullptr);
  EXPECT_TRUE(impl->scoped_weight_source->file.IsValid());
  auto it = impl->scoped_weight_source->sections.find("weights.group");
  ASSERT_NE(it, impl->scoped_weight_source->sections.end());
  EXPECT_EQ(it->second.offset, 4);
  EXPECT_EQ(it->second.length, 8);
  std::remove(path.c_str());
}

TEST(OptionsTest, GetOptionsGenericAccessor) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  LITERT_ASSERT_OK_AND_ASSIGN(auto& gpu_opts_1,
                              options.GetOptions<GpuOptions>());
  LITERT_EXPECT_OK(gpu_opts_1.SetPrecision(GpuOptions::Precision::kFp16));

  LITERT_ASSERT_OK_AND_ASSIGN(auto& gpu_opts_2,
                              options.GetOptions<GpuOptions>());
  EXPECT_EQ(&gpu_opts_1, &gpu_opts_2);

  LITERT_ASSERT_OK_AND_ASSIGN(auto& cpu_opts, options.GetOptions<CpuOptions>());
  LITERT_EXPECT_OK(cpu_opts.SetNumThreads(4));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_handle,
      internal::LiteRtOptionsPtrBuilder::Build(options, env.GetHolder()));
  ASSERT_NE(options_handle, nullptr);
}

class FakeEncodedOptions : public ConcreteOptionsBase {
 public:
  FakeEncodedOptions(std::string identifier, std::string payload)
      : identifier_(std::move(identifier)), payload_(std::move(payload)) {}

  LiteRtStatus GetOpaqueOptionsData(
      const char** identifier, void** payload,
      void (**payload_deleter)(void*)) const override {
    *identifier = identifier_.c_str();
    char* copy = static_cast<char*>(std::malloc(payload_.size() + 1));
    std::memcpy(copy, payload_.c_str(), payload_.size() + 1);
    *payload = copy;
    *payload_deleter = [](void* p) { std::free(p); };
    return kLiteRtStatusOk;
  }

 private:
  std::string identifier_;
  std::string payload_;
};

TEST(OptionsTest, AddConcreteOpaqueOptionsMaterializesMultiplePayloads) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  EXPECT_FALSE(
      options.AddOpaqueOptions(std::unique_ptr<ConcreteOptionsBase>(nullptr))
          .HasValue());
  LITERT_EXPECT_OK(options.AddOpaqueOptions(
      std::make_unique<FakeEncodedOptions>("opt_a", "a = 1\n")));
  LITERT_EXPECT_OK(options.AddOpaqueOptions(
      std::make_unique<FakeEncodedOptions>("opt_b", "b = true\n")));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_handle,
      internal::LiteRtOptionsPtrBuilder::Build(options, env.GetHolder()));
  auto* impl = reinterpret_cast<LiteRtOptionsT*>(options_handle.get());
  ASSERT_NE(impl->options, nullptr);

  void* data_a = nullptr;
  ASSERT_EQ(LiteRtFindOpaqueOptionsData(impl->options, "opt_a", &data_a),
            kLiteRtStatusOk);
  EXPECT_STREQ(static_cast<const char*>(data_a), "a = 1\n");

  void* data_b = nullptr;
  ASSERT_EQ(LiteRtFindOpaqueOptionsData(impl->options, "opt_b", &data_b),
            kLiteRtStatusOk);
  EXPECT_STREQ(static_cast<const char*>(data_b), "b = true\n");
}

class FailingEncodedOptions : public ConcreteOptionsBase {
 public:
  LiteRtStatus GetOpaqueOptionsData(
      const char** /*identifier*/, void** /*payload*/,
      void (** /*payload_deleter*/)(void*)) const override {
    return kLiteRtStatusErrorInvalidArgument;
  }
};

TEST(OptionsTest, AddConcreteOpaqueOptionsPropagatesPayloadError) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());

  LITERT_EXPECT_OK(
      options.AddOpaqueOptions(std::make_unique<FailingEncodedOptions>()));

  EXPECT_FALSE(
      internal::LiteRtOptionsPtrBuilder::Build(options, env.GetHolder())
          .HasValue());
}

}  // namespace
}  // namespace litert
