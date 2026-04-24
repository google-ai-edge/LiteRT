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

#include "litert/cc/internal/litert_options_wrapper.h"

#include <utility>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_context_wrapper.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"

namespace litert::compiler {
namespace {

TEST(OptionsTest, GetOpaqueOptions) {
  LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());

  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  ASSERT_NE(ctx, nullptr);

  litert::internal::OptionsWrapper compiler_options(
      litert::internal::ContextWrapper(ctx), options.Get());

  auto result = compiler_options.GetOpaqueOptions();
  // We expect success or NotFound depending on whether empty options are
  // allowed. Let's check if it's ok.
  if (!result.HasValue()) {
    EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorNotFound);
  }
}

TEST(OptionsTest, FindOpaqueOptionsData) {
  const char* kDummyOpaqueOptionsId = "dummy_opaque_options";
  struct DummyOpaqueOptions {
    int dummy_option;
  };
  auto dummy_opaque_options = new DummyOpaqueOptions{123};
  LITERT_ASSIGN_OR_ABORT(
      auto opaque_options,
      litert::OpaqueOptions::Create(
          kDummyOpaqueOptionsId, dummy_opaque_options,
          [](void* d) { delete reinterpret_cast<DummyOpaqueOptions*>(d); }));
  LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
  ASSERT_TRUE(options.AddOpaqueOptions(std::move(opaque_options)));

  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  ASSERT_NE(ctx, nullptr);

  litert::internal::OptionsWrapper compiler_options(
      litert::internal::ContextWrapper(ctx), options.Get());

  auto result = compiler_options.FindOpaqueOptionsData(kDummyOpaqueOptionsId);
  ASSERT_TRUE(result.HasValue());

  result = compiler_options.FindOpaqueOptionsData("not_found_id");
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorNotFound);
}

}  // namespace
}  // namespace litert::compiler
