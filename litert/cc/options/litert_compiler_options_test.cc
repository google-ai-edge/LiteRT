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
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/cc/options/litert_compiler_options.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

TEST(CompilerOptionsTest, CreateSetAndGetDummyOptionWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(::litert::CompilerOptions options,
                              ::litert::CompilerOptions::Create());
  LITERT_EXPECT_OK(options.SetDummyOption(true));
  LITERT_ASSERT_OK_AND_ASSIGN(bool dummy_option, options.GetDummyOption());
  EXPECT_TRUE(dummy_option);
}

TEST(CompilerOptionsTest, CreateOpaqueOptionsWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CompilerOptions::Create());
  LITERT_EXPECT_OK(options.SetDummyOption(true));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueCompilerOptionsData(options.Get(), &identifier,
                                            &payload, &payload_deleter),
            kLiteRtStatusOk);

  LiteRtOpaqueOptions opaque_opts = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_opts),
            kLiteRtStatusOk);

  litert::OpaqueOptions cpp_opaque_opts =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
}

TEST(CompilerOptionsTest, SetAndGetPartitionStrategyReturnsSetValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(::litert::CompilerOptions options,
                              ::litert::CompilerOptions::Create());
  LITERT_EXPECT_OK(options.SetPartitionStrategy(
      kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected));
  LITERT_ASSERT_OK_AND_ASSIGN(auto partition_strategy,
                              options.GetPartitionStrategy());
  EXPECT_EQ(partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
}

TEST(CompilerOptionsTest, AddInputShapesSerializeToToml) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CompilerOptions::Create());

  const int32_t image[] = {1, 224, 224, 3};
  const int32_t tokens[] = {1, -1};
  LITERT_EXPECT_OK(options.AddPositionalInputShape(absl::MakeConstSpan(image)));
  LITERT_EXPECT_OK(options.AddPositionalInputShape(
      "serving_default", absl::MakeConstSpan(tokens)));
  LITERT_EXPECT_OK(
      options.AddTensorInputShape("arg0", absl::MakeConstSpan(image)));
  LITERT_EXPECT_OK(options.AddTensorInputShape("serving_default", "arg0",
                                               absl::MakeConstSpan(image)));
  LITERT_EXPECT_OK(
      options.AddSignatureInputShape("image", absl::MakeConstSpan(image)));
  LITERT_EXPECT_OK(options.AddSignatureInputShape("decode", "tokens",
                                                  absl::MakeConstSpan(tokens)));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueCompilerOptionsData(options.Get(), &identifier,
                                            &payload, &payload_deleter),
            kLiteRtStatusOk);
  const std::string toml_str = static_cast<const char*>(payload);
  payload_deleter(payload);

  EXPECT_EQ(toml_str,
            "positional_input_shapes = [\"@1:224:224:3\", "
            "\"serving_default@1:-1\"]\n"
            "tensor_input_shapes = [\"@arg0@1:224:224:3\", "
            "\"serving_default@arg0@1:224:224:3\"]\n"
            "signature_input_shapes = [\"@image@1:224:224:3\", "
            "\"decode@tokens@1:-1\"]\n");
}

TEST(CompilerOptionsTest, AddInputShapesRejectInvalidArgs) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CompilerOptions::Create());

  const int32_t shape[] = {1, 2};
  EXPECT_FALSE(options.AddPositionalInputShape({}).HasValue());
  EXPECT_FALSE(
      options.AddTensorInputShape("", absl::MakeConstSpan(shape)).HasValue());
  EXPECT_FALSE(options.AddTensorInputShape("arg0", {}).HasValue());
  EXPECT_FALSE(options.AddSignatureInputShape("", absl::MakeConstSpan(shape))
                   .HasValue());
  EXPECT_FALSE(options.AddSignatureInputShape("image", {}).HasValue());
}

}  // namespace
}  // namespace litert
