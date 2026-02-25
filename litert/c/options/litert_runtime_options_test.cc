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

#include "litert/c/options/litert_runtime_options.h"

#include <stdlib.h>

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/runtime/litert_runtime_options.h"
#include "litert/test/matchers.h"
#include "toml.hpp"  // from @tomlplusplus

namespace {

using ::testing::StrEq;
using ::testing::litert::IsError;

TEST(LiteRtRuntimeOptionsTest, CreateErrorsOutWithNullptrParam) {
  EXPECT_THAT(LrtCreateRuntimeOptions(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(LiteRtRuntimeOptionsTest, CreateWorks) {
  LrtRuntimeOptions* options = nullptr;
  LITERT_ASSERT_OK(LrtCreateRuntimeOptions(&options));
  EXPECT_NE(options, nullptr);

  // Verify defaults
  bool enable_profiling;
  EXPECT_EQ(LrtGetRuntimeOptionsEnableProfiling(options, &enable_profiling),
            kLiteRtStatusErrorNotFound);

  LiteRtErrorReporterMode error_reporter_mode;
  EXPECT_EQ(
      LrtGetRuntimeOptionsErrorReporterMode(options, &error_reporter_mode),
      kLiteRtStatusErrorNotFound);

  bool compress_quantization_zero_points;
  EXPECT_EQ(LrtGetRuntimeOptionsCompressQuantizationZeroPoints(
                options, &compress_quantization_zero_points),
            kLiteRtStatusErrorNotFound);

  LrtDestroyRuntimeOptions(options);
}

TEST(LiteRtRuntimeOptionsTest, OpaqueOptionsSerialization) {
  LrtRuntimeOptions* options = nullptr;
  LITERT_ASSERT_OK(LrtCreateRuntimeOptions(&options));

  const bool kEnableProfiling = true;
  const auto kErrorReporterMode = kLiteRtErrorReporterModeStderr;
  const bool kCompressZeroPoints = true;

  LITERT_ASSERT_OK(
      LrtSetRuntimeOptionsEnableProfiling(options, kEnableProfiling));
  LITERT_ASSERT_OK(
      LrtSetRuntimeOptionsErrorReporterMode(options, kErrorReporterMode));
  LITERT_ASSERT_OK(LrtSetRuntimeOptionsCompressQuantizationZeroPoints(
      options, kCompressZeroPoints));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueRuntimeOptionsData(options, &identifier,
                                                  &payload, &payload_deleter));

  LiteRtOpaqueOptions opaque_options = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(identifier, payload,
                                             payload_deleter, &opaque_options));

  // Verify identifier
  const char* id = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsIdentifier(opaque_options, &id));
  EXPECT_THAT(id, StrEq(LrtGetRuntimeOptionsIdentifier()));

  // Verify payload with Toml parser
  void* payload_void = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(opaque_options, &payload_void));
  absl::string_view payload_str(static_cast<char*>(payload_void));

  auto toml_tbl = toml::parse(payload_str);
  EXPECT_EQ(toml_tbl["enable_profiling"].value<bool>().value(),
            kEnableProfiling);
  EXPECT_EQ(toml_tbl["error_reporter_mode"].value<int>().value(),
            static_cast<int>(kErrorReporterMode));
  EXPECT_EQ(toml_tbl["compress_quantization_zero_points"].value<bool>().value(),
            kCompressZeroPoints);

  // Verify payload with our parser
  LiteRtRuntimeOptionsT runtime_options;
  EXPECT_EQ(litert::internal::ParseLiteRtRuntimeOptions(
                payload_str.data(), payload_str.size(), &runtime_options),
            kLiteRtStatusOk);
  EXPECT_EQ(runtime_options.enable_profiling, kEnableProfiling);
  EXPECT_EQ(runtime_options.error_reporter_mode, kErrorReporterMode);
  EXPECT_EQ(runtime_options.compress_quantization_zero_points,
            kCompressZeroPoints);

  LiteRtDestroyOpaqueOptions(opaque_options);
  LrtDestroyRuntimeOptions(options);
}

TEST(LiteRtRuntimeOptionsTest, OpaqueOptionsSerializationOptionality) {
  LrtRuntimeOptions* options = nullptr;
  LITERT_ASSERT_OK(LrtCreateRuntimeOptions(&options));

  // Only set enable_profiling
  const bool kEnableProfiling = true;
  LITERT_ASSERT_OK(
      LrtSetRuntimeOptionsEnableProfiling(options, kEnableProfiling));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueRuntimeOptionsData(options, &identifier,
                                                  &payload, &payload_deleter));

  LiteRtOpaqueOptions opaque_options = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(identifier, payload,
                                             payload_deleter, &opaque_options));

  // Verify payload
  void* payload_void = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(opaque_options, &payload_void));
  absl::string_view payload_str(static_cast<char*>(payload_void));

  auto toml_tbl = toml::parse(payload_str);
  EXPECT_TRUE(toml_tbl["enable_profiling"]);
  EXPECT_EQ(toml_tbl["enable_profiling"].value<bool>().value(),
            kEnableProfiling);

  // Verify other options are NOT present
  EXPECT_FALSE(toml_tbl["error_reporter_mode"]);
  EXPECT_FALSE(toml_tbl["compress_quantization_zero_points"]);

  LiteRtDestroyOpaqueOptions(opaque_options);
  LrtDestroyRuntimeOptions(options);
}

}  // namespace
