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

#include "litert/c/options/litert_intel_openvino_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"

namespace {

void SerializeAndParse(LrtIntelOpenVinoOptions payload,
                       LrtIntelOpenVinoOptions* payload_from_toml) {
  const char* identifier = nullptr;
  void* opaque_payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;

  LITERT_ASSERT_OK(LrtGetOpaqueIntelOpenVinoOptionsData(
      payload, &identifier, &opaque_payload, &payload_deleter));

  LITERT_ASSERT_OK(LrtCreateIntelOpenVinoOptionsFromToml(
      static_cast<const char*>(opaque_payload), payload_from_toml));

  if (payload_deleter && opaque_payload) {
    payload_deleter(opaque_payload);
  }
}

using ::testing::Eq;
using ::testing::NotNull;
using ::testing::StrEq;
using ::testing::litert::IsError;

TEST(IntelOpenVinoOptions, CreationWorks) {
  EXPECT_THAT(LrtIntelOpenVinoOptionsCreate(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LrtIntelOpenVinoOptions payload = nullptr;
  LITERT_ASSERT_OK(LrtIntelOpenVinoOptionsCreate(&payload));
  EXPECT_THAT(payload, NotNull());

  LrtDestroyIntelOpenVinoOptions(payload);
}

TEST(IntelOpenVinoOptions, SetAndGetDeviceType) {
  LrtIntelOpenVinoOptions payload = nullptr;
  LITERT_ASSERT_OK(LrtIntelOpenVinoOptionsCreate(&payload));

  LiteRtIntelOpenVinoDeviceType device_type;
  // Check the default value.
  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsGetDeviceType(payload, &device_type));
  EXPECT_THAT(device_type, Eq(kLiteRtIntelOpenVinoDeviceTypeNPU));

  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsSetDeviceType(
      payload, kLiteRtIntelOpenVinoDeviceTypeCPU));
  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsGetDeviceType(payload, &device_type));
  EXPECT_EQ(device_type, kLiteRtIntelOpenVinoDeviceTypeCPU);

  LrtIntelOpenVinoOptions payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  LiteRtIntelOpenVinoDeviceType device_type_from_toml;
  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsGetDeviceType(
      payload_from_toml, &device_type_from_toml));
  EXPECT_THAT(device_type_from_toml, Eq(kLiteRtIntelOpenVinoDeviceTypeCPU));

  LrtDestroyIntelOpenVinoOptions(payload_from_toml);
  LrtDestroyIntelOpenVinoOptions(payload);
}

TEST(IntelOpenVinoOptions, SetAndGetPerformanceMode) {
  LrtIntelOpenVinoOptions payload = nullptr;
  LITERT_ASSERT_OK(LrtIntelOpenVinoOptionsCreate(&payload));

  LiteRtIntelOpenVinoPerformanceMode performance_mode;
  // Check the default value.
  LITERT_EXPECT_OK(
      LrtIntelOpenVinoOptionsGetPerformanceMode(payload, &performance_mode));
  EXPECT_THAT(performance_mode, Eq(kLiteRtIntelOpenVinoPerformanceModeLatency));

  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsSetPerformanceMode(
      payload, kLiteRtIntelOpenVinoPerformanceModeThroughput));
  LITERT_EXPECT_OK(
      LrtIntelOpenVinoOptionsGetPerformanceMode(payload, &performance_mode));
  EXPECT_EQ(performance_mode, kLiteRtIntelOpenVinoPerformanceModeThroughput);

  LrtIntelOpenVinoOptions payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  LiteRtIntelOpenVinoPerformanceMode performance_mode_from_toml;
  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsGetPerformanceMode(
      payload_from_toml, &performance_mode_from_toml));
  EXPECT_THAT(performance_mode_from_toml,
              Eq(kLiteRtIntelOpenVinoPerformanceModeThroughput));

  LrtDestroyIntelOpenVinoOptions(payload_from_toml);
  LrtDestroyIntelOpenVinoOptions(payload);
}

TEST(IntelOpenVinoOptions, TomlSerializationWithConfigsMap) {
  LrtIntelOpenVinoOptions payload = nullptr;
  LITERT_ASSERT_OK(LrtIntelOpenVinoOptionsCreate(&payload));

  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsSetConfigsMapOption(
      payload, "INFERENCE_PRECISION_HINT", "f16"));
  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsSetConfigsMapOption(
      payload, "CACHE_DIR", "/tmp/ov_cache"));

  const char* identifier = nullptr;
  void* opaque_payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;

  LITERT_EXPECT_OK(LrtGetOpaqueIntelOpenVinoOptionsData(
      payload, &identifier, &opaque_payload, &payload_deleter));
  EXPECT_THAT(identifier, StrEq("intel_openvino"));
  EXPECT_THAT(opaque_payload, NotNull());

  LrtIntelOpenVinoOptions payload_from_toml = nullptr;
  LITERT_ASSERT_OK(LrtCreateIntelOpenVinoOptionsFromToml(
      static_cast<const char*>(opaque_payload), &payload_from_toml));

  int num_options = 0;
  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(
      payload_from_toml, &num_options));
  EXPECT_THAT(num_options, Eq(2));

  const char* key0 = nullptr;
  const char* value0 = nullptr;
  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsGetConfigsMapOption(
      payload_from_toml, 0, &key0, &value0));
  EXPECT_THAT(key0, StrEq("INFERENCE_PRECISION_HINT"));
  EXPECT_THAT(value0, StrEq("f16"));

  const char* key1 = nullptr;
  const char* value1 = nullptr;
  LITERT_EXPECT_OK(LrtIntelOpenVinoOptionsGetConfigsMapOption(
      payload_from_toml, 1, &key1, &value1));
  EXPECT_THAT(key1, StrEq("CACHE_DIR"));
  EXPECT_THAT(value1, StrEq("/tmp/ov_cache"));

  payload_deleter(opaque_payload);
  LrtDestroyIntelOpenVinoOptions(payload);
  LrtDestroyIntelOpenVinoOptions(payload_from_toml);
}

}  // namespace
