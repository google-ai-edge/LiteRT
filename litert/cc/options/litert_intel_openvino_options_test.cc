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

#include "litert/cc/options/litert_intel_openvino_options.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_intel_openvino_options.h"

using litert::intel_openvino::IntelOpenVinoOptions;

namespace {

void SerializeAndParse(LrtIntelOpenVinoOptions payload,
                       LrtIntelOpenVinoOptions* payload_from_toml) {
  const char* identifier;
  void* data;
  void (*payload_deleter)(void*);
  ASSERT_EQ(LrtGetOpaqueIntelOpenVinoOptionsData(payload, &identifier, &data,
                                                 &payload_deleter),
            kLiteRtStatusOk);

  ASSERT_EQ(LrtCreateIntelOpenVinoOptionsFromToml(
                static_cast<const char*>(data), payload_from_toml),
            kLiteRtStatusOk);

  payload_deleter(data);
}

class IntelOpenVinoOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto options_result = IntelOpenVinoOptions::Create();
    ASSERT_TRUE(options_result);
    options_ = std::make_unique<IntelOpenVinoOptions>(
        std::move(options_result.Value()));
  }

  std::unique_ptr<IntelOpenVinoOptions> options_;
};

TEST_F(IntelOpenVinoOptionsTest, DefaultValues) {
  LrtIntelOpenVinoOptions parsed = nullptr;
  SerializeAndParse(options_->Get(), &parsed);

  // No per-graph overrides are configured by default.
  int num_overrides = -1;
  ASSERT_EQ(LrtIntelOpenVinoOptionsGetNumGraphOverrides(parsed, &num_overrides),
            kLiteRtStatusOk);
  EXPECT_EQ(num_overrides, 0);

  LiteRtIntelOpenVinoPerformanceMode perf_mode;
  ASSERT_EQ(LrtIntelOpenVinoOptionsGetPerformanceMode(parsed, &perf_mode),
            kLiteRtStatusOk);
  EXPECT_EQ(perf_mode, kLiteRtIntelOpenVinoPerformanceModeLatency);

  int num_configs;
  ASSERT_EQ(
      LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(parsed, &num_configs),
      kLiteRtStatusOk);
  EXPECT_EQ(num_configs, 0);

  LrtDestroyIntelOpenVinoOptions(parsed);
}

TEST_F(IntelOpenVinoOptionsTest, SetAndGetGraphBackend) {
  options_->SetGraphBackend(/*graph_index=*/0, kLiteRtIntelOpenVinoGraphBackendCPU);
  LrtIntelOpenVinoOptions parsed = nullptr;
  SerializeAndParse(options_->Get(), &parsed);

  LiteRtIntelOpenVinoGraphBackend graph_backend;
  ASSERT_EQ(LrtIntelOpenVinoOptionsGetGraphBackend(parsed, /*graph_index=*/0,
                                                &graph_backend),
            kLiteRtStatusOk);
  EXPECT_EQ(graph_backend, kLiteRtIntelOpenVinoGraphBackendCPU);
  LrtDestroyIntelOpenVinoOptions(parsed);

  options_->SetGraphBackend(/*graph_index=*/0, kLiteRtIntelOpenVinoGraphBackendGPU);
  LrtIntelOpenVinoOptions parsed2 = nullptr;
  SerializeAndParse(options_->Get(), &parsed2);

  ASSERT_EQ(LrtIntelOpenVinoOptionsGetGraphBackend(parsed2, /*graph_index=*/0,
                                                &graph_backend),
            kLiteRtStatusOk);
  EXPECT_EQ(graph_backend, kLiteRtIntelOpenVinoGraphBackendGPU);
  LrtDestroyIntelOpenVinoOptions(parsed2);
}

TEST_F(IntelOpenVinoOptionsTest, SetAndGetPerformanceMode) {
  options_->SetPerformanceMode(kLiteRtIntelOpenVinoPerformanceModeThroughput);
  LrtIntelOpenVinoOptions parsed = nullptr;
  SerializeAndParse(options_->Get(), &parsed);

  LiteRtIntelOpenVinoPerformanceMode perf_mode;
  ASSERT_EQ(LrtIntelOpenVinoOptionsGetPerformanceMode(parsed, &perf_mode),
            kLiteRtStatusOk);
  EXPECT_EQ(perf_mode, kLiteRtIntelOpenVinoPerformanceModeThroughput);
  LrtDestroyIntelOpenVinoOptions(parsed);

  options_->SetPerformanceMode(
      kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput);
  LrtIntelOpenVinoOptions parsed2 = nullptr;
  SerializeAndParse(options_->Get(), &parsed2);

  ASSERT_EQ(LrtIntelOpenVinoOptionsGetPerformanceMode(parsed2, &perf_mode),
            kLiteRtStatusOk);
  EXPECT_EQ(perf_mode, kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput);
  LrtDestroyIntelOpenVinoOptions(parsed2);
}

TEST_F(IntelOpenVinoOptionsTest, SetAndGetConfigsMapOption) {
  options_->SetConfigsMapOption("INFERENCE_PRECISION_HINT", "f16");
  options_->SetConfigsMapOption("NPU_COMPILATION_MODE_PARAMS",
                                "compute-layers-with-higher-precision=Sigmoid");

  LrtIntelOpenVinoOptions parsed = nullptr;
  SerializeAndParse(options_->Get(), &parsed);

  int num_configs;
  ASSERT_EQ(
      LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(parsed, &num_configs),
      kLiteRtStatusOk);
  EXPECT_EQ(num_configs, 2);

  std::map<std::string, std::string> parsed_map;
  for (int i = 0; i < num_configs; i++) {
    const char* key;
    const char* val;
    ASSERT_EQ(LrtIntelOpenVinoOptionsGetConfigsMapOption(parsed, i, &key, &val),
              kLiteRtStatusOk);
    parsed_map[key] = val;
  }

  EXPECT_EQ(parsed_map["INFERENCE_PRECISION_HINT"], "f16");
  EXPECT_EQ(parsed_map["NPU_COMPILATION_MODE_PARAMS"],
            "compute-layers-with-higher-precision=Sigmoid");

  LrtDestroyIntelOpenVinoOptions(parsed);
}

// Test C API
TEST(IntelOpenVinoOptionsCApiTest, CreateAndDestroy) {
  LrtIntelOpenVinoOptions options;
  EXPECT_EQ(LrtIntelOpenVinoOptionsCreate(&options), kLiteRtStatusOk);

  EXPECT_EQ(LrtIntelOpenVinoOptionsSetGraphBackend(
                options, /*graph_index=*/0, kLiteRtIntelOpenVinoGraphBackendCPU),
            kLiteRtStatusOk);
  EXPECT_EQ(LrtIntelOpenVinoOptionsSetPerformanceMode(
                options, kLiteRtIntelOpenVinoPerformanceModeThroughput),
            kLiteRtStatusOk);

  LiteRtIntelOpenVinoGraphBackend graph_backend;
  EXPECT_EQ(LrtIntelOpenVinoOptionsGetGraphBackend(options, /*graph_index=*/0,
                                                &graph_backend),
            kLiteRtStatusOk);
  EXPECT_EQ(graph_backend, kLiteRtIntelOpenVinoGraphBackendCPU);

  LrtDestroyIntelOpenVinoOptions(options);
}

TEST(IntelOpenVinoOptionsCApiTest, InvalidArguments) {
  EXPECT_EQ(LrtIntelOpenVinoOptionsCreate(nullptr),
            kLiteRtStatusErrorInvalidArgument);

  LrtIntelOpenVinoOptions options;
  EXPECT_EQ(LrtIntelOpenVinoOptionsCreate(&options), kLiteRtStatusOk);

  EXPECT_EQ(LrtIntelOpenVinoOptionsSetGraphBackend(
                nullptr, /*graph_index=*/0, kLiteRtIntelOpenVinoGraphBackendCPU),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(
      LrtIntelOpenVinoOptionsGetGraphBackend(options, /*graph_index=*/0, nullptr),
      kLiteRtStatusErrorInvalidArgument);

  LrtDestroyIntelOpenVinoOptions(options);
}

TEST(IntelOpenVinoOptionsCApiTest, RoundTripSerialization) {
  LrtIntelOpenVinoOptions options;
  EXPECT_EQ(LrtIntelOpenVinoOptionsCreate(&options), kLiteRtStatusOk);

  EXPECT_EQ(LrtIntelOpenVinoOptionsSetGraphBackend(
                options, /*graph_index=*/0, kLiteRtIntelOpenVinoGraphBackendGPU),
            kLiteRtStatusOk);
  EXPECT_EQ(
      LrtIntelOpenVinoOptionsSetConfigsMapOption(options, "key1", "value1"),
      kLiteRtStatusOk);

  LrtIntelOpenVinoOptions parsed = nullptr;
  SerializeAndParse(options, &parsed);

  LiteRtIntelOpenVinoGraphBackend graph_backend;
  ASSERT_EQ(LrtIntelOpenVinoOptionsGetGraphBackend(parsed, /*graph_index=*/0,
                                                &graph_backend),
            kLiteRtStatusOk);
  EXPECT_EQ(graph_backend, kLiteRtIntelOpenVinoGraphBackendGPU);

  int num_configs;
  ASSERT_EQ(
      LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(parsed, &num_configs),
      kLiteRtStatusOk);
  EXPECT_EQ(num_configs, 1);

  const char* key;
  const char* val;
  ASSERT_EQ(LrtIntelOpenVinoOptionsGetConfigsMapOption(parsed, 0, &key, &val),
            kLiteRtStatusOk);
  EXPECT_EQ(std::string(key), "key1");
  EXPECT_EQ(std::string(val), "value1");

  LrtDestroyIntelOpenVinoOptions(parsed);
  LrtDestroyIntelOpenVinoOptions(options);
}

}  // namespace
