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
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_intel_openvino_options.h"

using litert::intel_openvino::IntelOpenVinoOptions;

namespace {

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
  EXPECT_EQ(options_->GetDeviceType(), kLiteRtIntelOpenVinoDeviceTypeNPU);
  EXPECT_EQ(options_->GetPerformanceMode(),
            kLiteRtIntelOpenVinoPerformanceModeLatency);
}

TEST_F(IntelOpenVinoOptionsTest, SetAndGetDeviceType) {
  options_->SetDeviceType(kLiteRtIntelOpenVinoDeviceTypeCPU);
  EXPECT_EQ(options_->GetDeviceType(), kLiteRtIntelOpenVinoDeviceTypeCPU);

  options_->SetDeviceType(kLiteRtIntelOpenVinoDeviceTypeGPU);
  EXPECT_EQ(options_->GetDeviceType(), kLiteRtIntelOpenVinoDeviceTypeGPU);

  options_->SetDeviceType(kLiteRtIntelOpenVinoDeviceTypeAUTO);
  EXPECT_EQ(options_->GetDeviceType(), kLiteRtIntelOpenVinoDeviceTypeAUTO);
}

TEST_F(IntelOpenVinoOptionsTest, SetAndGetPerformanceMode) {
  options_->SetPerformanceMode(kLiteRtIntelOpenVinoPerformanceModeThroughput);
  EXPECT_EQ(options_->GetPerformanceMode(),
            kLiteRtIntelOpenVinoPerformanceModeThroughput);

  options_->SetPerformanceMode(
      kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput);
  EXPECT_EQ(options_->GetPerformanceMode(),
            kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput);
}

TEST_F(IntelOpenVinoOptionsTest, SetAndGetConfigsMapOption) {
  // Define expected config options
  std::map<std::string, std::string> expected_configs = {
      {"INFERENCE_PRECISION_HINT", "f16"},
      {"NPU_COMPILATION_MODE_PARAMS",
       "compute-layers-with-higher-precision=Sigmoid"},
      {"CACHE_DIR", "/tmp/ov_cache"}};

  // Set config options from the map
  for (const auto& [key, value] : expected_configs) {
    options_->SetConfigsMapOption(key.c_str(), value.c_str());
  }

  // Verify the number of options set
  EXPECT_EQ(options_->GetNumConfigsMapOptions(), 3);

  // Retrieve and verify each config option
  std::map<std::string, std::string> actual_configs;
  for (int i = 0; i < options_->GetNumConfigsMapOptions(); ++i) {
    auto [key, value] = options_->GetConfigsMapOption(i);
    if (!key.empty() && !value.empty()) {  // Skip empty pairs (error case)
      actual_configs[key] = value;
    }
  }

  // Verify all expected configs were set correctly
  EXPECT_EQ(actual_configs, expected_configs);
}

TEST_F(IntelOpenVinoOptionsTest, ConfigsMapMultipleOptions) {
  // Define expected configuration properties
  std::map<std::string, std::string> expected_configs = {
      {"PERFORMANCE_HINT", "LATENCY"},
      {"NUM_STREAMS", "4"},
      {"INFERENCE_PRECISION_HINT", "f16"}};

  // Set config options from the map
  for (const auto& [key, value] : expected_configs) {
    options_->SetConfigsMapOption(key.c_str(), value.c_str());
  }

  // Verify the number of options
  EXPECT_EQ(options_->GetNumConfigsMapOptions(), 3);

  // Verify the options object is still valid after multiple sets
  EXPECT_EQ(options_->GetDeviceType(), kLiteRtIntelOpenVinoDeviceTypeNPU);

  // Retrieve and verify the values
  std::map<std::string, std::string> actual_configs;
  for (int i = 0; i < options_->GetNumConfigsMapOptions(); ++i) {
    auto [key, value] = options_->GetConfigsMapOption(i);
    if (!key.empty() && !value.empty()) {  // Skip empty pairs (error case)
      actual_configs[key] = value;
    }
  }

  EXPECT_EQ(actual_configs, expected_configs);
}

TEST_F(IntelOpenVinoOptionsTest, OptionsIdentifier) {
  EXPECT_EQ(std::string(IntelOpenVinoOptions::Discriminator()),
            "intel_openvino");
}

// Test C API
TEST(IntelOpenVinoOptionsCApiTest, CreateAndDestroy) {
  LiteRtOpaqueOptions opaque_options;
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsCreate(&opaque_options), kLiteRtStatusOk);

  LiteRtIntelOpenVinoOptions options;
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsGet(opaque_options, &options),
            kLiteRtStatusOk);

  // Test setting and getting values through C API
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsSetDeviceType(
                options, kLiteRtIntelOpenVinoDeviceTypeCPU),
            kLiteRtStatusOk);

  LiteRtIntelOpenVinoDeviceType device_type;
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsGetDeviceType(options, &device_type),
            kLiteRtStatusOk);
  EXPECT_EQ(device_type, kLiteRtIntelOpenVinoDeviceTypeCPU);

  // Cleanup
  LiteRtDestroyOpaqueOptions(opaque_options);
}

TEST(IntelOpenVinoOptionsCApiTest, InvalidArguments) {
  // Test null pointer handling
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsCreate(nullptr),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtOpaqueOptions opaque_options;
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsCreate(&opaque_options), kLiteRtStatusOk);

  EXPECT_EQ(LiteRtIntelOpenVinoOptionsGet(nullptr, nullptr),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtIntelOpenVinoOptions options;
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsGet(opaque_options, &options),
            kLiteRtStatusOk);

  // Test setting with null options
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsSetDeviceType(
                nullptr, kLiteRtIntelOpenVinoDeviceTypeCPU),
            kLiteRtStatusErrorInvalidArgument);

  // Test getting with null output parameter
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsGetDeviceType(options, nullptr),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyOpaqueOptions(opaque_options);
}

TEST(IntelOpenVinoOptionsCApiTest, ConfigsMapOptions) {
  LiteRtOpaqueOptions opaque_options;
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsCreate(&opaque_options), kLiteRtStatusOk);

  LiteRtIntelOpenVinoOptions options;
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsGet(opaque_options, &options),
            kLiteRtStatusOk);

  // Set various configuration properties
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsSetConfigsMapOption(
                options, "INFERENCE_PRECISION_HINT", "f16"),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsSetConfigsMapOption(
                options, "NPU_COMPILATION_MODE_PARAMS",
                "compute-layers-with-higher-precision=Sigmoid"),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtIntelOpenVinoOptionsSetConfigsMapOption(options, "CACHE_DIR",
                                                          "/tmp/cache"),
            kLiteRtStatusOk);

  // Test null pointer handling for configs_map
  EXPECT_EQ(
      LiteRtIntelOpenVinoOptionsSetConfigsMapOption(nullptr, "KEY", "VALUE"),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtIntelOpenVinoOptionsSetConfigsMapOption(options, nullptr, "VALUE"),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtIntelOpenVinoOptionsSetConfigsMapOption(options, "KEY", nullptr),
      kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyOpaqueOptions(opaque_options);
}

}  // namespace
