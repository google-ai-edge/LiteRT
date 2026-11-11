// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/arm_vulkan_ml/dispatch/tensor_capabilities.h"

#include <cstdint>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"
#include "vulkan/vulkan_core.h"  // from @arm_vulkan_ml_dep_vulkan_headers
#include "vulkan/vk_platform.h"  // from @arm_vulkan_ml_dep_vulkan_headers

namespace litert::arm_vulkan_ml {
namespace {
TEST(TensorStorage, Int16AndInt64HaveExactWidthsAndFormats) {
  EXPECT_EQ(TensorElementBytes(TensorElement::kInt16), 2);
  EXPECT_EQ(TensorElementBytes(TensorElement::kInt64), 8);
  EXPECT_EQ(TensorFormat(TensorElement::kInt16), VK_FORMAT_R16_SINT);
  EXPECT_EQ(TensorFormat(TensorElement::kInt64), VK_FORMAT_R64_SINT);
  const auto invalid_element = static_cast<TensorElement>(99);
  EXPECT_EQ(TensorElementBytes(invalid_element), 0);
  EXPECT_EQ(TensorFormat(invalid_element), VK_FORMAT_UNDEFINED);
  EXPECT_NE(VulkanLibraryName(), nullptr);
  EXPECT_STRNE(VulkanLibraryName(), "");
}

TEST(TosaLevelTest, EnforcesEveryLevelBoundAndRejectsUncheckedOrUnknownLevels) {
  DeviceCapabilities device;
  GraphEngine engine{};
  engine.instruction_set = "TOSA.001000.1";
  engine.level = VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM;
  engine.profiles = {{"PRO-INT", VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM}};
  CompiledPartition artifact;
  std::string reason;
  EXPECT_FALSE(
      CheckEngineRequirements(artifact, device, engine, false, reason));
  EXPECT_NE(reason.find("no checked"), std::string::npos);
  artifact.level_requirements.checked = true;
  const std::pair<uint64_t TosaLevelRequirements::*, uint64_t> bounds[] = {
      {&TosaLevelRequirements::max_rank, 6},
      {&TosaLevelRequirements::max_tensor_bytes, INT32_MAX},
      {&TosaLevelRequirements::max_kernel, 8192},
      {&TosaLevelRequirements::max_stride, 8192},
      {&TosaLevelRequirements::max_scale, 256},
      {&TosaLevelRequirements::max_tensor_list_size, 64},
      {&TosaLevelRequirements::max_nesting, 6},
  };
  for (const auto& [member, maximum] : bounds) {
    artifact.level_requirements.*member = maximum;
    EXPECT_TRUE(
        CheckEngineRequirements(artifact, device, engine, false, reason))
        << reason;
    artifact.level_requirements.*member = maximum + 1;
    EXPECT_FALSE(
        CheckEngineRequirements(artifact, device, engine, false, reason));
    EXPECT_NE(reason.find("exceeded"), std::string::npos);
    engine.level = VK_DATA_GRAPH_TOSA_LEVEL_NONE_ARM;
    EXPECT_TRUE(
        CheckEngineRequirements(artifact, device, engine, false, reason))
        << reason;
    engine.level = VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM;
    artifact.level_requirements.*member = 0;
  }
  engine.level = VK_DATA_GRAPH_TOSA_LEVEL_NONE_ARM;
  artifact.level_requirements.max_scale = 2049;
  EXPECT_FALSE(
      CheckEngineRequirements(artifact, device, engine, false, reason));
  engine.level = VK_DATA_GRAPH_TOSALEVEL_MAX_ENUM_ARM;
  EXPECT_FALSE(
      CheckEngineRequirements(artifact, device, engine, false, reason));
  EXPECT_NE(reason.find("Unknown TOSA level"), std::string::npos);
}

TEST(TosaLevelTest, IntegerWidthFeaturesDoNotImplyConformantInt16Extension) {
  DeviceCapabilities device;
  GraphEngine engine{};
  engine.instruction_set = "TOSA.001000.1";
  engine.level = VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM;
  engine.profiles = {{"PRO-INT", VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM}};
  CompiledPartition artifact;
  artifact.level_requirements.checked = true;
  artifact.requires_shader_int16 = true;
  std::string reason;
  EXPECT_FALSE(CheckEngineRequirements(artifact, device, engine, true, reason));
  device.shader_int16 = true;
  EXPECT_TRUE(CheckEngineRequirements(artifact, device, engine, true, reason))
      << reason;
  artifact.extensions = {"EXT-INT16"};
  EXPECT_FALSE(CheckEngineRequirements(artifact, device, engine, true, reason));
  engine.extensions = {
      {"EXT-INT16", VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM}};
  artifact.requires_shader_int64 = true;
  EXPECT_FALSE(CheckEngineRequirements(artifact, device, engine, true, reason));
  device.shader_int64 = true;
  EXPECT_TRUE(CheckEngineRequirements(artifact, device, engine, true, reason))
      << reason;
}

class TensorCapabilitiesTest : public testing::Test {
 protected:
  static VKAPI_ATTR void VKAPI_CALL Properties(
      VkPhysicalDevice device, VkPhysicalDeviceProperties2* out) {
    auto* self = reinterpret_cast<TensorCapabilitiesTest*>(device);
    *static_cast<VkPhysicalDeviceTensorPropertiesARM*>(out->pNext) =
        self->limits_;
  }
  static VKAPI_ATTR void VKAPI_CALL FormatProperties(
      VkPhysicalDevice device, VkFormat format, VkFormatProperties2* out) {
    auto* self = reinterpret_cast<TensorCapabilitiesTest*>(device);
    ++self->queries_[format];
    static_cast<VkTensorFormatPropertiesARM*>(out->pNext)
        ->linearTilingTensorFeatures = self->formats_[format];
  }
  void SetUp() override {
    limits_ = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM};
    limits_.maxTensorDimensionCount = 6;
    limits_.maxTensorElements = 1024;
    limits_.maxPerDimensionTensorElements = 128;
    limits_.maxTensorSize = 4096;
    limits_.maxDescriptorSetStorageTensors = 8;
    limits_.maxPerStageDescriptorSetStorageTensors = 8;
    formats_.clear();
    for (auto format :
         {VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_SFLOAT}) {
      formats_[format] = VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM;
    }
    queries_.clear();
    artifact_.inputs = {{TensorElement::kInt8, {1, 4, 4, 3}, 0}};
    artifact_.outputs = {{TensorElement::kFloat32, {1, 8}, 1}};
    artifact_.constants = {{{TensorElement::kInt32, {8}, 0}, {}}};
  }
  bool Supported() {
    return QueryTensorCapabilities(reinterpret_cast<VkPhysicalDevice>(this),
                                   artifact_, Properties, FormatProperties,
                                   reason_);
  }
  VkPhysicalDeviceTensorPropertiesARM limits_{};
  absl::flat_hash_map<VkFormat, VkFormatFeatureFlags2> formats_;
  absl::flat_hash_map<VkFormat, int> queries_;
  CompiledPartition artifact_;
  std::string reason_;
};

TEST_F(TensorCapabilitiesTest, Int16AndInt64FormatsAndByteLimitsAreQueried) {
  artifact_.inputs = {{TensorElement::kInt16, {1, 4}, 0}};
  artifact_.outputs = {{TensorElement::kInt16, {1, 4}, 1}};
  artifact_.constants = {{{TensorElement::kInt64, {2}, 0}, {}}};
  formats_[VK_FORMAT_R16_SINT] = VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM;
  formats_[VK_FORMAT_R64_SINT] = VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM;
  EXPECT_TRUE(Supported()) << reason_;
  EXPECT_EQ(queries_[VK_FORMAT_R16_SINT], 1);
  EXPECT_EQ(queries_[VK_FORMAT_R64_SINT], 1);
  formats_[VK_FORMAT_R64_SINT] = 0;
  EXPECT_FALSE(Supported());
  formats_[VK_FORMAT_R64_SINT] = VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM;
  limits_.maxTensorSize = 15;
  EXPECT_FALSE(Supported());
  EXPECT_NE(reason_.find("byte size"), std::string::npos);
  limits_.maxTensorSize = 16;
  EXPECT_TRUE(Supported()) << reason_;
}

TEST_F(TensorCapabilitiesTest, QueriesAllResourceTypesAndCachesFormats) {
  artifact_.inputs.push_back(artifact_.inputs.front());
  EXPECT_TRUE(Supported()) << reason_;
  EXPECT_EQ(queries_.size(), 3);
  for (const auto& query : queries_) {
    EXPECT_EQ(query.second, 1) << "format=" << query.first;
  }
}
TEST_F(TensorCapabilitiesTest, RejectsMissingConstantFormatSupport) {
  formats_[VK_FORMAT_R32_SINT] = VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_ARM;
  EXPECT_FALSE(Supported());
  EXPECT_NE(reason_.find("linear-tiled"), std::string::npos);
}
TEST_F(TensorCapabilitiesTest, RejectsMissingOutputFormatSupport) {
  formats_[VK_FORMAT_R32_SFLOAT] = 0;
  EXPECT_FALSE(Supported());
}
TEST_F(TensorCapabilitiesTest, RejectsDimensionAndElementLimits) {
  limits_.maxPerDimensionTensorElements = 3;
  EXPECT_FALSE(Supported());
  limits_.maxPerDimensionTensorElements = 128;
  limits_.maxTensorElements = 47;
  EXPECT_FALSE(Supported());
  limits_.maxTensorElements = 48;
  EXPECT_TRUE(Supported()) << reason_;
}
TEST_F(TensorCapabilitiesTest, RejectsByteSizeAndRankLimits) {
  limits_.maxTensorSize = 31;
  EXPECT_FALSE(Supported());
  limits_.maxTensorSize = 4096;
  limits_.maxTensorDimensionCount = 3;
  EXPECT_FALSE(Supported());
}
TEST_F(TensorCapabilitiesTest, RejectsDescriptorLimitsAndZeroDimensions) {
  limits_.maxPerStageDescriptorSetStorageTensors = 1;
  EXPECT_FALSE(Supported());
  limits_.maxPerStageDescriptorSetStorageTensors = 8;
  artifact_.inputs.front().shape[1] = 0;
  EXPECT_FALSE(Supported());
}

TEST_F(TensorCapabilitiesTest,
       RejectsUnknownTensorElementAndInstructionSetMismatch) {
  artifact_.inputs.front().element = static_cast<TensorElement>(99);
  EXPECT_FALSE(Supported());
  EXPECT_EQ(reason_, "Unknown tensor element type");

  DeviceCapabilities device;
  GraphEngine engine{};
  engine.instruction_set = "TOSA.001000.1";
  engine.level = VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM;
  CompiledPartition mismatch_artifact;
  mismatch_artifact.instruction_set = "TOSA.999999.1";
  mismatch_artifact.level_requirements.checked = true;
  std::string engine_reason;
  EXPECT_FALSE(CheckEngineRequirements(mismatch_artifact, device, engine, false,
                                       engine_reason));
  EXPECT_EQ(engine_reason, "Unsupported instruction set: TOSA.999999.1");
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
