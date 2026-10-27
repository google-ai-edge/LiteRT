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

#include <map>
#include "gtest/gtest.h"

namespace litert::arm_vulkan_ml {
namespace {
VkPhysicalDeviceTensorPropertiesARM limits;
std::map<VkFormat, VkFormatFeatureFlags2> formats;
std::map<VkFormat, int> queries;

VKAPI_ATTR void VKAPI_CALL Properties(VkPhysicalDevice, VkPhysicalDeviceProperties2* out) {
  *static_cast<VkPhysicalDeviceTensorPropertiesARM*>(out->pNext) = limits;
}
VKAPI_ATTR void VKAPI_CALL FormatProperties(VkPhysicalDevice, VkFormat format,
                                           VkFormatProperties2* out) {
  ++queries[format];
  static_cast<VkTensorFormatPropertiesARM*>(out->pNext)->linearTilingTensorFeatures = formats[format];
}
class TensorCapabilitiesTest : public testing::Test {
 protected:
  void SetUp() override {
    limits = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM};
    limits.maxTensorDimensionCount = 6;
    limits.maxTensorElements = 1024;
    limits.maxPerDimensionTensorElements = 128;
    limits.maxTensorSize = 4096;
    limits.maxDescriptorSetStorageTensors = 8;
    limits.maxPerStageDescriptorSetStorageTensors = 8;
    for (auto format : {VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_SFLOAT})
      formats[format] = VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM;
    queries.clear();
    artifact.inputs = {{TensorElement::kInt8, {1, 4, 4, 3}, 0}};
    artifact.outputs = {{TensorElement::kFloat32, {1, 8}, 1}};
    artifact.constants = {{{TensorElement::kInt32, {8}, 0}, {}}};
  }
  bool Supported() {
    return QueryTensorCapabilities(VK_NULL_HANDLE, artifact, Properties,
                                   FormatProperties, reason);
  }
  CompiledPartition artifact;
  std::string reason;
};

TEST_F(TensorCapabilitiesTest, QueriesAllResourceTypesAndCachesFormats) {
  artifact.inputs.push_back(artifact.inputs.front());
  EXPECT_TRUE(Supported()) << reason;
  EXPECT_EQ(queries.size(), 3);
  for (const auto& query : queries) EXPECT_EQ(query.second, 1);
}
TEST_F(TensorCapabilitiesTest, RejectsMissingConstantFormatSupport) {
  formats[VK_FORMAT_R32_SINT] = VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_ARM;
  EXPECT_FALSE(Supported());
  EXPECT_NE(reason.find("linear-tiled"), std::string::npos);
}
TEST_F(TensorCapabilitiesTest, RejectsMissingOutputFormatSupport) {
  formats[VK_FORMAT_R32_SFLOAT] = 0;
  EXPECT_FALSE(Supported());
}
TEST_F(TensorCapabilitiesTest, RejectsDimensionAndElementLimits) {
  limits.maxPerDimensionTensorElements = 3;
  EXPECT_FALSE(Supported());
  limits.maxPerDimensionTensorElements = 128;
  limits.maxTensorElements = 47;
  EXPECT_FALSE(Supported());
  limits.maxTensorElements = 48;
  EXPECT_TRUE(Supported()) << reason;
}
TEST_F(TensorCapabilitiesTest, RejectsByteSizeAndRankLimits) {
  limits.maxTensorSize = 31;
  EXPECT_FALSE(Supported());
  limits.maxTensorSize = 4096;
  limits.maxTensorDimensionCount = 3;
  EXPECT_FALSE(Supported());
}
TEST_F(TensorCapabilitiesTest, RejectsDescriptorLimitsAndZeroDimensions) {
  limits.maxPerStageDescriptorSetStorageTensors = 1;
  EXPECT_FALSE(Supported());
  limits.maxPerStageDescriptorSetStorageTensors = 8;
  artifact.inputs.front().shape[1] = 0;
  EXPECT_FALSE(Supported());
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
