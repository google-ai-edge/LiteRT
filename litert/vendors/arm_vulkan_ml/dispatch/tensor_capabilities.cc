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

namespace litert::arm_vulkan_ml {

VkFormat TensorFormat(TensorElement element) {
  switch (element) {
    case TensorElement::kInt8: return VK_FORMAT_R8_SINT;
    case TensorElement::kInt32: return VK_FORMAT_R32_SINT;
    case TensorElement::kFloat32: return VK_FORMAT_R32_SFLOAT;
  }
  return VK_FORMAT_UNDEFINED;
}

bool QueryTensorCapabilities(
    VkPhysicalDevice device, const CompiledPartition& artifact,
    PFN_vkGetPhysicalDeviceProperties2 get_properties,
    PFN_vkGetPhysicalDeviceFormatProperties2 get_format_properties,
    std::string& reason) {
  VkPhysicalDeviceTensorPropertiesARM limits{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM};
  VkPhysicalDeviceProperties2 properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &limits;
  get_properties(device, &properties);
  const size_t descriptors = artifact.inputs.size() + artifact.outputs.size();
  if (descriptors > limits.maxDescriptorSetStorageTensors ||
      descriptors > limits.maxPerStageDescriptorSetStorageTensors) {
    reason = "Tensor descriptor count exceeds device limits";
    return false;
  }
  std::map<VkFormat, VkFormatFeatureFlags2> formats;
  auto check = [&](const TensorDescriptor& tensor) {
    const VkFormat format = TensorFormat(tensor.element);
    if (format == VK_FORMAT_UNDEFINED) {
      reason = "Unknown tensor element type";
      return false;
    }
    if (!formats.count(format)) {
      VkTensorFormatPropertiesARM tensor_format{
          VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_ARM};
      VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
      properties.pNext = &tensor_format;
      get_format_properties(device, format, &properties);
      formats.emplace(format, tensor_format.linearTilingTensorFeatures);
    }
    if (!(formats.find(format)->second & VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM)) {
      reason = "VkFormat " + std::to_string(format) +
          " does not support linear-tiled data-graph tensors";
      return false;
    }
    if (tensor.shape.empty() || tensor.shape.size() > limits.maxTensorDimensionCount) {
      reason = "Tensor rank exceeds device limits";
      return false;
    }
    uint64_t elements = 1;
    for (uint32_t dimension : tensor.shape) {
      if (!dimension || dimension > limits.maxPerDimensionTensorElements ||
          elements > limits.maxTensorElements / dimension) {
        reason = "Tensor dimensions or element count exceed device limits";
        return false;
      }
      elements *= dimension;
    }
    const uint64_t element_bytes = tensor.element == TensorElement::kInt8 ? 1 : 4;
    if (elements > limits.maxTensorSize / element_bytes) {
      reason = "Tensor byte size exceeds device limits";
      return false;
    }
    return true;
  };
  for (const auto& tensor : artifact.inputs) if (!check(tensor)) return false;
  for (const auto& tensor : artifact.outputs) if (!check(tensor)) return false;
  for (const auto& constant : artifact.constants)
    if (!check(constant.descriptor)) return false;
  reason.clear();
  return true;
}

}  // namespace litert::arm_vulkan_ml
