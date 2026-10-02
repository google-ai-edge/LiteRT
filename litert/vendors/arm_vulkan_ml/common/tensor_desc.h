/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_TENSOR_DESC_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_TENSOR_DESC_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>  // from @arm_vulkan_ml_dep_vulkan_headers

namespace litert::arm_vulkan_ml {

struct TensorDesc {
  std::string name;
  std::vector<int64_t> shape;
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t binding = 0;
  size_t bytes = 0;
};

struct ConstantTensorDesc : public TensorDesc {
  std::vector<uint8_t> data;
  // On-device VGF constants can refer directly into GraphDesc::vgf_data.
  // Other callers can retain owned data storage.
  size_t vgf_data_offset = 0;
  bool uses_vgf_data = false;
};

}  // namespace litert::arm_vulkan_ml

#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_TENSOR_DESC_H_
