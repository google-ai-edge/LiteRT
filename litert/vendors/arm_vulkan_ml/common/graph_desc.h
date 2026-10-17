/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_GRAPH_DESC_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_GRAPH_DESC_H_

#include <cstdint>
#include <string>
#include <vector>

#include "litert/vendors/arm_vulkan_ml/common/tensor_desc.h"

namespace litert::arm_vulkan_ml {

struct GraphDesc {
  std::string name;
  std::vector<uint32_t> spirv;
  std::vector<TensorDesc> inputs;
  std::vector<TensorDesc> outputs;
  std::vector<ConstantTensorDesc> const_inputs;
  std::vector<char> vgf_data;

  // Returns a constant's owned bytes or a view into the retained on-device VGF
  // buffer.
  const uint8_t* ConstantData(const ConstantTensorDesc& constant) const {
    if (!constant.uses_vgf_data) {
      return constant.data.data();
    }

    if (constant.vgf_data_offset > vgf_data.size() ||
        constant.bytes > vgf_data.size() - constant.vgf_data_offset) {
      return nullptr;
    }

    return reinterpret_cast<const uint8_t*>(vgf_data.data()) +
           constant.vgf_data_offset;
  }
};

}  // namespace litert::arm_vulkan_ml

#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_GRAPH_DESC_H_
