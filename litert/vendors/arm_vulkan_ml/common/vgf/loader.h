/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_VGF_LOADER_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_VGF_LOADER_H_

#include <cstddef>
#include <string_view>
#include <vector>

#include "litert/cc/litert_expected.h"
#include "litert/vendors/arm_vulkan_ml/common/graph_desc.h"

namespace litert::arm_vulkan_ml::vgf {

litert::Expected<litert::arm_vulkan_ml::GraphDesc> LoadVgfGraphFromBuffer(
    const void* data, size_t size,
    std::string_view preferred_graph_name = std::string_view{});

// Takes ownership of a VGF and keeps its constant payloads as zero copy views
// in the returned graph. This is intended for On Device flow, where the VGF
// buffer has no other consumer.
litert::Expected<litert::arm_vulkan_ml::GraphDesc> LoadVgfGraphFromOwnedBuffer(
    std::vector<char> data,
    std::string_view preferred_graph_name = std::string_view{});

}  // namespace litert::arm_vulkan_ml::vgf

#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_VGF_LOADER_H_
