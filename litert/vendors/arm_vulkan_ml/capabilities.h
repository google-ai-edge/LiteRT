// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_CAPABILITIES_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_CAPABILITIES_H_

#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"

namespace litert::arm_vulkan_ml {

bool IsSupportedOpCode(LiteRtOpCode op_code);

bool IsSupportedType(ElementType type);

}  // namespace litert::arm_vulkan_ml

#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_CAPABILITIES_H_
