//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0
//

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_ARM_VULKAN_ML_JIT_EXECUTABLE_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_ARM_VULKAN_ML_JIT_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace litert::arm_vulkan_ml {

enum class TensorElement { kInt8, kInt32, kFloat32 };

struct TensorDescriptor {
  TensorElement element;
  std::vector<uint32_t> shape;
  uint32_t binding;
};

struct ConstantTensor {
  TensorDescriptor descriptor;
  std::vector<uint8_t> bytes;
};

// Published as const after compilation. Contains no borrowed model memory,
// TFSC IR, VGF archive, Vulkan objects, or driver-private handles.
struct CompiledPartition {
  std::vector<uint32_t> spirv;
  std::vector<ConstantTensor> constants;
  std::vector<TensorDescriptor> inputs;
  std::vector<TensorDescriptor> outputs;
  std::string entry_point = "main";
  std::string instruction_set = "TOSA.001000.1";
  std::vector<std::string> profiles = {"PRO-INT"};
  std::vector<std::string> extensions;
};

// The compiled result owns this wrapper. An invocation must copy `partition`
// while the wrapper is alive, then retain that shared ownership until teardown.
struct ArmVulkanMLJitExecutable {
  std::shared_ptr<const CompiledPartition> partition;
};

}  // namespace litert::arm_vulkan_ml

#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_ARM_VULKAN_ML_JIT_EXECUTABLE_H_
