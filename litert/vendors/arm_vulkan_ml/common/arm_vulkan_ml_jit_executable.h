//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0
//

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_ARM_VULKAN_ML_JIT_EXECUTABLE_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_ARM_VULKAN_ML_JIT_EXECUTABLE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/litert_model_types.h"

namespace litert::arm_vulkan_ml {

enum class TensorElement { kInt8, kInt32, kFloat32, kInt16, kInt64 };
inline size_t TensorElementBytes(TensorElement element) {
  switch (element) {
    case TensorElement::kInt8:
      return 1;
    case TensorElement::kInt16:
      return 2;
    case TensorElement::kInt32:
    case TensorElement::kFloat32:
      return 4;
    case TensorElement::kInt64:
      return 8;
  }
  return 0;
}

struct TensorDescriptor {
  TensorElement element;
  std::vector<uint32_t> shape;
  uint32_t binding;
  // BOOL uses INT8 storage and UINT32 uses INT32 storage in the graph ABI.
  // COMPLEX64 uses interleaved FP32 storage with a trailing dimension of two.
  // Retain the model type for LiteRT buffer requirements and attachment checks.
  LiteRtElementType logical_element =
      element == TensorElement::kInt8      ? kLiteRtElementTypeInt8
      : element == TensorElement::kFloat32 ? kLiteRtElementTypeFloat32
      : element == TensorElement::kInt16   ? kLiteRtElementTypeInt16
      : element == TensorElement::kInt64   ? kLiteRtElementTypeInt64
                                           : kLiteRtElementTypeInt32;
  // Vulkan tensors have positive rank. A logical LiteRT scalar is stored as
  // one physical element and retains strict rank-zero buffer validation.
  bool logical_scalar = false;
};

// Requirements of the emitted graph, including intermediate tensors and
// generated operators. The specification database remains a development tool.
struct TosaLevelRequirements {
  bool checked = false;
  uint64_t max_rank = 0;
  uint64_t max_tensor_bytes = 0;
  uint64_t max_kernel = 0;
  uint64_t max_stride = 0;
  uint64_t max_scale = 0;
  uint64_t max_tensor_list_size = 0;
  uint64_t max_nesting = 0;
};

// Static logical storage volumes, not allocations or peak live memory.
struct GraphMetrics {
  std::map<uint32_t, size_t> operators;
  size_t identity_reshapes = 0;
  uint64_t reshape_bytes = 0;
  uint64_t identity_reshape_bytes = 0;
  uint64_t largest_result_bytes = 0;
};

struct ConstantTensor {
  TensorDescriptor descriptor;
  std::vector<uint8_t> bytes;
};

// Published as const after compilation. Contains no borrowed model memory,
// TFSC IR, VGF archive, Vulkan objects, or driver-private handles.
struct CompiledPartition {
  bool requires_shader_int16 = false;
  bool requires_shader_int64 = false;
  std::vector<uint32_t> spirv;
  std::vector<ConstantTensor> constants;
  std::vector<TensorDescriptor> inputs;
  std::vector<TensorDescriptor> outputs;
  std::string entry_point = "main";
  std::string instruction_set = "TOSA.001000.1";
  std::vector<std::string> profiles = {"PRO-INT"};
  std::vector<std::string> extensions;
  TosaLevelRequirements level_requirements;
  GraphMetrics metrics;
};

// The compiled result owns this wrapper. An invocation must copy `partition`
// while the wrapper is alive, then retain that shared ownership until teardown.
struct ArmVulkanMLJitExecutable {
  std::shared_ptr<const CompiledPartition> partition;
};

}  // namespace litert::arm_vulkan_ml

#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMMON_ARM_VULKAN_ML_JIT_EXECUTABLE_H_
