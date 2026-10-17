/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_JIT_JIT_COMPILER_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_JIT_JIT_COMPILER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/arm_vulkan_ml/common/graph_desc.h"

namespace litert::arm_vulkan_ml {

namespace jit {

struct JitCompiledModel {
  std::vector<std::vector<char>> byte_codes;
  std::vector<std::unique_ptr<litert::arm_vulkan_ml::GraphDesc>>
      jit_executables;
  std::vector<std::string> call_infos;
  std::vector<size_t> call_executable_indices;
};

LiteRtStatus PartitionForJit(LiteRtSubgraph subgraph,
                             LiteRtOpList selected_ops);

litert::Expected<JitCompiledModel> CompileJit(LiteRtModel partitions);

}  // namespace jit

}  // namespace litert::arm_vulkan_ml

#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_JIT_JIT_COMPILER_H_
