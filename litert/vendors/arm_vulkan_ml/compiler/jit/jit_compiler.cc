/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "litert/vendors/arm_vulkan_ml/compiler/jit/jit_compiler.h"

#include <string_view>
#include <utility>

#include "litert/cc/litert_macros.h"
#include "litert/vendors/arm_vulkan_ml/compiler/tcft_model_converter.h"
#include "litert/vendors/arm_vulkan_ml/common/vgf/loader.h"

namespace litert::arm_vulkan_ml::jit {

LiteRtStatus PartitionForJit(LiteRtSubgraph subgraph,
                             LiteRtOpList selected_ops) {
  if (subgraph == nullptr || selected_ops == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  return litert::arm_vulkan_ml::PartitionForTcftModelConverter(subgraph,
                                                               selected_ops);
}

litert::Expected<JitCompiledModel> CompileJit(LiteRtModel partitions) {
  if (partitions == nullptr) {
    return Unexpected{kLiteRtStatusErrorInvalidArgument, "partitions"};
  }

  litert::arm_vulkan_ml::TcftModelConverterCompiledModel tcft_compiled_model;
  LITERT_RETURN_IF_ERROR(litert::arm_vulkan_ml::CompileWithTcftModelConverter(
      partitions, &tcft_compiled_model));

  JitCompiledModel result;
  result.byte_codes.reserve(tcft_compiled_model.byte_codes.size());
  result.jit_executables.reserve(tcft_compiled_model.byte_codes.size());
  for (size_t i = 0; i < tcft_compiled_model.byte_codes.size(); ++i) {
    std::string_view graph_name;
    for (size_t call_idx = 0;
         call_idx < tcft_compiled_model.call_byte_code_indices.size();
         ++call_idx) {
      if (tcft_compiled_model.call_byte_code_indices[call_idx] == i &&
          call_idx < tcft_compiled_model.call_infos.size()) {
        graph_name = tcft_compiled_model.call_infos[call_idx];
        break;
      }
    }

    LITERT_ASSIGN_OR_RETURN(
        auto graph_desc,
        litert::arm_vulkan_ml::vgf::LoadVgfGraphFromOwnedBuffer(
            std::move(tcft_compiled_model.byte_codes[i]), graph_name),
        _ << "Arm ML Extensions for Vulkan JIT compiler: VGF graph decode "
             "failed for partition "
          << i);

    auto executable = std::make_unique<litert::arm_vulkan_ml::GraphDesc>(
        std::move(graph_desc));
    result.byte_codes.emplace_back();
    result.jit_executables.emplace_back(std::move(executable));
  }

  result.call_infos = std::move(tcft_compiled_model.call_infos);
  result.call_executable_indices =
      std::move(tcft_compiled_model.call_byte_code_indices);

  return result;
}

}  // namespace litert::arm_vulkan_ml::jit
