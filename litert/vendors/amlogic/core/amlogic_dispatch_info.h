// Copyright 2026 Amlogic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file    amlogic_dispatch_info.h
 * @module  amlogic_compiler_plugin / aml_dispatch
 * @brief   Per-call metadata shared by the AML plugin and dispatch runtime.
 * @note    Not owned by aml_compiler_core. The ADLA binary itself is the single
 *          shared bytecode module; this header only describes per-call info.
 */

#ifndef ODML_LITERT_LITERT_VENDORS_AMLOGIC_CORE_AMLOGIC_DISPATCH_INFO_H_
#define ODML_LITERT_LITERT_VENDORS_AMLOGIC_CORE_AMLOGIC_DISPATCH_INFO_H_

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief Call-info text. Every compiled call uses this name.
 *
 * Format: "aml_partition_<subgraph_idx>". The index selects a subgraph inside
 * the single shared ADLA bytecode module (byte_code_index is always 0).
 */
inline constexpr char kAmlogicPartitionCallInfoFmt[] = "aml_partition_%d";

/**
 * @brief Parse @c kAmlogicPartitionCallInfoFmt.
 * @param call_info Null-terminated call info from the compiled result.
 * @param[out] subgraph_idx Subgraph index on success.
 * @return false when @p call_info is missing or not exactly that format.
 */
bool ParseAmlogicPartitionCallInfo(const char* call_info, int* subgraph_idx);

/**
 * @brief Optional per-partition metadata. Not the compiled bytecode.
 *
 * The compiled result stores one raw ADLA binary as bytecode module 0 and
 * points every call's byte_code_index at it. Dispatch reads the subgraph
 * index from the call-info string, not from a process-global registry or a
 * file path.
 *
 * @var model_path    Unused by dispatch. Do not bake host paths into a model.
 * @var model_names   Unused by dispatch.
 * @var graph_names   Graph / entry-point name for this partition.
 * @var graph_inputs  Input tensor names.
 * @var graph_outputs Output tensor names.
 * @var subgraph_idx  Subgraph index inside the shared ADLA binary.
 * @var adla_bin      Not used as the compiled bytecode. Left empty.
 * @var adla_bin_size Byte size of @c adla_bin when populated.
 */
struct AML_Dispatch_Info {
  std::string model_path;
  std::string model_names;
  std::string graph_names;
  std::vector<std::string> graph_inputs;
  std::vector<std::string> graph_outputs;
  int subgraph_idx = 0;
  std::vector<uint8_t> adla_bin;
  size_t adla_bin_size = 0;
};

/**
 * @brief Serialize AML_Dispatch_Info to bytes for LiteRT context_bin.
 * @param dispatch_info Source payload.
 * @return Opaque byte buffer.
 */
std::string SerializeDispatchInfo(const AML_Dispatch_Info& dispatch_info);

/**
 * @brief Deserialize buffer produced by SerializeDispatchInfo().
 * @param buffer Serialized bytes.
 * @return Reconstructed AML_Dispatch_Info.
 */
AML_Dispatch_Info DeserializeDispatchInfo(const std::string& buffer);

#endif  // ODML_LITERT_LITERT_VENDORS_AMLOGIC_CORE_AMLOGIC_DISPATCH_INFO_H_
