/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file    aml_dispatch_info.h
 * @module  aml_compiler_plugin / aml_dispatch
 * @brief   Packager bytecode (AML_Dispatch_Info) shared by plugin and runtime.
 * @note    Not owned by aml_compiler_core. Plugin fills after compile; dispatch
 *          deserializes from LiteRT context_bin.
 ******************************************************************************/

#ifndef ODML_LITERT_LITERT_VENDORS_AML_COMPILER_AML_DISPATCH_INFO_H_
#define ODML_LITERT_LITERT_VENDORS_AML_COMPILER_AML_DISPATCH_INFO_H_

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief Dispatch payload describing one compiled partition for runtime load.
 *
 * Offline multi-subgraph: one shared ADLA binary is written only into
 * @c dispatch_info[0].adla_bin; partitions 1..N-1 keep metadata
 * (@c subgraph_idx / IO names) with empty @c adla_bin and reuse at runtime.
 *
 * @var model_path    Directory or path hint for ADLA asset (may be empty).
 * @var model_names   Model / artifact name (shared-load key with model_path).
 * @var graph_names   Graph / entry-point name for this partition.
 * @var graph_inputs  Input tensor names.
 * @var graph_outputs Output tensor names.
 * @var subgraph_idx  Subgraph index inside the multi-partition ADLA.
 * @var adla_bin      Embedded ADLA bytes (typically only partition 0 offline).
 * @var adla_bin_size Byte size of @c adla_bin.
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

#endif  // ODML_LITERT_LITERT_VENDORS_AML_COMPILER_AML_DISPATCH_INFO_H_
