// Copyright 2026 Google LLC.
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
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_TCFT_MODEL_CONVERTER_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_TCFT_MODEL_CONVERTER_H_

#include <cstddef>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"

namespace litert::arm_vulkan_ml {

struct TcftModelConverterCompiledModel {
  std::vector<std::vector<char>> byte_codes;
  std::vector<std::string> call_infos;
  std::vector<size_t> call_byte_code_indices;
};

LiteRtStatus PartitionForTcftModelConverter(LiteRtSubgraph subgraph,
                                            LiteRtOpList selected_ops);

// Compiles every subgraph in partitions. Each subgraph is serialized as TFLite,
// lowered by TCFT to TOSA MLIR bytecode, then converted to a VGF bytecode blob
// by Model Converter using its default options.
LiteRtStatus CompileWithTcftModelConverter(
    LiteRtModel partitions, TcftModelConverterCompiledModel* compiled_model);

}  // namespace litert::arm_vulkan_ml

#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_TCFT_MODEL_CONVERTER_H_
