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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SUBBYTE_GEMV_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SUBBYTE_GEMV_PLUGIN_H_

#include <cstdint>

#include "litert/vendors/nvidia/tensorrt_rtx/include/NvInferRuntime.h"

namespace litert::nvidia {

nvinfer1::IPluginV3* CreateSubbyteGemvPlugin(int32_t bit_width, int32_t rows,
                                             int32_t columns) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureSubbyteGemvPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SUBBYTE_GEMV_PLUGIN_H_
