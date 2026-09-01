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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_TENSORRT_RTX_PLUGIN_COMPAT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_TENSORRT_RTX_PLUGIN_COMPAT_H_

#include <cstdint>

#include "NvInfer.h"

extern "C" void* getPrivateApi_INTERNAL(uint64_t api_id,
                                        int32_t version) noexcept;

namespace litert::nvidia::tensorrt_rtx_1_5_0_99 {

// TensorRT-RTX 1.5.0.99 does not expose INetworkDefinition::addPluginV3 in
// its public headers. Keep the version-specific function table isolated here.
struct NetworkDefinitionPrivate {
  static constexpr uint64_t kApiId = 0xB38F642A771BA941ULL;
  static constexpr int32_t kApiVersion = 1;

  nvinfer1::IPluginV3Layer* (*add_plugin_v3)(
      nvinfer1::INetworkDefinition& network, nvinfer1::ITensor* const* inputs,
      int32_t num_inputs, nvinfer1::ITensor* const* shape_inputs,
      int32_t num_shape_inputs, nvinfer1::IPluginV3& plugin) noexcept;
};

inline nvinfer1::IPluginV3Layer* AddPluginV3(
    nvinfer1::INetworkDefinition& network, nvinfer1::ITensor* const* inputs,
    int32_t num_inputs, nvinfer1::IPluginV3& plugin) noexcept {
  static auto* api = static_cast<NetworkDefinitionPrivate*>(
      getPrivateApi_INTERNAL(NetworkDefinitionPrivate::kApiId,
                             NetworkDefinitionPrivate::kApiVersion));
  return api == nullptr ? nullptr
                        : api->add_plugin_v3(network, inputs, num_inputs,
                                             nullptr, 0, plugin);
}

}  // namespace litert::nvidia::tensorrt_rtx_1_5_0_99

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_TENSORRT_RTX_PLUGIN_COMPAT_H_
