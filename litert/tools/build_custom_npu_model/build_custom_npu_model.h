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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_BUILD_CUSTOM_NPU_MODEL_BUILD_CUSTOM_NPU_MODEL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_BUILD_CUSTOM_NPU_MODEL_BUILD_CUSTOM_NPU_MODEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"

namespace litert::tools {

// Description of a single input or output tensor.
struct TensorInfo {
  std::string name;
  LiteRtElementType element_type = kLiteRtElementTypeFloat32;
  std::vector<int32_t> dimensions;
};

// Options configuring custom NPU model packaging.
struct BuildCustomNpuModelOptions {
  // Path to compiled NPU bytecode binary file.
  std::string npu_bytecode_path;

  // Path where output .tflite model will be written.
  std::string output_model_path;

  // Target SoC Manufacturer (e.g. Qualcomm, MediaTek, GoogleTensor).
  std::string soc_manufacturer;

  // Target SoC Model (e.g. SM8750, PTL).
  std::string soc_model;

  // List of input tensor specifications.
  std::vector<TensorInfo> input_tensors;

  // List of output tensor specifications.
  std::vector<TensorInfo> output_tensors;

  // Name of the entry point function within the NPU bytecode (default: "main").
  std::string entry_point_name = "main";

  // TFLite signature key name for the model (default: "serving_default").
  std::string signature_key = "serving_default";
};

// Parse element type string (e.g. "f32", "i32", "u8", "i8", "i16", "f16",
// "bool").
Expected<LiteRtElementType> ParseElementType(absl::string_view dtype_str);

// Parse shape string in AxBxCxD format (e.g. "1x224x224x3" or "1x10").
Expected<std::vector<int32_t>> ParseDimensions(absl::string_view shape_str);

// Parse comma-separated list of shapes, data types, and custom names into
// TensorInfo items.
Expected<std::vector<TensorInfo>> ParseTensorInfoList(
    absl::string_view shapes_flag, absl::string_view dtypes_flag,
    absl::string_view names_flag, absl::string_view default_prefix);

// Build a custom NPU model in memory returning serialized .tflite buffer.
Expected<OwningBufferRef<uint8_t>> BuildCustomNpuModelMemory(
    const BuildCustomNpuModelOptions& options,
    BufferRef<uint8_t> bytecode_data);

// Build a custom NPU model from options and save to options.output_model_path.
Expected<void> BuildCustomNpuModel(const BuildCustomNpuModelOptions& options);

}  // namespace litert::tools

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_BUILD_CUSTOM_NPU_MODEL_BUILD_CUSTOM_NPU_MODEL_H_
