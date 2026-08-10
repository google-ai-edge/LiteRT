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

#include <iostream>
#include <string>
#include <utility>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "litert/tools/build_custom_npu_model/build_custom_npu_model.h"

ABSL_FLAG(std::string, model, "",
          "Path to input compiled NPU model bytecode blob (e.g. Qualcomm "
          "context binary, MediaTek binary)");

ABSL_FLAG(std::string, o, "",
          "Path to save serialized output .tflite model containing custom "
          "DispatchOp");
ABSL_FLAG(std::string, output_model, "", "Alias for -o");

ABSL_FLAG(std::string, soc_manufacturer, "",
          "Target SoC manufacturer (e.g. Qualcomm, MediaTek, GoogleTensor)");
ABSL_FLAG(std::string, soc_model, "", "Target SoC model (e.g. SM8750, PTL)");

ABSL_FLAG(std::string, input_shapes, "",
          "Input tensor dimensions in AxBxCxD format (e.g. '1x224x224x3' or "
          "'1x224x224x3,1x10' for multiple inputs)");
ABSL_FLAG(std::string, output_shapes, "",
          "Output tensor dimensions in AxBxCxD format (e.g. '1x1000' or "
          "'1x1000,1x10')");

ABSL_FLAG(std::string, input_dtypes, "f32",
          "Input element types (e.g. 'f32', 'i32', 'u8', 'i8', 'i16', 'f16', "
          "'bool')");
ABSL_FLAG(std::string, output_dtypes, "f32",
          "Output element types (e.g. 'f32', 'i32', 'u8', 'i8', 'i16', 'f16', "
          "'bool')");

ABSL_FLAG(std::string, input_names, "",
          "Input tensor names (e.g. 'image,mask', default: 'input_0,input_1')");
ABSL_FLAG(
    std::string, output_names, "",
    "Output tensor names (e.g. 'logits,scores', default: 'output_0,output_1')");

ABSL_FLAG(
    std::string, entry_point, "main",
    "Entry point graph/function name within NPU bytecode (default: 'main')");

ABSL_FLAG(std::string, signature_key, "serving_default",
          "TFLite model signature key name (default: 'serving_default')");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  litert::tools::BuildCustomNpuModelOptions options;

  options.npu_bytecode_path = absl::GetFlag(FLAGS_model);

  std::string output_path = absl::GetFlag(FLAGS_o);
  if (output_path.empty()) {
    output_path = absl::GetFlag(FLAGS_output_model);
  }
  options.output_model_path = output_path;

  options.soc_manufacturer = absl::GetFlag(FLAGS_soc_manufacturer);
  options.soc_model = absl::GetFlag(FLAGS_soc_model);
  options.entry_point_name = absl::GetFlag(FLAGS_entry_point);
  options.signature_key = absl::GetFlag(FLAGS_signature_key);

  if (options.npu_bytecode_path.empty()) {
    std::cerr << "Error: Must specify --model path.\n";
    return 1;
  }
  if (options.output_model_path.empty()) {
    std::cerr << "Error: Must specify -o or --output_model path.\n";
    return 1;
  }

  std::string input_shapes_str = absl::GetFlag(FLAGS_input_shapes);
  std::string output_shapes_str = absl::GetFlag(FLAGS_output_shapes);
  std::string input_dtypes_str = absl::GetFlag(FLAGS_input_dtypes);
  std::string output_dtypes_str = absl::GetFlag(FLAGS_output_dtypes);
  std::string input_names_str = absl::GetFlag(FLAGS_input_names);
  std::string output_names_str = absl::GetFlag(FLAGS_output_names);

  if (input_shapes_str.empty() || output_shapes_str.empty()) {
    std::cerr << "Error: Must specify --input_shapes and --output_shapes.\n";
    return 1;
  }

  auto parsed_inputs = litert::tools::ParseTensorInfoList(
      input_shapes_str, input_dtypes_str, input_names_str, "input");
  if (!parsed_inputs) {
    std::cerr << "Error parsing input shapes/dtypes: "
              << parsed_inputs.Error().Message() << "\n";
    return 1;
  }
  options.input_tensors = std::move(*parsed_inputs);

  auto parsed_outputs = litert::tools::ParseTensorInfoList(
      output_shapes_str, output_dtypes_str, output_names_str, "output");
  if (!parsed_outputs) {
    std::cerr << "Error parsing output shapes/dtypes: "
              << parsed_outputs.Error().Message() << "\n";
    return 1;
  }
  options.output_tensors = std::move(*parsed_outputs);

  auto res = litert::tools::BuildCustomNpuModel(options);
  if (!res) {
    std::cerr << "Failed to build custom NPU model: " << res.Error().Message()
              << "\n";
    return 1;
  }

  std::cout << "Successfully generated custom NPU model at: "
            << options.output_model_path << "\n";
  return 0;
}
