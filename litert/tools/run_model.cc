// Copyright 2025 Google LLC.
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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/profiling/time.h"

ABSL_FLAG(std::string, graph, "", "Model filename to use for testing.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Path to the dispatch library.");
ABSL_FLAG(bool, use_gpu, false, "Use GPU Accelerator.");
ABSL_FLAG(size_t, signature_index, 0, "Index of the signature to run.");
ABSL_FLAG(bool, print_tensors, false, "Print tensor values after execution.");
ABSL_FLAG(bool, compare_numerical, false,
          "Decide if random value should be filled into tensor buffer to "
          "perform numerical check.");
ABSL_FLAG(size_t, max_elements_to_print, 20,
          "Maximum number of elements to print per tensor.");

namespace litert {
namespace {

// Fills a tensor buffer with sample data based on element type
Expected<void> FillInputBuffer(TensorBuffer& buffer) {
  if (!absl::GetFlag(FLAGS_compare_numerical)) {
    return {};
  }
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());

  // Calculate the total number of elements from dimensions
  const auto& layout = type.Layout();
  size_t total_elements = 1;
  for (size_t d = 0; d < layout.Rank(); ++d) {
    total_elements *= layout.Dimensions()[d];
  }

  // Always treat input as float and fill with rotating values from 0.0 to 0.9
  std::vector<float> data(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    // Rotate through 0.0, 0.1, 0.2, ..., 0.9, 0.0, 0.1, ...
    data[i] = static_cast<float>(i % 10) * 0.1f;
  }

  // Write the data to the tensor buffer
  return buffer.Write<float>(absl::MakeConstSpan(data));
}

// Prints tensor buffer information and data
Expected<void> PrintTensorBuffer(TensorBuffer& buffer,
                                 const std::string& buffer_type, size_t index) {
  LITERT_ASSIGN_OR_RETURN(size_t size, buffer.Size());
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto buffer_type_enum, buffer.BufferType());

  ABSL_LOG(INFO) << buffer_type << " buffer " << index << ":";
  ABSL_LOG(INFO) << "  Size: " << size << " bytes";
  ABSL_LOG(INFO) << "  ElementType: " << static_cast<int>(type.ElementType());
  ABSL_LOG(INFO) << "  BufferType: " << static_cast<int>(buffer_type_enum);

  // Calculate the total number of elements from dimensions
  const auto& layout = type.Layout();
  size_t total_elements = 1;

  ABSL_LOG(INFO) << "  Dimensions: [";
  for (size_t d = 0; d < layout.Rank(); ++d) {
    ABSL_LOG(INFO) << "    " << layout.Dimensions()[d];
    total_elements *= layout.Dimensions()[d];
  }
  ABSL_LOG(INFO) << "  ]";
  ABSL_LOG(INFO) << "  Total logical elements: " << total_elements;

  // Print elements up to max_elements_to_print

  if (!absl::GetFlag(FLAGS_compare_numerical)) {
    return {};
  }
  size_t max_elements = absl::GetFlag(FLAGS_max_elements_to_print);
  size_t elements_to_print = std::min(total_elements, max_elements);
  ABSL_LOG(INFO) << "  Data (first " << elements_to_print << " elements):";

  if (type.ElementType() == ElementType::Float32) {
    std::vector<float> data(elements_to_print);

    // Only read the elements we need to print
    LITERT_RETURN_IF_ERROR(buffer.Read<float>(absl::MakeSpan(data)));

    for (size_t j = 0; j < elements_to_print; ++j) {
      ABSL_LOG(INFO) << "    " << j << ": " << data[j];
    }
  } else if (type.ElementType() == ElementType::Int32) {
    std::vector<int32_t> data(elements_to_print);

    LITERT_RETURN_IF_ERROR(buffer.Read<int32_t>(absl::MakeSpan(data)));

    for (size_t j = 0; j < elements_to_print; ++j) {
      ABSL_LOG(INFO) << "    " << j << ": " << data[j];
    }
  } else {
    // Just print bytes for other types
    std::vector<uint8_t> data(elements_to_print);

    LITERT_RETURN_IF_ERROR(buffer.Read<uint8_t>(absl::MakeSpan(data)));

    for (size_t j = 0; j < elements_to_print; ++j) {
      ABSL_LOG(INFO) << "    " << j << ": " << static_cast<int>(data[j]);
    }
  }

  return {};
}

Expected<void> RunModel() {
  if (absl::GetFlag(FLAGS_graph).empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Model filename is empty. Use --graph to provide it.");
  }

  ABSL_LOG(INFO) << "Model: " << absl::GetFlag(FLAGS_graph);
  LITERT_ASSIGN_OR_RETURN(auto model,
                          Model::CreateFromFile(absl::GetFlag(FLAGS_graph)));

  const std::string dispatch_library_dir =
      absl::GetFlag(FLAGS_dispatch_library_dir);

  std::vector<litert::Environment::Option> environment_options = {};
  if (!dispatch_library_dir.empty()) {
    environment_options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::DispatchLibraryDir,
        absl::string_view(dispatch_library_dir)});
  };

  LITERT_ASSIGN_OR_RETURN(
      auto env,
      litert::Environment::Create(absl::MakeConstSpan(environment_options)));

  ABSL_LOG(INFO) << "Create CompiledModel";
  auto accelerator = absl::GetFlag(FLAGS_use_gpu) ? kLiteRtHwAcceleratorGpu
                                                  : kLiteRtHwAcceleratorNone;
  if (accelerator == kLiteRtHwAcceleratorGpu) {
    ABSL_LOG(INFO) << "Using GPU Accelerator";
  }
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, model, accelerator));

  LITERT_ASSIGN_OR_RETURN(auto signatures, model.GetSignatures());
  size_t signature_index = absl::GetFlag(FLAGS_signature_index);
  ABSL_LOG(INFO) << "Signature index: " << signature_index;

  ABSL_LOG(INFO) << "Prepare input buffers";

  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          compiled_model.CreateInputBuffers(signature_index));

  // Fill input buffers with sample data
  for (size_t i = 0; i < input_buffers.size(); ++i) {
    auto& buffer = input_buffers[i];
    LITERT_RETURN_IF_ERROR(FillInputBuffer(buffer));

    // Print tensor info and data if requested
    if (absl::GetFlag(FLAGS_print_tensors)) {
      LITERT_RETURN_IF_ERROR(PrintTensorBuffer(buffer, "Input", i));
    }
  }

  ABSL_LOG(INFO) << "Prepare output buffers";

  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          compiled_model.CreateOutputBuffers(signature_index));

  ABSL_LOG(INFO) << "Run model";
  uint64_t start = tflite::profiling::time::NowMicros();
  auto status =
      compiled_model.Run(signature_index, input_buffers, output_buffers);
  uint64_t end = tflite::profiling::time::NowMicros();
  LITERT_LOG(LITERT_INFO, "Run took %lu microseconds", end - start);

  // Print output tensor information and values if requested
  if (absl::GetFlag(FLAGS_print_tensors)) {
    for (size_t i = 0; i < output_buffers.size(); ++i) {
      auto& buffer = output_buffers[i];
      LITERT_RETURN_IF_ERROR(PrintTensorBuffer(buffer, "Output", i));
    }
  }

  ABSL_LOG(INFO) << "Model run completed";

  return status;
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  auto res = litert::RunModel();
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().c_str());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
