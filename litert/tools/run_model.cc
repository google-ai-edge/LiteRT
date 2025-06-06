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

#define INCLUDE_QUALCOMM_RUNTIME_FLAGS
#define INCLUDE_MEDIATEK_RUNTIME_FLAGS
#define INCLUDE_GOOGLE_TENSOR_RUNTIME_FLAGS

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/tools/flags/vendors/google_tensor_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/mediatek_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/qualcomm_flags.h"  // IWYU pragma: keep
#include "litert/tools/tensor_utils.h"
#include "tflite/profiling/time.h"

ABSL_FLAG(std::string, graph, "", "Model filename to use for testing.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Path to the dispatch library.");
ABSL_FLAG(std::string, compiler_plugin_library_dir, "",
          "Path to the compiler plugin library. Only for JIT compilation.");
ABSL_FLAG(std::string, accelerator, "cpu", "Which backend to use.");
ABSL_FLAG(size_t, signature_index, 0, "Index of the signature to run.");
ABSL_FLAG(bool, print_tensors, false, "Print tensor values after execution.");
ABSL_FLAG(bool, compare_numerical, false,
          "Decide if random value should be filled into tensor buffer to "
          "perform numerical check.");
ABSL_FLAG(size_t, sample_size, 5,
          "Number of sample elements to print from beginning, middle, and end "
          "of tensor.");
ABSL_FLAG(size_t, iterations, 1,
          "The number of iterations the graph will execute.");

namespace litert {
namespace {

using ::litert::google_tensor::GoogleTensorOptionsFromFlags;
using ::litert::mediatek::MediatekOptionsFromFlags;
using ::litert::qualcomm::QualcommOptionsFromFlags;

LiteRtHwAccelerators GetAccelerator() {
  const auto accelerator = absl::GetFlag(FLAGS_accelerator);
  if (accelerator == "gpu") {
    return kLiteRtHwAcceleratorGpu;
  } else if (accelerator == "npu") {
    return kLiteRtHwAcceleratorNpu;
  } else {
    return kLiteRtHwAcceleratorCpu;
  }
}

Expected<Environment> GetEnvironment() {
  std::vector<litert::Environment::Option> environment_options = {};

  const auto dispatch_library_dir = absl::GetFlag(FLAGS_dispatch_library_dir);
  if (!dispatch_library_dir.empty()) {
    environment_options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::DispatchLibraryDir,
        absl::string_view(dispatch_library_dir)});
  }

  const auto compiler_plugin_library_dir =
      absl::GetFlag(FLAGS_compiler_plugin_library_dir);
  if (!compiler_plugin_library_dir.empty()) {
    environment_options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::CompilerPluginLibraryDir,
        absl::string_view(compiler_plugin_library_dir)});
  }

  return Environment::Create(absl::MakeConstSpan(environment_options));
}

Expected<Options> GetOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(GetAccelerator());
  if (auto qnn_opts = QualcommOptionsFromFlags()) {
    options.AddOpaqueOptions(std::move(*qnn_opts));
  }
  if (auto google_tensor_opts = GoogleTensorOptionsFromFlags()) {
    options.AddOpaqueOptions(std::move(*google_tensor_opts));
  }
  if (auto mediatek_opts = MediatekOptionsFromFlags()) {
    options.AddOpaqueOptions(std::move(*mediatek_opts));
  }
  return options;
}

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

// Using tensor utility functions from the tensor_utils.h header

// Function to print tensor buffer information and data
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

  if (!absl::GetFlag(FLAGS_compare_numerical)) {
    return {};
  }

  size_t sample_size = absl::GetFlag(FLAGS_sample_size);

  // Handle different data types using the tensor_utils library
  if (type.ElementType() == ElementType::Float32) {
    std::vector<float> data(total_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<float>(absl::MakeSpan(data)));

    auto stats = tensor_utils::CalculateTensorStats(data, total_elements);
    tensor_utils::PrintTensorStats(stats);

    tensor_utils::PrintTensorSamples(data, total_elements, sample_size);
  } else if (type.ElementType() == ElementType::Int32) {
    std::vector<int32_t> data(total_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<int32_t>(absl::MakeSpan(data)));

    auto stats = tensor_utils::CalculateTensorStats(data, total_elements);
    tensor_utils::PrintTensorStats(stats);

    tensor_utils::PrintTensorSamples(data, total_elements, sample_size);
  } else {
    // Default to uint8_t for other types
    std::vector<uint8_t> data(total_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<uint8_t>(absl::MakeSpan(data)));

    auto stats = tensor_utils::CalculateTensorStats(data, total_elements);
    tensor_utils::PrintTensorStats(stats);

    tensor_utils::PrintTensorSamples(data, total_elements, sample_size);
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

  LITERT_ASSIGN_OR_RETURN(auto env, GetEnvironment());
  LITERT_ASSIGN_OR_RETURN(auto options, GetOptions());

  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, model, options));

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

  const size_t iterations = absl::GetFlag(FLAGS_iterations);
  if (iterations <= 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "iterations should be large than zero.");
  }

  ABSL_LOG(INFO) << "Run model for " << iterations << " times";
  litert::Expected<void> status;
  std::vector<uint64_t> timers(iterations, 0);
  for (auto& timer : timers) {
    uint64_t start = tflite::profiling::time::NowMicros();
    status = compiled_model.Run(signature_index, input_buffers, output_buffers);
    uint64_t end = tflite::profiling::time::NowMicros();
    timer = end - start;
  }
  ABSL_LOG(INFO) << "First run took " << timers[0] << " microseconds";
  ABSL_LOG(INFO) << "Slowest run took "
                 << *std::max_element(timers.begin(), timers.end())
                 << " microseconds";
  ABSL_LOG(INFO) << "Fastest run took "
                 << *std::min_element(timers.begin(), timers.end())
                 << " microseconds";
  ABSL_LOG(INFO) << "All runs took average "
                 << std::accumulate(timers.begin(), timers.end(), uint64_t{0}) /
                        timers.size()
                 << " microseconds";

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
    ABSL_LOG(ERROR) << res.Error().Message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
