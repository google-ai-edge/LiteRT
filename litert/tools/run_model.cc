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
#define INCLUDE_INTEL_OPENVINO_RUNTIME_FLAGS
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
#include "absl/random/random.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/tools/flags/vendors/google_tensor_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/intel_openvino_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/mediatek_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/qualcomm_flags.h"  // IWYU pragma: keep
#include "litert/tools/tensor_utils.h"
#include "tflite/profiling/time.h"

ABSL_FLAG(std::string, graph, "", "Model filename to use for testing.");
ABSL_FLAG(std::string, dispatch_library_dir, "/data/local/tmp/run_model_test",
          "Path to the dispatch library.");
ABSL_FLAG(std::string, compiler_plugin_library_dir, "",
          "Path to the compiler plugin library. Only for JIT compilation.");
ABSL_FLAG(std::string, compiler_cache_dir, "",
          "Path to the compiler cache directory. Only for JIT compilation.");
ABSL_FLAG(std::string, accelerator, "cpu",
          "Which backend to use. Comma delimited string of accelerators (e.g. "
          "cpu,gpu,npu). Will delegate to NPU, GPU, then CPU if they are "
          "specified in this flag.");
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
ABSL_FLAG(bool, language_model, false,
          "Whether the model is a language model,"
          " so that the input tensors will be reasonable.");

ABSL_FLAG(std::string, input_dir, "",
          "An input folder containing .raw files with model input signatures "
          "as their file names.");

namespace litert {
namespace {

using ::litert::google_tensor::UpdateGoogleTensorOptionsFromFlags;
using ::litert::intel_openvino::UpdateIntelOpenVinoOptionsFromFlags;
using ::litert::mediatek::UpdateMediatekOptionsFromFlags;
using ::litert::qualcomm::UpdateQualcommOptionsFromFlags;

litert::HwAcceleratorSet GetAccelerator() {
  const std::string accelerator_str = absl::GetFlag(FLAGS_accelerator);
  litert::HwAcceleratorSet accelerators(
      static_cast<int>(litert::HwAccelerators::kNone));
  for (absl::string_view accelerator : absl::StrSplit(accelerator_str, ',')) {
    if (accelerator == "gpu") {
      accelerators |= litert::HwAccelerators::kGpu;
    } else if (accelerator == "npu") {
      accelerators |= litert::HwAccelerators::kNpu;
    } else if (accelerator == "cpu") {
      accelerators |= litert::HwAccelerators::kCpu;
    }
  }
  return accelerators;
}

Expected<Environment> GetEnvironment() {
  std::vector<litert::EnvironmentOptions::Option> environment_options = {};

  const auto dispatch_library_dir = absl::GetFlag(FLAGS_dispatch_library_dir);
  if (!dispatch_library_dir.empty()) {
    environment_options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
        absl::string_view(dispatch_library_dir)});
  }

  const auto compiler_plugin_library_dir =
      absl::GetFlag(FLAGS_compiler_plugin_library_dir);
  if (!compiler_plugin_library_dir.empty()) {
    environment_options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
        absl::string_view(compiler_plugin_library_dir)});
    const auto compiler_cache_dir = absl::GetFlag(FLAGS_compiler_cache_dir);
    if (!compiler_cache_dir.empty()) {
      environment_options.push_back(litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerCacheDir,
          absl::string_view(compiler_cache_dir)});
    }
  }

  return Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
}

Expected<Options> GetOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(GetAccelerator());
  LITERT_ASSIGN_OR_RETURN(auto& qnn_opts, options.GetQualcommOptions());
  LITERT_RETURN_IF_ERROR(UpdateQualcommOptionsFromFlags(qnn_opts));
  LITERT_ASSIGN_OR_RETURN(auto& google_tensor_opts,
                          options.GetGoogleTensorOptions());
  LITERT_RETURN_IF_ERROR(
      UpdateGoogleTensorOptionsFromFlags(google_tensor_opts));
  LITERT_ASSIGN_OR_RETURN(auto& intel_openvino_opts,
                          options.GetIntelOpenVinoOptions());
  LITERT_RETURN_IF_ERROR(
      UpdateIntelOpenVinoOptionsFromFlags(intel_openvino_opts));
  LITERT_ASSIGN_OR_RETURN(auto& mediatek_opts, options.GetMediatekOptions());
  LITERT_RETURN_IF_ERROR(UpdateMediatekOptionsFromFlags(mediatek_opts));
  return options;
}

// Helper function to get the total number of elements in a tensor.
Expected<size_t> GetTotalElements(const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  const auto& layout = type.Layout();
  size_t total_elements = 1;
  for (size_t d = 0; d < layout.Rank(); ++d) {
    total_elements *= layout.Dimensions()[d];
  }
  return total_elements;
}

// Fills a tensor buffer with sample data based on element type
Expected<void> FillInputBuffer(TensorBuffer& buffer) {
  if (!absl::GetFlag(FLAGS_compare_numerical)) {
    return {};
  }

  LITERT_ASSIGN_OR_RETURN(const size_t total_elements,
                          GetTotalElements(buffer));

  // Always treat input as float and fill with rotating values from 0.0 to 0.9
  std::vector<float> data(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    // Rotate through 0.0, 0.1, 0.2, ..., 0.9, 0.0, 0.1, ...
    data[i] = static_cast<float>(i % 10) * 0.1f;
  }

  // Write the data to the tensor buffer
  return buffer.Write<float>(absl::MakeConstSpan(data));
}

// Fills input buffers for a language model with sample data.
Expected<void> FillLanguageModelInputBuffers(
    absl::Span<TensorBuffer> input_buffers) {
  if (input_buffers.size() < 3) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Language model requires at least 3 input tensors.");
  }

  // 1. First tensor: Randomized token IDs with padding.
  auto& token_buffer = input_buffers[0];
  LITERT_ASSIGN_OR_RETURN(const size_t max_length,
                          GetTotalElements(token_buffer));

  // Generate a random sequence length between 1 and max_length.
  absl::BitGen gen;
  // Ensure sequence_length is at least 1.
  size_t sequence_length =
      absl::uniform_int_distribution<size_t>(1, max_length)(gen);

  // Distribution for token IDs in the range [0, 30521].
  absl::uniform_int_distribution<int32_t> token_dist(0, 30521);

  // Initialize with 0s for padding.
  std::vector<int32_t> token_data(max_length, 0);
  for (size_t i = 0; i < sequence_length; ++i) {
    token_data[i] = token_dist(gen);
  }
  LITERT_RETURN_IF_ERROR(
      token_buffer.Write<int32_t>(absl::MakeConstSpan(token_data)));

  // 2. Second tensor: Attention mask (1s for tokens, 0s for padding).
  auto& mask_buffer = input_buffers[1];
  LITERT_ASSIGN_OR_RETURN(const size_t mask_length,
                          GetTotalElements(mask_buffer));
  if (mask_length != max_length) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Token ID and mask tensors must have the same length.");
  }
  std::vector<int32_t> mask_data(max_length, 0);
  // Set 1s for the valid tokens.
  std::fill(mask_data.begin(), mask_data.begin() + sequence_length, 1);
  LITERT_RETURN_IF_ERROR(
      mask_buffer.Write<int32_t>(absl::MakeConstSpan(mask_data)));

  // 3. Third tensor: All zeros.
  auto& third_buffer = input_buffers[2];
  LITERT_ASSIGN_OR_RETURN(const size_t third_length,
                          GetTotalElements(third_buffer));
  std::vector<int32_t> third_data(third_length, 0);
  LITERT_RETURN_IF_ERROR(
      third_buffer.Write<int32_t>(absl::MakeConstSpan(third_data)));

  return {};
}

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

  LITERT_ASSIGN_OR_RETURN(auto env, GetEnvironment());
  LITERT_ASSIGN_OR_RETURN(auto options, GetOptions());

  ABSL_LOG(INFO) << "Model: " << absl::GetFlag(FLAGS_graph);
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(env, absl::GetFlag(FLAGS_graph), options));

  size_t signature_index = absl::GetFlag(FLAGS_signature_index);
  ABSL_LOG(INFO) << "Signature index: " << signature_index;

  ABSL_LOG(INFO) << "Prepare input buffers";

  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          compiled_model.CreateInputBuffers(signature_index));

  std::string input_dir = absl::GetFlag(FLAGS_input_dir);
  if (!input_dir.empty()) {
    // Use the inputs given by the user.
    LITERT_RETURN_IF_ERROR(tensor_utils::FillInputBuffersWithCustomData(
        compiled_model, signature_index, input_buffers, input_dir));
  } else if (absl::GetFlag(FLAGS_language_model)) {
    // Language model, hard assumption on tensor order here TODO: generalize
    // when we find other examples.
    LITERT_RETURN_IF_ERROR(
        FillLanguageModelInputBuffers(absl::MakeSpan(input_buffers)));
  } else {
    // Non-language model, Fill input buffers with sample data.
    for (size_t i = 0; i < input_buffers.size(); ++i) {
      auto& buffer = input_buffers[i];
      LITERT_RETURN_IF_ERROR(FillInputBuffer(buffer));
    }
  }

  if (absl::GetFlag(FLAGS_print_tensors)) {
    for (size_t i = 0; i < input_buffers.size(); ++i) {
      LITERT_RETURN_IF_ERROR(PrintTensorBuffer(input_buffers[i], "Input", i));
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
