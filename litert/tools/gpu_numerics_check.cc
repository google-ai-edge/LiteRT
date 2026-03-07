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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
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
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/core/filesystem.h"
#include "tflite/c/c_api_types.h"
#include "tflite/tools/utils.h"

ABSL_FLAG(bool, print_diff_stats, false,
          "Whether to print the diff stats CSV.");
ABSL_FLAG(std::string, model_dir, "",
          "Optional base directory to prepend to models provide in --graph.");
ABSL_FLAG(std::vector<std::string>, graph, {},
          "Model file(s) to use for testing.");
ABSL_FLAG(size_t, signature_index, 0, "Index of the signature to run.");
ABSL_FLAG(float, epsilon, 1e-4f,
          "Threshold value for gpu / cpu inference comparison");
ABSL_FLAG(bool, check_element_type, false,
          "Whether to check the element type of the output buffers.");
ABSL_FLAG(std::string, gpu_backend, "opencl",
          "GPU backend to use for testing. Can be opencl, webgpu.");
ABSL_FLAG(bool, external_tensor_mode, false,
          "Whether to enable external tensor mode.");
ABSL_FLAG(bool, print_outputs, false, "Whether to print the output tensors.");
ABSL_FLAG(bool, enable_constant_tensors_sharing, false,
          "Whether to enable constant tensors sharing.");
ABSL_FLAG(bool, use_fp16, false, "Whether to use FP32 precision.");

namespace litert {

namespace {

struct BufferDiffStats {
  // Index of output buffer.
  size_t buffer_idx;
  // Total number of elements in the buffer.
  size_t total_elements;
  // Number of elements with absolute difference greater than epsilon.
  size_t diff_elements;
  // Epsilon value used for comparison.
  double epsilon;
  // Maximum absolute difference between CPU and GPU values.
  double max_diff;
  // Minimum absolute difference between CPU and GPU values.
  double min_diff;
  // Mean absolute difference between CPU and GPU values.
  double mean_diff;
  // Mean squared error between CPU and GPU values.
  double mse;
};

struct ModelRunResult {
  std::string model_name;
  std::vector<BufferDiffStats> diff_stats;
};

Expected<Environment> GetEnvironment() {
  std::vector<litert::EnvironmentOptions::Option> environment_options = {};

  return Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
}

Expected<Options> GetGpuOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kGpu);
  LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
  gpu_options.EnableExternalTensorsMode(
      absl::GetFlag(FLAGS_external_tensor_mode));
  if (absl::GetFlag(FLAGS_use_fp16)) {
    gpu_options.SetPrecision(GpuOptions::Precision::kFp16);
  } else {
    gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
  }
  if (absl::GetFlag(FLAGS_gpu_backend) == "webgpu") {
    gpu_options.SetBackend(GpuOptions::Backend::kWebGpu);
  } else if (absl::GetFlag(FLAGS_gpu_backend) == "opengl") {
    gpu_options.SetBackend(GpuOptions::Backend::kOpenGl);
  }

  if (absl::GetFlag(FLAGS_enable_constant_tensors_sharing)) {
    gpu_options.EnableConstantTensorSharing(true);
  }
  return options;
}

Expected<Options> GetCpuOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);
  return options;
}

Expected<void> FillInputTensor(TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  const auto& layout = type.Layout();
  size_t total_elements =
      std::accumulate(layout.Dimensions().begin(), layout.Dimensions().end(), 1,
                      std::multiplies<size_t>());
  float low_range = 0;
  float high_range = 0;
  tflite::utils::GetDataRangesForType(
      static_cast<TfLiteType>(type.ElementType()), &low_range, &high_range);
  auto tensor_data = tflite::utils::CreateRandomTensorData(
      /*name=*/"", static_cast<TfLiteType>(type.ElementType()), total_elements,
      low_range, high_range);

  return buffer.Write<char>(absl::MakeSpan(
      reinterpret_cast<char*>(tensor_data.data.get()), tensor_data.bytes));
}

// Creates and fills input buffers for a given compiled model.
Expected<std::vector<TensorBuffer>> CreateAndFillInputBuffers(
    const CompiledModel& compiled_model, size_t signature_index) {
  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          compiled_model.CreateInputBuffers(signature_index));

  for (auto& buffer : input_buffers) {
    LITERT_RETURN_IF_ERROR(FillInputTensor(buffer));
  }
  return input_buffers;
}

// Creates output buffers for a given compiled model.
Expected<std::vector<TensorBuffer>> CreateOutputBuffers(
    const CompiledModel& compiled_model, size_t signature_index) {
  return compiled_model.CreateOutputBuffers(signature_index);
}

// Compares a single pair of output buffers and prints the results.
Expected<BufferDiffStats> CompareSingleOutputBuffer(TensorBuffer& cpu_buffer,
                                                    TensorBuffer& gpu_buffer,
                                                    size_t buffer_index,
                                                    float epsilon) {
  std::vector<std::pair<float, int>> all_diffs;
  const int kMaxPrint = 20;
  int printed = 0;
  size_t total_different = 0;
  double mean_squared_error = 0;
  float mean_diff = 0;

  LITERT_ASSIGN_OR_RETURN(auto cpu_type, cpu_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto gpu_type, gpu_buffer.TensorType());
  if (absl::GetFlag(FLAGS_check_element_type)) {
    if (cpu_type.ElementType() != gpu_type.ElementType()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Element type mismatch between CPU and GPU.");
    }
    LITERT_ASSIGN_OR_RETURN(size_t cpu_size, cpu_buffer.Size());
    LITERT_ASSIGN_OR_RETURN(size_t gpu_size, gpu_buffer.Size());
    if (cpu_size != gpu_size) {
      return Error(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrFormat("Size mismatch for output buffer %d", buffer_index));
    }
  }

  // Calculate the total number of elements from dimensions and check that the
  // dimensions are the same.
  size_t total_elements = 1;
  const auto& cpu_layout = cpu_type.Layout();
  const auto& gpu_layout = gpu_type.Layout();
  for (size_t d = 0; d < cpu_layout.Rank(); ++d) {
    total_elements *= cpu_layout.Dimensions()[d];
    if (cpu_layout.Dimensions()[d] != gpu_layout.Dimensions()[d]) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   absl::StrFormat("Dimension mismatch for output buffer %d",
                                   buffer_index));
    }
  }

  ABSL_LOG(INFO) << "Comparing output buffer " << buffer_index << ":";

  auto get_val = [&](TensorBuffer& buffer,
                     std::vector<float>& buffer_data) -> Expected<void> {
    auto tensor_type = buffer.TensorType();
    if (!tensor_type.HasValue()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Tensor type is not available.");
    }
    auto element_type = tensor_type->ElementType();
    auto copy_data_and_return = [&](auto& dst, auto& src,
                                    size_t size) -> Expected<void> {
      for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i];
      }
      return {};
    };
    if (element_type == ElementType::Float32) {
      std::vector<float> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<float>(absl::MakeSpan(data)));
      copy_data_and_return(buffer_data, data, total_elements);
    }
    if (element_type == ElementType::Int32) {
      std::vector<int32_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<int32_t>(absl::MakeSpan(data)));
      copy_data_and_return(buffer_data, data, total_elements);
    }
    if (element_type == ElementType::Int16) {
      std::vector<int16_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<int16_t>(absl::MakeSpan(data)));
      copy_data_and_return(buffer_data, data, total_elements);
    }
    if (element_type == ElementType::Int8) {
      std::vector<int8_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<int8_t>(absl::MakeSpan(data)));
      copy_data_and_return(buffer_data, data, total_elements);
    }
    if (element_type == ElementType::UInt8) {
      std::vector<uint8_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<uint8_t>(absl::MakeSpan(data)));
      copy_data_and_return(buffer_data, data, total_elements);
    }
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Unsupported element type for reading tensor.");
  };

  std::vector<float> cpu_data(total_elements);
  std::vector<float> gpu_data(total_elements);
  get_val(cpu_buffer, cpu_data);
  get_val(gpu_buffer, gpu_data);
  bool print_outputs = absl::GetFlag(FLAGS_print_outputs);
  for (int element_index = 0; element_index < total_elements; ++element_index) {
    if (print_outputs) {
      std::cout << "Element #" << element_index
                << ": CPU value = " << cpu_data[element_index]
                << ", GPU value = " << gpu_data[element_index] << std::endl;
    }

    const float abs_diff =
        fabs(cpu_data[element_index] - gpu_data[element_index]);
    const double diff_square =
        (cpu_data[element_index] - gpu_data[element_index]) *
        (cpu_data[element_index] - gpu_data[element_index]);
    mean_squared_error += diff_square;
    mean_diff += abs_diff;

    all_diffs.push_back(std::make_pair(abs_diff, element_index));
    if (abs_diff > epsilon) {
      total_different++;
      if (printed < kMaxPrint) {
        std::cout << "Element #" << element_index << ": CPU value - "
                  << cpu_data[element_index] << ", GPU value - "
                  << gpu_data[element_index] << ", abs diff - " << abs_diff
                  << std::endl;
        printed++;
      }
      if (printed == kMaxPrint) {
        std::cout << "Printed " << kMaxPrint
                  << " different elements, threshold - " << epsilon
                  << ", next different elements skipped" << std::endl;
        printed++;
      }
    }
  }

  std::sort(all_diffs.begin(), all_diffs.end());
  std::sort(all_diffs.begin(), all_diffs.end(),
            [](auto& left, auto& right) { return left.first < right.first; });
  std::cout << "Max diff: " << all_diffs.back().first << std::endl;
  std::cout << "Min diff: " << all_diffs.front().first << std::endl;

  for (int ii = 0; ii < kMaxPrint && ii < all_diffs.size(); ++ii) {
    const int reversed_index = all_diffs.size() - ii - 1;
    std::cout << "Top " << ii << " diff: " << all_diffs[reversed_index].first
              << " @ element #: " << all_diffs[reversed_index].second
              << ", CPU val: " << cpu_data[all_diffs[reversed_index].second]
              << " , GPU val: " << gpu_data[all_diffs[reversed_index].second]
              << std::endl;
  }

  std::cout << "Mean diff: " << mean_diff / all_diffs.size() << std::endl;
  std::cout << "MSE: " << mean_squared_error / total_elements << std::endl;
  std::cout << "Total " << total_different << " out of " << total_elements
            << " are different elements, for output #" << buffer_index
            << ", threshold - " << epsilon << std::endl;
  return BufferDiffStats{
      .buffer_idx = buffer_index,
      .total_elements = total_elements,
      .diff_elements = total_different,
      .epsilon = epsilon,
      .max_diff = all_diffs.back().first,
      .min_diff = all_diffs.front().first,
      .mean_diff = mean_diff / all_diffs.size(),
      .mse = mean_squared_error / total_elements,
  };
}

Expected<std::vector<BufferDiffStats>> CompareOutputBuffers(
    std::vector<TensorBuffer>& cpu_output_buffers,
    std::vector<TensorBuffer>& gpu_output_buffers) {
  if (cpu_output_buffers.size() != gpu_output_buffers.size()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Number of output buffers mismatch between CPU and GPU.");
  }

  float epsilon = absl::GetFlag(FLAGS_epsilon);
  size_t num_output_buffers = cpu_output_buffers.size();
  std::vector<BufferDiffStats> diff_stats;
  for (size_t i = 0; i < num_output_buffers; ++i) {
    auto& cpu_buffer = cpu_output_buffers[i];
    auto& gpu_buffer = gpu_output_buffers[i];
    LITERT_ASSIGN_OR_RETURN(
        auto diff_stat,
        CompareSingleOutputBuffer(cpu_buffer, gpu_buffer, i, epsilon));
    diff_stats.push_back(std::move(diff_stat));
  }
  return diff_stats;
}

Expected<std::vector<BufferDiffStats>> RunModel(absl::string_view model_path) {
  ABSL_LOG(INFO) << "Model: " << model_path;

  LITERT_ASSIGN_OR_RETURN(auto env, GetEnvironment());

  LITERT_ASSIGN_OR_RETURN(auto cpu_options, GetCpuOptions());
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model_cpu,
      CompiledModel::Create(env, std::string(model_path), cpu_options));

  LITERT_ASSIGN_OR_RETURN(auto gpu_options, GetGpuOptions());
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model_gpu,
      CompiledModel::Create(env, std::string(model_path), gpu_options));

  size_t signature_index = absl::GetFlag(FLAGS_signature_index);
  ABSL_LOG(INFO) << "Signature index: " << signature_index;

  // Create and fill input buffers
  LITERT_ASSIGN_OR_RETURN(
      auto cpu_input_buffers,
      CreateAndFillInputBuffers(compiled_model_cpu, signature_index));
  LITERT_ASSIGN_OR_RETURN(
      auto gpu_input_buffers,
      compiled_model_gpu.CreateInputBuffers(signature_index));
  // Copy input buffers from CPU to GPU.
  for (size_t i = 0; i < cpu_input_buffers.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size, cpu_input_buffers[i].Size());
    std::vector<char> data(buffer_size);
    LITERT_RETURN_IF_ERROR(
        cpu_input_buffers[i].Read<char>(absl::MakeSpan(data)));
    LITERT_RETURN_IF_ERROR(
        gpu_input_buffers[i].Write<char>(absl::MakeSpan(data)));
  }

  // Create output buffers
  LITERT_ASSIGN_OR_RETURN(
      auto cpu_output_buffers,
      CreateOutputBuffers(compiled_model_cpu, signature_index));
  LITERT_ASSIGN_OR_RETURN(
      auto gpu_output_buffers,
      CreateOutputBuffers(compiled_model_gpu, signature_index));

  // Run models
  LITERT_RETURN_IF_ERROR(compiled_model_cpu.Run(
      signature_index, cpu_input_buffers, cpu_output_buffers));
  LITERT_RETURN_IF_ERROR(compiled_model_gpu.Run(
      signature_index, gpu_input_buffers, gpu_output_buffers));

  // Compare output buffers
  LITERT_ASSIGN_OR_RETURN(
      auto diff_stats,
      CompareOutputBuffers(cpu_output_buffers, gpu_output_buffers));
  return diff_stats;
}

Expected<std::vector<ModelRunResult>> RunModels() {
  std::vector<std::string> relative_model_paths = absl::GetFlag(FLAGS_graph);
  if (relative_model_paths.empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "No model provided. Use --graph to provide it.");
  }

  std::string model_dir = absl::GetFlag(FLAGS_model_dir);
  std::vector<std::string> full_model_paths;
  full_model_paths.reserve(relative_model_paths.size());
  for (auto& model_path : relative_model_paths) {
    full_model_paths.push_back(internal::Join({model_dir, model_path}));
  }

  std::vector<ModelRunResult> results;
  for (const auto& model_path : full_model_paths) {
    LITERT_ASSIGN_OR_RETURN(std::vector<BufferDiffStats> diff_stats,
                            RunModel(model_path));
    results.push_back(ModelRunResult{
        .model_name = internal::Stem(model_path),
        .diff_stats = std::move(diff_stats),
    });
  }

  return results;
}

void PrintDiffStats(const std::vector<litert::ModelRunResult>& results) {
  // Print CSV header
  std::cout << "model_name, buffer_idx, total_elements, diff_elements, "
               "epsilon, max_diff, min_diff, mean_diff, mse"
            << std::endl;
  for (const auto& result : results) {
    for (const auto& diff_stat : result.diff_stats) {
      std::cout << result.model_name << ", " << diff_stat.buffer_idx << ", "
                << diff_stat.total_elements << ", " << diff_stat.diff_elements
                << ", " << diff_stat.epsilon << ", " << diff_stat.max_diff
                << ", " << diff_stat.min_diff << ", " << diff_stat.mean_diff
                << ", " << diff_stat.mse << std::endl;
    }
  }
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  auto res = litert::RunModels();
  if (!res) {
    ABSL_LOG(ERROR) << res.Error().Message();
    return EXIT_FAILURE;
  }

  if (absl::GetFlag(FLAGS_print_diff_stats)) {
    litert::PrintDiffStats(*res);
  }

  return EXIT_SUCCESS;
}
