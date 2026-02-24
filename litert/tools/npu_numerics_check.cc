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
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment_options.h"
#define INCLUDE_QUALCOMM_RUNTIME_FLAGS
#define INCLUDE_MEDIATEK_RUNTIME_FLAGS
#define INCLUDE_GOOGLE_TENSOR_RUNTIME_FLAGS
#define INCLUDE_INTEL_OPENVINO_RUNTIME_FLAGS

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/tools/flags/vendors/google_tensor_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/mediatek_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/qualcomm_flags.h"  // IWYU pragma: keep
#include "litert/tools/flags/vendors/intel_openvino_flags.h"  // IWYU pragma: keep
#include "litert/tools/tensor_utils.h"  // IWYU pragma: keep

// NPU and CPU models must have the same input signature
ABSL_FLAG(std::string, cpu_model, "", "CPU Model filename to use for testing.");
ABSL_FLAG(std::string, npu_model, "", "NPU Model filename to use for testing.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Path to the dispatch library.");
ABSL_FLAG(size_t, signature_index, 0, "Index of the signature to run.");
ABSL_FLAG(float, epsilon, 1e-4f,
          "Threshold value for npu / cpu inference comparison");
ABSL_FLAG(bool, check_element_type, false,
          "Whether to check the element type of the output buffers.");
ABSL_FLAG(bool, print_distribution, false,
          "Whether to print the distribution of the output buffers.");
ABSL_FLAG(
    bool, print_difference_distribution, false,
    "Whether to print the difference distribution of the output buffers.");
ABSL_FLAG(std::string, input_dir, "",
          "An input folder containing .raw files with model input signatures "
          "as their file names.");

namespace litert {
namespace {

using ::litert::google_tensor::UpdateGoogleTensorOptionsFromFlags;
using ::litert::mediatek::UpdateMediatekOptionsFromFlags;
using ::litert::qualcomm::UpdateQualcommOptionsFromFlags;
using ::litert::intel_openvino::UpdateIntelOpenVinoOptionsFromFlags;

Expected<Environment> GetEnvironment() {
  std::vector<EnvironmentOptions::Option> env_options;
  const auto dispatch_library_dir = absl::GetFlag(FLAGS_dispatch_library_dir);
  if (!dispatch_library_dir.empty()) {
    env_options.push_back(EnvironmentOptions::Option{
        EnvironmentOptions::Tag::kDispatchLibraryDir, dispatch_library_dir});
  }
  auto env_options_obj = EnvironmentOptions(env_options);
  return Environment::Create(env_options_obj);
}

Expected<Options> GetOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);
  LITERT_ASSIGN_OR_RETURN(auto& qnn_opts, options.GetQualcommOptions());
  LITERT_RETURN_IF_ERROR(UpdateQualcommOptionsFromFlags(qnn_opts));
  LITERT_ASSIGN_OR_RETURN(auto& google_tensor_opts,
                          options.GetGoogleTensorOptions());
  LITERT_RETURN_IF_ERROR(
      UpdateGoogleTensorOptionsFromFlags(google_tensor_opts));
  LITERT_ASSIGN_OR_RETURN(auto& mediatek_opts, options.GetMediatekOptions());
  LITERT_RETURN_IF_ERROR(UpdateMediatekOptionsFromFlags(mediatek_opts));
  LITERT_ASSIGN_OR_RETURN(auto& intel_openvino_opts,
                          options.GetIntelOpenVinoOptions());
  LITERT_RETURN_IF_ERROR(
      UpdateIntelOpenVinoOptionsFromFlags(intel_openvino_opts));
  return options;
}

// Creates output buffers for a given compiled model.
Expected<std::vector<TensorBuffer>> CreateOutputBuffers(
    const CompiledModel& compiled_model, size_t signature_index) {
  return compiled_model.CreateOutputBuffers(signature_index);
}

void PrintDistribution(const std::vector<float>& cpu_data,
                       const std::vector<float>& npu_data) {
  if (cpu_data.empty() || npu_data.empty()) {
    return;
  }

  float min_val = cpu_data[0];
  float max_val = cpu_data[0];
  for (const auto& val : cpu_data) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
  }
  for (const auto& val : npu_data) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
  }

  if (min_val == max_val) {
    std::cout << "All values are the same: " << min_val << std::endl;
    return;
  }

  const int kNumBins = 20;
  std::vector<int> cpu_hist(kNumBins, 0);
  std::vector<int> npu_hist(kNumBins, 0);
  const float bin_width = (max_val - min_val) / kNumBins;

  for (const auto& val : cpu_data) {
    int bin = (val - min_val) / bin_width;
    if (bin >= kNumBins) bin = kNumBins - 1;
    cpu_hist[bin]++;
  }
  for (const auto& val : npu_data) {
    int bin = (val - min_val) / bin_width;
    if (bin >= kNumBins) bin = kNumBins - 1;
    npu_hist[bin]++;
  }

  int max_hist_count = 0;
  for (int count : cpu_hist) {
    if (count > max_hist_count) max_hist_count = count;
  }
  for (int count : npu_hist) {
    if (count > max_hist_count) max_hist_count = count;
  }

  if (max_hist_count == 0) {
    return;
  }

  const int kMaxBarWidth = 50;
  std::cout << "\n--- Value Distribution ---" << std::endl;
  std::cout << absl::StrFormat("%-22s | %-50s | %-50s\n", "Value Range", "CPU",
                               "NPU");
  std::cout << std::string(22 + 3 + 50 + 3 + 50, '-') << std::endl;

  for (int i = 0; i < kNumBins; ++i) {
    float bin_start = min_val + i * bin_width;
    float bin_end = bin_start + bin_width;
    std::string range_str = absl::StrFormat("[%.4f, %.4f)", bin_start, bin_end);

    int cpu_bar_width = (cpu_hist[i] * kMaxBarWidth) / max_hist_count;
    int npu_bar_width = (npu_hist[i] * kMaxBarWidth) / max_hist_count;
    std::string cpu_bar(cpu_bar_width, '#');
    std::string npu_bar(npu_bar_width, '#');

    std::cout << absl::StrFormat("%-22s | %-50s | %-50s\n", range_str, cpu_bar,
                                 npu_bar);
  }
  std::cout << "--------------------------\n" << std::endl;
}

void PrintDifferenceDistribution(const std::vector<float>& cpu_data,
                                 const std::vector<float>& npu_data) {
  if (cpu_data.size() != npu_data.size() || cpu_data.empty()) {
    return;
  }

  std::vector<float> diffs(cpu_data.size());
  for (size_t i = 0; i < cpu_data.size(); ++i) {
    diffs[i] = cpu_data[i] - npu_data[i];
  }

  float min_diff = diffs[0];
  float max_diff = diffs[0];
  for (const auto& diff : diffs) {
    if (diff < min_diff) min_diff = diff;
    if (diff > max_diff) max_diff = diff;
  }

  if (min_diff == max_diff) {
    std::cout << "All differences are the same: " << min_diff << std::endl;
    return;
  }

  const int kNumBins = 20;
  std::vector<int> hist(kNumBins, 0);
  const float bin_width = (max_diff - min_diff) / kNumBins;

  for (const auto& diff : diffs) {
    int bin = (diff - min_diff) / bin_width;
    if (bin >= kNumBins) bin = kNumBins - 1;
    hist[bin]++;
  }

  int max_hist_count = 0;
  for (int count : hist) {
    if (count > max_hist_count) max_hist_count = count;
  }

  if (max_hist_count == 0) {
    return;
  }

  const int kMaxBarWidth = 100;
  std::cout << "\n--- Difference (CPU - NPU) Distribution ---" << std::endl;
  std::cout << absl::StrFormat("%-22s | %s\n", "Difference Range",
                               "Distribution");
  std::cout << std::string(22 + 3 + 100, '-') << std::endl;

  for (int i = 0; i < kNumBins; ++i) {
    float bin_start = min_diff + i * bin_width;
    float bin_end = bin_start + bin_width;
    std::string range_str = absl::StrFormat("[%.4f, %.4f)", bin_start, bin_end);

    int bar_width = (hist[i] * kMaxBarWidth) / max_hist_count;
    std::string bar(bar_width, '#');

    std::cout << absl::StrFormat("%-22s | %s\n", range_str, bar);
  }
  std::cout << "---------------------------------------------\n" << std::endl;
}

// Compares a single pair of output buffers and prints the results.
Expected<void> CompareSingleOutputBuffer(TensorBuffer& cpu_buffer,
                                         TensorBuffer& npu_buffer,
                                         size_t buffer_index,
                                         absl::string_view output_name,
                                         float epsilon) {
  std::vector<std::pair<float, int>> all_diffs;
  const int kMaxPrint = 20;
  int printed = 0;
  int total_different = 0;
  double mean_squared_error = 0;
  float mean_diff = 0;
  double dot_product = 0.0;
  double magnitude_cpu = 0.0;
  double magnitude_npu = 0.0;
  float max_abs_cpu = 0.0f;
  double sum_cpu = 0.0;
  double sum_npu = 0.0;

  LITERT_ASSIGN_OR_RETURN(auto cpu_type, cpu_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto npu_type, npu_buffer.TensorType());
  if (absl::GetFlag(FLAGS_check_element_type)) {
    if (cpu_type.ElementType() != npu_type.ElementType()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Element type mismatch between CPU and NPU.");
    }
    LITERT_ASSIGN_OR_RETURN(size_t cpu_size, cpu_buffer.Size());
    LITERT_ASSIGN_OR_RETURN(size_t npu_size, npu_buffer.Size());
    if (cpu_size != npu_size) {
      return Error(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrFormat("Size mismatch for output buffer %d", buffer_index));
    }
  }

  // Calculate the total number of elements from dimensions and check that the
  // dimensions are the same.
  size_t total_elements = 1;
  const auto& cpu_layout = cpu_type.Layout();
  const auto& npu_layout = npu_type.Layout();
  for (size_t d = 0; d < cpu_layout.Rank(); ++d) {
    total_elements *= cpu_layout.Dimensions()[d];
    if (cpu_layout.Dimensions()[d] != npu_layout.Dimensions()[d]) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   absl::StrFormat("Dimension mismatch for output buffer %d",
                                   buffer_index));
    }
  }

  ABSL_LOG(INFO) << "Comparing output buffer " << buffer_index
                 << " (name: " << output_name << "):";

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
    if (element_type == ElementType::Int64) {
      std::vector<int64_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<int64_t>(absl::MakeSpan(data)));
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
  std::vector<float> npu_data(total_elements);
  get_val(cpu_buffer, cpu_data);
  get_val(npu_buffer, npu_data);

  if (absl::GetFlag(FLAGS_print_distribution)) {
    PrintDistribution(cpu_data, npu_data);
  }
  if (absl::GetFlag(FLAGS_print_difference_distribution)) {
    PrintDifferenceDistribution(cpu_data, npu_data);
  }

  for (int element_index = 0; element_index < total_elements; ++element_index) {
    const float abs_diff =
        fabs(cpu_data[element_index] - npu_data[element_index]);
    const double diff_square =
        (cpu_data[element_index] - npu_data[element_index]) *
        (cpu_data[element_index] - npu_data[element_index]);
    mean_squared_error += diff_square;
    mean_diff += abs_diff;
    dot_product += cpu_data[element_index] * npu_data[element_index];
    magnitude_cpu += cpu_data[element_index] * cpu_data[element_index];
    magnitude_npu += npu_data[element_index] * npu_data[element_index];
    max_abs_cpu = std::max(max_abs_cpu, std::abs(cpu_data[element_index]));
    sum_cpu += cpu_data[element_index];
    sum_npu += npu_data[element_index];

    all_diffs.push_back(std::make_pair(abs_diff, element_index));
    if (abs_diff > epsilon) {
      total_different++;
      if (printed < kMaxPrint) {
        std::cout << "Element #" << element_index << ": CPU value - "
                  << cpu_data[element_index] << ", NPU value - "
                  << npu_data[element_index] << ", abs diff - " << abs_diff
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

  const double cosine_similarity =
      dot_product / (sqrt(magnitude_cpu) * sqrt(magnitude_npu));
  const double mse = mean_squared_error / total_elements;
  const double snr = 10 * log10(magnitude_cpu / mean_squared_error);
  const double psnr = 10 * log10(max_abs_cpu * max_abs_cpu / mse);

  const double numerator = total_elements * dot_product - sum_cpu * sum_npu;
  const double denominator_cpu =
      total_elements * magnitude_cpu - sum_cpu * sum_cpu;
  const double denominator_npu =
      total_elements * magnitude_npu - sum_npu * sum_npu;
  double pearson_correlation = 0.0;
  if (denominator_cpu > 0 && denominator_npu > 0) {
    pearson_correlation = numerator / (sqrt(denominator_cpu * denominator_npu));
  } else if (mean_squared_error == 0) {
    pearson_correlation = 1.0;
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
              << " , NPU val: " << npu_data[all_diffs[reversed_index].second]
              << std::endl;
  }

  std::cout << "CPU magnitude: " << magnitude_cpu << std::endl;
  std::cout << "NPU magnitude: " << magnitude_npu << std::endl;
  std::cout << "Mean diff: " << mean_diff / all_diffs.size() << std::endl;
  std::cout << "Cosine similarity: " << cosine_similarity << std::endl;
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "SNR: " << snr << " dB" << std::endl;
  std::cout << "PSNR: " << psnr << " dB" << std::endl;
  std::cout << "Pearson correlation: " << pearson_correlation << std::endl;
  std::cout << "Total " << total_different << " out of " << total_elements
            << " are different elements, for output #" << buffer_index
            << " (name: " << output_name << "), threshold - " << epsilon
            << std::endl;
  return {};
}

Expected<void> CompareOutputBuffers(
    std::vector<TensorBuffer>& cpu_output_buffers,
    std::vector<TensorBuffer>& npu_output_buffers,
    const std::vector<absl::string_view>& output_names) {
  if (cpu_output_buffers.size() != npu_output_buffers.size()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Number of output buffers mismatch between CPU and NPU.");
  }
  if (cpu_output_buffers.size() != output_names.size()) {
    return Error(
        kLiteRtStatusErrorInvalidArgument,
        "Number of output buffers mismatch between CPU buffers and names.");
  }

  float epsilon = absl::GetFlag(FLAGS_epsilon);
  size_t num_output_buffers = cpu_output_buffers.size();
  for (size_t i = 0; i < num_output_buffers; ++i) {
    auto& cpu_buffer = cpu_output_buffers[i];
    auto& npu_buffer = npu_output_buffers[i];
    LITERT_RETURN_IF_ERROR(CompareSingleOutputBuffer(cpu_buffer, npu_buffer, i,
                                                     output_names[i], epsilon));
  }
  return {};
}

Expected<void> RunModel() {
  if (absl::GetFlag(FLAGS_cpu_model).empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "CPU model filename is empty. Use --cpu_model to provide it.");
  }

  if (absl::GetFlag(FLAGS_npu_model).empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "NPU model filename is empty. Use --npu_model to provide it.");
  }

  if (absl::GetFlag(FLAGS_dispatch_library_dir).empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Dispatch library directory is empty. Use "
                 "--dispatch_library_dir to provide it.");
  }

  LITERT_ASSIGN_OR_RETURN(auto env, GetEnvironment());
  LITERT_ASSIGN_OR_RETURN(auto options, GetOptions());

  ABSL_LOG(INFO) << "CPU Model: " << absl::GetFlag(FLAGS_cpu_model);
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model_cpu,
      CompiledModel::Create(env, absl::GetFlag(FLAGS_cpu_model), options));

  ABSL_LOG(INFO) << "NPU Model: " << absl::GetFlag(FLAGS_npu_model);
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model_npu,
      CompiledModel::Create(env, absl::GetFlag(FLAGS_npu_model), options));

  size_t signature_index = absl::GetFlag(FLAGS_signature_index);
  ABSL_LOG(INFO) << "Signature index: " << signature_index;
  std::string input_dir = absl::GetFlag(FLAGS_input_dir);

  LITERT_ASSIGN_OR_RETURN(
      auto cpu_input_buffers,
      compiled_model_cpu.CreateInputBuffers(signature_index));

  // Create and fill input buffers
  if (!input_dir.empty()) {
    LITERT_RETURN_IF_ERROR(tensor_utils::FillInputBuffersWithCustomData(
        compiled_model_cpu, signature_index, cpu_input_buffers, input_dir));
  } else {
    for (auto& buffer : cpu_input_buffers) {
      LITERT_RETURN_IF_ERROR(tensor_utils::FillBufferWithRandomData(buffer));
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto npu_input_buffers,
      compiled_model_npu.CreateInputBuffers(signature_index));
  // Copy input buffers from CPU to NPU.
  for (size_t i = 0; i < cpu_input_buffers.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size, cpu_input_buffers[i].Size());
    std::vector<char> data(buffer_size);
    LITERT_RETURN_IF_ERROR(
        cpu_input_buffers[i].Read<char>(absl::MakeSpan(data)));
    LITERT_RETURN_IF_ERROR(
        npu_input_buffers[i].Write<char>(absl::MakeSpan(data)));
  }

  // Create output buffers
  LITERT_ASSIGN_OR_RETURN(
      auto cpu_output_buffers,
      CreateOutputBuffers(compiled_model_cpu, signature_index));
  LITERT_ASSIGN_OR_RETURN(
      auto npu_output_buffers,
      CreateOutputBuffers(compiled_model_npu, signature_index));

  // Run models
  LITERT_RETURN_IF_ERROR(compiled_model_cpu.Run(
      signature_index, cpu_input_buffers, cpu_output_buffers));
  LITERT_RETURN_IF_ERROR(compiled_model_npu.Run(
      signature_index, npu_input_buffers, npu_output_buffers));

  // Get output names
  LITERT_ASSIGN_OR_RETURN(
      auto output_names,
      compiled_model_cpu.GetSignatureOutputNames(signature_index));

  // Compare output buffers
  LITERT_RETURN_IF_ERROR(CompareOutputBuffers(
      cpu_output_buffers, npu_output_buffers, output_names));

  return {};
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
