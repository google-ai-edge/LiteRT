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

// Tool to compare inference results between CPU and NPU for the same model.
// Uses randomly generated inputs and reports per-output numerical differences.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"       // from @com_google_absl
#include "absl/flags/parse.h"      // from @com_google_absl
#include "absl/log/absl_log.h"     // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"       // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/tools/tensor_utils.h"
#include "tflite/profiling/time.h"

ABSL_FLAG(std::string, model, "", "Model filename (.tflite) to compare.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Path to the NPU dispatch library directory.");
ABSL_FLAG(std::string, compiler_plugin_library_dir, "",
          "Path to the compiler plugin library directory (for JIT).");
ABSL_FLAG(std::string, compiler_cache_dir, "",
          "Path to the compiler cache directory. When --use_caching is set "
          "and this is empty, a default cache directory next to the model "
          "file is used.");
ABSL_FLAG(bool, use_caching, false,
          "Enable caching of compiled model artifacts. Uses "
          "--compiler_cache_dir if set, otherwise creates a cache directory "
          "next to the model file.");
ABSL_FLAG(size_t, signature_index, 0, "Index of the signature to run.");
ABSL_FLAG(float, epsilon, 1e-4f,
          "Threshold for per-element difference reporting.");
ABSL_FLAG(size_t, iterations, 1,
          "Number of inference iterations to run on each accelerator.");
ABSL_FLAG(size_t, top_diffs, 10,
          "Number of top differences to print per output.");
ABSL_FLAG(bool, print_distribution, false,
          "Print value-distribution histograms for each output.");

namespace litert {
namespace {

// Resolve the compiler cache directory. When --use_caching is set and no
// explicit --compiler_cache_dir is provided, derive one from the model path.
std::string ResolveCacheDir(const std::string& model_path) {
  std::string cache_dir = absl::GetFlag(FLAGS_compiler_cache_dir);
  if (!cache_dir.empty()) return cache_dir;

  if (!absl::GetFlag(FLAGS_use_caching)) return "";

  // Default: <model_dir>/<model_stem>_cache
  std::filesystem::path p(model_path);
  std::filesystem::path dir =
      p.parent_path() / (p.stem().string() + "_cache");
  std::filesystem::create_directories(dir);
  return dir.string();
}

Expected<Environment> GetEnvironment(const std::string& model_path) {
  std::vector<EnvironmentOptions::Option> env_options;
  const auto dispatch_library_dir = absl::GetFlag(FLAGS_dispatch_library_dir);
  if (!dispatch_library_dir.empty()) {
    env_options.push_back(EnvironmentOptions::Option{
        EnvironmentOptions::Tag::kDispatchLibraryDir, dispatch_library_dir});
  }
  const auto compiler_plugin_library_dir =
      absl::GetFlag(FLAGS_compiler_plugin_library_dir);
  if (!compiler_plugin_library_dir.empty()) {
    env_options.push_back(EnvironmentOptions::Option{
        EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
        compiler_plugin_library_dir});
  }
  const std::string cache_dir = ResolveCacheDir(model_path);
  if (!cache_dir.empty()) {
    ABSL_LOG(INFO) << "Compiler cache directory: " << cache_dir;
    env_options.push_back(EnvironmentOptions::Option{
        EnvironmentOptions::Tag::kCompilerCacheDir, cache_dir});
  }
  return Environment::Create(EnvironmentOptions(absl::MakeConstSpan(
      env_options)));
}

// Read tensor data as float regardless of the underlying element type.
Expected<std::vector<float>> ReadBufferAsFloat(TensorBuffer& buffer,
                                               size_t total_elements) {
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  std::vector<float> result(total_elements);

  switch (type.ElementType()) {
    case ElementType::Float32: {
      std::vector<float> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<float>(absl::MakeSpan(data)));
      result.assign(data.begin(), data.end());
      break;
    }
    case ElementType::Int32: {
      std::vector<int32_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<int32_t>(absl::MakeSpan(data)));
      for (size_t i = 0; i < total_elements; ++i) result[i] = data[i];
      break;
    }
    case ElementType::Int16: {
      std::vector<int16_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<int16_t>(absl::MakeSpan(data)));
      for (size_t i = 0; i < total_elements; ++i) result[i] = data[i];
      break;
    }
    case ElementType::Int64: {
      std::vector<int64_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<int64_t>(absl::MakeSpan(data)));
      for (size_t i = 0; i < total_elements; ++i) result[i] = data[i];
      break;
    }
    case ElementType::Int8: {
      std::vector<int8_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<int8_t>(absl::MakeSpan(data)));
      for (size_t i = 0; i < total_elements; ++i) result[i] = data[i];
      break;
    }
    case ElementType::UInt8: {
      std::vector<uint8_t> data(total_elements);
      LITERT_RETURN_IF_ERROR(buffer.Read<uint8_t>(absl::MakeSpan(data)));
      for (size_t i = 0; i < total_elements; ++i) result[i] = data[i];
      break;
    }
    default:
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unsupported element type for comparison.");
  }
  return result;
}

void PrintDistribution(const std::vector<float>& cpu_data,
                       const std::vector<float>& npu_data) {
  if (cpu_data.empty() || npu_data.empty()) return;

  float min_val = cpu_data[0];
  float max_val = cpu_data[0];
  for (float v : cpu_data) {
    min_val = std::min(min_val, v);
    max_val = std::max(max_val, v);
  }
  for (float v : npu_data) {
    min_val = std::min(min_val, v);
    max_val = std::max(max_val, v);
  }
  if (min_val == max_val) {
    std::cout << "  All values are identical: " << min_val << std::endl;
    return;
  }

  constexpr int kNumBins = 20;
  constexpr int kMaxBarWidth = 50;
  std::vector<int> cpu_hist(kNumBins, 0);
  std::vector<int> npu_hist(kNumBins, 0);
  const float bin_width = (max_val - min_val) / kNumBins;

  auto bin_for = [&](float v) {
    int bin = static_cast<int>((v - min_val) / bin_width);
    return std::min(bin, kNumBins - 1);
  };
  for (float v : cpu_data) cpu_hist[bin_for(v)]++;
  for (float v : npu_data) npu_hist[bin_for(v)]++;

  int max_count = 0;
  for (int i = 0; i < kNumBins; ++i)
    max_count = std::max(max_count, std::max(cpu_hist[i], npu_hist[i]));
  if (max_count == 0) return;

  std::cout << "\n  --- Value Distribution ---\n";
  std::cout << absl::StrFormat("  %-22s | %-50s | %-50s\n", "Range", "CPU",
                               "NPU");
  std::cout << "  " << std::string(22 + 3 + 50 + 3 + 50, '-') << "\n";
  for (int i = 0; i < kNumBins; ++i) {
    float lo = min_val + i * bin_width;
    float hi = lo + bin_width;
    int cpu_w = (cpu_hist[i] * kMaxBarWidth) / max_count;
    int npu_w = (npu_hist[i] * kMaxBarWidth) / max_count;
    std::cout << absl::StrFormat("  [%9.4f, %9.4f) | %-50s | %-50s\n", lo, hi,
                                 std::string(cpu_w, '#'),
                                 std::string(npu_w, '#'));
  }
  std::cout << std::endl;
}

Expected<void> CompareOutputBuffer(TensorBuffer& cpu_buf,
                                   TensorBuffer& npu_buf,
                                   size_t buffer_index,
                                   absl::string_view output_name,
                                   float epsilon, size_t top_diffs) {
  LITERT_ASSIGN_OR_RETURN(auto cpu_type, cpu_buf.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto npu_type, npu_buf.TensorType());

  const auto& cpu_layout = cpu_type.Layout();
  const auto& npu_layout = npu_type.Layout();
  if (cpu_layout.Rank() != npu_layout.Rank()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 absl::StrFormat("Rank mismatch for output %d", buffer_index));
  }
  size_t total_elements = 1;
  for (size_t d = 0; d < cpu_layout.Rank(); ++d) {
    if (cpu_layout.Dimensions()[d] != npu_layout.Dimensions()[d]) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   absl::StrFormat("Dimension mismatch at axis %d for output %d",
                                   d, buffer_index));
    }
    total_elements *= cpu_layout.Dimensions()[d];
  }

  LITERT_ASSIGN_OR_RETURN(auto cpu_data,
                          ReadBufferAsFloat(cpu_buf, total_elements));
  LITERT_ASSIGN_OR_RETURN(auto npu_data,
                          ReadBufferAsFloat(npu_buf, total_elements));

  std::cout << "\n=== Output " << buffer_index << " (name: " << output_name
            << ", elements: " << total_elements << ") ===" << std::endl;

  if (absl::GetFlag(FLAGS_print_distribution)) {
    PrintDistribution(cpu_data, npu_data);
  }

  // Compute statistics.
  double sum_sq_err = 0.0;
  double sum_abs_err = 0.0;
  double dot_product = 0.0;
  double mag_cpu = 0.0;
  double mag_npu = 0.0;
  double sum_cpu = 0.0;
  double sum_npu = 0.0;
  float max_abs_cpu = 0.0f;
  size_t num_different = 0;

  std::vector<std::pair<float, size_t>> diffs;  // (abs_diff, index)
  diffs.reserve(total_elements);

  for (size_t i = 0; i < total_elements; ++i) {
    float c = cpu_data[i];
    float n = npu_data[i];
    float abs_diff = std::fabs(c - n);
    diffs.push_back({abs_diff, i});

    sum_sq_err += static_cast<double>(c - n) * (c - n);
    sum_abs_err += abs_diff;
    dot_product += static_cast<double>(c) * n;
    mag_cpu += static_cast<double>(c) * c;
    mag_npu += static_cast<double>(n) * n;
    sum_cpu += c;
    sum_npu += n;
    max_abs_cpu = std::max(max_abs_cpu, std::fabs(c));
    if (abs_diff > epsilon) ++num_different;
  }

  double mse = sum_sq_err / std::max(total_elements, size_t{1});
  double cosine_sim =
      (mag_cpu > 0 && mag_npu > 0)
          ? dot_product / (std::sqrt(mag_cpu) * std::sqrt(mag_npu))
          : (sum_sq_err == 0 ? 1.0 : 0.0);
  double snr = (sum_sq_err > 0) ? 10.0 * std::log10(mag_cpu / sum_sq_err)
                                 : std::numeric_limits<double>::infinity();
  double psnr =
      (mse > 0) ? 10.0 * std::log10(max_abs_cpu * max_abs_cpu / mse)
                : std::numeric_limits<double>::infinity();

  // Pearson correlation.
  double num = total_elements * dot_product - sum_cpu * sum_npu;
  double den_cpu = total_elements * mag_cpu - sum_cpu * sum_cpu;
  double den_npu = total_elements * mag_npu - sum_npu * sum_npu;
  double pearson = (den_cpu > 0 && den_npu > 0)
                       ? num / (std::sqrt(den_cpu) * std::sqrt(den_npu))
                       : (sum_sq_err == 0 ? 1.0 : 0.0);

  std::cout << "  MSE:                 " << mse << std::endl;
  std::cout << "  Mean abs error:      "
            << sum_abs_err / std::max(total_elements, size_t{1}) << std::endl;
  std::cout << "  Cosine similarity:   " << cosine_sim << std::endl;
  std::cout << "  SNR:                 " << snr << " dB" << std::endl;
  std::cout << "  PSNR:                " << psnr << " dB" << std::endl;
  std::cout << "  Pearson correlation: " << pearson << std::endl;
  std::cout << "  Elements > epsilon:  " << num_different << " / "
            << total_elements << " (epsilon=" << epsilon << ")" << std::endl;

  // Top diffs.
  std::sort(diffs.begin(), diffs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  size_t print_count = std::min(top_diffs, diffs.size());
  if (print_count > 0) {
    std::cout << "  Top " << print_count << " differences:" << std::endl;
    for (size_t i = 0; i < print_count; ++i) {
      size_t idx = diffs[i].second;
      std::cout << absl::StrFormat(
                       "    #%zu  element[%zu]  CPU=%.6g  NPU=%.6g  diff=%.6g",
                       i, idx, cpu_data[idx], npu_data[idx], diffs[i].first)
                << std::endl;
    }
  }
  return {};
}

Expected<void> Run() {
  const std::string model_path = absl::GetFlag(FLAGS_model);
  if (model_path.empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Model filename is empty. Use --model to provide it.");
  }

  LITERT_ASSIGN_OR_RETURN(auto env, GetEnvironment(model_path));
  const size_t signature_index = absl::GetFlag(FLAGS_signature_index);
  const size_t iterations = absl::GetFlag(FLAGS_iterations);
  const float epsilon = absl::GetFlag(FLAGS_epsilon);
  const size_t top_diffs = absl::GetFlag(FLAGS_top_diffs);

  // ---- Create CPU compiled model ----
  ABSL_LOG(INFO) << "Creating CPU compiled model for: " << model_path;
  LITERT_ASSIGN_OR_RETURN(auto cpu_options, Options::Create());
  cpu_options.SetHardwareAccelerators(HwAccelerators::kCpu);
  LITERT_ASSIGN_OR_RETURN(
      auto cpu_model, CompiledModel::Create(env, model_path, cpu_options));

  // ---- Create NPU compiled model ----
  ABSL_LOG(INFO) << "Creating NPU compiled model for: " << model_path;
  LITERT_ASSIGN_OR_RETURN(auto npu_options, Options::Create());
  npu_options.SetHardwareAccelerators(HwAccelerators::kNpu);
  LITERT_ASSIGN_OR_RETURN(
      auto npu_model, CompiledModel::Create(env, model_path, npu_options));

  // ---- Prepare input buffers with random data ----
  ABSL_LOG(INFO) << "Preparing input buffers (random data)";
  LITERT_ASSIGN_OR_RETURN(auto cpu_inputs,
                          cpu_model.CreateInputBuffers(signature_index));
  for (auto& buf : cpu_inputs) {
    LITERT_RETURN_IF_ERROR(tensor_utils::FillBufferWithRandomData(buf));
  }

  // Copy the same random input data to NPU input buffers.
  LITERT_ASSIGN_OR_RETURN(auto npu_inputs,
                          npu_model.CreateInputBuffers(signature_index));
  for (size_t i = 0; i < cpu_inputs.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(size_t buf_size, cpu_inputs[i].Size());
    std::vector<char> raw(buf_size);
    LITERT_RETURN_IF_ERROR(cpu_inputs[i].Read<char>(absl::MakeSpan(raw)));
    LITERT_RETURN_IF_ERROR(npu_inputs[i].Write<char>(absl::MakeSpan(raw)));
  }

  // ---- Create output buffers ----
  LITERT_ASSIGN_OR_RETURN(auto cpu_outputs,
                          cpu_model.CreateOutputBuffers(signature_index));
  LITERT_ASSIGN_OR_RETURN(auto npu_outputs,
                          npu_model.CreateOutputBuffers(signature_index));

  // ---- Run CPU inference ----
  ABSL_LOG(INFO) << "Running CPU inference (" << iterations << " iterations)";
  std::vector<uint64_t> cpu_times(iterations);
  for (size_t iter = 0; iter < iterations; ++iter) {
    uint64_t start = tflite::profiling::time::NowMicros();
    LITERT_RETURN_IF_ERROR(
        cpu_model.Run(signature_index, cpu_inputs, cpu_outputs));
    cpu_times[iter] = tflite::profiling::time::NowMicros() - start;
  }

  // ---- Run NPU inference ----
  ABSL_LOG(INFO) << "Running NPU inference (" << iterations << " iterations)";
  std::vector<uint64_t> npu_times(iterations);
  for (size_t iter = 0; iter < iterations; ++iter) {
    uint64_t start = tflite::profiling::time::NowMicros();
    LITERT_RETURN_IF_ERROR(
        npu_model.Run(signature_index, npu_inputs, npu_outputs));
    npu_times[iter] = tflite::profiling::time::NowMicros() - start;
  }

  // ---- Print timing ----
  auto avg = [](const std::vector<uint64_t>& v) {
    uint64_t s = 0;
    for (auto t : v) s += t;
    return s / std::max(v.size(), size_t{1});
  };
  std::cout << "\n--- Latency (us) ---" << std::endl;
  std::cout << "  CPU:  first=" << cpu_times[0]
            << "  avg=" << avg(cpu_times)
            << "  min=" << *std::min_element(cpu_times.begin(), cpu_times.end())
            << "  max=" << *std::max_element(cpu_times.begin(), cpu_times.end())
            << std::endl;
  std::cout << "  NPU:  first=" << npu_times[0]
            << "  avg=" << avg(npu_times)
            << "  min=" << *std::min_element(npu_times.begin(), npu_times.end())
            << "  max=" << *std::max_element(npu_times.begin(), npu_times.end())
            << std::endl;

  // ---- Compare outputs ----
  LITERT_ASSIGN_OR_RETURN(
      auto output_names,
      cpu_model.GetSignatureOutputNames(signature_index));

  if (cpu_outputs.size() != npu_outputs.size()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Number of output buffers mismatch between CPU and NPU.");
  }

  std::cout << "\n--- Numerical Comparison ---" << std::endl;
  for (size_t i = 0; i < cpu_outputs.size(); ++i) {
    absl::string_view name =
        (i < output_names.size()) ? output_names[i] : "unknown";
    LITERT_RETURN_IF_ERROR(CompareOutputBuffer(
        cpu_outputs[i], npu_outputs[i], i, name, epsilon, top_diffs));
  }

  std::cout << "\nDone." << std::endl;
  return {};
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  auto res = litert::Run();
  if (!res) {
    ABSL_LOG(ERROR) << res.Error().Message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
