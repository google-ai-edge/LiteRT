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

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/core/model/model_load.h"
#include "litert/tools/accuracy_debugger/accuracy_debugger_util.h"

#define INCLUDE_QUALCOMM_COMPILE_FLAGS
#define INCLUDE_QUALCOMM_RUNTIME_FLAGS
#define INCLUDE_MEDIATEK_COMPILE_FLAGS
#define INCLUDE_MEDIATEK_RUNTIME_FLAGS
#define INCLUDE_INTEL_OPENVINO_COMPILE_FLAGS
#define INCLUDE_INTEL_OPENVINO_RUNTIME_FLAGS
#define INCLUDE_GOOGLE_TENSOR_COMPILE_FLAGS
#define INCLUDE_GOOGLE_TENSOR_RUNTIME_FLAGS

#include "litert/tools/flags/vendors/google_tensor_flags.h"
#include "litert/tools/flags/vendors/intel_openvino_flags.h"
#include "litert/tools/flags/vendors/mediatek_flags.h"
#include "litert/tools/flags/vendors/qualcomm_flags.h"

ABSL_FLAG(std::string, model_path, "", "(Required) Path to the tflite model");
ABSL_FLAG(std::string, input_dir, "",
          "Directory containing real input data (.raw files)");
ABSL_FLAG(std::string, output_dir, "/tmp/accuracy_debugger",
          "Output directory for failing ops");

// Accuracy Threshold Flags
ABSL_FLAG(float, max_diff, 5e-3f, "Threshold for maximum absolute difference.");
ABSL_FLAG(float, mse, 1e-5f, "Threshold for mean squared error.");
ABSL_FLAG(float, cosine_similarity, 0.99f,
          "Threshold for cosine similarity (minimum).");
ABSL_FLAG(float, snr, 30.0f,
          "Threshold for Signal-to-Noise Ratio (minimum dB).");
ABSL_FLAG(float, psnr, 35.0f,
          "Threshold for Peak Signal-to-Noise Ratio (minimum dB).");
ABSL_FLAG(float, pearson_correlation, 0.98f,
          "Threshold for Pearson correlation (minimum).");

ABSL_FLAG(size_t, signature_index, 0, "Index of the signature to run.");
ABSL_FLAG(int, max_ops, -1,
          "Maximum number of operations to check. -1 for all.");
ABSL_FLAG(bool, skip_unsupported_npu_ops, true,
          "Whether to skip operations that are known to be unsupported or "
          "unstable on the NPU (e.g. EMBEDDING_LOOKUP, CUSTOM).");
ABSL_FLAG(bool, use_accel_output_as_input, true,
          "Whether to use accelerator output as input for next ops to mimic "
          "error propagation.");
ABSL_FLAG(bool, save_failing_models, false,
          "Whether to save failing single-op models and their inputs.");
ABSL_FLAG(std::string, accelerator, "gpu", "Target accelerator (gpu, npu)");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "(Required for NPU) Path to the dispatch library.");
ABSL_FLAG(std::string, compiler_plugin_library_dir, "",
          "(Required for NPU) Path to the compiler plugin library.");
ABSL_FLAG(std::string, compiler_cache_dir, "",
          "(Required for NPU) Path to the compiler cache directory.");
ABSL_FLAG(int, summary_max_rows, 50,
          "Maximum number of rows to print in the summary table (-1 for all)");
ABSL_FLAG(std::string, sort_by, "cos_sim",
          "Metric to sort the summary table by (index, max_diff, mse, cos_sim, "
          "snr, psnr)");
ABSL_FLAG(
    std::string, boundary_tensors, "",
    "Comma-separated list of boundary tensor names for multi-op chunking. "
    "If empty, single-op mode is used.");
ABSL_FLAG(bool, use_gpu_ref, false,
          "Use GPU FP32 as reference path instead of CPU");

namespace litert::tools {

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
    }
  }
  return accelerators;
}

litert::Expected<litert::Environment> GetEnvironment() {
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
  return litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
}

litert::Expected<litert::Options> GetOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
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

}  // namespace litert::tools

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    std::cerr << "Please provide --model_path" << std::endl;
    return 1;
  }

  auto env_res = litert::tools::GetEnvironment();
  if (!env_res) {
    ABSL_LOG(ERROR) << "Failed to create environment: "
                    << env_res.Error().Message();
    return 1;
  }

  auto options_res = litert::tools::GetOptions();
  if (!options_res) {
    ABSL_LOG(ERROR) << "Failed to create options: "
                    << options_res.Error().Message();
    return 1;
  }

  auto model_res = litert::internal::LoadModelFromFile(model_path);
  if (!model_res) {
    ABSL_LOG(ERROR) << "Failed to load model: " << model_res.Error().Message();
    return 1;
  }

  litert::tools::AccuracyDebuggerOptions checker_options;
  checker_options.input_dir = absl::GetFlag(FLAGS_input_dir);
  checker_options.output_dir = absl::GetFlag(FLAGS_output_dir);
  checker_options.accelerator = absl::GetFlag(FLAGS_accelerator);

  checker_options.thresholds.max_diff = absl::GetFlag(FLAGS_max_diff);
  checker_options.thresholds.mse = absl::GetFlag(FLAGS_mse);
  checker_options.thresholds.cosine_similarity =
      absl::GetFlag(FLAGS_cosine_similarity);
  checker_options.thresholds.snr = absl::GetFlag(FLAGS_snr);
  checker_options.thresholds.psnr = absl::GetFlag(FLAGS_psnr);
  checker_options.thresholds.pearson_correlation =
      absl::GetFlag(FLAGS_pearson_correlation);

  checker_options.signature_index = absl::GetFlag(FLAGS_signature_index);
  checker_options.max_ops = absl::GetFlag(FLAGS_max_ops);
  checker_options.skip_unsupported_npu_ops =
      absl::GetFlag(FLAGS_skip_unsupported_npu_ops);
  checker_options.summary_max_rows = absl::GetFlag(FLAGS_summary_max_rows);
  checker_options.sort_by = absl::GetFlag(FLAGS_sort_by);
  checker_options.use_accel_output_as_input =
      absl::GetFlag(FLAGS_use_accel_output_as_input);
  checker_options.save_failing_models =
      absl::GetFlag(FLAGS_save_failing_models);
  checker_options.use_gpu_ref = absl::GetFlag(FLAGS_use_gpu_ref);

  std::string boundary_tensors_str = absl::GetFlag(FLAGS_boundary_tensors);
  if (!boundary_tensors_str.empty()) {
    checker_options.boundary_tensors =
        absl::StrSplit(boundary_tensors_str, ',');
  }

  litert::tools::AccuracyDebuggerSummary summary;
  auto status = litert::tools::RunAccuracyDebugger(
      *env_res, **model_res, *options_res, checker_options, &summary);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Accuracy debugger failed: " << status.message();
    return 1;
  }

  return 0;
}
