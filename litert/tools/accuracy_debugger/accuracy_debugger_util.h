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

#ifndef ODML_LITERT_LITERT_TOOLS_ACCURACY_DEBUGGER_UTIL_H_
#define ODML_LITERT_LITERT_TOOLS_ACCURACY_DEBUGGER_UTIL_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/core/model/model.h"

namespace litert::tools {

struct AccuracyThresholds {
  float max_diff = 1e-3f;
  float mse = 1e-6f;
  float cosine_similarity = 0.99f;
  float snr = 30.0f;
  float psnr = 40.0f;
  float pearson_correlation = 0.98f;
};

struct ComparisonResult {
  float max_diff = 0;
  double mse = 0;
  double mean_diff = 0;
  double cosine_similarity = 0;
  double snr = 0;
  double psnr = 0;
  double pearson_correlation = 0;
  int diff_count = 0;
  bool failed = false;
  std::vector<std::string> failing_metrics;

  bool has_nan_accel = false;
  float min_ref = 0;
  float max_ref = 0;
  float min_accel = 0;
  float max_accel = 0;
};

namespace internal {

struct ExtractedModel {
  LiteRtModelT model;
  std::vector<const LiteRtTensorT*> inputs;
  std::vector<const LiteRtTensorT*> outputs;
};

ExtractedModel ExtractOp(const LiteRtOpT& op,
                         const LiteRtModelT& original_model);

litert::Expected<ComparisonResult> CompareBuffers(
    TensorBuffer& cpu_buffer, TensorBuffer& accel_buffer,
    const AccuracyThresholds& thresholds, const std::string& op_info);

litert::Expected<std::vector<double>> GetFloats(TensorBuffer& buffer,
                                                size_t num_elements);

}  // namespace internal

struct AccuracyDebuggerOptions {
  std::string input_dir;
  std::string output_dir = "/tmp/accuracy_debugger";
  std::string accelerator = "gpu";
  bool save_failing_models = true;
  size_t signature_index = 0;
  bool use_accel_output_as_input = false;
  int max_ops = -1;
  bool skip_unsupported_npu_ops = true;
  int summary_max_rows = -1;
  std::string sort_by = "index";
  std::vector<std::string> boundary_tensors;
  bool use_gpu_ref = false;
  AccuracyThresholds thresholds;
};

// Summary of the accuracy debugging run.
struct AccuracyDebuggerSummary {
  struct OpStats {
    int global_index;
    std::string op_code;
    std::string tensor_name;
    ComparisonResult metrics;
  };
  std::vector<OpStats> all_ops;
  std::vector<int> failing_op_indices;

  void LogSummary(const AccuracyDebuggerOptions& options) const;
  void SaveToCsv(const std::string& path) const;
};

// Runs the accuracy debugger.
// It splits the model into single-op models, runs each on CPU and the targeted
// accelerator, and compares the results using multiple metrics.
absl::Status RunAccuracyDebugger(litert::Environment& env, LiteRtModelT& model,
                                 litert::Options& accel_opts,
                                 const AccuracyDebuggerOptions& options,
                                 AccuracyDebuggerSummary* summary = nullptr);

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_TOOLS_ACCURACY_DEBUGGER_UTIL_H_
