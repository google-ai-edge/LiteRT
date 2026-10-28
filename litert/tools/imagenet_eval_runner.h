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

#ifndef LITERT_TOOLS_IMAGENET_EVAL_RUNNER_H_
#define LITERT_TOOLS_IMAGENET_EVAL_RUNNER_H_
#include <string>
#include "litert/tools/imagenet_eval_data.h"
namespace litert::imagenet {
struct EvaluationConfig {
  std::string model;
  std::string baseline_model, summary_json, runtime_library_dir;
  std::string gpu_use_ir_model = "default";
  std::string interpolation = "bilinear", preprocessing_config;
  int require_full_acceleration = 0, require_output_parity = 0;
  double atol = 1e-4, rtol = 1e-3;
  std::string manifest;
  std::string image_root;
  std::string out;
  std::string scores_out;
  std::string inputs_out;
  int num_threads = 0;
  int xnnpack_only = 0;
  int warmup_runs = 0;
  int start_sample = 0;
  std::string accelerator = "cpu";
  std::string compiler_plugin_library_dir;
  int arm_vulkan_ml_enable_jit = 0;
  std::string baseline_accelerator = "inherit";
  int baseline_num_threads = 1;
  std::string dispatch_library_dir;
  std::string openvino_device;
  std::string gpu_backend = "automatic";
  std::string gpu_precision = "default";
  int gpu_benchmark_mode = 0;
  int resize = 232;
  int crop_h = 224;
  int crop_w = 224;
  int antialias = 1;
  std::string mean = "0.485,0.456,0.406";
  std::string stddev = "0.229,0.224,0.225";
  int max_samples = 0;
  // Optional code-space parity gates for backend development tools.
  bool capture_records = false;
  bool compare_int8_codes = false;
  double max_code_error = 4, mean_code_error = 0.5;
  bool require_prediction_agreement = false;
};
Expected<void> Evaluate(const EvaluationConfig& config, Json& summary);
int EvaluationMain(int argc, char** argv);
}  // namespace litert::imagenet
#endif
