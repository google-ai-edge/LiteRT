/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef ODML_LITERT_LITERT_TOOLS_BENCHMARK_LITERT_MODEL_H_
#define ODML_LITERT_LITERT_TOOLS_BENCHMARK_LITERT_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_profiler.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/profiling/model_runtime_info.h"
#include "tflite/profiling/profile_buffer.h"
#include "tflite/tools/benchmark/benchmark_model.h"
#include "tflite/tools/benchmark/benchmark_params.h"
#include "tflite/tools/benchmark/proto/benchmark_result.pb.h"
#include "tflite/tools/command_line_flags.h"
#include "tflite/tools/utils.h"

namespace litert {
namespace benchmark {
using ::tflite::tools::benchmark::BenchmarkResult;

// Custom logging listener with better output
class BenchmarkLoggingListener : public ::tflite::benchmark::BenchmarkListener {
 private:
  std::string result_file_path_ = "";
  std::function<std::string()> summary_provider_;

 public:
  explicit BenchmarkLoggingListener(
      std::function<std::string()> summary_provider)
      : summary_provider_(summary_provider) {}

  void OnBenchmarkStart(
      const ::tflite::benchmark::BenchmarkParams& params) override {
    if (!params.Get<std::string>("result_file_path").empty()) {
      result_file_path_ =
          std::string(params.Get<std::string>("result_file_path"));
    }
  }

  void OnBenchmarkEnd(
      const ::tflite::benchmark::BenchmarkResults& results) override {
    auto inference_us = results.inference_time_us();
    auto init_us = results.startup_latency_us();
    auto warmup_us = results.warmup_time_us();

    LITERT_LOG(LITERT_INFO, "\n========== BENCHMARK RESULTS ==========");
    LITERT_LOG(LITERT_INFO, "Model initialization: %.2f ms", init_us / 1000.0);
    LITERT_LOG(LITERT_INFO, "Warmup (first):       %.2f ms",
               warmup_us.first() / 1000.0);
    LITERT_LOG(LITERT_INFO, "Warmup (avg):         %.2f ms (%d runs)",
               warmup_us.avg() / 1000.0, warmup_us.count());
    LITERT_LOG(LITERT_INFO, "Inference (avg):      %.2f ms (%d runs)",
               inference_us.avg() / 1000.0, inference_us.count());

    if (inference_us.count() > 0) {
      LITERT_LOG(LITERT_INFO, "Inference (min):      %.2f ms",
                 inference_us.min() / 1000.0);
      LITERT_LOG(LITERT_INFO, "Inference (max):      %.2f ms",
                 inference_us.max() / 1000.0);
      LITERT_LOG(LITERT_INFO, "Inference (std):      %.2f",
                 inference_us.std_deviation() / 1000.0);
    } else {
      LITERT_LOG(LITERT_INFO, "No inference runs were performed.");
    }

    double throughput = results.throughput_MB_per_second();
    if (throughput > 0) {
      LITERT_LOG(LITERT_INFO, "Throughput:           %.2f MB/s", throughput);
    } else {
      LITERT_LOG(LITERT_INFO,
                 "No throughput data available (throughput <= 0).");
    }

    auto init_mem_usage = results.init_mem_usage();
    auto overall_mem_usage = results.overall_mem_usage();
    if (init_mem_usage.IsSupported()) {
      LITERT_LOG(LITERT_INFO, "\nMemory Usage:");
      LITERT_LOG(LITERT_INFO, "Init footprint:       %.2f MB",
                 init_mem_usage.mem_footprint_kb / 1024.0);
      LITERT_LOG(LITERT_INFO, "Overall footprint:    %.2f MB",
                 overall_mem_usage.mem_footprint_kb / 1024.0);

      float peak_mem_mb = results.peak_mem_mb();
      if (peak_mem_mb > 0) {
        LITERT_LOG(LITERT_INFO, "Peak memory:          %.2f MB", peak_mem_mb);
      } else {
        LITERT_LOG(LITERT_INFO,
                   "Peak memory usage not available. (peak_mem_mb <= 0)");
      }
    }
    LITERT_LOG(LITERT_INFO, "======================================\n");
    if (!result_file_path_.empty()) {
      BenchmarkResult result;
      result.mutable_latency_metrics()->set_init_ms(init_us / 1000.0);
      result.mutable_latency_metrics()->set_first_inference_ms(
          warmup_us.first() / 1000.0);
      result.mutable_latency_metrics()->set_average_warm_up_ms(warmup_us.avg() /
                                                               1000.0);
      result.mutable_latency_metrics()->set_min_ms(inference_us.min() / 1000.0);
      result.mutable_latency_metrics()->set_max_ms(inference_us.max() / 1000.0);
      result.mutable_latency_metrics()->set_stddev_ms(
          inference_us.std_deviation() / 1000.0);
      result.mutable_latency_metrics()->set_avg_ms(inference_us.avg() / 1000.0);
      result.mutable_latency_metrics()->set_median_ms(
          inference_us.percentile(50) / 1000.0);
      result.mutable_latency_metrics()->set_p5_ms(inference_us.percentile(5) /
                                                  1000.0);
      result.mutable_latency_metrics()->set_p95_ms(inference_us.percentile(95) /
                                                   1000.0);
      if (init_mem_usage.IsSupported()) {
        result.mutable_memory_metrics()->set_init_footprint_kb(
            init_mem_usage.mem_footprint_kb);
        result.mutable_memory_metrics()->set_overall_footprint_kb(
            overall_mem_usage.mem_footprint_kb);
        if (results.peak_mem_mb() > 0) {
          result.mutable_memory_metrics()->set_peak_mem_mb(
              results.peak_mem_mb());
        }
      }

      result.mutable_misc_metrics()->set_model_size_mb(results.model_size_mb());
      result.mutable_misc_metrics()->set_num_runs(inference_us.count());
      result.mutable_misc_metrics()->set_num_warmup_runs(warmup_us.count());
      result.mutable_misc_metrics()->set_model_throughput_in_mb_per_sec(
          results.throughput_MB_per_second());

      std::ofstream out_file(result_file_path_,
                             std::ios::binary | std::ios::out);
      if (out_file.good()) {
        LITERT_LOG(LITERT_INFO, "Saving benchmark result to: %s",
                   result_file_path_.c_str());
        result.SerializeToOstream(&out_file);
        out_file.close();
        LITERT_LOG(LITERT_INFO, "Saved benchmark result to: %s",
                   result_file_path_.c_str());
      } else {
        LITERT_LOG(LITERT_ERROR, "Failed to save benchmark result to: %s",
                   result_file_path_.c_str());
      }
    }

    if (summary_provider_) {
      std::string summary = summary_provider_();
      if (!summary.empty()) {
        LITERT_LOG(LITERT_INFO, "\n%s", summary.c_str());
      }
    }
  }
};

// Dumps the Model Runtime Info if enabled when export_model_runtime_info is
// set to true.
class ModelRuntimeInfoListener : public ::tflite::benchmark::BenchmarkListener {
 public:
  explicit ModelRuntimeInfoListener(::tflite::Interpreter* interpreter_ptr)
      : interpreter_(interpreter_ptr) {}

  // At this stage, the graph is fully modified with delegates.
  // So the interpreter can be used to capture the ModelRuntimeDetails.
  void OnBenchmarkStart(
      const ::tflite::benchmark::BenchmarkParams& params) override {
    const std::string output_file_path =
        std::string(params.Get<std::string>("model_runtime_info_output_file"));
    const auto status = tflite::profiling::GenerateModelRuntimeInfo(
        *interpreter_, output_file_path);
    if (status != kTfLiteOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to generate model runtime info: %s",
                 status);
    }
    LITERT_LOG(LITERT_INFO, "Generated model runtime info: %s",
               output_file_path.c_str());
  }

 private:
  ::tflite::Interpreter* interpreter_ = nullptr;
};
using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::Model;
using ::litert::TensorBuffer;
using ::tflite::benchmark::BenchmarkModel;
using ::tflite::benchmark::BenchmarkParam;
using ::tflite::benchmark::BenchmarkParams;
using ::tflite::utils::InputTensorData;

class BenchmarkLiteRtModel : public BenchmarkModel {
 public:
  explicit BenchmarkLiteRtModel(BenchmarkParams params = DefaultParams())
      : BenchmarkModel(std::move(params)) {
    model_runtime_info_listener_ = nullptr;
  }
  ~BenchmarkLiteRtModel() override = default;
  static BenchmarkParams DefaultParams() {
    BenchmarkParams default_params = BenchmarkModel::DefaultParams();
    default_params.AddParam("graph", BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("signature_to_run_for",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("use_cpu", BenchmarkParam::Create<bool>(true));
    default_params.AddParam("use_gpu", BenchmarkParam::Create<bool>(false));
    default_params.AddParam("use_npu", BenchmarkParam::Create<bool>(false));
    default_params.AddParam("dispatch_library_path",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("compiler_plugin_library_path",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("compiler_cache_path",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("require_full_delegation",
                            BenchmarkParam::Create<bool>(false));
    default_params.AddParam("use_profiler",
                            BenchmarkParam::Create<bool>(false));
    default_params.AddParam("enable_perfetto",
                            BenchmarkParam::Create<bool>(false));
    default_params.AddParam("gpu_backend",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("allow_fp16", BenchmarkParam::Create<bool>(true));
    default_params.AddParam("gpu_low_priority",
                            BenchmarkParam::Create<bool>(false));
    default_params.AddParam("enable_weight_sharing",
                            BenchmarkParam::Create<bool>(false));
    default_params.AddParam("result_file_path",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("model_runtime_info_output_file",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("mediatek_nerun_pilot_version",
                            BenchmarkParam::Create<std::string>("version8"));
    return default_params;
  }

  TfLiteStatus Init() override;

  int64_t MayGetModelFileSize() override {
    std::string fd_or_graph_path = params_.Get<std::string>("graph");
    // Path can be one of the following:
    // 1) File descriptor path: path must be in the format of
    // "fd:%model_fd%:%model_offset%:%model_size%".
    // 2) File path: path to the model file.
    // Please see tensorflow/lite/tools/model_loader.h for more information.
    std::vector<absl::string_view> parts =
        absl::StrSplit(fd_or_graph_path, ':');
    if (!parts.empty() && parts[0] == "fd") {
      int64_t model_size = -1;
      if (parts.size() != 4 || !absl::SimpleAtoi(parts[3], &model_size)) {
        LITERT_LOG(LITERT_ERROR, "Failed to parse model file size: %s",
                   fd_or_graph_path.c_str());
      }
      return model_size;
    }
    std::ifstream in_file(fd_or_graph_path, std::ios::binary | std::ios::ate);
    return in_file.tellg();
  }

  bool isFullyAccelerated() {
    auto is_fully_accelerated = compiled_model_->IsFullyAccelerated();
    if (!is_fully_accelerated.HasValue()) {
      LITERT_LOG(LITERT_ERROR,
                 "Failed to get is_fully_accelerated. Returning false.");
      return false;
    }
    return *is_fully_accelerated;
  }

  TfLiteStatus RunImpl() override {
    if (!compiled_model_) {
      LITERT_LOG(LITERT_ERROR, "Compiled model not initialized");
      return kTfLiteError;
    }
    auto signature = params_.Get<std::string>("signature_to_run_for");
    if (auto res = compiled_model_->Run(signature, *input_buffers_,
                                        *output_buffers_)) {
      return kTfLiteOk;
    } else {
      LITERT_LOG(LITERT_ERROR, "Run failed: %s", res.Error().Message().c_str());
      return kTfLiteError;
    }
  }

  uint64_t ComputeInputBytes() override {
    uint64_t total_bytes = 0;
    for (const auto& buffer : *input_buffers_) {
      LITERT_ASSIGN_OR_ABORT(const size_t buffer_bytes, buffer.Size());
      total_bytes += buffer_bytes;
    }
    return total_bytes;
  }

  InputTensorData CreateRandomTensorData(const litert::TensorBuffer& t,
                                         std::string name) {
    float low_range = 0;
    float high_range = 0;
    LITERT_ASSIGN_OR_ABORT(const auto t_tensor_type, t.TensorType());
    LITERT_ASSIGN_OR_ABORT(const size_t t_size, t.Size());
    size_t num_elements = t_size / *GetByteWidth(t_tensor_type.ElementType());
    tflite::utils::GetDataRangesForType(
        static_cast<TfLiteType>(t_tensor_type.ElementType()), &low_range,
        &high_range);
    return tflite::utils::CreateRandomTensorData(
        name, static_cast<TfLiteType>(t_tensor_type.ElementType()),
        num_elements, low_range, high_range);
  }

  TfLiteStatus PrepareInputData() override {
    int index = 0;
    for (auto& buffer : *input_buffers_) {
      auto t_data =
          CreateRandomTensorData(buffer, "input_" + std::to_string(index));
      auto res = buffer.Write<char>(absl::MakeSpan(
          reinterpret_cast<char*>(t_data.data.get()), t_data.bytes));
      if (!res.HasValue()) {
        LITERT_LOG(LITERT_ERROR, "PrepareInputData: %s",
                   res.Error().Message().c_str());
        return kTfLiteError;
      }

      ++index;
    }
    return kTfLiteOk;
  }

  TfLiteStatus ResetInputsAndOutputs() override {
    if (profiler_) {
      profiler_->StopProfiling();
      profiler_->GetProfileSummary(compiled_model_->Get());
      profiler_->Reset();
      profiler_->StartProfiling();
    }
    return kTfLiteOk;
  }
  std::vector<tflite::Flag> GetFlags() override {
    std::vector<tflite::Flag> flags = BenchmarkModel::GetFlags();
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "graph", &params_, "The path to the model file."));
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "signature_to_run_for", &params_, "The signature to run for."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "use_cpu", &params_, "Whether to use CPU accelerator."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "use_gpu", &params_, "Whether to use GPU accelerator."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "use_npu", &params_, "Whether to use NPU accelerator."));
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "dispatch_library_path", &params_, "Dispatch library path."));
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "compiler_plugin_library_path", &params_,
        "Compiler plugin library path. Only for JIT compilation."));
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "compiler_cache_path", &params_,
        "Compiler plugin cache path, used to store JIT-compiled models."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "require_full_delegation", &params_,
        "Whether to require full delegation."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "use_profiler", &params_, "Whether to use profiler."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "enable_perfetto", &params_, "Whether to enable perfetto tracing."));
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "gpu_backend", &params_,
        "GPU backend to use when using GPU accelerator."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "allow_fp16", &params_, "Whether to allow FP16."));
    flags.push_back(tflite::benchmark::CreateFlag<bool>(
        "gpu_low_priority", &params_,
        "Whether to use low priority for GPU accelerator."));
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "result_file_path", &params_,
        "Path to save the benchmark result in binary proto format."));
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "model_runtime_info_output_file", &params_,
        "Path to save the model runtime info in binary proto format."));
    flags.push_back(tflite::benchmark::CreateFlag<std::string>(
        "mediatek_nerun_pilot_version", &params_,
        "Which version of the MediaTek NPU SDK to use."));
    return flags;
  }

 protected:
  virtual TfLiteStatus LoadModel();
  std::unique_ptr<Model> model_;

 private:
  std::unique_ptr<litert::Environment> environment_;
  std::unique_ptr<litert::CompiledModel> compiled_model_;
  std::unique_ptr<std::vector<litert::TensorBuffer>> input_buffers_;
  std::unique_ptr<std::vector<litert::TensorBuffer>> output_buffers_;
  std::unique_ptr<litert::Profiler> profiler_;
  std::unique_ptr<BenchmarkLoggingListener> log_output_;
  std::unique_ptr<ModelRuntimeInfoListener> model_runtime_info_listener_;

  // TFLite Interpreter is needed for run_summarizer_
  ::tflite::Interpreter* interpreter_ = nullptr;
};

}  // namespace benchmark
}  // namespace litert

#endif  // ODML_LITERT_LITERT_TOOLS_BENCHMARK_LITERT_MODEL_H_
