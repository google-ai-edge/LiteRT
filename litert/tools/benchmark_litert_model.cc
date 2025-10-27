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
#include "litert/tools/benchmark_litert_model.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/internal/litert_tflite_error_status_builder.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_profiler.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/runtime/compiled_model.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/profiling/profile_summarizer.h"

namespace litert::benchmark {
namespace {
using ::litert::CompiledModel;
using ::litert::Options;
using ::litert::RuntimeOptions;
using ::litert::TensorBuffer;

Options CreateCompiledModelOptions(const BenchmarkParams& params) {
  auto use_gpu = params.Get<bool>("use_gpu");
  auto use_npu = params.Get<bool>("use_npu");
  auto use_cpu = params.Get<bool>("use_cpu");
  auto gpu_backend = params.Get<std::string>("gpu_backend");
  auto allow_fp16 = params.Get<bool>("allow_fp16");
  auto gpu_low_priority = params.Get<bool>("gpu_low_priority");
  auto use_profiler = params.Get<bool>("use_profiler");
  auto require_full_delegation = params.Get<bool>("require_full_delegation");
  auto num_threads = params.Get<int>("num_threads");
  LITERT_ASSIGN_OR_ABORT(Options compilation_options,
                         litert::Options::Create());

  if (use_cpu && require_full_delegation) {
    LITERT_LOG(
        LITERT_ERROR,
        "Requesting full delegation and CPU acceleration are incompatible.");
    std::abort();
  }

  LiteRtHwAcceleratorSet hardware_accelerators = 0;

  if (use_npu) {
    hardware_accelerators |= LiteRtHwAccelerators::kLiteRtHwAcceleratorNpu;
    // QNN options
    LITERT_ASSIGN_OR_ABORT(auto qnn_opts,
                           ::litert::qualcomm::QualcommOptions::Create());
    qnn_opts.SetLogLevel(kLiteRtQualcommLogOff);
    qnn_opts.SetHtpPerformanceMode(kLiteRtQualcommHtpPerformanceModeBurst);
    qnn_opts.SetUseFoldReLU(true);
    qnn_opts.SetUseConvHMX(true);
    qnn_opts.SetUseHtpPreference(true);
    qnn_opts.SetOptimizationLevel(kHtpOptimizeForInferenceO3);
    compilation_options.AddOpaqueOptions(std::move(qnn_opts));
    // TODO(yunandrew): Add options for other NPU backends.
  }

  if (use_gpu) {
    hardware_accelerators |= LiteRtHwAccelerators::kLiteRtHwAcceleratorGpu;
    LITERT_ASSIGN_OR_ABORT(auto gpu_options, GpuOptions::Create());
    // Enable benchmark mode to run clFinish() after each inference.
    gpu_options.EnableBenchmarkMode(/*enabled=*/true);
    if (gpu_backend == "webgpu") {
      gpu_options.SetGpuBackend(kLiteRtGpuBackendWebGpu);
    } else if (gpu_backend == "opengl") {
      gpu_options.SetGpuBackend(kLiteRtGpuBackendOpenGl);
    }
    if (allow_fp16 == false) {
      gpu_options.SetDelegatePrecision(kLiteRtDelegatePrecisionFp32);
    }
    if (gpu_low_priority) {
      gpu_options.SetGpuPriority(kLiteRtGpuPriorityLow);
    }

    auto use_profiler = params.Get<bool>("use_profiler");
    if (use_profiler) {
      gpu_options.SetGpuPriority(kLiteRtGpuPriorityLow);
    }

    compilation_options.AddOpaqueOptions(std::move(gpu_options));
  }

  if (use_cpu || !require_full_delegation) {
    hardware_accelerators |= LiteRtHwAccelerators::kLiteRtHwAcceleratorCpu;

    if (num_threads > 0) {
      LITERT_ASSIGN_OR_ABORT(auto cpu_options, CpuOptions::Create());
      cpu_options.SetNumThreads(num_threads);
      compilation_options.AddOpaqueOptions(std::move(cpu_options));
    }
  }

  compilation_options.SetHardwareAccelerators(hardware_accelerators);

  if (use_profiler) {
    LITERT_ASSIGN_OR_ABORT(auto runtime_options, RuntimeOptions::Create());
    runtime_options.SetEnableProfiling(/*enabled=*/true);
    compilation_options.AddOpaqueOptions(std::move(runtime_options));
  }

  return compilation_options;
}

litert::Expected<Environment> CreateDefaultEnvironment(
    const BenchmarkParams& params) {
  if (!params.Get<bool>("use_npu")) {
    // If NPU is not used, we don't need to set the dispatch library directory.
    return litert::Environment::Create({});
  }
  auto dispatch_library_path = params.Get<std::string>("dispatch_library_path");
  LITERT_LOG(LITERT_INFO, "dispatch_library_path: %s",
             dispatch_library_path.c_str());
  auto compiler_plugin_library_path =
      params.Get<std::string>("compiler_plugin_library_path");
  LITERT_LOG(LITERT_INFO, "compiler_plugin_library_path: %s",
             compiler_plugin_library_path.c_str());
  auto compiler_cache_path = params.Get<std::string>("compiler_cache_path");
  LITERT_LOG(LITERT_INFO, "compiler_cache_path: %s",
             compiler_cache_path.c_str());

  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          dispatch_library_path.c_str(),
      },
      litert::Environment::Option{
          litert::Environment::OptionTag::CompilerPluginLibraryDir,
          compiler_plugin_library_path.c_str(),
      },
      litert::Environment::Option{
          litert::Environment::OptionTag::CompilerCacheDir,
          compiler_cache_path.c_str(),
      },
  };
  return litert::Environment::Create(absl::MakeConstSpan(environment_options));
}
}  // namespace

TfLiteStatus BenchmarkLiteRtModel::LoadModel() {
  std::string fd_or_graph_path = params_.Get<std::string>("graph");
  LITERT_LOG(LITERT_INFO, "Loading model from: %s", fd_or_graph_path.c_str());
  LITERT_ASSIGN_OR_RETURN(auto model_result,
                          litert::Model::CreateFromFile(fd_or_graph_path),
                          AsTfLiteStatus(_ << "Failed to load model."));
  model_ = std::make_unique<litert::Model>(std::move(model_result));
  return kTfLiteOk;
}

TfLiteStatus BenchmarkLiteRtModel::Init() {
  TF_LITE_ENSURE_STATUS(LoadModel());

  LITERT_ASSIGN_OR_RETURN(
      auto env_result, CreateDefaultEnvironment(params_),
      AsTfLiteStatus(_ << "Failed to create litert environment."));
  environment_ = std::make_unique<litert::Environment>(std::move(env_result));

  auto compilation_options = CreateCompiledModelOptions(params_);
  LITERT_ASSIGN_OR_RETURN(auto compiled_model_result,
                          litert::CompiledModel::Create(*environment_, *model_,
                                                        compilation_options),
                          AsTfLiteStatus(_ << "Failed to compile model."));

  compiled_model_ =
      std::make_unique<litert::CompiledModel>(std::move(compiled_model_result));

  LiteRtCompiledModelT* compiled_model_ptr = compiled_model_->Get();
  if (compiled_model_ptr == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Compiled model is null");
    return kTfLiteError;
  }
  LITERT_ASSIGN_OR_RETURN(interpreter_, GetInterpreter(compiled_model_ptr),
                          AsTfLiteStatus(_ << "Failed to get interpreter."));

  if (!params_.Get<std::string>("model_runtime_info_output_file").empty()) {
    model_runtime_info_listener_ =
        std::make_unique<ModelRuntimeInfoListener>(interpreter_);
    AddListener(model_runtime_info_listener_.get());
  }

  auto use_profiler = params_.Get<bool>("use_profiler");
  if (use_profiler) {
    run_summarizer_ = std::make_unique<tflite::profiling::ProfileSummarizer>();
    LITERT_ASSIGN_OR_ABORT(profiler_, compiled_model_->GetProfiler());
    profiler_.StartProfiling();
  }
  log_output_ =
      std::make_unique<BenchmarkLoggingListener>(run_summarizer_.get());
  AddListener(log_output_.get());

  auto signature = params_.Get<std::string>("signature_to_run_for");
  LITERT_ASSIGN_OR_RETURN(
      auto input_buffers_result, compiled_model_->CreateInputBuffers(signature),
      AsTfLiteStatus(_ << "Failed to create input buffer."));
  input_buffers_ = std::make_unique<std::vector<litert::TensorBuffer>>(
      std::move(input_buffers_result));

  LITERT_ASSIGN_OR_RETURN(
      auto output_buffers_result,
      compiled_model_->CreateOutputBuffers(signature),
      AsTfLiteStatus(_ << "Failed to create output buffer."));
  output_buffers_ = std::make_unique<std::vector<litert::TensorBuffer>>(
      std::move(output_buffers_result));

  return kTfLiteOk;
}
}  // namespace litert::benchmark
