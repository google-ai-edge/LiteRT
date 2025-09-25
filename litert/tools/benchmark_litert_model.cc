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
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_profiler.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tflite_error_status_builder.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/runtime/compiled_model.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

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
  }

  if (use_gpu) {
    hardware_accelerators |= LiteRtHwAccelerators::kLiteRtHwAcceleratorGpu;
    LITERT_ASSIGN_OR_ABORT(auto gpu_options, GpuOptions::Create());
    // Enable no external tensors mode.
    gpu_options.EnableNoExternalTensorsMode(/*enabled=*/true);
    // Enable benchmark mode to run clFinish() after each inference.
    gpu_options.EnableBenchmarkMode(/*enabled=*/true);
    if (gpu_backend == "webgpu") {
      gpu_options.SetGpuBackend(kLiteRtGpuBackendWebGpu);
    }
    if (allow_fp16 == false) {
      gpu_options.SetDelegatePrecision(kLiteRtDelegatePrecisionFp32);
    }
    if (gpu_low_priority) {
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

  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          dispatch_library_path.c_str(),
      },
      litert::Environment::Option{
          litert::Environment::OptionTag::CompilerPluginLibraryDir,
          compiler_plugin_library_path.c_str(),
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

  if (!params_.Get<std::string>("model_runtime_info_output_file").empty()) {
    ::tflite::Interpreter* interpreter_ptr = nullptr;
    LiteRtCompiledModelT* compiled_model_ptr = compiled_model_->Get();
    if (compiled_model_ptr == nullptr) {
      LITERT_LOG(LITERT_ERROR, "Compiled model is null");
      return kTfLiteError;
    }
    LITERT_ASSIGN_OR_RETURN(interpreter_ptr, GetInterpreter(compiled_model_ptr),
                            AsTfLiteStatus(_ << "Failed to get interpreter."));
    model_runtime_info_listener_ =
        std::make_unique<ModelRuntimeInfoListener>(interpreter_ptr);
    AddListener(model_runtime_info_listener_.get());
  }

  auto use_profiler = params_.Get<bool>("use_profiler");
  if (use_profiler) {
    LITERT_ASSIGN_OR_ABORT(profiler_, compiled_model_->GetProfiler());
    profiler_.StartProfiling();
  }

  auto signature = params_.Get<std::string>("signature_to_run_for");
  if (signature.empty()) {
    LITERT_ASSIGN_OR_RETURN(auto s, model_->GetSignature(0), AsTfLiteStatus(_));
    signature = s.Key();
  }

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
