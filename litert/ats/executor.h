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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_EXECUTOR_H_

// Type and implementations for executing the compiled model for ATS.
// Different ATS configurations may require different hardware accelerators
// and backend specific configurations, hence the need for polymorphism.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/inference_capture.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_compiled_model_next.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/test/simple_buffer.h"

namespace litert::testing {

// Base class for executing the compiled model for ATS.
class CompiledModelExecutor {
 public:
  using Ptr = std::unique_ptr<CompiledModelExecutor>;

  CompiledModelExecutor(CompiledModelExecutor&& other) = default;
  CompiledModelExecutor& operator=(CompiledModelExecutor&& other) = default;
  CompiledModelExecutor(const CompiledModelExecutor&) = delete;
  CompiledModelExecutor& operator=(const CompiledModelExecutor&) = delete;

  // Run the compiled model against the given model and with the given inputs.
  template <typename Iter>
  Expected<std::vector<SimpleBuffer>> Run(
      Iter start, Iter end, std::optional<Latency::Ref> stats = {}) {
    LITERT_ASSIGN_OR_RETURN(auto api_inputs, api_.CreateInputBuffers());
    if (api_inputs.size() != std::distance(start, end)) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   absl::StrFormat("Expected %d inputs, got %d",
                                   api_inputs.size(), api_inputs.size()));
    }
    for (auto it = start; it != end; ++it) {
      api_inputs[std::distance(start, it)].Write(it->template Span<uint8_t>());
    }
    LITERT_ASSIGN_OR_RETURN(auto api_outputs, api_.CreateOutputBuffers());

    auto time_start = Latency::Start(stats);
    LITERT_RETURN_IF_ERROR(api_.Run(api_inputs, api_outputs));
    Latency::Stop(stats, time_start);

    std::vector<SimpleBuffer> output_buffers;
    for (const auto& output : api_outputs) {
      LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                              SimpleBuffer::FromTensorBuffer(output));
      output_buffers.push_back(std::move(output_buffer));
    }
    return output_buffers;
  }

  template <typename Inputs>
  Expected<std::vector<SimpleBuffer>> Run(
      const Inputs& inputs, std::optional<Latency::Ref> stats = {}) {
    return Run(std::cbegin(inputs), std::cend(inputs), stats);
  }

  virtual ~CompiledModelExecutor() = default;

 protected:
  CompiledModelExecutor(CompiledModelNext&& api, Environment&& env)
      : env_(std::move(env)), api_(std::move(api)) {}

  // env_ must be destroyed after api_ to avoid use-after-free.
  Environment env_;
  CompiledModelNext api_;
};

// Executor for the CPU backend.
class CpuCompiledModelExecutor : public CompiledModelExecutor {
 public:
  CpuCompiledModelExecutor(CpuCompiledModelExecutor&& other) = default;
  CpuCompiledModelExecutor& operator=(CpuCompiledModelExecutor&& other) =
      default;
  CpuCompiledModelExecutor(const CpuCompiledModelExecutor&) = delete;
  CpuCompiledModelExecutor& operator=(const CpuCompiledModelExecutor&) = delete;

  struct Args {};

  static constexpr absl::string_view Name() { return "cpu"; }

  static Expected<CpuCompiledModelExecutor> Create(LiteRtModelT& model,
                                                   const Options& options) {
    // Setup options.
    LITERT_ASSIGN_OR_RETURN(
        auto env, litert::Environment::Create(litert::EnvironmentOptions({})));
    // Init compiled model api.
    LITERT_ASSIGN_OR_RETURN(
        auto api, CompiledModelNext::Create(
                      env, Model::CreateFromNonOwnedHandle(&model), options));

    return CpuCompiledModelExecutor(std::move(api), std::move(env));
  }

  static Expected<CpuCompiledModelExecutor> Create(LiteRtModelT& model,
                                                   const Options& options,
                                                   const Args& args) {
    return Create(model, options);
  }

 private:
  CpuCompiledModelExecutor(CompiledModelNext&& api, Environment&& env)
      : CompiledModelExecutor(std::move(api), std::move(env)) {}
};

// Executor for the NPU backend.
class NpuCompiledModelExecutor : public CompiledModelExecutor {
 public:
  NpuCompiledModelExecutor(NpuCompiledModelExecutor&& other) = default;
  NpuCompiledModelExecutor& operator=(NpuCompiledModelExecutor&& other) =
      default;
  NpuCompiledModelExecutor(const NpuCompiledModelExecutor&) = delete;
  NpuCompiledModelExecutor& operator=(const NpuCompiledModelExecutor&) = delete;

  struct Args {
    std::string dispatch_dir;
    std::optional<std::string> plugin_dir;
  };

  static constexpr absl::string_view Name() { return "npu"; }

  static Expected<NpuCompiledModelExecutor> Create(LiteRtModelT& model,
                                                   const Options& options,
                                                   const Args& args) {
    return Create(model, options, args.dispatch_dir, args.plugin_dir);
  }

  static Expected<NpuCompiledModelExecutor> Create(
      LiteRtModelT& model, const Options& options,
      const std::string& dispatch_dir,
      const std::optional<std::string>& plugin_dir = std::nullopt) {
    std::vector<litert::EnvironmentOptions::Option> environment_options = {
        litert::EnvironmentOptions::Option{
            litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
            absl::string_view(dispatch_dir),
        }};

    if (plugin_dir) {
      environment_options.push_back(litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          absl::string_view(*plugin_dir),
      });
      environment_options.push_back(litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerCacheDir,
          // TODO: Make this configurable.
          "/data/local/tmp/litert_compiler_cache",
      });
    }

    LITERT_ASSIGN_OR_RETURN(
        auto env, Environment::Create(EnvironmentOptions(environment_options)));
    LITERT_ASSIGN_OR_RETURN(
        auto api, CompiledModelNext::Create(
                      env, Model::CreateFromNonOwnedHandle(&model), options));

    return NpuCompiledModelExecutor(std::move(api), std::move(env));
  }

 private:
  NpuCompiledModelExecutor(CompiledModelNext&& api, Environment&& env)
      : CompiledModelExecutor(std::move(api), std::move(env)) {}
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_EXECUTOR_H_
