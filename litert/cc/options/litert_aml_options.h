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

#ifndef ODML_LITERT_LITERT_CC_OPTIONS_LITERT_AML_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_OPTIONS_LITERT_AML_OPTIONS_H_

#include <memory>
#include <string>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_aml_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_concrete_options_base.h"

namespace litert::aml {

class AmlOptions : public ConcreteOptionsBase {
 public:
  AmlOptions() = delete;
  explicit AmlOptions(LrtAmlOptions options)
      : options_(options, LrtDestroyAmlOptions) {}

  AmlOptions(const AmlOptions&) = delete;
  AmlOptions& operator=(const AmlOptions&) = delete;
  AmlOptions(AmlOptions&&) = default;
  AmlOptions& operator=(AmlOptions&&) = default;
  ~AmlOptions() override = default;

  static const char* Discriminator() { return LrtAmlOptionsGetIdentifier(); }

  static Expected<AmlOptions> Create() {
    LrtAmlOptions options = nullptr;
    LITERT_RETURN_IF_ERROR(LrtCreateAmlOptions(&options));
    return AmlOptions(options);
  }

  LrtAmlOptions Get() const { return options_.get(); }

  LiteRtStatus GetOpaqueOptionsData(
      const char** identifier, void** payload,
      void (**payload_deleter)(void*)) const override {
    return LrtGetOpaqueAmlOptionsData(Get(), identifier, payload,
                                      payload_deleter);
  }

  Expected<void> SetLogLevel(LrtAmlOptionsLogLevel log_level) {
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsSetLogLevel(Get(), log_level));
    return {};
  }
  Expected<LrtAmlOptionsLogLevel> GetLogLevel() const {
    LrtAmlOptionsLogLevel log_level;
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsGetLogLevel(Get(), &log_level));
    return log_level;
  }

  Expected<void> SetPerformanceMode(LrtAmlOptionsPerfMode perf_mode) {
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsSetPerfMode(Get(), perf_mode));
    return {};
  }
  Expected<LrtAmlOptionsPerfMode> GetPerformanceMode() const {
    LrtAmlOptionsPerfMode perf_mode;
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsGetPerfMode(Get(), &perf_mode));
    return perf_mode;
  }

  Expected<void> SetMemoryPolicy(LrtAmlOptionsMemoryPolicy policy) {
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsSetMemoryPolicy(Get(), policy));
    return {};
  }
  Expected<LrtAmlOptionsMemoryPolicy> GetMemoryPolicy() const {
    LrtAmlOptionsMemoryPolicy policy;
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsGetMemoryPolicy(Get(), &policy));
    return policy;
  }

  Expected<void> SetModelPath(const std::string& model_path) {
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsSetModelPath(Get(), model_path.c_str()));
    return {};
  }
  Expected<const char*> GetModelPath() const {
    const char* model_path = nullptr;
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsGetModelPath(Get(), &model_path));
    return model_path;
  }

  Expected<void> SetModelName(const std::string& model_name) {
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsSetModelName(Get(), model_name.c_str()));
    return {};
  }
  Expected<const char*> GetModelName() const {
    const char* model_name = nullptr;
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsGetModelName(Get(), &model_name));
    return model_name;
  }

  Expected<void> SetProfiling(LrtAmlOptionsProfiling profiling) {
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsSetProfiling(Get(), profiling));
    return {};
  }
  Expected<LrtAmlOptionsProfiling> GetProfiling() const {
    LrtAmlOptionsProfiling profiling;
    LITERT_RETURN_IF_ERROR(LrtAmlOptionsGetProfiling(Get(), &profiling));
    return profiling;
  }

 private:
  std::unique_ptr<LrtAmlOptionsT, void (*)(LrtAmlOptions)> options_;
};

}  // namespace litert::aml

#endif  // ODML_LITERT_LITERT_CC_OPTIONS_LITERT_AML_OPTIONS_H_
