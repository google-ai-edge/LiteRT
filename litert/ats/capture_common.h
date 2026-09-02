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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_CAPTURE_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_CAPTURE_COMMON_H_

#include <cstddef>
#include <string>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/ats/common.h"
#include "litert/ats/configure.h"
#include "litert/ats/print.h"
#include "litert/c/litert_any.h"
#include "litert/cc/internal/litert_compiled_model_next.h"
#include "litert/core/model/model.h"

namespace litert::testing {

// Information about the input model.
struct ModelDetail : Printable<std::string, std::string, bool> {
  // File name, if in memory only graph, an identifier of the graph.
  std::string name = "";
  // Optional description or representation of the model.
  std::string desc = "";
  // Was the input model precompiled offline?
  bool precompiled = false;

  void SetFields(const TestNames& names, const LiteRtModelT& model) {
    name = names.report_id;
    desc = names.desc;
    precompiled = GetBuildStamp(model).has_value();
  }

  ModelDetail() : Printable("ModelDetail", "name", "desc", "precompiled") {}

 private:
  Fields GetFields() const override {
    return Fields{name, absl::StrFormat("\"%s\"", desc), precompiled};
  }
};

// Information about the accelerator used if any.
struct AcceleratorDetail
    : Printable<ExecutionBackend, std::string, std::string> {
  // The type of accelerator used.
  ExecutionBackend a_type = ExecutionBackend::kCpu;

  // Only applicable in the NPU case.
  std::string soc_man = "n/a";
  std::string soc_model = "n/a";

  void SetFields(const AtsConf& conf) {
    a_type = conf.Backend();
    if (conf.IsNpu()) {
      soc_man = conf.SocManufacturer();
      soc_model = conf.SocModel();
    }
  }

  AcceleratorDetail()
      : Printable("AcceleratorDetail", "backend", "soc_man", "soc_model") {}

 private:
  Fields GetFields() const override {
    return Fields{a_type, soc_man, soc_model};
  }
};

// Information about any compilation that was done.
struct CompilationDetail : Printable<CompilationStatus> {
  // The status of the compilation.
  CompilationStatus status = CompilationStatus::kNotRequested;

  CompilationDetail() : Printable("CompilationDetail", "status") {}

  void SetFields(const AtsConf& conf, const LiteRtModelT& model, bool error) {
    if (!conf.IsNpu()) {
      return;
    }
    if (error) {
      status = CompilationStatus::kError;
    } else if (!internal::HasAnyCompiled(model)) {
      status = CompilationStatus::kNoOpsCompiled;
    } else if (!internal::IsFullyCompiled(model)) {
      status = CompilationStatus::kPartiallyCompiled;
    } else {
      status = CompilationStatus::kFullyCompiled;
    }
  }

 private:
  Fields GetFields() const override { return Fields{status}; }
};

// Information about GPU/accelerator intermediate working memory and constant
// weights.
struct MemoryDetail : Printable<size_t, size_t> {
  // Intermediate working memory allocated on GPU in kilobytes.
  size_t gpu_intermediate_memory_kb = 0;

  // Constant weight memory allocated on GPU in kilobytes.
  size_t gpu_constant_memory_kb = 0;

  MemoryDetail()
      : Printable("MemoryDetail", "gpu_intermediate_memory_kb",
                  "gpu_constant_memory_kb") {}

  void SetFields(const CompiledModelNext::Metrics& metrics) {
    for (const auto& metric : metrics.metrics) {
      if (metric.value.type == kLiteRtAnyTypeInt) {
        if (metric.name == "gpu_intermediate_memory_bytes") {
          gpu_intermediate_memory_kb =
              static_cast<size_t>(metric.value.int_value / 1024);
        } else if (metric.name == "gpu_constant_memory_bytes") {
          gpu_constant_memory_kb =
              static_cast<size_t>(metric.value.int_value / 1024);
        }
      }
    }
  }

  void SetFields() {}

 private:
  Fields GetFields() const override {
    return Fields{gpu_intermediate_memory_kb, gpu_constant_memory_kb};
  }
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_CAPTURE_COMMON_H_
