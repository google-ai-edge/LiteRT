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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMMON_H_

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/core/model/model.h"

namespace litert::testing {

// Names, ids and descriptions for a given test.
struct TestNames {
  // Gtest suite name.
  std::string suite;
  // Gtest test name.
  std::string test;
  // Description of the test.
  std::string desc;
  // Identifier for the test in the report.
  std::string report_id;

  // Create using repr of ops as desc. Only use if the model has 1-ish ops.
  static TestNames Create(size_t test_id, absl::string_view family,
                          absl::string_view logic, const LiteRtModelT& graph) {
    auto suite = MakeSuite(test_id, family, logic);
    auto test = absl::StrFormat("%v", graph.Subgraph(0).Ops());
    auto desc = test;
    auto report_id = suite;
    return {suite, test, desc, report_id};
  }

  // Create with an explicit desc.
  static TestNames Create(size_t test_id, absl::string_view family,
                          absl::string_view logic, absl::string_view test,
                          absl::string_view desc = "") {
    auto suite = MakeSuite(test_id, family, logic);
    return {suite, std::string(logic), std::string(desc), std::string(test)};
  }

 private:
  static std::string MakeSuite(size_t test_id, absl::string_view family,
                               absl::string_view logic) {
    return absl::StrFormat("ats_%lu_%s_%s", test_id, family, logic);
  }
};

// The type of reference to use for validation.
enum class ReferenceType {
  kNone,
  // Standard CPU inference.
  kCpu,
  // Custom c++ reference.
  kCustom,
};

enum class RunStatus {
  // End never recorded.
  kUnknown,
  // The runs completed successfully.
  kOk,
  // The runs failed due to an error.
  kError,
  // The runs failed due to timeout.
  kTimeout,
};

enum class CompilationStatus {
  // End never recorded.
  kNotRequested,
  // The compilation failed due to an error.
  kError,
  // Compilation succeeded, but no ops were compiled.
  kNoOpsCompiled,
  // Compilation succeeded, not all ops were compiled.
  kPartiallyCompiled,
  // Compilation succeeded, all ops were compiled.
  kFullyCompiled,
};

// Timing related types.
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Nanoseconds = uint64_t;

// Which backend to use as the "actual".
enum class ExecutionBackend { kCpu, kGpu, kNpu };

// Printing.

template <typename Sink>
void AbslStringify(Sink& sink, const ExecutionBackend& backend) {
  switch (backend) {
    case ExecutionBackend::kCpu:
      sink.Append("cpu");
      break;
    case ExecutionBackend::kGpu:
      sink.Append("gpu");
      break;
    case ExecutionBackend::kNpu:
      sink.Append("npu");
      break;
  }
}

template <typename Sink>
void AbslStringify(Sink& sink, const ReferenceType& type) {
  switch (type) {
    case ReferenceType::kNone:
      sink.Append("none");
      break;
    case ReferenceType::kCpu:
      sink.Append("cpu");
      break;
    case ReferenceType::kCustom:
      sink.Append("custom");
      break;
  }
}

template <typename Sink>
void AbslStringify(Sink& sink, const RunStatus& status) {
  switch (status) {
    case RunStatus::kUnknown:
      sink.Append("unknown");
      break;
    case RunStatus::kOk:
      sink.Append("ok");
      break;
    case RunStatus::kError:
      sink.Append("error");
      break;
    case RunStatus::kTimeout:
      sink.Append("timeout");
      break;
  }
}

template <typename Sink>
void AbslStringify(Sink& sink, const CompilationStatus& status) {
  switch (status) {
    case CompilationStatus::kNotRequested:
      sink.Append("not_requested");
      break;
    case CompilationStatus::kError:
      sink.Append("error");
      break;
    case CompilationStatus::kNoOpsCompiled:
      sink.Append("no_ops_compiled");
      break;
    case CompilationStatus::kPartiallyCompiled:
      sink.Append("partially_compiled");
      break;
    case CompilationStatus::kFullyCompiled:
      sink.Append("fully_compiled");
      break;
  }
}

template <typename Sink>
void AbslStringify(Sink& sink, const Nanoseconds& ns) {
  absl::Format(&sink, "%e", ns);
}

}  // namespace litert::testing
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMMON_H_
