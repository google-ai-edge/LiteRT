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
#include <cstdint>

#include "absl/strings/str_format.h"  // from @com_google_absl

namespace litert::testing {

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
void AbslStringify(Sink& sink, const Nanoseconds& ns) {
  absl::Format(&sink, "%e", ns);
}

}  // namespace litert::testing
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMMON_H_
