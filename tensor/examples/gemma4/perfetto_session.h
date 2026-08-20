/* Copyright 2026 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_PERFETTO_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_PERFETTO_SESSION_H_

#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "perfetto/tracing/track_event.h"  // from @perfetto
#include "perfetto/tracing/track_event_category_registry.h"  // from @perfetto

namespace litert::tensor::examples::gemma4 {

inline constexpr char kGemma4Category[] = "gemma4";

// Manages the lifecycle of an in-process Perfetto tracing session.
class PerfettoSession {
 public:
  // Creates and initializes a Perfetto tracing session. If output_path is
  // non-empty, tracing will begin immediately and will be written to
  // output_path upon StopAndSave() or destruction.
  static absl::StatusOr<std::unique_ptr<PerfettoSession>> Create(
      absl::string_view output_path);

  ~PerfettoSession();

  // Stops the tracing session and writes trace data to the configured output
  // path.
  absl::Status StopAndSave();

 private:
  explicit PerfettoSession(absl::string_view output_path);

  absl::Status Initialize();

  std::string output_path_;
  std::unique_ptr<perfetto::TracingSession> tracing_session_;
};

}  // namespace litert::tensor::examples::gemma4

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category(litert::tensor::examples::gemma4::kGemma4Category)
        .SetDescription("Gemma 4 End-to-End Runner Events"));

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_PERFETTO_SESSION_H_
