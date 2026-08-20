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

#include "tensor/examples/gemma4/perfetto_session.h"

#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/utils/macros.h"
#include "perfetto/tracing/backend_type.h"  // from @perfetto
#include "perfetto/tracing/core/data_source_config.h"  // from @perfetto
#include "perfetto/tracing/core/trace_config.h"  // from @perfetto  // IWYU pragma: keep
#include "perfetto/tracing/tracing.h"  // from @perfetto
#include "perfetto/tracing/track_event.h"  // from @perfetto

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace litert::tensor::examples::gemma4 {

PerfettoSession::PerfettoSession(absl::string_view output_path)
    : output_path_(output_path) {}

PerfettoSession::~PerfettoSession() { StopAndSave().IgnoreError(); }

absl::StatusOr<std::unique_ptr<PerfettoSession>> PerfettoSession::Create(
    absl::string_view output_path) {
  auto session =
      std::unique_ptr<PerfettoSession>(new PerfettoSession(output_path));
  LRT_TENSOR_RETURN_IF_ERROR(session->Initialize());
  return session;
}

absl::Status PerfettoSession::Initialize() {
  perfetto::TracingInitArgs args;
  args.backends |= perfetto::kInProcessBackend;
  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();

  if (!output_path_.empty()) {
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(65 * 1024);

    perfetto::DataSourceConfig* ds_cfg =
        cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");

    tracing_session_ = perfetto::Tracing::NewTrace();
    tracing_session_->Setup(cfg);
    tracing_session_->StartBlocking();
    ABSL_LOG(INFO) << "Perfetto trace session started. Trace will be saved to: "
                   << output_path_;
  }
  return absl::OkStatus();
}

absl::Status PerfettoSession::StopAndSave() {
  if (tracing_session_) {
    tracing_session_->StopBlocking();
    std::vector<char> trace_data = tracing_session_->ReadTraceBlocking();
    tracing_session_.reset();

    if (trace_data.empty()) {
      ABSL_LOG(WARNING) << "Perfetto trace data is empty!";
      return absl::OkStatus();
    }

    ABSL_LOG(INFO) << "Captured " << trace_data.size()
                   << " bytes of trace data.";
    std::ofstream ofs(output_path_, std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
      return absl::InternalError(
          absl::StrCat("Failed to open trace output file: ", output_path_));
    }
    ofs.write(trace_data.data(), trace_data.size());
    ofs.close();
    ABSL_LOG(INFO) << "Successfully saved Perfetto trace to: " << output_path_;
  }
  return absl::OkStatus();
}

}  // namespace litert::tensor::examples::gemma4
