/* Copyright 2025 Google LLC.

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

#include "tensor/runners/xnnpack/runner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

// `LRT_XNN_PROFILE=1` logs a per-operator time summary of each runtime's first
// invocation. Read once.
bool IsXnnProfilingEnabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("LRT_XNN_PROFILE");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

void PrintXnnProfileSummary(xnn_runtime_t rt) {
  size_t num_ops = 0;
  size_t required = 0;
  if (xnn_get_runtime_profiling_info(rt, xnn_profile_info_num_operators,
                                     sizeof(num_ops), &num_ops,
                                     &required) != xnn_status_success ||
      num_ops == 0) {
    return;
  }
  std::vector<uint64_t> timings(num_ops, 0);
  if (xnn_get_runtime_profiling_info(rt, xnn_profile_info_operator_timing,
                                     timings.size() * sizeof(uint64_t),
                                     timings.data(),
                                     &required) != xnn_status_success) {
    return;
  }
  size_t names_size = 0;
  xnn_get_runtime_profiling_info(rt, xnn_profile_info_operator_name, 0, nullptr,
                                 &names_size);
  std::vector<char> names_buf(names_size, 0);
  if (xnn_get_runtime_profiling_info(rt, xnn_profile_info_operator_name,
                                     names_buf.size(), names_buf.data(),
                                     &required) != xnn_status_success) {
    return;
  }
  struct OperatorStats {
    size_t count = 0;
    uint64_t total_us = 0;
  };
  // The names point into `names_buf`, which outlives both containers.
  absl::flat_hash_map<absl::string_view, OperatorStats> by_name;
  uint64_t total_us = 0;
  const char* name_ptr = names_buf.data();
  for (size_t i = 0; i < num_ops; ++i) {
    const absl::string_view name(name_ptr);
    name_ptr += name.size() + 1;
    OperatorStats& stats = by_name[name];
    stats.count += 1;
    stats.total_us += timings[i];
    total_us += timings[i];
  }
  std::vector<std::pair<absl::string_view, OperatorStats>> sorted(
      by_name.begin(), by_name.end());
  absl::c_sort(sorted, [](const auto& a, const auto& b) {
    return a.second.total_us > b.second.total_us;
  });
  std::string summary =
      absl::StrFormat("XNNPACK profile (%zu ops, total %.3f ms):", num_ops,
                      total_us / 1000.0);
  for (const auto& [name, stats] : sorted) {
    absl::StrAppendFormat(
        &summary,
        "\n  %-52s count=%4zu  total=%8.3f ms (%5.1f%%)  avg=%6.1f us", name,
        stats.count, stats.total_us / 1000.0,
        total_us > 0 ? (100.0 * stats.total_us / total_us) : 0.0,
        static_cast<double>(stats.total_us) / stats.count);
  }
  ABSL_LOG(INFO) << summary;
}

}  // namespace

absl::Status XnnpackRunner::CreateRuntime() {
  auto& xnn_graph = static_cast<XnnpackGraph&>(*graph_);
  xnn_subgraph_t sg = xnn_graph.GetSubgraph();
  xnn_runtime* raw_runtime = nullptr;
  const uint32_t flags = IsXnnProfilingEnabled() ? XNN_FLAG_BASIC_PROFILING : 0;
  LRT_TENSOR_RETURN_IF_ERROR(XnnStatusToAbsl(
      xnn_create_runtime_v4(sg, weights_cache_, workspace_, threadpool_, flags,
                            &raw_runtime),
      "xnn_create_runtime_v4"));
  runtime_.reset(raw_runtime);
  // Blockwise FP16 scale buffers are only consumed when packing weights during
  // `xnn_create_runtime_v4`; release them now so multi-signature runtimes do
  // not hold duplicate host copies per pass.
  xnn_graph.fp16_buffers().clear();
  xnn_graph.fp16_buffers().shrink_to_fit();
  return absl::OkStatus();
}

absl::Status XnnpackRunner::SetExternalValueShape(
    uint32_t id, absl::Span<const size_t> dims) {
  return XnnStatusToAbsl(
      xnn_reshape_external_value(runtime_.get(), id, dims.size(),
                                 dims.empty() ? nullptr : dims.data()),
      "xnn_reshape_external_value");
}

absl::Status XnnpackRunner::ReshapeRuntime() {
  return XnnStatusToAbsl(xnn_reshape_runtime(runtime_.get()),
                         "xnn_reshape_runtime");
}

absl::Status XnnpackRunner::GetExternalValueShape(uint32_t id,
                                                  std::vector<size_t>& dims) {
  size_t num_dims = 0;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> shape_arr{};
  LRT_TENSOR_RETURN_IF_ERROR(
      XnnStatusToAbsl(xnn_get_external_value_shape(runtime_.get(), id,
                                                   &num_dims, shape_arr.data()),
                      "xnn_get_external_value_shape"));
  dims.assign(shape_arr.begin(), shape_arr.begin() + num_dims);
  return absl::OkStatus();
}

absl::Status XnnpackRunner::SetupExternalValues(
    absl::Span<NnpackValue> values,
    const absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>>&
        external_buffers,
    std::vector<LockedBufferSpan<const std::byte>>& locks) {
  std::vector<xnn_external_value> externals;
  externals.reserve(values.size());
  locks.reserve(values.size());
  for (NnpackValue& value : values) {
    if (value.flags == 0) {
      continue;
    }
    auto it = external_buffers.find(value.id);
    if (it == external_buffers.end() || it->second == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrFormat("External value %u missing host buffer", value.id));
    }
    LockedBufferSpan<const std::byte> lock = it->second->Lock();
    if (lock.data() == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrFormat("External value %u could not be locked", value.id));
    }
    externals.push_back(
        {.id = value.id, .data = const_cast<std::byte*>(lock.data())});
    locks.push_back(std::move(lock));
  }
  return XnnStatusToAbsl(
      xnn_setup_runtime_v2(runtime_.get(), externals.size(), externals.data()),
      "xnn_setup_runtime_v2");
}

absl::Status XnnpackRunner::InvokeRuntime() {
  LRT_TENSOR_RETURN_IF_ERROR(XnnStatusToAbsl(
      xnn_invoke_runtime(runtime_.get()), "xnn_invoke_runtime"));
  if (IsXnnProfilingEnabled() && !profile_logged_) {
    profile_logged_ = true;
    PrintXnnProfileSummary(runtime_.get());
  }
  return absl::OkStatus();
}

}  // namespace litert::tensor
