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
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

absl::Status XnnpackRunner::CreateRuntime(size_t num_threads) {
  xnn_subgraph_t sg = static_cast<XnnpackGraph&>(*graph_).GetSubgraph();
  xnn_runtime* raw_runtime = nullptr;
  LRT_TENSOR_RETURN_IF_ERROR(
      XnnStatusToAbsl(xnn_create_runtime_v3(sg, weights_cache_, threadpool_,
                                            /*flags=*/0, &raw_runtime),
                      "xnn_create_runtime_v3"));
  runtime_.reset(raw_runtime);
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
  return XnnStatusToAbsl(xnn_invoke_runtime(runtime_.get()),
                         "xnn_invoke_runtime");
}

}  // namespace litert::tensor
