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

#include "tensor/runners/ynnpack/runner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/ynnpack/graph.h"
#include "tensor/backends/ynnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

int YnnpackScheduler::NumThreads(void* self) {
  return reinterpret_cast<YnnpackScheduler*>(self)->pool_.thread_count();
}

void YnnpackScheduler::Schedule(void* self, void* context,
                                void (*task)(void* context)) {
  YnnpackScheduler* scheduler = reinterpret_cast<YnnpackScheduler*>(self);
  scheduler->pool_.enqueue([task, context]() { (*task)(context); });
}

const ynn_scheduler* YnnpackScheduler::Vtable() {
  static const ynn_scheduler kScheduler = {NumThreads, Schedule};
  return &kScheduler;
}

absl::Status YnnpackRunner::CreateRuntime(size_t num_threads) {
  if (scheduler_ != nullptr && threadpool_ == nullptr) {
    ynn_threadpool_t raw_threadpool = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(YnnStatusToAbsl(
        ynn_create_threadpool(YnnpackScheduler::Vtable(), scheduler_.get(),
                              /*flags=*/0, &raw_threadpool),
        "ynn_create_threadpool"));
    threadpool_.reset(raw_threadpool);
  }

  ynn_subgraph_t sg = static_cast<YnnpackGraph&>(*graph_).GetSubgraph();
  LRT_TENSOR_RETURN_IF_ERROR(
      YnnStatusToAbsl(ynn_optimize_subgraph(sg, threadpool_.get(), /*flags=*/0),
                      "ynn_optimize_subgraph"));

  ynn_runtime* raw_runtime = nullptr;
  LRT_TENSOR_RETURN_IF_ERROR(YnnStatusToAbsl(
      ynn_create_runtime(sg, threadpool_.get(), /*flags=*/0, &raw_runtime),
      "ynn_create_runtime"));
  runtime_.reset(raw_runtime);
  return absl::OkStatus();
}

absl::Status YnnpackRunner::SetExternalValueShape(
    uint32_t id, absl::Span<const size_t> dims) {
  return YnnStatusToAbsl(
      ynn_set_external_value_shape(runtime_.get(), id, dims.size(),
                                   dims.empty() ? nullptr : dims.data()),
      "ynn_set_external_value_shape");
}

absl::Status YnnpackRunner::ReshapeRuntime() {
  return YnnStatusToAbsl(ynn_reshape_runtime(runtime_.get()),
                         "ynn_reshape_runtime");
}

absl::Status YnnpackRunner::GetExternalValueShape(uint32_t id,
                                                  std::vector<size_t>& dims) {
  // `rank` is an in/out parameter: on input it is the capacity of `shape_arr`.
  size_t rank = YNN_MAX_TENSOR_RANK;
  std::array<size_t, YNN_MAX_TENSOR_RANK> shape_arr{};
  LRT_TENSOR_RETURN_IF_ERROR(YnnStatusToAbsl(
      ynn_get_external_value_shape(runtime_.get(), id, &rank, shape_arr.data()),
      "ynn_get_external_value_shape"));
  dims.assign(shape_arr.begin(), shape_arr.begin() + rank);
  return absl::OkStatus();
}

absl::Status YnnpackRunner::SetupExternalValues(
    absl::Span<NnpackValue> values,
    const absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>>&
        external_buffers,
    std::vector<LockedBufferSpan<const std::byte>>& locks) {
  locks.reserve(values.size());
  for (NnpackValue& value : values) {
    if (value.flags == 0) {
      continue;
    }
    const auto it = external_buffers.find(value.id);
    if (it == external_buffers.end() || it->second == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrFormat("External value %u missing host buffer", value.id));
    }
    LockedBufferSpan<const std::byte> lock = it->second->Lock();
    if (lock.data() == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrFormat("External value %u could not be locked", value.id));
    }
    LRT_TENSOR_RETURN_IF_ERROR(YnnStatusToAbsl(
        ynn_set_external_value_data(runtime_.get(), value.id,
                                    const_cast<std::byte*>(lock.data())),
        "ynn_set_external_value_data"));
    locks.push_back(std::move(lock));
  }
  return absl::OkStatus();
}

absl::Status YnnpackRunner::InvokeRuntime() {
  return YnnStatusToAbsl(ynn_invoke_runtime(runtime_.get()),
                         "ynn_invoke_runtime");
}

}  // namespace litert::tensor
