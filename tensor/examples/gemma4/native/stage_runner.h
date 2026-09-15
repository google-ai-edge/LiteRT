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

// Local experiment only: XNNPACK tensor stages borrowing one pool/cache/arena.
// This header adds a derived runner; it does not change existing library ABIs.
#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_STAGE_RUNNER_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_STAGE_RUNNER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <pthreadpool.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xnnpack.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "xnnpack/subgraph.h"

namespace litert::tensor {

// All stages may borrow the same caller-owned pool, weights cache, and workspace.
// The caller must destroy every StageRunner before destroying these resources.
// nullptr means serial execution / no weights cache / private workspace. The caller
// also controls cache finalization after preparing all runtimes that add weights.
// This class never changes pool/cache ownership or finalizes the cache. XNNPACK
// retains one workspace reference per runtime and grows its arena as needed.
// Stage preparation, complete Run calls, and cache population must be serialized.
// Internal values may not escape an invocation: another stage overwrites shared
// scratch. External outputs stay in separate NnpackRunner buffers, so they remain
// usable by later stages and survive workspace growth. On arena relocation,
// XNNPACK repairs all attached runtimes' internal pointers and operator setup.
//
// SetInput, ReadOutput, ReshapeInput, and Run retain NnpackRunner's contracts:
// non-copy inputs must remain live through Run; owning-output locks retain their
// storage, while caller-provided outputs retain the caller's lifetime obligation;
// external-value locks remain live through the synchronous runtime invocation.
class StageRunner final : public NnpackRunner {
 public:
  static absl::StatusOr<std::unique_ptr<StageRunner>> Create(
      std::vector<TensorHandle> outputs, pthreadpool_t pool,
      xnn_weights_cache_t weights, uint32_t runtime_flags = 0,
      xnn_workspace_t workspace = nullptr) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto graph,
                               BuildXnnpackGraph(std::move(outputs)));
    return std::unique_ptr<StageRunner>(
        new StageRunner(std::move(graph), pool, weights, runtime_flags,
                        workspace));
  }

  // runtime_ is destroyed before the NnpackRunner graph/buffers. Borrowed
  // handles have no deleters, and remain valid for other stages afterward.
  ~StageRunner() override = default;
  StageRunner(const StageRunner&) = delete;
  StageRunner& operator=(const StageRunner&) = delete;
  StageRunner(StageRunner&&) = delete;
  StageRunner& operator=(StageRunner&&) = delete;

  // Compatibility with the runner API, without mutating the borrowed pool.
  // A count other than the supplied pool's count causes runtime creation or Run
  // to return FailedPrecondition. Create already records the correct pool count.
  void SetNumThreads(size_t num_threads) override {
    NnpackRunner::SetNumThreads(num_threads);
  }

  pthreadpool_t threadpool() const { return threadpool_; }
  xnn_weights_cache_t weights_cache() const { return weights_cache_; }
  xnn_runtime_t runtime() const { return runtime_.get(); }
  uint32_t runtime_flags() const { return runtime_flags_; }

  // Local diagnostic accessors use this checkout's internal XNNPACK structs.
  // workspace_bytes() is the arena's retained high-watermark, not this stage's
  // current scratch requirement. Count an identical workspace() only once when
  // aggregating stages. A private workspace is null until PrepareRuntime().
  xnn_workspace_t workspace() const {
    return runtime_ == nullptr ? shared_workspace_ : runtime_->workspace;
  }
  size_t workspace_bytes() const {
    return workspace() == nullptr ? 0 : workspace()->size;
  }

  struct ExternalBufferMemory {
    const Buffer* buffer = nullptr;
    const void* data = nullptr;
    size_t bytes = 0;
    bool owns_storage = false;
    uint32_t flags = 0;
  };
  struct MemoryInfo {
    xnn_workspace_t workspace = nullptr;
    const void* workspace_data = nullptr;
    size_t workspace_bytes = 0;
    size_t owned_external_bytes = 0;
    size_t owned_output_bytes = 0;
    size_t viewed_external_bytes = 0;
    size_t graph_constant_capacity_bytes = 0;
    size_t graph_dequantized_capacity_bytes = 0;
    size_t graph_fp16_capacity_bytes = 0;
    std::vector<ExternalBufferMemory> external_buffers;
  };

  // These are host buffer payload/capacity counts, not allocator totals or RSS.
  // Buffer records are deduplicated within this runner and permit cross-stage
  // identity checks. Viewed bytes overlap producer outputs/model mappings and
  // must not be added to owned bytes. Source weights retained by graph values or
  // keep_alive_buffers are excluded; account for their shared backing separately.
  // Graph conversion vectors are owned here; capacity includes retained slack.
  absl::StatusOr<MemoryInfo> MemorySnapshot() const {
    MemoryInfo info;
    info.workspace = workspace();
    info.workspace_data = info.workspace == nullptr ? nullptr : info.workspace->data;
    info.workspace_bytes = workspace_bytes();
    absl::flat_hash_map<const Buffer*, size_t> seen;
    for (const NnpackValue& value : graph_->values()) {
      const auto it = external_buffers_.find(value.id);
      if (it == external_buffers_.end() || it->second == nullptr) continue;
      const Buffer* buffer = it->second.get();
      const auto existing = seen.find(buffer);
      if (existing != seen.end()) {
        info.external_buffers[existing->second].flags |= value.flags;
        continue;
      }
      LRT_TENSOR_ASSIGN_OR_RETURN(const size_t bytes, buffer->ByteSize());
      auto lock = it->second->Lock();
      seen.emplace(buffer, info.external_buffers.size());
      info.external_buffers.push_back({buffer, lock.data(), bytes,
          buffer->IsA(OwningCpuBuffer::TypeId()), value.flags});
    }
    for (const ExternalBufferMemory& buffer : info.external_buffers) {
      if (buffer.owns_storage) {
        info.owned_external_bytes += buffer.bytes;
        if (buffer.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) {
          info.owned_output_bytes += buffer.bytes;
        }
      } else {
        info.viewed_external_bytes += buffer.bytes;
      }
    }
    for (const auto& buffer : graph_->constant_buffers()) {
      info.graph_constant_capacity_bytes += buffer.capacity() * sizeof(char);
    }
    for (const auto& buffer : graph_->dequantized_buffers()) {
      info.graph_dequantized_capacity_bytes += buffer.capacity() * sizeof(float);
    }
    for (const auto& buffer : graph_->fp16_buffers()) {
      info.graph_fp16_capacity_bytes += buffer.capacity() * sizeof(fp16_t);
    }
    return info;
  }

 protected:
  uint32_t FlagExternalInput() const override {
    return XNN_VALUE_FLAG_EXTERNAL_INPUT;
  }
  uint32_t FlagExternalOutput() const override {
    return XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  absl::Status CreateRuntime(size_t /*num_threads*/) override {
    LRT_TENSOR_RETURN_IF_ERROR(CheckThreadCount());
    xnn_subgraph_t subgraph =
        static_cast<XnnpackGraph&>(*graph_).GetSubgraph();
    xnn_runtime_t raw_runtime = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(XnnStatusToAbsl(
        xnn_create_runtime_v4(subgraph, weights_cache_, shared_workspace_,
                              threadpool_, runtime_flags_, &raw_runtime),
        "xnn_create_runtime_v4"));
    runtime_.reset(raw_runtime);
    return absl::OkStatus();
  }

  absl::Status SetExternalValueShape(
      uint32_t id, absl::Span<const size_t> dims) override {
    return XnnStatusToAbsl(
        xnn_reshape_external_value(runtime_.get(), id, dims.size(),
                                   dims.empty() ? nullptr : dims.data()),
        "xnn_reshape_external_value");
  }

  absl::Status ReshapeRuntime() override {
    LRT_TENSOR_RETURN_IF_ERROR(CheckThreadCount());
    return XnnStatusToAbsl(xnn_reshape_runtime(runtime_.get()),
                           "xnn_reshape_runtime");
  }

  absl::Status GetExternalValueShape(
      uint32_t id, std::vector<size_t>& dims) override {
    size_t num_dims = 0;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> shape{};
    LRT_TENSOR_RETURN_IF_ERROR(XnnStatusToAbsl(
        xnn_get_external_value_shape(runtime_.get(), id, &num_dims,
                                      shape.data()),
        "xnn_get_external_value_shape"));
    dims.assign(shape.begin(), shape.begin() + num_dims);
    return absl::OkStatus();
  }

  absl::Status SetupExternalValues(
      absl::Span<NnpackValue> values,
      const absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>>& buffers,
      std::vector<LockedBufferSpan<const std::byte>>& locks) override {
    std::vector<xnn_external_value> externals;
    externals.reserve(values.size());
    locks.reserve(values.size());
    for (NnpackValue& value : values) {
      if (value.flags == 0) continue;
      const auto it = buffers.find(value.id);
      if (it == buffers.end() || it->second == nullptr) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "External value %u missing host buffer", value.id));
      }
      LockedBufferSpan<const std::byte> lock = it->second->Lock();
      if (lock.data() == nullptr) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "External value %u could not be locked", value.id));
      }
      xnn_external_value external{};
      external.id = value.id;
      external.data = const_cast<std::byte*>(lock.data());
      externals.push_back(external);
      locks.push_back(std::move(lock));
    }
    return XnnStatusToAbsl(
        xnn_setup_runtime_v2(runtime_.get(), externals.size(), externals.data()),
        "xnn_setup_runtime_v2");
  }

  absl::Status InvokeRuntime() override {
    return XnnStatusToAbsl(xnn_invoke_runtime(runtime_.get()),
                           "xnn_invoke_runtime");
  }

 private:
  struct RuntimeDeleter {
    void operator()(::xnn_runtime* runtime) const {
      if (runtime != nullptr) xnn_delete_runtime(runtime);
    }
  };

  StageRunner(std::unique_ptr<XnnpackGraph> graph, pthreadpool_t pool,
              xnn_weights_cache_t weights, uint32_t runtime_flags,
              xnn_workspace_t workspace)
      : NnpackRunner(std::move(graph)), threadpool_(pool),
        weights_cache_(weights), runtime_flags_(runtime_flags),
        shared_workspace_(workspace) {
    num_threads_ = BorrowedThreadCount();
  }

  size_t BorrowedThreadCount() const {
    return threadpool_ == nullptr ? 1 : pthreadpool_get_threads_count(threadpool_);
  }

  absl::Status CheckThreadCount() const {
    const size_t actual = BorrowedThreadCount();
    if (num_threads_ != actual) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "StageRunner borrows a %u-thread pool; requested %u threads. "
          "Configure the shared pool before creating stages.",
          actual, num_threads_));
    }
    return absl::OkStatus();
  }

  pthreadpool_t const threadpool_;          // Borrowed; never destroyed here.
  xnn_weights_cache_t const weights_cache_;  // Borrowed; never finalized here.
  uint32_t const runtime_flags_;
  xnn_workspace_t const shared_workspace_;  // Borrowed; runtime retains a ref.
  std::unique_ptr<::xnn_runtime, RuntimeDeleter> runtime_;
};

}  // namespace litert::tensor
#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_STAGE_RUNNER_H_
