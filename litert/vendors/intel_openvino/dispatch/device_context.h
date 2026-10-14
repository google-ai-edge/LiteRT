// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#ifndef LITERT_WINDOWS_OS
#include <sys/mman.h>
#endif

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/intel_openvino/dispatch/openvino_shared_core.h"
#include "litert/vendors/intel_openvino/dispatch/weight_bank_runtime.h"

class LiteRtDispatchInvocationContextT;

class LiteRtDispatchDeviceContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchDeviceContextT>;

  ~LiteRtDispatchDeviceContextT() = default;
  static litert::Expected<Ptr> Create(
      const LiteRtRuntimeContext* runtime_context);
  litert::Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer);

  litert::Expected<void> UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<ov::Tensor> getOVTensor(
      const LiteRtTensorBufferHandle& handle) const {
    absl::MutexLock lock(&tensor_handle_mutex_);
    auto it = tensor_handle_map_.find(handle);
    if (it != tensor_handle_map_.end()) {
      return it->second.tensor;
    } else {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Failed to get Remote Tensor");
    }
  }

  // Return the core shared_pointer.
  std::shared_ptr<ov::Core> getCore() const {
    return OpenVINOSharedCore::GetInstance()->getCore();
  }

  const LiteRtRuntimeContext* runtime_context() const {
    return runtime_context_;
  }

  // --- Deferred-import eviction -------------------------------------------
  //
  // Every dispatch node imports its own blob on first use and can drop it
  // again (LiteRtDispatchInvocationContextT::EnsureImported /
  // ReleaseImported). The device context is the only place that sees all of a
  // model's nodes at once, so the release decision lives here.
  //
  // The case this exists for: a multi-context bundle crosses from its 4096
  // tier to its 16383 tier. The 4096 tier's nodes are never used again, but
  // nothing says so -- the executor just stops invoking them, and no detach or
  // unregister reaches the delegate.
  //
  // The rule is therefore the simplest one that needs no such signal: a node
  // about to import takes the slot, and the nodes of other tiers give theirs
  // up. No recency, no thresholds, no bookkeeping -- "release the others" is
  // decided entirely by the import that is already happening.
  //
  // A node that is busy -- running, or mid-import -- is skipped and stays
  // imported: the release never waits for it.
  void RegisterInvocationContext(LiteRtDispatchInvocationContextT* context);
  void UnregisterInvocationContext(LiteRtDispatchInvocationContextT* context);

  // Releases the imported models of every other tier's nodes. Called from
  // EnsureImported BEFORE the import, because the host peak is set while the
  // old tier's blob and the new one are both resident -- a release that runs
  // after the fact cannot lower it. Must NOT be called while holding an
  // invocation context's import lock.
  void ReleaseOthersForImport(LiteRtDispatchInvocationContextT* incoming);

  // After `imported` finished importing, start background imports of the
  // other not-yet-imported nodes of the SAME tier.
  void PrefetchSameTier(LiteRtDispatchInvocationContextT* imported);

  // The tier of the last foreground import (0 before any). A background import
  // that finishes for another tier releases itself: the crossing that moved
  // away from its tier skipped it while it was busy importing.
  int64_t ActiveTier() const { return active_tier_.load(); }

 private:
  class CleanupAction {
   public:
    CleanupAction() = default;
    explicit CleanupAction(std::function<void()> action)
        : action_(std::move(action)) {}
    ~CleanupAction() {
      if (action_) {
        action_();
      }
    }
    CleanupAction(const CleanupAction&) = delete;
    CleanupAction& operator=(const CleanupAction&) = delete;
    CleanupAction(CleanupAction&& other) : action_(std::move(other.action_)) {
      other.action_ = nullptr;
    }
    CleanupAction& operator=(CleanupAction&& other) {
      if (this != &other) {
        if (action_ != nullptr) {
          action_();
        }
        action_ = std::move(other.action_);
        other.action_ = nullptr;
      }
      return *this;
    }

   private:
    std::function<void()> action_;
  };

  struct RegisteredTensor {
    ov::Tensor tensor;
    CleanupAction cleanup;
  };

  // Inserts `tensor` under a freshly allocated handle while holding
  // `tensor_handle_mutex_`, and returns that handle.
  LiteRtTensorBufferHandle InsertTensor(RegisteredTensor tensor) {
    absl::MutexLock lock(&tensor_handle_mutex_);
    LiteRtTensorBufferHandle handle = (LiteRtTensorBufferHandle)next_handle_;
    tensor_handle_map_.emplace(handle, std::move(tensor));
    ++next_handle_;
    return handle;
  }

  explicit LiteRtDispatchDeviceContextT(
      const LiteRtRuntimeContext* runtime_context)
      : runtime_context_(runtime_context), next_handle_(0) {}
  const LiteRtRuntimeContext* runtime_context_;
  // Guards `tensor_handle_map_` and `next_handle_`, which are mutated by
  // Register/Unregister and read by getOVTensor from potentially concurrent
  // host threads. `mutable` so the const getOVTensor accessor can lock it.
  mutable absl::Mutex tensor_handle_mutex_;
  std::unordered_map<LiteRtTensorBufferHandle, RegisteredTensor>
      tensor_handle_map_ ABSL_GUARDED_BY(tensor_handle_mutex_);
  uint64_t next_handle_ ABSL_GUARDED_BY(tensor_handle_mutex_);
  // Per-model shared weight pools. Each guards its own state internally, so no
  // external locking is needed here. Held by value rather than lazily
  // allocated, so concurrent partitions creating their invocation contexts
  // cannot race on the allocation itself.
  litert::openvino::GpuSharedBank gpu_shared_bank_;
  litert::openvino::NpuSharedBank npu_shared_bank_;

  // Every invocation context created against this device context, so a node
  // about to import can see the others. Raw pointers: the contexts unregister
  // themselves in their destructor.
  std::atomic<int64_t> active_tier_{0};
  mutable absl::Mutex invocation_contexts_mutex_;
  std::vector<LiteRtDispatchInvocationContextT*> invocation_contexts_
      ABSL_GUARDED_BY(invocation_contexts_mutex_);

 public:
  // Bound at dispatch by the invocation context; see GpuSharedBank::Bind and
  // NpuSharedBank::EnsureOnDisk.
  litert::openvino::GpuSharedBank& GpuBank() { return gpu_shared_bank_; }
  litert::openvino::NpuSharedBank& NpuBank() { return npu_shared_bank_; }
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
