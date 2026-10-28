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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/intel_openvino/dispatch/device_context.h"

class LiteRtDispatchDeviceContextT;

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;
  using IntelOpenVinoOptions = ::litert::intel_openvino::IntelOpenVinoOptions;

  ~LiteRtDispatchInvocationContextT();

  static litert::Expected<Ptr> Create(
      LiteRtDispatchDeviceContextT& device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs,
      const IntelOpenVinoOptions* intel_openvino_opts);

  litert::Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
      const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<void> AttachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> AttachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);
  litert::Expected<void> DetachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
    // Nothing to do.
    return {};
  }

  litert::Expected<void> DetachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
    // Nothing to do.
    return {};
  }

  litert::Expected<void> Invoke();

  // Performs the deferred OpenVINO import (weights-bank staging +
  // core->import_model() + create_infer_request() + GPU weight binding) if it
  // has not happened yet, then replays any buffer attachments recorded before
  // the import. Idempotent and safe to call from any of the entry points that
  // need real OV state.
  //
  // Import is deferred out of Create() so that delegating a signature costs
  // nothing until that signature actually runs: a multi-context bundle
  // delegates every tier's signatures, but a session that never crosses its
  // first tier must never pay for the other tiers' blob imports.
  litert::Expected<void> EnsureImported();

  // Runs EnsureImported() on a background thread (once). Used to import a
  // tier's remaining nodes while its first node is already running.
  void StartBackgroundImport();

  // Drops the imported model: the infer request holds the only reference to
  // this node's ov::CompiledModel, so releasing it frees the imported blob and
  // its device-side state. The recorded buffer attachments are deliberately
  // KEPT -- the next EnsureImported() replays them onto the new infer request,
  // which is what makes a release transparent to the caller.
  //
  // The shared weights bank is unaffected: it is a per-model file mapping
  // owned by the device context, not by any one node.
  //
  // Never waits: returns false, and leaves the node imported, if it is busy --
  // in AttachInput/AttachOutput/Invoke on another thread (use_mutex_), or
  // mid-import (import_mutex_). Returns true if the node is not imported
  // afterwards, including when it never was.
  bool TryReleaseImported();

  bool IsImported() const { return imported_; }
  const std::string& FunctionName() const { return function_name_; }
  // KV sequence length this node was compiled for (its context tier), as
  // recorded by the compiler in the GlobalGraph. 0 if none (no KV inputs, or
  // not a GlobalGraph container).
  int64_t KvTier() const { return kv_tier_; }

  void SetSchedulingInfo(const LiteRtSchedulingInfo* scheduling_info) {
    if (scheduling_info == nullptr) {
      scheduling_info_ = std::nullopt;
      return;
    }
    scheduling_info_ = *scheduling_info;
  }

  const LiteRtSchedulingInfo* GetSchedulingInfo() const {
    return scheduling_info_.has_value() ? &scheduling_info_.value() : nullptr;
  }

 private:
  LiteRtDispatchInvocationContextT(LiteRtDispatchDeviceContextT& device_context,
                                   const LiteRtMemBuffer& exec_bytecode_buffer,
                                   std::string function_name, int64_t kv_tier,
                                   int num_inputs, int num_outputs)
      : device_context_(device_context),
        kv_tier_(kv_tier),
        exec_bytecode_buffer_(exec_bytecode_buffer),
        function_name_(std::move(function_name)) {}

  // `foreground` = called by the kernel (AttachInput/AttachOutput/Invoke),
  // not by a background prefetch: only the foreground releases other tiers and
  // starts prefetches.
  litert::Expected<void> EnsureImported(bool foreground);

  // Must be called with |import_mutex_| held.
  litert::Expected<void> ImportLocked()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(import_mutex_);
  void ReleaseImportedLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(import_mutex_);

  // Bind one already-resolved buffer handle onto the live infer request.
  // Requires the import to have happened; takes no lock.
  litert::Expected<void> BindInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);
  litert::Expected<void> BindOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  LiteRtDispatchDeviceContextT& device_context_;
  // See KvTier(); fixed at Create().
  const int64_t kv_tier_;
  ov::InferRequest infer_request_;
  // Views into the shared usm-host weight buffer bound to weight-Parameter
  // inputs; held for the infer request's lifetime (set_input_tensor does not
  // take ownership).
  std::vector<ov::Tensor> bound_weights_;

  // The import inputs, kept verbatim so EnsureImported() can do the whole job
  // later. Nothing derived from them is cached: the container parse and device
  // resolution are cheap next to the import itself, and re-deriving them at
  // import time is what keeps this class from accumulating a second,
  // half-initialized copy of the model's description.
  //
  // LIFETIME: |exec_bytecode_buffer_.base_addr| points into the model
  // allocation (LiteRtCompiledModelT::model_buf_ / fb_model_, or the
  // caller-owned LiteRtModelT it points into). That is declared BEFORE interp_
  // in LiteRtCompiledModelT, so the interpreter -- and with it every
  // invocation context -- is destroyed first. Referencing it past Create() is
  // therefore safe, but it does mean the owning litert Model must stay alive
  // for the session; do not relax that.
  LiteRtMemBuffer exec_bytecode_buffer_;
  std::string function_name_;
  // Written under import_mutex_; also read without it, as a cheap hint.
  std::atomic<bool> imported_{false};
  size_t imported_bytes_ ABSL_GUARDED_BY(import_mutex_) = 0;
  absl::Mutex import_mutex_;
  // Held shared while the infer request is in use (AttachInput, AttachOutput,
  // Invoke), so another thread's tier crossing cannot release it mid-use.
  absl::Mutex use_mutex_;
  // Background import started by StartBackgroundImport(); waited on in the
  // destructor.
  absl::Mutex prefetch_mutex_;
  std::future<void> prefetch_ ABSL_GUARDED_BY(prefetch_mutex_);
  // Every buffer the kernel attached to this node. LiteRT's
  // DispatchDelegateKernel attaches a node's buffers only once, on the node's
  // first run. If the node is later released to free memory and imported
  // again, the kernel will not attach them a second time, so we re-bind them
  // from this record.
  std::vector<std::pair<int, LiteRtTensorBufferHandle>> attached_inputs_;
  std::vector<std::pair<int, LiteRtTensorBufferHandle>> attached_outputs_;

  // Timeout is in milliseconds
  static constexpr int kInferRequestTimeoutMs = 10000;

  std::optional<LiteRtSchedulingInfo> scheduling_info_;
#if defined(__ANDROID__)
  void* ctx = nullptr;
#endif
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
