// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include <optional>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_dispatch.h"
#if LITERT_HAS_DARWINN_OPTIONS_SUPPORT
#include "litert/vendors/google_tensor/dispatch/darwinn_options.h"
#endif  // LITERT_HAS_DARWINN_OPTIONS_SUPPORT
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

// This class is thread-compatible.
class LiteRtDispatchDeviceContextT {
 public:
  static LiteRtStatus Create(LiteRtDispatchDeviceContext& device_context);

  LiteRtStatus Destroy();

  LiteRtStatus RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer,
      LiteRtTensorBufferHandle& tensor_buffer_handle);

  LiteRtStatus UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

  LiteRtStatus LoadExecutable(LiteRtDispatchExecutableType type,
                              const LiteRtMemBuffer& bytecode_buffer,
                              LiteRtDispatchExecutableHandle& exec_handle);

  LiteRtStatus UnloadExecutable(LiteRtDispatchExecutableHandle exec_handle);

  // Registers a graph with the device context.
  //
  // This has the effect of guaranteeing that the device context remains alive
  // until the graph is unregistered, meaning that a subsequent call to
  // `UnregisterGraph` is required to permit the device context to be destroyed.
  //
  // NOTE: a graph may only be registered once.
  LiteRtStatus RegisterGraph(LiteRtDispatchGraph graph);

  LiteRtStatus UnregisterGraph(LiteRtDispatchGraph graph);

  ThrContext* absl_nonnull thr_context() { return thr_context_; }

#if LITERT_HAS_DARWINN_OPTIONS_SUPPORT
  std::optional<litert::google_tensor::DarwinnOptionsData>& darwinn_options() {
    return darwinn_options_;
  }
#endif  // LITERT_HAS_DARWINN_OPTIONS_SUPPORT

 private:
  explicit LiteRtDispatchDeviceContextT(ThrContext* absl_nonnull thr_context)
      : thr_context_(thr_context) {}

  // Consumers of this class must use `Destroy` to delete the instance.
  ~LiteRtDispatchDeviceContextT() = default;

  ThrContext* absl_nonnull thr_context_;
#if LITERT_HAS_DARWINN_OPTIONS_SUPPORT
  std::optional<litert::google_tensor::DarwinnOptionsData> darwinn_options_;
#endif  // LITERT_HAS_DARWINN_OPTIONS_SUPPORT
  // A device context cannot be destroyed with any registered graphs.
  absl::flat_hash_set<LiteRtDispatchGraph> registered_graphs_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
