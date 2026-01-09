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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

class LiteRtDispatchDeviceContextT {
 public:
  static LiteRtStatus Create(LiteRtDispatchDeviceContext& device_context);

  ~LiteRtDispatchDeviceContextT();

  LiteRtStatus RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer,
      LiteRtTensorBufferHandle& tensor_buffer_handle);

  LiteRtStatus UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

  LiteRtStatus CreateGraph(LiteRtDispatchGraph& graph);

  LiteRtStatus DestroyGraph(LiteRtDispatchGraph graph);

  LiteRtStatus LoadExecutable(LiteRtDispatchExecutableType type,
                              const LiteRtMemBuffer& bytecode_buffer,
                              LiteRtDispatchExecutableHandle& exec_handle);

  LiteRtStatus UnloadExecutable(LiteRtDispatchExecutableHandle exec_handle);

  ThrContext* thr_context() { return thr_context_.get(); }

  void add_graph(ThrGraph* graph) { thr_graphs_.insert(graph); }

 private:
  // Struct to store DarwiNN options for later application.
  struct DarwinnOptionsData {
    std::optional<uint32_t> inference_power_state;
    std::optional<uint32_t> inference_memory_power_state;
    std::optional<int8_t> inference_priority;
    bool atomic_inference = false;
    bool prefer_coherent = false;
  };

  static void ThrContextDeleter(ThrContext* thr_context);

  using ThrContextPtr = std::unique_ptr<ThrContext,
                                        decltype(&ThrContextDeleter)>;

  LiteRtDispatchDeviceContextT(
      ThrContextPtr thr_context,
      std::optional<DarwinnOptionsData> darwinn_options)
      : thr_context_(std::move(thr_context)),
        darwinn_options_(std::move(darwinn_options)) {}

  ThrContextPtr thr_context_;
  std::optional<DarwinnOptionsData> darwinn_options_;
  absl::flat_hash_set<ThrGraph*> thr_graphs_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
