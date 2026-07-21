// Copyright (C) 2026 Intel Corporation
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

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_WEIGHT_BANK_RUNTIME_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_WEIGHT_BANK_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/intel_openvino/compiler/global_graph.h"

namespace litert::openvino {

// Binds one weight-Parameter to a view into the shared buffer.
struct BoundWeight {
  size_t input_index;  // port on the compiled model
  ov::Tensor view;     // view into the shared usm-host buffer (zero-copy)
};

// One shared usm-host buffer holding a model's deduplicated weight pool, owned
// by the model's device context (so the pool is allocated once and shared by
// the prefill and decode partitions, but distinct across models). The first
// partition to Bind() allocates and fills the buffer; later partitions bind
// views into it. Thread-safe: Bind() serializes concurrent partitions.
class GpuSharedBank {
 public:
  GpuSharedBank() = default;
  ~GpuSharedBank() = default;
  GpuSharedBank(const GpuSharedBank&) = delete;
  GpuSharedBank& operator=(const GpuSharedBank&) = delete;
  GpuSharedBank(GpuSharedBank&&) = delete;
  GpuSharedBank& operator=(GpuSharedBank&&) = delete;

  // Allocates the pool once (on first call) from |global_graph|.buffers, then
  // returns a zero-copy view into it for each of |compiled_model|'s
  // weight-Parameters named in |const_map| (friendly_name -> BufferId). Weights
  // are matched by the Parameter's friendly_name, so binding is robust to input
  // reordering across import_model. The caller sets each view on the infer
  // request and keeps the views alive for its lifetime.
  litert::Expected<std::vector<BoundWeight>> Bind(
      ov::Core& core, const OpenVinoGlobalGraph& global_graph,
      const ov::CompiledModel& compiled_model,
      const std::map<std::string, uint32_t>& const_map);

 private:
  absl::Mutex gpu_bank_mutex_;
  ov::RemoteTensor gpu_usm_ ABSL_GUARDED_BY(gpu_bank_mutex_);  // usm-host buffer for the pool
  void* base_ ABSL_GUARDED_BY(gpu_bank_mutex_) = nullptr;  // usm host pointer into usm_
  // buffer_id -> byte offset in the pool.
  std::map<uint32_t, size_t> weight_offset_ ABSL_GUARDED_BY(gpu_bank_mutex_);
  bool bank_ready_ ABSL_GUARDED_BY(gpu_bank_mutex_) = false;
};

}  // namespace litert::openvino

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_WEIGHT_BANK_RUNTIME_H_
