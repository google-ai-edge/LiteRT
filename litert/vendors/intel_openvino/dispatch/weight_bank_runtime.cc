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

#include "litert/vendors/intel_openvino/dispatch/weight_bank_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/intel_openvino/compiler/global_graph.h"

namespace litert::openvino {

litert::Expected<std::vector<BoundWeight>> GpuSharedBank::Bind(
    ov::Core& core, const OpenVinoGlobalGraph& global_graph,
    const ov::CompiledModel& compiled_model,
    const std::map<std::string, uint32_t>& const_map) {
  ov::RemoteContext ctx = core.get_default_context("GPU");

  // Hold the lock across the whole body: the first partition allocates+fills the
  // pool, later partitions read base_/offset_. Serializes concurrent partitions
  // driving the same model (this bank is owned per device context / per model).
  absl::MutexLock lock(&gpu_bank_mutex_);
  if (!bank_ready_) {
    size_t total = 0;
    for (const auto& [buffer_id, bytes] : global_graph.buffers)
      total += bytes.size();
    // Allocate the pool as one usm-host buffer of bytes; per-weight views
    // reinterpret slices as the weight's element type below.
    gpu_usm_ = ctx.create_tensor(
        ov::element::u8, ov::Shape{total},
        {{ov::intel_gpu::shared_mem_type.name(),
          ov::intel_gpu::SharedMemType::USM_HOST_BUFFER}});
    base_ = gpu_usm_.get_params()
                .at(ov::intel_gpu::mem_handle.name())
                .as<void*>();
    auto* base = static_cast<uint8_t*>(base_);
    size_t off = 0;
    // TODO(PR #8745 #7): recheck USM view alignment.
    for (const auto& [buffer_id, bytes] : global_graph.buffers) {  // ascending
      std::memcpy(base + off, bytes.data(), bytes.size());
      weight_offset_[buffer_id] = off;
      off += bytes.size();
    }
    bank_ready_ = true;
    LITERT_LOG(LITERT_INFO,
               "GlobalGraph: allocated shared usm-host bank (%zu bytes)", total);
  }

  auto* base = static_cast<uint8_t*>(base_);
  std::vector<BoundWeight> bound;
  const auto& inputs = compiled_model.inputs();
  for (size_t p = 0; p < inputs.size(); ++p) {
    // Match by the input's friendly_name (set at compile when the weight was
    // promoted to a Parameter), not by port index -- import_model may reorder
    // inputs, so positional keying is unsafe.
    const auto it = const_map.find(inputs[p].get_node()->get_friendly_name());
    if (it == const_map.end()) {
      continue;  // real activation input, not a shared weight
    }
    const auto off_it = weight_offset_.find(it->second);
    if (off_it == weight_offset_.end()) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "GlobalGraph: const_map buffer_id not in pool");
    }
    // Wrap the slice of the shared usm-host buffer as a remote tensor view of the
    // weight's own element type (zero-copy: same USM pointer, no reorder).
    ov::RemoteTensor view = ctx.create_tensor(
        inputs[p].get_element_type(), inputs[p].get_shape(),
        {{ov::intel_gpu::shared_mem_type.name(),
          ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
         {ov::intel_gpu::mem_handle.name(),
          static_cast<ov::intel_gpu::gpu_handle_param>(base + off_it->second)}});
    bound.push_back(BoundWeight{p, std::move(view)});
  }
  return bound;
}

}  // namespace litert::openvino
