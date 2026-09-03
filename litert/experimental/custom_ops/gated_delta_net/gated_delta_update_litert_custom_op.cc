// Copyright 2026 Google LLC.
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

#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_litert_custom_op.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_impl.h"

namespace litert {
namespace gated_delta_net {

// ============================================================================
// 1. TrilInv Custom Op
// ============================================================================

const std::string& TrilInvCustomOpKernel::OpName() const { return kOpName; }

Expected<void> TrilInvCustomOpKernel::GetOutputLayouts(
    const std::vector<Layout>& input_layouts,
    std::vector<Layout>& output_layouts) {
  if (input_layouts.size() != 1 || output_layouts.size() != 1) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Invalid number of arguments for TrilInv");
  }
  output_layouts[0] = input_layouts[0];
  return {};
}

Expected<void> TrilInvCustomOpKernel::Run(
    const std::vector<TensorBuffer>& inputs,
    std::vector<TensorBuffer>& outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Invalid tensor count for TrilInv");
  }
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, inputs[0].TensorType());
  auto dims = tensor_type.Layout().Dimensions();
  if (dims.size() < 2) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Input rank must be >= 2 for TrilInv");
  }
  const int C = dims[dims.size() - 1];
  if (dims[dims.size() - 2] != C) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Inner matrix must be square for TrilInv");
  }
  LITERT_ASSIGN_OR_RETURN(size_t total_elements,
                          tensor_type.Layout().NumElements());

  LITERT_ASSIGN_OR_RETURN(auto in_lock,
                          TensorBufferScopedLock::Create<const float>(
                              inputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto out_lock,
                          TensorBufferScopedLock::Create<float>(
                              outputs[0], TensorBuffer::LockMode::kWrite));

  ComputeTrilInv(in_lock.second, out_lock.second, total_elements, C);
  return {};
}

// ============================================================================
// 2. GatedDeltaUpdate Custom Op
// ============================================================================

const std::string& GatedDeltaUpdateCustomOpKernel::OpName() const {
  return kOpName;
}

Expected<void> GatedDeltaUpdateCustomOpKernel::Init(const void* init_data,
                                                    size_t init_data_size) {
  if (init_data != nullptr && init_data_size > 0) {
    const flexbuffers::Map fb_map =
        flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(init_data),
                             init_data_size)
            .AsMap();
    if (!fb_map["mode"].IsNull()) {
      mode_ = fb_map["mode"].AsInt32();
    }
  }
  return {};
}

Expected<void> GatedDeltaUpdateCustomOpKernel::GetOutputLayouts(
    const std::vector<Layout>& input_layouts,
    std::vector<Layout>& output_layouts) {
  if (input_layouts.size() != 6 || output_layouts.size() != 2) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Invalid number of arguments for GatedDeltaUpdate");
  }
  output_layouts[0] = input_layouts[2];  // v_t shape
  output_layouts[1] = input_layouts[5];  // rec_state shape
  return {};
}

Expected<void> GatedDeltaUpdateCustomOpKernel::Run(
    const std::vector<TensorBuffer>& inputs,
    std::vector<TensorBuffer>& outputs) {
  if (inputs.size() != 6 || outputs.size() != 2) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Invalid tensor count for GatedDeltaUpdate");
  }
  LITERT_ASSIGN_OR_RETURN(auto q_type, inputs[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto v_type, inputs[2].TensorType());
  auto q_dims = q_type.Layout().Dimensions();
  auto v_dims = v_type.Layout().Dimensions();
  if (q_dims.size() < 4 || v_dims.size() < 4) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Input rank must be >= 4 for GatedDeltaUpdate");
  }
  const int B = q_dims[0];
  const int H = q_dims[1];
  const int N = q_dims[2];
  const int D_k = q_dims[3];
  const int D_v = v_dims[3];

  LITERT_ASSIGN_OR_RETURN(auto q_lock,
                          TensorBufferScopedLock::Create<const float>(
                              inputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto k_lock,
                          TensorBufferScopedLock::Create<const float>(
                              inputs[1], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto v_lock,
                          TensorBufferScopedLock::Create<const float>(
                              inputs[2], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto beta_lock,
                          TensorBufferScopedLock::Create<const float>(
                              inputs[3], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto g_lock,
                          TensorBufferScopedLock::Create<const float>(
                              inputs[4], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto rec_lock,
                          TensorBufferScopedLock::Create<const float>(
                              inputs[5], TensorBuffer::LockMode::kRead));

  LITERT_ASSIGN_OR_RETURN(auto out_lock,
                          TensorBufferScopedLock::Create<float>(
                              outputs[0], TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(auto new_rec_lock,
                          TensorBufferScopedLock::Create<float>(
                              outputs[1], TensorBuffer::LockMode::kWrite));

  // Dispatch to only recurrent implementation for now
  ComputeGatedDeltaUpdateRecurrent(q_lock.second, k_lock.second, v_lock.second,
                                   beta_lock.second, g_lock.second,
                                   rec_lock.second, out_lock.second,
                                   new_rec_lock.second, B, H, N, D_k, D_v);

  return {};
}

Expected<void> RegisterGatedDeltaNetCustomOps(Options& options) {
  static auto* const gdn_update_custom_op_kernel =
      new GatedDeltaUpdateCustomOpKernel();
  static auto* const tril_inv_custom_op_kernel =
      new TrilInvCustomOpKernel();
  LITERT_RETURN_IF_ERROR(
      options.AddCustomOpKernel(*gdn_update_custom_op_kernel));
  LITERT_RETURN_IF_ERROR(
      options.AddCustomOpKernel(*tril_inv_custom_op_kernel));
  return {};
}

}  // namespace gated_delta_net
}  // namespace litert
