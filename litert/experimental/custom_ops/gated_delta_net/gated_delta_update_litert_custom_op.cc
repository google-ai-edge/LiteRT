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
#include "litert/cc/litert_element_type.h"
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
  mode_ = 0;
  block_.reset();
  if (init_data != nullptr && init_data_size > 0) {
    const flexbuffers::Map fb_map =
        flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(init_data),
                             init_data_size)
            .AsMap();
    if (!fb_map["mode"].IsNull()) {
      mode_ = fb_map["mode"].AsInt32();
    }
    if (!fb_map["num_k_heads"].IsNull() &&
        fb_map["num_k_heads"].AsInt32() > 0) {
      GatedDeltaBlockParams params;
      params.num_k_heads = fb_map["num_k_heads"].AsInt32();
      if (!fb_map["num_v_heads"].IsNull()) {
        params.num_v_heads = fb_map["num_v_heads"].AsInt32();
      }
      if (!fb_map["head_k_dim"].IsNull()) {
        params.head_k_dim = fb_map["head_k_dim"].AsInt32();
      }
      if (!fb_map["head_v_dim"].IsNull()) {
        params.head_v_dim = fb_map["head_v_dim"].AsInt32();
      }
      if (!fb_map["eps"].IsNull()) {
        params.eps = fb_map["eps"].AsFloat();
      }
      if (!fb_map["norm_eps"].IsNull()) {
        params.norm_eps = fb_map["norm_eps"].AsFloat();
      }
      if (params.num_v_heads < params.num_k_heads ||
          params.num_v_heads % params.num_k_heads != 0 ||
          params.head_k_dim <= 0 || params.head_v_dim <= 0) {
        return Unexpected(Status::kErrorInvalidArgument,
                          "Invalid GatedDeltaUpdate block attributes");
      }
      block_ = params;
    }
  }
  return {};
}

Expected<void> GatedDeltaUpdateCustomOpKernel::GetOutputLayouts(
    const std::vector<Layout>& input_layouts,
    std::vector<Layout>& output_layouts) {
  if ((input_layouts.size() != 6 && input_layouts.size() != 7) ||
      output_layouts.size() != 2) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Invalid number of arguments for GatedDeltaUpdate");
  }
  if (block_.has_value()) {
    auto conv_dims = input_layouts[0].Dimensions();
    if (conv_dims.size() != 3) {
      return Unexpected(Status::kErrorInvalidArgument,
                        "conv_out rank must be 3 for GatedDeltaUpdate block");
    }
    const int V_dim = block_->num_v_heads * block_->head_v_dim;
    output_layouts[0] = Layout(Dimensions({conv_dims[0], conv_dims[1], V_dim}));
    output_layouts[1] = input_layouts[5];
    return {};
  }
  output_layouts[0] = input_layouts[2];  // v_t shape
  output_layouts[1] = input_layouts[5];  // rec_state shape
  return {};
}

Expected<void> GatedDeltaUpdateCustomOpKernel::Run(
    const std::vector<TensorBuffer>& inputs,
    std::vector<TensorBuffer>& outputs) {
  if ((inputs.size() != 6 && inputs.size() != 7) || outputs.size() != 2) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Invalid tensor count for GatedDeltaUpdate");
  }

  if (block_.has_value()) {
    const int H_k = block_->num_k_heads;
    const int H_v = block_->num_v_heads;
    const int D_k = block_->head_k_dim;
    const int D_v = block_->head_v_dim;
    const int K_dim = H_k * D_k;
    const int V_dim = H_v * D_v;

    LITERT_ASSIGN_OR_RETURN(auto conv_type, inputs[0].TensorType());
    LITERT_ASSIGN_OR_RETURN(auto zba_type, inputs[1].TensorType());
    LITERT_ASSIGN_OR_RETURN(auto a_log_type, inputs[2].TensorType());
    LITERT_ASSIGN_OR_RETURN(auto dt_bias_type, inputs[3].TensorType());
    LITERT_ASSIGN_OR_RETURN(auto norm_w_type, inputs[4].TensorType());
    LITERT_ASSIGN_OR_RETURN(auto rec_type, inputs[5].TensorType());

    auto conv_dims = conv_type.Layout().Dimensions();
    auto zba_dims = zba_type.Layout().Dimensions();
    auto rec_dims = rec_type.Layout().Dimensions();
    if (conv_dims.size() != 3 || zba_dims.size() != 3 || rec_dims.size() != 4) {
      return Unexpected(Status::kErrorInvalidArgument,
                        "Invalid input ranks for GatedDeltaUpdate block");
    }
    const int B = conv_dims[0];
    const int T = conv_dims[1];
    const int conv_dim = 2 * K_dim + V_dim;
    const int zba_dim = V_dim + 2 * H_v;
    if (conv_dims[2] != conv_dim || zba_dims[0] != B || zba_dims[1] != T ||
        (zba_dims[2] != zba_dim && zba_dims[2] != conv_dim + zba_dim) ||
        rec_dims[0] != B || rec_dims[1] != H_v || rec_dims[2] != D_k ||
        rec_dims[3] != D_v) {
      return Unexpected(Status::kErrorInvalidArgument,
                        "Invalid tensor shapes for GatedDeltaUpdate block");
    }
    LITERT_ASSIGN_OR_RETURN(size_t a_log_elems,
                            a_log_type.Layout().NumElements());
    LITERT_ASSIGN_OR_RETURN(size_t dt_bias_elems,
                            dt_bias_type.Layout().NumElements());
    LITERT_ASSIGN_OR_RETURN(size_t norm_w_elems,
                            norm_w_type.Layout().NumElements());
    if (a_log_elems != static_cast<size_t>(H_v) ||
        dt_bias_elems != static_cast<size_t>(H_v) ||
        norm_w_elems != static_cast<size_t>(D_v)) {
      return Unexpected(
          Status::kErrorInvalidArgument,
          "Invalid parameter tensor sizes for GatedDeltaUpdate block");
    }

    std::vector<float> mask_vec;
    const float* mask_ptr = nullptr;
    if (inputs.size() == 7) {
      LITERT_ASSIGN_OR_RETURN(auto mask_type, inputs[6].TensorType());
      LITERT_ASSIGN_OR_RETURN(size_t mask_elems,
                              mask_type.Layout().NumElements());
      if (mask_elems != static_cast<size_t>(B * T)) {
        return Unexpected(
            Status::kErrorInvalidArgument,
            "Invalid valid_mask tensor size for GatedDeltaUpdate block");
      }
      mask_vec.resize(B * T);
      if (mask_type.ElementType() == ElementType::Float32) {
        LITERT_ASSIGN_OR_RETURN(auto mask_lock,
                                TensorBufferScopedLock::Create<const float>(
                                    inputs[6], TensorBuffer::LockMode::kRead));
        for (int i = 0; i < B * T; ++i) {
          mask_vec[i] = mask_lock.second[i];
        }
      } else if (mask_type.ElementType() == ElementType::Int32) {
        LITERT_ASSIGN_OR_RETURN(auto mask_lock,
                                TensorBufferScopedLock::Create<const int32_t>(
                                    inputs[6], TensorBuffer::LockMode::kRead));
        for (int i = 0; i < B * T; ++i) {
          mask_vec[i] = static_cast<float>(mask_lock.second[i]);
        }
      } else if (mask_type.ElementType() == ElementType::Bool) {
        LITERT_ASSIGN_OR_RETURN(auto mask_lock,
                                TensorBufferScopedLock::Create<const bool>(
                                    inputs[6], TensorBuffer::LockMode::kRead));
        for (int i = 0; i < B * T; ++i) {
          mask_vec[i] = mask_lock.second[i] ? 1.0f : 0.0f;
        }
      } else {
        return Unexpected(
            Status::kErrorInvalidArgument,
            "Unsupported valid_mask element type for GatedDeltaUpdate block");
      }
      mask_ptr = mask_vec.data();
    }

    LITERT_ASSIGN_OR_RETURN(auto conv_lock,
                            TensorBufferScopedLock::Create<const float>(
                                inputs[0], TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(auto zba_lock,
                            TensorBufferScopedLock::Create<const float>(
                                inputs[1], TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(auto a_log_lock,
                            TensorBufferScopedLock::Create<const float>(
                                inputs[2], TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(auto dt_bias_lock,
                            TensorBufferScopedLock::Create<const float>(
                                inputs[3], TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(auto norm_w_lock,
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

    ComputeGatedDeltaUpdateBlock(
        conv_lock.second, zba_lock.second, a_log_lock.second,
        dt_bias_lock.second, norm_w_lock.second, rec_lock.second, mask_ptr,
        out_lock.second, new_rec_lock.second, B, T, *block_, zba_dims[2]);
    return {};
  }

  LITERT_ASSIGN_OR_RETURN(auto q_type, inputs[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto v_type, inputs[2].TensorType());
  auto q_dims = q_type.Layout().Dimensions();
  auto v_dims = v_type.Layout().Dimensions();
  if (q_dims.size() < 4 || v_dims.size() < 4) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Input rank must be >= 4 for GatedDeltaUpdate");
  }
  const int B = v_dims[0];
  const int H = v_dims[1];
  const int H_k = q_dims[1];
  const int N = v_dims[2];
  const int D_k = q_dims[3];
  const int D_v = v_dims[3];

  std::vector<int> valid_len_vec;
  const int* valid_len_ptr = nullptr;
  if (inputs.size() == 7) {
    LITERT_ASSIGN_OR_RETURN(auto valid_len_type, inputs[6].TensorType());
    LITERT_ASSIGN_OR_RETURN(size_t num_elems,
                            valid_len_type.Layout().NumElements());
    if (num_elems != static_cast<size_t>(B)) {
      return Unexpected(Status::kErrorInvalidArgument,
                        "Invalid valid_len tensor size for GatedDeltaUpdate");
    }
    valid_len_vec.resize(B);
    if (valid_len_type.ElementType() == ElementType::Int32) {
      LITERT_ASSIGN_OR_RETURN(auto valid_len_lock,
                              TensorBufferScopedLock::Create<const int32_t>(
                                  inputs[6], TensorBuffer::LockMode::kRead));
      for (int b = 0; b < B; ++b) {
        valid_len_vec[b] = valid_len_lock.second[b];
      }
    } else if (valid_len_type.ElementType() == ElementType::Float32) {
      LITERT_ASSIGN_OR_RETURN(auto valid_len_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  inputs[6], TensorBuffer::LockMode::kRead));
      for (int b = 0; b < B; ++b) {
        valid_len_vec[b] = static_cast<int>(valid_len_lock.second[b]);
      }
    } else {
      return Unexpected(
          Status::kErrorInvalidArgument,
          "Unsupported valid_len element type for GatedDeltaUpdate");
    }
    valid_len_ptr = valid_len_vec.data();
  }

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
  ComputeGatedDeltaUpdateRecurrent(
      q_lock.second, k_lock.second, v_lock.second, beta_lock.second,
      g_lock.second, rec_lock.second, out_lock.second, new_rec_lock.second, B,
      H, N, D_k, D_v, H_k, valid_len_ptr);

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
