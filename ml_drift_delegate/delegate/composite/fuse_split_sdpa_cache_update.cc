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

#include "ml_drift_delegate/delegate/composite/fuse_split_sdpa_cache_update.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"

namespace litert::ml_drift {
namespace {

constexpr float kExpectedMaskFillValue = -10000.0f;
constexpr float kMaskFillTolerance = 1.0f;
// Softmax infinite-float capping clamps to +/- kMaxHalf (65504) or +/- FLT_MAX.
constexpr float kMinFloatCapMagnitude = ::ml_drift::kMaxHalf - 1.0f;

std::optional<float> GetScalarFloatFromConstAttr(const std::any& attr_any) {
  const auto* attr =
      std::any_cast<::ml_drift::ConstTensorAttributes>(&attr_any);
  if (attr == nullptr) {
    return std::nullopt;
  }
  if (const auto* t_f32 = std::get_if<::ml_drift::TensorFloat32>(&attr->tensor);
      t_f32 != nullptr && t_f32->data.size() == 1) {
    return t_f32->data[0];
  }
  if (const auto* t_f16 = std::get_if<::ml_drift::TensorFloat16>(&attr->tensor);
      t_f16 != nullptr && t_f16->data.size() == 1) {
    return static_cast<float>(t_f16->data[0]);
  }
  return std::nullopt;
}

std::optional<int> GetScalarIntFromConstAttr(const std::any& attr_any) {
  const auto* attr =
      std::any_cast<::ml_drift::ConstTensorAttributes>(&attr_any);
  if (attr == nullptr) {
    return std::nullopt;
  }
  if (const auto* t_i32 = std::get_if<::ml_drift::TensorInt32>(&attr->tensor);
      t_i32 != nullptr && t_i32->data.size() == 1) {
    return t_i32->data[0];
  }
  return std::nullopt;
}

std::optional<float> GetScalarFloatFromElementwiseAttr(
    const std::any& attr_any) {
  const auto* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&attr_any);
  if (attr == nullptr) {
    return std::nullopt;
  }
  const float* val = ::ml_drift::GetIfFloatScalar(&attr->param);
  if (val == nullptr) {
    return std::nullopt;
  }
  return *val;
}

std::optional<int> GetScalarIntFromElementwiseAttr(const std::any& attr_any) {
  const auto* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&attr_any);
  if (attr == nullptr) {
    return std::nullopt;
  }
  const auto* scalar = std::get_if<::ml_drift::ScalarValue>(&attr->param);
  if (scalar == nullptr) {
    return std::nullopt;
  }
  if (const int* v = std::get_if<int>(scalar); v != nullptr) {
    return *v;
  }
  if (const unsigned int* v = std::get_if<unsigned int>(scalar); v != nullptr) {
    return static_cast<int>(*v);
  }
  if (const float* v = std::get_if<float>(scalar); v != nullptr) {
    if (std::isfinite(*v) && std::floor(*v) == *v) {
      return static_cast<int>(*v);
    }
  }
  return std::nullopt;
}

bool IsUnitStrideSlice(const std::any& attr_any, const ::ml_drift::BHWC& starts,
                       const ::ml_drift::BHWC& ends) {
  const auto* attr = std::any_cast<::ml_drift::SliceAttributes>(&attr_any);
  if (attr == nullptr) {
    return false;
  }
  return attr->starts == starts && attr->ends == ends &&
         attr->strides == ::ml_drift::BHWC(1, 1, 1, 1);
}

bool IsConcatOnAxis(const std::any& attr_any, ::ml_drift::Axis axis) {
  const auto* attr = std::any_cast<::ml_drift::ConcatAttributes>(&attr_any);
  return attr != nullptr && attr->axis == axis;
}

bool IsSwapLastTwoDimsTranspose(const std::any& attr_any) {
  const auto* attr = std::any_cast<::ml_drift::TransposeAttributes>(&attr_any);
  return attr != nullptr && attr->perm == ::ml_drift::BHWC(0, 1, 3, 2);
}

bool IsValidFloatCapPair(float min_cap, float max_cap) {
  return std::isfinite(min_cap) && std::isfinite(max_cap) &&
         min_cap >= kMinFloatCapMagnitude &&
         std::abs(max_cap + min_cap) <= 1e-3f * min_cap;
}

bool IsMaskFillNeg10000(float value) {
  return std::isfinite(value) &&
         std::abs(value - kExpectedMaskFillValue) <= kMaskFillTolerance;
}

// Validates that `attr` is a float external-weights runtime BMM on KV cache.
// `is_src` selects V-cache (`src_end_ch_index`) vs K-cache
// (`dst_end_ch_index`).
bool ValidateRuntimeBmmAttr(const std::any& attr_any, bool is_src) {
  const auto* attr = std::any_cast<RuntimeBatchedMatMulAttributes>(&attr_any);
  if (attr == nullptr || !attr->external_weights.has_value() ||
      attr->scale.has_value()) {
    return false;
  }
  if (is_src) {
    return attr->runtime_check.src_end_ch_index.has_value() &&
           !attr->runtime_check.dst_end_ch_index.has_value();
  }
  return attr->runtime_check.dst_end_ch_index.has_value() &&
         !attr->runtime_check.src_end_ch_index.has_value();
}

bool ValidateRingCacheUpdateAttr(const std::any& attr_any, int heads,
                                 int cache_size, int head_dim,
                                 int expected_v_update_ts_idx) {
  const auto* attr = std::any_cast<AddValuesToCacheAttributes>(&attr_any);
  if (attr == nullptr) {
    return false;
  }
  return attr->kv_cache_batch_size == heads && attr->cache_size == cache_size &&
         attr->head_size == head_dim && attr->is_ring_buffer.value_or(false) &&
         attr->kv_cache_dtype == ::ml_drift::DataType::kUnknown &&
         !attr->scale_k.has_value() && !attr->scale_v.has_value() &&
         attr->token_index_offset_param_idx == 0 &&
         attr->active_tokens_param_idx == 1 && attr->k_cache_ts_idx == 2 &&
         attr->v_cache_ts_idx == 3 && attr->k_update_ts_idx == 2 &&
         attr->v_update_ts_idx == expected_v_update_ts_idx;
}

// ============================================================================
// GraphFloat32 implementation
// ============================================================================

::ml_drift::Value* PeelCopy(const ::ml_drift::GraphFloat32* graph,
                            ::ml_drift::Value* value) {
  const std::string copy_op = ToString(::ml_drift::OperationType::kCopy);
  while (value != nullptr) {
    ::ml_drift::Node* prod = graph->FindProducer(value->id);
    if (prod == nullptr || prod->operation.type != copy_op) {
      break;
    }
    std::vector<::ml_drift::Value*> ins = graph->FindInputs(prod->id);
    if (ins.size() != 1) {
      break;
    }
    value = ins[0];
  }
  return value;
}

bool IsSingleUseInternal(
    const ::ml_drift::GraphFloat32* graph, const ::ml_drift::Value* value,
    const absl::flat_hash_set<::ml_drift::ValueId>& orig_graph_outputs) {
  return value != nullptr && !orig_graph_outputs.contains(value->id) &&
         graph->FindConsumers(value->id).size() == 1;
}

// Checks whether `node` is `op_type(input, scalar_int)` where `scalar_int` is
// either in `ElementwiseAttributes::param` or a constant int32 second input.
// When `require_first_runtime` is true (e.g. FloorMod), requires the runtime
// tensor to be the first operand (`runtime % scalar_int`).
::ml_drift::Value* MatchElementwiseIntConst(
    const ::ml_drift::GraphFloat32* graph, const ::ml_drift::Node* node,
    ::ml_drift::OperationType op_type, int expected_const,
    bool require_first_runtime) {
  if (node == nullptr || node->operation.type != ToString(op_type)) {
    return nullptr;
  }
  std::vector<::ml_drift::Value*> ins = graph->FindInputs(node->id);
  if (ins.size() == 1) {
    const auto* attr = std::any_cast<::ml_drift::ElementwiseAttributes>(
        &node->operation.attributes);
    if (attr == nullptr) {
      return nullptr;
    }
    if (require_first_runtime && attr->runtime_tensor_is_second) {
      return nullptr;
    }
    std::optional<int> c =
        GetScalarIntFromElementwiseAttr(node->operation.attributes);
    if (!c.has_value() || *c != expected_const) {
      return nullptr;
    }
    return ins[0];
  }
  if (ins.size() == 2) {
    const std::string const_op = ToString(::ml_drift::OperationType::kConstant);
    ::ml_drift::Node* prod1 = graph->FindProducer(ins[1]->id);
    if (prod1 != nullptr && prod1->operation.type == const_op) {
      std::optional<int> c =
          GetScalarIntFromConstAttr(prod1->operation.attributes);
      if (c.has_value() && *c == expected_const) {
        return ins[0];
      }
    }
    if (!require_first_runtime) {
      ::ml_drift::Node* prod0 = graph->FindProducer(ins[0]->id);
      if (prod0 != nullptr && prod0->operation.type == const_op) {
        std::optional<int> c =
            GetScalarIntFromConstAttr(prod0->operation.attributes);
        if (c.has_value() && *c == expected_const) {
          return ins[1];
        }
      }
    }
  }
  return nullptr;
}

// Extracts the original `[1, 1, T, W + T]` bool mask from `mask_all`
// `[1, 1, G * T, W + T]`.
::ml_drift::Value* MatchOriginalMask(const ::ml_drift::GraphFloat32* graph,
                                     ::ml_drift::Value* mask_all, int g, int t,
                                     int w) {
  if (mask_all == nullptr ||
      mask_all->tensor.type != ::ml_drift::DataType::kBool ||
      mask_all->tensor.shape != ::ml_drift::BHWC(1, 1, g * t, w + t)) {
    return nullptr;
  }
  const std::string concat_op = ToString(::ml_drift::OperationType::kConcat);
  const std::string slice_op = ToString(::ml_drift::OperationType::kSlice);

  auto unwrap_repeated_rows = [&](::ml_drift::Value* val,
                                  int expected_c) -> ::ml_drift::Value* {
    if (g == 1) {
      return PeelCopy(graph, val);
    }
    ::ml_drift::Node* rep_node = graph->FindProducer(val->id);
    if (rep_node == nullptr || rep_node->operation.type != concat_op ||
        !IsConcatOnAxis(rep_node->operation.attributes,
                        ::ml_drift::Axis::kWidth)) {
      return nullptr;
    }
    std::vector<::ml_drift::Value*> rep_ins = graph->FindInputs(rep_node->id);
    if (static_cast<int>(rep_ins.size()) != g) {
      return nullptr;
    }
    ::ml_drift::Value* base = PeelCopy(graph, rep_ins[0]);
    if (base == nullptr ||
        base->tensor.shape != ::ml_drift::BHWC(1, 1, t, expected_c)) {
      return nullptr;
    }
    for (int i = 1; i < g; ++i) {
      if (PeelCopy(graph, rep_ins[i]) != base) {
        return nullptr;
      }
    }
    return base;
  };

  // Direct row repeat without past/new split: concat_W([mask_raw] * G).
  if (g > 1) {
    ::ml_drift::Value* direct = unwrap_repeated_rows(mask_all, w + t);
    if (direct != nullptr &&
        direct->tensor.type == ::ml_drift::DataType::kBool) {
      return direct;
    }
  }

  ::ml_drift::Node* concat_node = graph->FindProducer(mask_all->id);
  if (concat_node != nullptr && concat_node->operation.type == concat_op &&
      IsConcatOnAxis(concat_node->operation.attributes,
                     ::ml_drift::Axis::kChannels)) {
    std::vector<::ml_drift::Value*> c_ins = graph->FindInputs(concat_node->id);
    if (c_ins.size() == 2) {
      ::ml_drift::Value* m_past = unwrap_repeated_rows(c_ins[0], w);
      ::ml_drift::Value* m_new = unwrap_repeated_rows(c_ins[1], t);
      if (m_past != nullptr && m_new != nullptr) {
        ::ml_drift::Node* sp = graph->FindProducer(m_past->id);
        ::ml_drift::Node* sn = graph->FindProducer(m_new->id);
        if (sp != nullptr && sn != nullptr && sp->operation.type == slice_op &&
            sn->operation.type == slice_op &&
            IsUnitStrideSlice(sp->operation.attributes,
                              ::ml_drift::BHWC(0, 0, 0, 0),
                              ::ml_drift::BHWC(1, 1, t, w)) &&
            IsUnitStrideSlice(sn->operation.attributes,
                              ::ml_drift::BHWC(0, 0, 0, w),
                              ::ml_drift::BHWC(1, 1, t, w + t))) {
          std::vector<::ml_drift::Value*> sp_ins = graph->FindInputs(sp->id);
          std::vector<::ml_drift::Value*> sn_ins = graph->FindInputs(sn->id);
          if (sp_ins.size() == 1 && sn_ins.size() == 1 &&
              sp_ins[0] == sn_ins[0] &&
              sp_ins[0]->tensor.type == ::ml_drift::DataType::kBool &&
              sp_ins[0]->tensor.shape == ::ml_drift::BHWC(1, 1, t, w + t)) {
            return sp_ins[0];
          }
        }
      }
    }
  }

  if (g == 1) {
    return mask_all;
  }
  return nullptr;
}

struct MatchedParam {
  ::ml_drift::Value* param_raw = nullptr;
  ::ml_drift::Value* param_src = nullptr;
  ::ml_drift::Value* offset_val = nullptr;
};

// Recovers the raw runtime param tensor (`[0] = start`, `[1] = end`) from
// `param_past = concat_C(start % W, min(start, W), min(start, W), rest)`.
std::optional<MatchedParam> MatchOriginalParam(
    const ::ml_drift::GraphFloat32* graph, ::ml_drift::Value* param_past,
    int w) {
  if (param_past == nullptr ||
      param_past->tensor.type != ::ml_drift::DataType::kInt32 ||
      param_past->tensor.shape != ::ml_drift::BHWC(1, 1, 1, 7)) {
    return std::nullopt;
  }
  ::ml_drift::Node* concat_node = graph->FindProducer(param_past->id);
  if (concat_node == nullptr ||
      concat_node->operation.type !=
          ToString(::ml_drift::OperationType::kConcat) ||
      !IsConcatOnAxis(concat_node->operation.attributes,
                      ::ml_drift::Axis::kChannels)) {
    return std::nullopt;
  }
  std::vector<::ml_drift::Value*> ins = graph->FindInputs(concat_node->id);
  if (ins.size() != 4) {
    return std::nullopt;
  }
  ::ml_drift::Value* offset_val = PeelCopy(graph, ins[0]);
  ::ml_drift::Value* active0 = PeelCopy(graph, ins[1]);
  ::ml_drift::Value* active1 = PeelCopy(graph, ins[2]);
  if (offset_val == nullptr || active0 == nullptr || active0 != active1) {
    return std::nullopt;
  }

  ::ml_drift::Value* mod_in = MatchElementwiseIntConst(
      graph, graph->FindProducer(offset_val->id),
      ::ml_drift::OperationType::kFloorMod, w, /*require_first_runtime=*/true);
  ::ml_drift::Value* min_in = MatchElementwiseIntConst(
      graph, graph->FindProducer(active0->id),
      ::ml_drift::OperationType::kMinimum, w, /*require_first_runtime=*/false);
  if (mod_in == nullptr || min_in == nullptr) {
    return std::nullopt;
  }
  ::ml_drift::Value* start_slice = PeelCopy(graph, mod_in);
  if (start_slice != PeelCopy(graph, min_in)) {
    return std::nullopt;
  }

  ::ml_drift::Node* slice_start = graph->FindProducer(start_slice->id);
  if (slice_start == nullptr ||
      slice_start->operation.type !=
          ToString(::ml_drift::OperationType::kSlice) ||
      !IsUnitStrideSlice(slice_start->operation.attributes,
                         ::ml_drift::BHWC(0, 0, 0, 0),
                         ::ml_drift::BHWC(1, 1, 1, 1))) {
    return std::nullopt;
  }
  std::vector<::ml_drift::Value*> slice_start_ins =
      graph->FindInputs(slice_start->id);
  if (slice_start_ins.size() != 1) {
    return std::nullopt;
  }
  ::ml_drift::Value* param_src = slice_start_ins[0];
  ::ml_drift::Value* param_raw = param_src;

  // Step through an optional prefix slice `param_raw[..., 0:k]` (e.g. 0:3).
  if (::ml_drift::Node* src_prod = graph->FindProducer(param_src->id);
      src_prod != nullptr &&
      src_prod->operation.type == ToString(::ml_drift::OperationType::kSlice)) {
    const auto* slice_attr = std::any_cast<::ml_drift::SliceAttributes>(
        &src_prod->operation.attributes);
    std::vector<::ml_drift::Value*> outer_ins = graph->FindInputs(src_prod->id);
    if (slice_attr != nullptr && outer_ins.size() == 1 &&
        slice_attr->starts == ::ml_drift::BHWC(0, 0, 0, 0) &&
        slice_attr->strides == ::ml_drift::BHWC(1, 1, 1, 1) &&
        slice_attr->ends.b == 1 && slice_attr->ends.h == 1 &&
        slice_attr->ends.w == 1 && slice_attr->ends.c >= 2) {
      param_raw = outer_ins[0];
    }
  }

  if (param_raw == nullptr ||
      param_raw->tensor.type != ::ml_drift::DataType::kInt32 ||
      param_raw->tensor.shape.b != 1 || param_raw->tensor.shape.h != 1 ||
      param_raw->tensor.shape.w != 1 || param_raw->tensor.shape.c < 2) {
    return std::nullopt;
  }
  return MatchedParam{
      .param_raw = param_raw,
      .param_src = param_src,
      .offset_val = offset_val,
  };
}

// Verifies that `param_upd` is built from the same `offset_val` and
// `min(param[1:2], W)` so `param_raw[0]` and `param_raw[1]` are confirmed to be
// `(start, end)`.
bool ValidateUpdateParam(const ::ml_drift::GraphFloat32* graph,
                         ::ml_drift::Value* param_upd,
                         const MatchedParam& matched_param, int w) {
  if (param_upd == nullptr ||
      param_upd->tensor.type != ::ml_drift::DataType::kInt32 ||
      param_upd->tensor.shape != ::ml_drift::BHWC(1, 1, 1, 7)) {
    return false;
  }
  ::ml_drift::Node* concat_node = graph->FindProducer(param_upd->id);
  if (concat_node == nullptr ||
      concat_node->operation.type !=
          ToString(::ml_drift::OperationType::kConcat) ||
      !IsConcatOnAxis(concat_node->operation.attributes,
                      ::ml_drift::Axis::kChannels)) {
    return false;
  }
  std::vector<::ml_drift::Value*> ins = graph->FindInputs(concat_node->id);
  if (ins.size() != 4) {
    return false;
  }
  if (PeelCopy(graph, ins[0]) != matched_param.offset_val) {
    return false;
  }
  ::ml_drift::Value* active_val = PeelCopy(graph, ins[1]);
  ::ml_drift::Value* min_in = MatchElementwiseIntConst(
      graph, graph->FindProducer(active_val->id),
      ::ml_drift::OperationType::kMinimum, w, /*require_first_runtime=*/false);
  if (min_in == nullptr) {
    return false;
  }
  ::ml_drift::Node* end_slice =
      graph->FindProducer(PeelCopy(graph, min_in)->id);
  if (end_slice == nullptr ||
      end_slice->operation.type !=
          ToString(::ml_drift::OperationType::kSlice) ||
      !IsUnitStrideSlice(end_slice->operation.attributes,
                         ::ml_drift::BHWC(0, 0, 0, 1),
                         ::ml_drift::BHWC(1, 1, 1, 2))) {
    return false;
  }
  std::vector<::ml_drift::Value*> end_ins = graph->FindInputs(end_slice->id);
  return end_ins.size() == 1 && (end_ins[0] == matched_param.param_src ||
                                 end_ins[0] == matched_param.param_raw);
}

// Optionally peels `logits = tanh(logits / cap) * cap` (or `* (1 / cap)`).
// Updates `cur_val`, `softcap`, and `extra_nodes` on match.
void TryPeelSoftcap(
    const ::ml_drift::GraphFloat32* graph, ::ml_drift::Value*& cur_val,
    std::optional<float>& softcap, std::vector<::ml_drift::NodeId>& extra_nodes,
    const absl::flat_hash_set<::ml_drift::ValueId>& orig_graph_outputs) {
  const std::string mul_op = ToString(::ml_drift::OperationType::kMul);
  const std::string div_op = ToString(::ml_drift::OperationType::kDiv);
  const std::string tanh_op = ToString(::ml_drift::OperationType::kTanh);

  if (!IsSingleUseInternal(graph, cur_val, orig_graph_outputs)) {
    return;
  }
  ::ml_drift::Node* mul_out = graph->FindProducer(cur_val->id);
  if (mul_out == nullptr || mul_out->operation.type != mul_op) {
    return;
  }
  std::vector<::ml_drift::Value*> mul_out_ins = graph->FindInputs(mul_out->id);
  if (mul_out_ins.size() != 1) {
    return;
  }
  std::optional<float> cap =
      GetScalarFloatFromElementwiseAttr(mul_out->operation.attributes);
  if (!cap.has_value() || !std::isfinite(*cap) || *cap <= 0.0f) {
    return;
  }

  ::ml_drift::Value* tanh_out = mul_out_ins[0];
  if (!IsSingleUseInternal(graph, tanh_out, orig_graph_outputs)) {
    return;
  }
  ::ml_drift::Node* tanh_node = graph->FindProducer(tanh_out->id);
  if (tanh_node == nullptr || tanh_node->operation.type != tanh_op) {
    return;
  }
  std::vector<::ml_drift::Value*> tanh_ins = graph->FindInputs(tanh_node->id);
  if (tanh_ins.size() != 1) {
    return;
  }

  ::ml_drift::Value* scaled_in = tanh_ins[0];
  if (!IsSingleUseInternal(graph, scaled_in, orig_graph_outputs)) {
    return;
  }
  ::ml_drift::Node* scale_node = graph->FindProducer(scaled_in->id);
  if (scale_node == nullptr) {
    return;
  }
  std::vector<::ml_drift::Value*> scale_ins = graph->FindInputs(scale_node->id);
  if (scale_ins.size() != 1) {
    return;
  }
  std::optional<float> scale_param =
      GetScalarFloatFromElementwiseAttr(scale_node->operation.attributes);
  if (!scale_param.has_value() || !std::isfinite(*scale_param)) {
    return;
  }

  bool scale_matches = false;
  if (scale_node->operation.type == mul_op) {
    scale_matches = std::abs(*scale_param * (*cap) - 1.0f) <= 1e-3f;
  } else if (scale_node->operation.type == div_op) {
    const auto* div_attr = std::any_cast<::ml_drift::ElementwiseAttributes>(
        &scale_node->operation.attributes);
    scale_matches = div_attr != nullptr &&
                    !div_attr->runtime_tensor_is_second &&
                    std::abs(*scale_param - *cap) <= 1e-3f * (*cap);
  }
  if (!scale_matches) {
    return;
  }

  softcap = *cap;
  extra_nodes.push_back(mul_out->id);
  extra_nodes.push_back(tanh_node->id);
  extra_nodes.push_back(scale_node->id);
  cur_val = scale_ins[0];
}

struct SplitSdpaMatch {
  ::ml_drift::Value* query = nullptr;
  ::ml_drift::Value* key_cache = nullptr;
  ::ml_drift::Value* value_cache = nullptr;
  ::ml_drift::Value* key_new = nullptr;
  ::ml_drift::Value* value_new = nullptr;
  ::ml_drift::Value* mask_raw = nullptr;
  ::ml_drift::Value* param_raw = nullptr;

  ::ml_drift::Value* attention = nullptr;
  ::ml_drift::Value* key_cache_out = nullptr;
  ::ml_drift::Value* value_cache_out = nullptr;

  std::optional<float> softcap;
  bool value_new_needs_transpose = false;
  int heads = 0;
  int head_dim = 0;
  int new_len = 0;

  ::ml_drift::NodeId insert_after_id = 0;
  std::vector<::ml_drift::NodeId> nodes_to_delete;
  std::vector<::ml_drift::ValueId> dead_value_candidates;
};

absl::Status PruneDeadNodes(
    ::ml_drift::GraphFloat32* graph, std::vector<::ml_drift::ValueId> worklist,
    const absl::flat_hash_set<::ml_drift::ValueId>& orig_graph_outputs) {
  while (!worklist.empty()) {
    const ::ml_drift::ValueId vid = worklist.back();
    worklist.pop_back();
    if (orig_graph_outputs.contains(vid)) {
      continue;
    }
    ::ml_drift::Value* val = graph->GetValue(vid);
    if (val == nullptr || !graph->FindConsumers(vid).empty()) {
      continue;
    }
    ::ml_drift::Node* prod = graph->FindProducer(vid);
    if (prod == nullptr) {
      continue;
    }
    std::vector<::ml_drift::Value*> prod_outs = graph->FindOutputs(prod->id);
    bool all_outputs_dead = true;
    for (const ::ml_drift::Value* out : prod_outs) {
      if (out == nullptr || orig_graph_outputs.contains(out->id) ||
          !graph->FindConsumers(out->id).empty()) {
        all_outputs_dead = false;
        break;
      }
    }
    if (!all_outputs_dead) {
      continue;
    }
    std::vector<::ml_drift::Value*> prod_ins = graph->FindInputs(prod->id);
    for (const ::ml_drift::Value* in : prod_ins) {
      if (in != nullptr) {
        worklist.push_back(in->id);
      }
    }
    std::vector<::ml_drift::ValueId> out_ids;
    out_ids.reserve(prod_outs.size());
    for (const ::ml_drift::Value* out : prod_outs) {
      out_ids.push_back(out->id);
    }
    ABSL_RETURN_IF_ERROR(graph->DeleteNode(prod->id));
    for (::ml_drift::ValueId out_id : out_ids) {
      ABSL_RETURN_IF_ERROR(graph->DeleteValue(out_id));
    }
  }
  return absl::OkStatus();
}

// ============================================================================
// IrModel implementation
// ============================================================================

::ml_drift::ir::IrTensorId PeelCopyIr(const ::ml_drift::ir::IrModel* model,
                                      ::ml_drift::ir::IrTensorId tid) {
  const std::string copy_op = ToString(::ml_drift::OperationType::kCopy);
  while (true) {
    ::ml_drift::ir::IrOp* prod = model->FindProducer(tid);
    if (prod == nullptr || prod->name != copy_op || prod->inputs.size() != 1) {
      break;
    }
    tid = prod->inputs[0];
  }
  return tid;
}

bool IsSingleUseInternalIr(
    const ::ml_drift::ir::IrModel* model, ::ml_drift::ir::IrTensorId tid,
    const absl::flat_hash_set<::ml_drift::ir::IrTensorId>& orig_graph_outputs) {
  const ::ml_drift::ir::IrTensor* t = model->tensor(tid);
  return t != nullptr && !orig_graph_outputs.contains(tid) &&
         t->consumers.size() == 1;
}

std::optional<::ml_drift::ir::IrTensorId> MatchElementwiseIntConstIr(
    const ::ml_drift::ir::IrModel* model, const ::ml_drift::ir::IrOp* op,
    ::ml_drift::OperationType op_type, int expected_const,
    bool require_first_runtime) {
  if (op == nullptr || op->name != ToString(op_type)) {
    return std::nullopt;
  }
  if (op->inputs.size() == 1) {
    const auto* attr =
        std::any_cast<::ml_drift::ElementwiseAttributes>(&op->attr);
    if (attr == nullptr) {
      return std::nullopt;
    }
    if (require_first_runtime && attr->runtime_tensor_is_second) {
      return std::nullopt;
    }
    std::optional<int> c = GetScalarIntFromElementwiseAttr(op->attr);
    if (!c.has_value() || *c != expected_const) {
      return std::nullopt;
    }
    return op->inputs[0];
  }
  if (op->inputs.size() == 2) {
    const std::string const_op = ToString(::ml_drift::OperationType::kConstant);
    ::ml_drift::ir::IrOp* prod1 = model->FindProducer(op->inputs[1]);
    if (prod1 != nullptr && prod1->name == const_op) {
      std::optional<int> c = GetScalarIntFromConstAttr(prod1->attr);
      if (c.has_value() && *c == expected_const) {
        return op->inputs[0];
      }
    }
    if (!require_first_runtime) {
      ::ml_drift::ir::IrOp* prod0 = model->FindProducer(op->inputs[0]);
      if (prod0 != nullptr && prod0->name == const_op) {
        std::optional<int> c = GetScalarIntFromConstAttr(prod0->attr);
        if (c.has_value() && *c == expected_const) {
          return op->inputs[1];
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<::ml_drift::ir::IrTensorId> MatchOriginalMaskIr(
    const ::ml_drift::ir::IrModel* model, ::ml_drift::ir::IrTensorId mask_all,
    int g, int t, int w) {
  const ::ml_drift::ir::IrTensor* mask_tensor = model->tensor(mask_all);
  if (mask_tensor == nullptr ||
      mask_tensor->desc.GetDataType() != ::ml_drift::DataType::kBool ||
      mask_tensor->desc.GetBHWCShape() !=
          ::ml_drift::BHWC(1, 1, g * t, w + t)) {
    return std::nullopt;
  }
  const std::string concat_op = ToString(::ml_drift::OperationType::kConcat);
  const std::string slice_op = ToString(::ml_drift::OperationType::kSlice);

  auto unwrap_repeated_rows =
      [&](::ml_drift::ir::IrTensorId tid,
          int expected_c) -> std::optional<::ml_drift::ir::IrTensorId> {
    if (g == 1) {
      return PeelCopyIr(model, tid);
    }
    ::ml_drift::ir::IrOp* rep_op = model->FindProducer(tid);
    if (rep_op == nullptr || rep_op->name != concat_op ||
        !IsConcatOnAxis(rep_op->attr, ::ml_drift::Axis::kWidth) ||
        static_cast<int>(rep_op->inputs.size()) != g) {
      return std::nullopt;
    }
    const ::ml_drift::ir::IrTensorId base =
        PeelCopyIr(model, rep_op->inputs[0]);
    const ::ml_drift::ir::IrTensor* base_tensor = model->tensor(base);
    if (base_tensor == nullptr || base_tensor->desc.GetBHWCShape() !=
                                      ::ml_drift::BHWC(1, 1, t, expected_c)) {
      return std::nullopt;
    }
    for (int i = 1; i < g; ++i) {
      if (PeelCopyIr(model, rep_op->inputs[i]) != base) {
        return std::nullopt;
      }
    }
    return base;
  };

  if (g > 1) {
    std::optional<::ml_drift::ir::IrTensorId> direct =
        unwrap_repeated_rows(mask_all, w + t);
    if (direct.has_value()) {
      const ::ml_drift::ir::IrTensor* direct_tensor = model->tensor(*direct);
      if (direct_tensor != nullptr &&
          direct_tensor->desc.GetDataType() == ::ml_drift::DataType::kBool) {
        return direct;
      }
    }
  }

  ::ml_drift::ir::IrOp* concat_node = model->FindProducer(mask_all);
  if (concat_node != nullptr && concat_node->name == concat_op &&
      IsConcatOnAxis(concat_node->attr, ::ml_drift::Axis::kChannels) &&
      concat_node->inputs.size() == 2) {
    std::optional<::ml_drift::ir::IrTensorId> m_past =
        unwrap_repeated_rows(concat_node->inputs[0], w);
    std::optional<::ml_drift::ir::IrTensorId> m_new =
        unwrap_repeated_rows(concat_node->inputs[1], t);
    if (m_past.has_value() && m_new.has_value()) {
      ::ml_drift::ir::IrOp* sp = model->FindProducer(*m_past);
      ::ml_drift::ir::IrOp* sn = model->FindProducer(*m_new);
      if (sp != nullptr && sn != nullptr && sp->name == slice_op &&
          sn->name == slice_op && sp->inputs.size() == 1 &&
          sn->inputs.size() == 1 && sp->inputs[0] == sn->inputs[0] &&
          IsUnitStrideSlice(sp->attr, ::ml_drift::BHWC(0, 0, 0, 0),
                            ::ml_drift::BHWC(1, 1, t, w)) &&
          IsUnitStrideSlice(sn->attr, ::ml_drift::BHWC(0, 0, 0, w),
                            ::ml_drift::BHWC(1, 1, t, w + t))) {
        const ::ml_drift::ir::IrTensor* src = model->tensor(sp->inputs[0]);
        if (src != nullptr &&
            src->desc.GetDataType() == ::ml_drift::DataType::kBool &&
            src->desc.GetBHWCShape() == ::ml_drift::BHWC(1, 1, t, w + t)) {
          return sp->inputs[0];
        }
      }
    }
  }

  if (g == 1) {
    return mask_all;
  }
  return std::nullopt;
}

struct MatchedParamIr {
  ::ml_drift::ir::IrTensorId param_raw = 0;
  ::ml_drift::ir::IrTensorId param_src = 0;
  ::ml_drift::ir::IrTensorId offset_id = 0;
};

std::optional<MatchedParamIr> MatchOriginalParamIr(
    const ::ml_drift::ir::IrModel* model, ::ml_drift::ir::IrTensorId param_past,
    int w) {
  const ::ml_drift::ir::IrTensor* pt = model->tensor(param_past);
  if (pt == nullptr || pt->desc.GetDataType() != ::ml_drift::DataType::kInt32 ||
      pt->desc.GetBHWCShape() != ::ml_drift::BHWC(1, 1, 1, 7)) {
    return std::nullopt;
  }
  ::ml_drift::ir::IrOp* concat_op = model->FindProducer(param_past);
  if (concat_op == nullptr ||
      concat_op->name != ToString(::ml_drift::OperationType::kConcat) ||
      !IsConcatOnAxis(concat_op->attr, ::ml_drift::Axis::kChannels) ||
      concat_op->inputs.size() != 4) {
    return std::nullopt;
  }
  const ::ml_drift::ir::IrTensorId offset_id =
      PeelCopyIr(model, concat_op->inputs[0]);
  const ::ml_drift::ir::IrTensorId active0 =
      PeelCopyIr(model, concat_op->inputs[1]);
  const ::ml_drift::ir::IrTensorId active1 =
      PeelCopyIr(model, concat_op->inputs[2]);
  if (active0 != active1) {
    return std::nullopt;
  }

  std::optional<::ml_drift::ir::IrTensorId> mod_in = MatchElementwiseIntConstIr(
      model, model->FindProducer(offset_id),
      ::ml_drift::OperationType::kFloorMod, w, /*require_first_runtime=*/true);
  std::optional<::ml_drift::ir::IrTensorId> min_in = MatchElementwiseIntConstIr(
      model, model->FindProducer(active0), ::ml_drift::OperationType::kMinimum,
      w, /*require_first_runtime=*/false);
  if (!mod_in.has_value() || !min_in.has_value()) {
    return std::nullopt;
  }
  const ::ml_drift::ir::IrTensorId start_slice = PeelCopyIr(model, *mod_in);
  if (start_slice != PeelCopyIr(model, *min_in)) {
    return std::nullopt;
  }

  ::ml_drift::ir::IrOp* slice_start = model->FindProducer(start_slice);
  if (slice_start == nullptr ||
      slice_start->name != ToString(::ml_drift::OperationType::kSlice) ||
      slice_start->inputs.size() != 1 ||
      !IsUnitStrideSlice(slice_start->attr, ::ml_drift::BHWC(0, 0, 0, 0),
                         ::ml_drift::BHWC(1, 1, 1, 1))) {
    return std::nullopt;
  }
  const ::ml_drift::ir::IrTensorId param_src = slice_start->inputs[0];
  ::ml_drift::ir::IrTensorId param_raw = param_src;

  if (::ml_drift::ir::IrOp* src_prod = model->FindProducer(param_src);
      src_prod != nullptr &&
      src_prod->name == ToString(::ml_drift::OperationType::kSlice) &&
      src_prod->inputs.size() == 1) {
    const auto* slice_attr =
        std::any_cast<::ml_drift::SliceAttributes>(&src_prod->attr);
    if (slice_attr != nullptr &&
        slice_attr->starts == ::ml_drift::BHWC(0, 0, 0, 0) &&
        slice_attr->strides == ::ml_drift::BHWC(1, 1, 1, 1) &&
        slice_attr->ends.b == 1 && slice_attr->ends.h == 1 &&
        slice_attr->ends.w == 1 && slice_attr->ends.c >= 2) {
      param_raw = src_prod->inputs[0];
    }
  }

  const ::ml_drift::ir::IrTensor* raw_t = model->tensor(param_raw);
  if (raw_t == nullptr ||
      raw_t->desc.GetDataType() != ::ml_drift::DataType::kInt32) {
    return std::nullopt;
  }
  const ::ml_drift::BHWC raw_shape = raw_t->desc.GetBHWCShape();
  if (raw_shape.b != 1 || raw_shape.h != 1 || raw_shape.w != 1 ||
      raw_shape.c < 2) {
    return std::nullopt;
  }
  return MatchedParamIr{
      .param_raw = param_raw,
      .param_src = param_src,
      .offset_id = offset_id,
  };
}

bool ValidateUpdateParamIr(const ::ml_drift::ir::IrModel* model,
                           ::ml_drift::ir::IrTensorId param_upd,
                           const MatchedParamIr& matched_param, int w) {
  const ::ml_drift::ir::IrTensor* pt = model->tensor(param_upd);
  if (pt == nullptr || pt->desc.GetDataType() != ::ml_drift::DataType::kInt32 ||
      pt->desc.GetBHWCShape() != ::ml_drift::BHWC(1, 1, 1, 7)) {
    return false;
  }
  ::ml_drift::ir::IrOp* concat_op = model->FindProducer(param_upd);
  if (concat_op == nullptr ||
      concat_op->name != ToString(::ml_drift::OperationType::kConcat) ||
      !IsConcatOnAxis(concat_op->attr, ::ml_drift::Axis::kChannels) ||
      concat_op->inputs.size() != 4) {
    return false;
  }
  if (PeelCopyIr(model, concat_op->inputs[0]) != matched_param.offset_id) {
    return false;
  }
  const ::ml_drift::ir::IrTensorId active_id =
      PeelCopyIr(model, concat_op->inputs[1]);
  std::optional<::ml_drift::ir::IrTensorId> min_in = MatchElementwiseIntConstIr(
      model, model->FindProducer(active_id),
      ::ml_drift::OperationType::kMinimum, w, /*require_first_runtime=*/false);
  if (!min_in.has_value()) {
    return false;
  }
  ::ml_drift::ir::IrOp* end_slice =
      model->FindProducer(PeelCopyIr(model, *min_in));
  if (end_slice == nullptr ||
      end_slice->name != ToString(::ml_drift::OperationType::kSlice) ||
      end_slice->inputs.size() != 1 ||
      !IsUnitStrideSlice(end_slice->attr, ::ml_drift::BHWC(0, 0, 0, 1),
                         ::ml_drift::BHWC(1, 1, 1, 2))) {
    return false;
  }
  return end_slice->inputs[0] == matched_param.param_src ||
         end_slice->inputs[0] == matched_param.param_raw;
}

void TryPeelSoftcapIr(
    const ::ml_drift::ir::IrModel* model, ::ml_drift::ir::IrTensorId& cur_tid,
    std::optional<float>& softcap,
    std::vector<::ml_drift::ir::IrOpId>& extra_ops,
    const absl::flat_hash_set<::ml_drift::ir::IrTensorId>& orig_graph_outputs) {
  const std::string mul_op = ToString(::ml_drift::OperationType::kMul);
  const std::string div_op = ToString(::ml_drift::OperationType::kDiv);
  const std::string tanh_op = ToString(::ml_drift::OperationType::kTanh);

  if (!IsSingleUseInternalIr(model, cur_tid, orig_graph_outputs)) {
    return;
  }
  ::ml_drift::ir::IrOp* mul_out = model->FindProducer(cur_tid);
  if (mul_out == nullptr || mul_out->name != mul_op ||
      mul_out->inputs.size() != 1) {
    return;
  }
  std::optional<float> cap = GetScalarFloatFromElementwiseAttr(mul_out->attr);
  if (!cap.has_value() || !std::isfinite(*cap) || *cap <= 0.0f) {
    return;
  }

  const ::ml_drift::ir::IrTensorId tanh_out = mul_out->inputs[0];
  if (!IsSingleUseInternalIr(model, tanh_out, orig_graph_outputs)) {
    return;
  }
  ::ml_drift::ir::IrOp* tanh_node = model->FindProducer(tanh_out);
  if (tanh_node == nullptr || tanh_node->name != tanh_op ||
      tanh_node->inputs.size() != 1) {
    return;
  }

  const ::ml_drift::ir::IrTensorId scaled_in = tanh_node->inputs[0];
  if (!IsSingleUseInternalIr(model, scaled_in, orig_graph_outputs)) {
    return;
  }
  ::ml_drift::ir::IrOp* scale_node = model->FindProducer(scaled_in);
  if (scale_node == nullptr || scale_node->inputs.size() != 1) {
    return;
  }
  std::optional<float> scale_param =
      GetScalarFloatFromElementwiseAttr(scale_node->attr);
  if (!scale_param.has_value() || !std::isfinite(*scale_param)) {
    return;
  }

  bool scale_matches = false;
  if (scale_node->name == mul_op) {
    scale_matches = std::abs(*scale_param * (*cap) - 1.0f) <= 1e-3f;
  } else if (scale_node->name == div_op) {
    const auto* div_attr =
        std::any_cast<::ml_drift::ElementwiseAttributes>(&scale_node->attr);
    scale_matches = div_attr != nullptr &&
                    !div_attr->runtime_tensor_is_second &&
                    std::abs(*scale_param - *cap) <= 1e-3f * (*cap);
  }
  if (!scale_matches) {
    return;
  }

  softcap = *cap;
  extra_ops.push_back(mul_out->id);
  extra_ops.push_back(tanh_node->id);
  extra_ops.push_back(scale_node->id);
  cur_tid = scale_node->inputs[0];
}

struct IrSplitSdpaMatch {
  ::ml_drift::ir::IrTensorId query = 0;
  ::ml_drift::ir::IrTensorId key_cache = 0;
  ::ml_drift::ir::IrTensorId value_cache = 0;
  ::ml_drift::ir::IrTensorId key_new = 0;
  ::ml_drift::ir::IrTensorId value_new = 0;
  ::ml_drift::ir::IrTensorId mask_raw = 0;
  ::ml_drift::ir::IrTensorId param_raw = 0;

  ::ml_drift::ir::IrTensorId attention = 0;
  std::optional<::ml_drift::ir::IrTensorId> key_cache_out;
  std::optional<::ml_drift::ir::IrTensorId> value_cache_out;
  std::optional<::ml_drift::ir::IrOpId> cache_update_op_id;

  std::optional<float> softcap;
  bool value_new_needs_transpose = false;
  int heads = 0;
  int head_dim = 0;
  int new_len = 0;

  // Existing op slots reused in place so `model->ops()` remains in valid
  // topological execution order (`transpose_slot_id < fused_slot_id`).
  ::ml_drift::ir::IrOpId transpose_slot_id = 0;
  ::ml_drift::ir::IrOpId fused_slot_id = 0;
  std::vector<::ml_drift::ir::IrOpId> ops_to_delete;
  std::vector<::ml_drift::ir::IrTensorId> dead_tensor_candidates;
};

void ClearIrOpInPlace(::ml_drift::ir::IrModel* model,
                      ::ml_drift::ir::IrOp* op) {
  for (::ml_drift::ir::IrTensorId in_id : op->inputs) {
    if (::ml_drift::ir::IrTensor* t = model->GetMutableTensor(in_id);
        t != nullptr) {
      t->consumers.erase(op->id);
    }
  }
  for (::ml_drift::ir::IrTensorId out_id : op->outputs) {
    if (::ml_drift::ir::IrTensor* t = model->GetMutableTensor(out_id);
        t != nullptr && t->producer == op->id) {
      t->producer.reset();
    }
  }
  op->inputs.clear();
  op->outputs.clear();
  op->attr.reset();
}

absl::Status PruneDeadOpsIr(
    ::ml_drift::ir::IrModel* model,
    std::vector<::ml_drift::ir::IrTensorId> worklist,
    const absl::flat_hash_set<::ml_drift::ir::IrTensorId>& orig_graph_outputs) {
  while (!worklist.empty()) {
    const ::ml_drift::ir::IrTensorId tid = worklist.back();
    worklist.pop_back();
    if (orig_graph_outputs.contains(tid)) {
      continue;
    }
    const ::ml_drift::ir::IrTensor* t = model->tensor(tid);
    if (t == nullptr || !t->consumers.empty()) {
      continue;
    }
    ::ml_drift::ir::IrOp* prod = model->FindProducer(tid);
    if (prod == nullptr) {
      continue;
    }
    bool all_outputs_dead = true;
    for (::ml_drift::ir::IrTensorId out_id : prod->outputs) {
      const ::ml_drift::ir::IrTensor* out_t = model->tensor(out_id);
      if (orig_graph_outputs.contains(out_id) || out_t == nullptr ||
          !out_t->consumers.empty()) {
        all_outputs_dead = false;
        break;
      }
    }
    if (!all_outputs_dead) {
      continue;
    }
    for (::ml_drift::ir::IrTensorId in_id : prod->inputs) {
      worklist.push_back(in_id);
    }
    ABSL_RETURN_IF_ERROR(model->RemoveOp(prod->id));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status FuseSplitSdpaCacheUpdate(::ml_drift::GraphFloat32* graph) {
  if (graph == nullptr) {
    return absl::OkStatus();
  }

  const std::string add_op = ToString(::ml_drift::OperationType::kAdd);
  const std::string bmm_op =
      ToString(::ml_drift::OperationType::kBatchedMatmul);
  const std::string concat_op = ToString(::ml_drift::OperationType::kConcat);
  const std::string const_op = ToString(::ml_drift::OperationType::kConstant);
  const std::string max_op = ToString(::ml_drift::OperationType::kMaximum);
  const std::string min_op = ToString(::ml_drift::OperationType::kMinimum);
  const std::string select_op = ToString(::ml_drift::OperationType::kSelectV2);
  const std::string slice_op = ToString(::ml_drift::OperationType::kSlice);
  const std::string softmax_op = ToString(::ml_drift::OperationType::kSoftmax);
  const std::string transpose_op =
      ToString(::ml_drift::OperationType::kTranspose);

  absl::flat_hash_set<::ml_drift::ValueId> orig_graph_outputs;
  for (const ::ml_drift::Value* out : graph->outputs()) {
    if (out != nullptr) {
      orig_graph_outputs.insert(out->id);
    }
  }

  std::vector<SplitSdpaMatch> matches;
  absl::flat_hash_set<::ml_drift::NodeId> matched_node_ids;

  for (::ml_drift::Node* add_node : graph->nodes()) {
    if (add_node == nullptr || matched_node_ids.contains(add_node->id) ||
        add_node->operation.type != add_op) {
      continue;
    }
    std::vector<::ml_drift::Value*> add_ins = graph->FindInputs(add_node->id);
    std::vector<::ml_drift::Value*> add_outs = graph->FindOutputs(add_node->id);
    if (add_ins.size() != 2 || add_outs.size() != 1) {
      continue;
    }
    if (!IsSingleUseInternal(graph, add_ins[0], orig_graph_outputs) ||
        !IsSingleUseInternal(graph, add_ins[1], orig_graph_outputs)) {
      continue;
    }

    ::ml_drift::Node* prod0 = graph->FindProducer(add_ins[0]->id);
    ::ml_drift::Node* prod1 = graph->FindProducer(add_ins[1]->id);
    if (prod0 == nullptr || prod1 == nullptr) {
      continue;
    }

    ::ml_drift::Node* bmm_past_v = nullptr;
    ::ml_drift::Node* bmm_new_v = nullptr;
    if (prod0->operation.type == kRuntimeBatchedMatMulType &&
        prod1->operation.type == bmm_op) {
      bmm_past_v = prod0;
      bmm_new_v = prod1;
    } else if (prod1->operation.type == kRuntimeBatchedMatMulType &&
               prod0->operation.type == bmm_op) {
      bmm_past_v = prod1;
      bmm_new_v = prod0;
    } else {
      continue;
    }

    if (!ValidateRuntimeBmmAttr(bmm_past_v->operation.attributes,
                                /*is_src=*/true)) {
      continue;
    }
    const auto* bmm_new_v_attr =
        std::any_cast<::ml_drift::BatchedMatMulAttributes>(
            &bmm_new_v->operation.attributes);
    if (bmm_new_v_attr == nullptr || bmm_new_v_attr->transpose_left) {
      continue;
    }

    std::vector<::ml_drift::Value*> pv_ins = graph->FindInputs(bmm_past_v->id);
    std::vector<::ml_drift::Value*> nv_ins = graph->FindInputs(bmm_new_v->id);
    if (pv_ins.size() != 3 || nv_ins.size() != 2) {
      continue;
    }
    ::ml_drift::Value* probs_past = pv_ins[0];
    ::ml_drift::Value* value_cache = pv_ins[1];
    ::ml_drift::Value* param_past = pv_ins[2];
    ::ml_drift::Value* probs_new = nv_ins[0];
    ::ml_drift::Value* value_new = nv_ins[1];

    if (value_cache == nullptr || value_new == nullptr ||
        !graph->IsGraphInput(value_cache->id) ||
        !::ml_drift::IsFloatType(value_cache->tensor.type) ||
        !::ml_drift::IsFloatType(value_new->tensor.type) ||
        !::ml_drift::IsFloatType(add_outs[0]->tensor.type)) {
      continue;
    }

    const ::ml_drift::BHWC& vc_shape = value_cache->tensor.shape;
    const ::ml_drift::BHWC& out_shape = add_outs[0]->tensor.shape;
    const int heads = vc_shape.h;
    const int head_dim = vc_shape.w;
    const int cache_size = vc_shape.c;
    const int rows = out_shape.w;
    if (vc_shape.b != 1 || out_shape.b != 1 || out_shape.h != heads ||
        out_shape.c != head_dim || heads <= 0 || head_dim <= 0 ||
        cache_size <= 0 || head_dim % 4 != 0 || cache_size % 4 != 0) {
      continue;
    }

    const bool value_new_needs_transpose = !bmm_new_v_attr->transpose_right;
    const int new_len = value_new_needs_transpose ? value_new->tensor.shape.w
                                                  : value_new->tensor.shape.c;
    const ::ml_drift::BHWC expected_vn_shape =
        value_new_needs_transpose
            ? ::ml_drift::BHWC(1, heads, new_len, head_dim)
            : ::ml_drift::BHWC(1, heads, head_dim, new_len);
    if (new_len <= 1 || rows <= 0 || rows % new_len != 0 ||
        value_new->tensor.shape != expected_vn_shape) {
      continue;
    }
    const int g = rows / new_len;

    if (!IsSingleUseInternal(graph, probs_past, orig_graph_outputs) ||
        !IsSingleUseInternal(graph, probs_new, orig_graph_outputs)) {
      continue;
    }
    ::ml_drift::Node* slice_past = graph->FindProducer(probs_past->id);
    ::ml_drift::Node* slice_new = graph->FindProducer(probs_new->id);
    if (slice_past == nullptr || slice_new == nullptr ||
        slice_past->operation.type != slice_op ||
        slice_new->operation.type != slice_op ||
        !IsUnitStrideSlice(slice_past->operation.attributes,
                           ::ml_drift::BHWC(0, 0, 0, 0),
                           ::ml_drift::BHWC(1, heads, rows, cache_size)) ||
        !IsUnitStrideSlice(
            slice_new->operation.attributes,
            ::ml_drift::BHWC(0, 0, 0, cache_size),
            ::ml_drift::BHWC(1, heads, rows, cache_size + new_len))) {
      continue;
    }
    std::vector<::ml_drift::Value*> sp_ins = graph->FindInputs(slice_past->id);
    std::vector<::ml_drift::Value*> sn_ins = graph->FindInputs(slice_new->id);
    if (sp_ins.size() != 1 || sn_ins.size() != 1 || sp_ins[0] != sn_ins[0]) {
      continue;
    }
    ::ml_drift::Value* probs_all = sp_ins[0];
    if (orig_graph_outputs.contains(probs_all->id) ||
        graph->FindConsumers(probs_all->id).size() != 2 ||
        probs_all->tensor.shape !=
            ::ml_drift::BHWC(1, heads, rows, cache_size + new_len)) {
      continue;
    }

    ::ml_drift::Node* softmax_node = graph->FindProducer(probs_all->id);
    if (softmax_node == nullptr || softmax_node->operation.type != softmax_op) {
      continue;
    }
    const auto* softmax_attr = std::any_cast<::ml_drift::SoftmaxAttributes>(
        &softmax_node->operation.attributes);
    if (softmax_attr == nullptr ||
        softmax_attr->axis != ::ml_drift::Axis::kChannels) {
      continue;
    }
    std::vector<::ml_drift::Value*> softmax_ins =
        graph->FindInputs(softmax_node->id);
    if (softmax_ins.size() != 1) {
      continue;
    }

    std::vector<::ml_drift::NodeId> extra_nodes;
    ::ml_drift::Value* cur_val = softmax_ins[0];

    // Peel optional infinite-float capping (`max(x, -cap) -> min(x, +cap)`)
    // inserted by `SoftmaxOperationParser`.
    if (IsSingleUseInternal(graph, cur_val, orig_graph_outputs)) {
      ::ml_drift::Node* min_node = graph->FindProducer(cur_val->id);
      if (min_node != nullptr && min_node->operation.type == min_op) {
        std::vector<::ml_drift::Value*> min_ins =
            graph->FindInputs(min_node->id);
        std::optional<float> min_cap =
            GetScalarFloatFromElementwiseAttr(min_node->operation.attributes);
        if (min_ins.size() == 1 && min_cap.has_value() &&
            IsSingleUseInternal(graph, min_ins[0], orig_graph_outputs)) {
          ::ml_drift::Node* max_node = graph->FindProducer(min_ins[0]->id);
          if (max_node != nullptr && max_node->operation.type == max_op) {
            std::vector<::ml_drift::Value*> max_ins =
                graph->FindInputs(max_node->id);
            std::optional<float> max_cap = GetScalarFloatFromElementwiseAttr(
                max_node->operation.attributes);
            if (max_ins.size() == 1 && max_cap.has_value() &&
                IsValidFloatCapPair(*min_cap, *max_cap)) {
              extra_nodes.push_back(min_node->id);
              extra_nodes.push_back(max_node->id);
              cur_val = max_ins[0];
            }
          }
        }
      }
    }

    if (!IsSingleUseInternal(graph, cur_val, orig_graph_outputs)) {
      continue;
    }
    ::ml_drift::Node* select_node = graph->FindProducer(cur_val->id);
    if (select_node == nullptr || select_node->operation.type != select_op) {
      continue;
    }
    std::vector<::ml_drift::Value*> sel_ins =
        graph->FindInputs(select_node->id);
    if (sel_ins.size() != 3) {
      continue;
    }
    ::ml_drift::Value* mask_all = sel_ins[0];
    ::ml_drift::Value* logits_pre_mask = sel_ins[1];
    ::ml_drift::Value* fill_val = sel_ins[2];

    ::ml_drift::Node* fill_prod = graph->FindProducer(fill_val->id);
    if (fill_prod == nullptr || fill_prod->operation.type != const_op) {
      continue;
    }
    std::optional<float> fill_scalar =
        GetScalarFloatFromConstAttr(fill_prod->operation.attributes);
    if (!fill_scalar.has_value() || !IsMaskFillNeg10000(*fill_scalar)) {
      continue;
    }

    ::ml_drift::Value* mask_raw =
        MatchOriginalMask(graph, mask_all, g, new_len, cache_size);
    if (mask_raw == nullptr) {
      continue;
    }

    std::optional<float> softcap;
    ::ml_drift::Value* logits_all = logits_pre_mask;
    TryPeelSoftcap(graph, logits_all, softcap, extra_nodes, orig_graph_outputs);

    if (!IsSingleUseInternal(graph, logits_all, orig_graph_outputs)) {
      continue;
    }
    ::ml_drift::Node* concat_logits = graph->FindProducer(logits_all->id);
    if (concat_logits == nullptr ||
        concat_logits->operation.type != concat_op ||
        !IsConcatOnAxis(concat_logits->operation.attributes,
                        ::ml_drift::Axis::kChannels)) {
      continue;
    }
    std::vector<::ml_drift::Value*> cl_ins =
        graph->FindInputs(concat_logits->id);
    if (cl_ins.size() != 2 ||
        !IsSingleUseInternal(graph, cl_ins[0], orig_graph_outputs) ||
        !IsSingleUseInternal(graph, cl_ins[1], orig_graph_outputs)) {
      continue;
    }
    ::ml_drift::Value* logits_past = cl_ins[0];
    ::ml_drift::Value* logits_new = cl_ins[1];
    if (logits_past->tensor.shape !=
            ::ml_drift::BHWC(1, heads, rows, cache_size) ||
        logits_new->tensor.shape != ::ml_drift::BHWC(1, heads, rows, new_len)) {
      continue;
    }

    ::ml_drift::Node* bmm_past_k = graph->FindProducer(logits_past->id);
    ::ml_drift::Node* bmm_new_k = graph->FindProducer(logits_new->id);
    if (bmm_past_k == nullptr || bmm_new_k == nullptr ||
        bmm_past_k->operation.type != kRuntimeBatchedMatMulType ||
        bmm_new_k->operation.type != bmm_op) {
      continue;
    }
    if (!ValidateRuntimeBmmAttr(bmm_past_k->operation.attributes,
                                /*is_src=*/false)) {
      continue;
    }
    const auto* bmm_new_k_attr =
        std::any_cast<::ml_drift::BatchedMatMulAttributes>(
            &bmm_new_k->operation.attributes);
    if (bmm_new_k_attr == nullptr || bmm_new_k_attr->transpose_left ||
        !bmm_new_k_attr->transpose_right) {
      continue;
    }

    std::vector<::ml_drift::Value*> pk_ins = graph->FindInputs(bmm_past_k->id);
    std::vector<::ml_drift::Value*> nk_ins = graph->FindInputs(bmm_new_k->id);
    if (pk_ins.size() != 3 || nk_ins.size() != 2) {
      continue;
    }
    ::ml_drift::Value* query = pk_ins[0];
    ::ml_drift::Value* key_cache = pk_ins[1];
    ::ml_drift::Value* param_past_k = pk_ins[2];
    ::ml_drift::Value* key_new = nk_ins[1];

    if (nk_ins[0] != query || param_past_k != param_past ||
        key_cache == nullptr || key_new == nullptr ||
        !graph->IsGraphInput(key_cache->id) ||
        key_cache->tensor.type != value_cache->tensor.type ||
        !::ml_drift::IsFloatType(query->tensor.type) ||
        !::ml_drift::IsFloatType(key_new->tensor.type) ||
        query->tensor.shape != ::ml_drift::BHWC(1, heads, rows, head_dim) ||
        key_cache->tensor.shape !=
            ::ml_drift::BHWC(1, heads, cache_size, head_dim) ||
        key_new->tensor.shape !=
            ::ml_drift::BHWC(1, heads, new_len, head_dim)) {
      continue;
    }

    std::optional<MatchedParam> matched_param =
        MatchOriginalParam(graph, param_past, cache_size);
    if (!matched_param.has_value()) {
      continue;
    }

    // Check for an accompanying ring-buffer `add_values_to_cache` consumer on
    // `key_new`. If one exists, it must validate cleanly so we never drop an
    // incompatible cache update or reorder a shared-KV cache write ahead of
    // another reader.
    ::ml_drift::Node* cache_update_node = nullptr;
    ::ml_drift::Value* key_cache_out = nullptr;
    ::ml_drift::Value* value_cache_out = nullptr;
    ::ml_drift::Value* param_upd = nullptr;
    ::ml_drift::Value* cu_v_in = nullptr;
    bool has_incompatible_cache_update = false;

    for (::ml_drift::Node* consumer : graph->FindConsumers(key_new->id)) {
      if (consumer == nullptr ||
          consumer->operation.type != kAddValuesToCacheType) {
        continue;
      }
      if (cache_update_node != nullptr) {
        has_incompatible_cache_update = true;
        break;
      }
      std::vector<::ml_drift::Value*> cu_ins = graph->FindInputs(consumer->id);
      std::vector<::ml_drift::Value*> cu_outs =
          graph->FindOutputs(consumer->id);
      if (cu_ins.size() != 3 || cu_outs.size() != 2 || cu_ins[0] != key_new) {
        has_incompatible_cache_update = true;
        break;
      }
      int expected_v_update_ts_idx = -1;
      if (cu_ins[1] == value_new) {
        expected_v_update_ts_idx = value_new_needs_transpose ? 2 : 3;
      } else {
        ::ml_drift::Node* cu_v_prod = graph->FindProducer(cu_ins[1]->id);
        if (cu_v_prod != nullptr && cu_v_prod->operation.type == transpose_op &&
            IsSwapLastTwoDimsTranspose(cu_v_prod->operation.attributes)) {
          std::vector<::ml_drift::Value*> tr_ins =
              graph->FindInputs(cu_v_prod->id);
          if (tr_ins.size() == 1 && tr_ins[0] == value_new) {
            expected_v_update_ts_idx = value_new_needs_transpose ? 3 : 2;
          }
        }
      }
      if (expected_v_update_ts_idx < 0 || new_len > cache_size ||
          !ValidateRingCacheUpdateAttr(consumer->operation.attributes, heads,
                                       cache_size, head_dim,
                                       expected_v_update_ts_idx) ||
          !ValidateUpdateParam(graph, cu_ins[2], *matched_param, cache_size)) {
        ABSL_VLOG(1) << "FuseSplitSdpaCacheUpdate: rejecting add_node="
                     << add_node->id << " due to cache_update attr/param.";
        has_incompatible_cache_update = true;
        break;
      }
      // Require in-place cache safety: no other consumers read the pre-write
      // caches, and the cache outputs are pure graph outputs.
      if (graph->FindConsumers(key_cache->id).size() != 1 ||
          graph->FindConsumers(value_cache->id).size() != 1 ||
          !orig_graph_outputs.contains(cu_outs[0]->id) ||
          !orig_graph_outputs.contains(cu_outs[1]->id) ||
          !graph->FindConsumers(cu_outs[0]->id).empty() ||
          !graph->FindConsumers(cu_outs[1]->id).empty() ||
          cu_outs[0]->tensor.shape != key_cache->tensor.shape ||
          cu_outs[1]->tensor.shape != value_cache->tensor.shape ||
          cu_outs[0]->tensor.type != key_cache->tensor.type ||
          cu_outs[1]->tensor.type != value_cache->tensor.type) {
        has_incompatible_cache_update = true;
        break;
      }
      cache_update_node = consumer;
      key_cache_out = cu_outs[0];
      value_cache_out = cu_outs[1];
      param_upd = cu_ins[2];
      cu_v_in = cu_ins[1];
    }
    if (has_incompatible_cache_update) {
      continue;
    }

    std::vector<::ml_drift::NodeId> to_delete = {
        add_node->id,   bmm_new_v->id,    bmm_past_v->id,  slice_new->id,
        slice_past->id, softmax_node->id, select_node->id, concat_logits->id,
        bmm_new_k->id,  bmm_past_k->id,
    };
    to_delete.insert(to_delete.end(), extra_nodes.begin(), extra_nodes.end());
    if (cache_update_node != nullptr) {
      to_delete.push_back(cache_update_node->id);
    }

    bool overlaps = false;
    for (::ml_drift::NodeId nid : to_delete) {
      if (matched_node_ids.contains(nid)) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) {
      continue;
    }
    for (::ml_drift::NodeId nid : to_delete) {
      matched_node_ids.insert(nid);
    }

    SplitSdpaMatch match;
    match.query = query;
    match.key_cache = key_cache;
    match.value_cache = value_cache;
    match.key_new = key_new;
    match.value_new = value_new;
    match.mask_raw = mask_raw;
    match.param_raw = matched_param->param_raw;
    match.attention = add_outs[0];
    match.key_cache_out = key_cache_out;
    match.value_cache_out = value_cache_out;
    match.softcap = softcap;
    match.value_new_needs_transpose = value_new_needs_transpose;
    match.heads = heads;
    match.head_dim = head_dim;
    match.new_len = new_len;
    match.insert_after_id = add_node->id;
    match.nodes_to_delete = std::move(to_delete);
    match.dead_value_candidates = {mask_all->id, fill_val->id, param_past->id};
    if (param_upd != nullptr) {
      match.dead_value_candidates.push_back(param_upd->id);
    }
    if (cu_v_in != nullptr && cu_v_in != value_new) {
      match.dead_value_candidates.push_back(cu_v_in->id);
    }
    matches.push_back(std::move(match));
  }

  if (matches.empty()) {
    return absl::OkStatus();
  }

  ABSL_VLOG(1) << "FuseSplitSdpaCacheUpdate: fusing " << matches.size()
               << " sliding-window SDPA subgraph(s) into "
               << kFusedSdpaCacheUpdateType << ".";

  std::vector<::ml_drift::ValueId> dead_value_worklist;
  for (const SplitSdpaMatch& match : matches) {
    ::ml_drift::NodeId insert_after = match.insert_after_id;
    ::ml_drift::Value* value_new_for_sdpa = match.value_new;
    if (match.value_new_needs_transpose) {
      ::ml_drift::Node* transpose_node = nullptr;
      ABSL_RETURN_IF_ERROR(
          graph->InsertNodeAfter(insert_after, &transpose_node));
      transpose_node->operation.type = transpose_op;
      ::ml_drift::TransposeAttributes t_attr;
      t_attr.perm = ::ml_drift::BHWC(0, 1, 3, 2);
      transpose_node->operation.attributes = t_attr;
      ::ml_drift::Value* v_transposed = graph->NewValue();
      v_transposed->tensor.type = match.value_new->tensor.type;
      v_transposed->tensor.shape =
          ::ml_drift::BHWC(1, match.heads, match.head_dim, match.new_len);
      v_transposed->tensor.ref = -1;
      graph->AddConsumer(transpose_node->id, match.value_new->id);
      graph->SetProducer(transpose_node->id, v_transposed->id);
      value_new_for_sdpa = v_transposed;
      insert_after = transpose_node->id;
    }

    ::ml_drift::Node* fused_node = nullptr;
    ABSL_RETURN_IF_ERROR(graph->InsertNodeAfter(insert_after, &fused_node));
    fused_node->operation.type = kFusedSdpaCacheUpdateType;
    FusedSdpaCacheUpdateAttributes fused_attr;
    fused_attr.softcap = match.softcap;
    fused_attr.update_cache = match.key_cache_out != nullptr;
    fused_node->operation.attributes = fused_attr;

    graph->AddConsumer(fused_node->id, match.query->id);
    graph->AddConsumer(fused_node->id, match.key_cache->id);
    graph->AddConsumer(fused_node->id, match.value_cache->id);
    graph->AddConsumer(fused_node->id, match.key_new->id);
    graph->AddConsumer(fused_node->id, value_new_for_sdpa->id);
    graph->AddConsumer(fused_node->id, match.mask_raw->id);
    graph->AddConsumer(fused_node->id, match.param_raw->id);

    graph->SetProducer(fused_node->id, match.attention->id);
    if (match.key_cache_out != nullptr && match.value_cache_out != nullptr) {
      graph->SetProducer(fused_node->id, match.key_cache_out->id);
      graph->SetProducer(fused_node->id, match.value_cache_out->id);
    }

    std::vector<::ml_drift::ValueId> internal_values_to_delete;
    for (::ml_drift::NodeId nid : match.nodes_to_delete) {
      for (const ::ml_drift::Value* out : graph->FindOutputs(nid)) {
        if (out != nullptr && out != match.attention &&
            out != match.key_cache_out && out != match.value_cache_out) {
          internal_values_to_delete.push_back(out->id);
        }
      }
    }
    for (::ml_drift::NodeId nid : match.nodes_to_delete) {
      ABSL_RETURN_IF_ERROR(graph->DeleteNode(nid));
    }
    for (::ml_drift::ValueId vid : internal_values_to_delete) {
      if (graph->GetValue(vid) != nullptr &&
          graph->FindProducer(vid) == nullptr &&
          graph->FindConsumers(vid).empty()) {
        ABSL_RETURN_IF_ERROR(graph->DeleteValue(vid));
      }
    }
    dead_value_worklist.insert(dead_value_worklist.end(),
                               match.dead_value_candidates.begin(),
                               match.dead_value_candidates.end());
  }

  return PruneDeadNodes(graph, std::move(dead_value_worklist),
                        orig_graph_outputs);
}

namespace ir {

absl::Status FuseSplitSdpaCacheUpdate(::ml_drift::ir::IrModel* model) {
  if (model == nullptr) {
    return absl::OkStatus();
  }

  const std::string add_op = ToString(::ml_drift::OperationType::kAdd);
  const std::string bmm_op =
      ToString(::ml_drift::OperationType::kBatchedMatmul);
  const std::string concat_op = ToString(::ml_drift::OperationType::kConcat);
  const std::string const_op = ToString(::ml_drift::OperationType::kConstant);
  const std::string max_op = ToString(::ml_drift::OperationType::kMaximum);
  const std::string min_op = ToString(::ml_drift::OperationType::kMinimum);
  const std::string select_op = ToString(::ml_drift::OperationType::kSelectV2);
  const std::string slice_op = ToString(::ml_drift::OperationType::kSlice);
  const std::string softmax_op = ToString(::ml_drift::OperationType::kSoftmax);
  const std::string transpose_op =
      ToString(::ml_drift::OperationType::kTranspose);

  absl::flat_hash_set<::ml_drift::ir::IrTensorId> orig_graph_outputs(
      model->outputs().begin(), model->outputs().end());

  std::vector<IrSplitSdpaMatch> matches;
  absl::flat_hash_set<::ml_drift::ir::IrOpId> matched_op_ids;

  for (size_t i = 0; i < model->ops().size(); ++i) {
    ::ml_drift::ir::IrOp* add_node = model->ops()[i].get();
    if (add_node == nullptr || matched_op_ids.contains(add_node->id) ||
        add_node->name != add_op || add_node->inputs.size() != 2 ||
        add_node->outputs.size() != 1) {
      continue;
    }
    if (!IsSingleUseInternalIr(model, add_node->inputs[0],
                               orig_graph_outputs) ||
        !IsSingleUseInternalIr(model, add_node->inputs[1],
                               orig_graph_outputs)) {
      continue;
    }

    ::ml_drift::ir::IrOp* prod0 = model->FindProducer(add_node->inputs[0]);
    ::ml_drift::ir::IrOp* prod1 = model->FindProducer(add_node->inputs[1]);
    if (prod0 == nullptr || prod1 == nullptr) {
      continue;
    }

    ::ml_drift::ir::IrOp* bmm_past_v = nullptr;
    ::ml_drift::ir::IrOp* bmm_new_v = nullptr;
    if (prod0->name == kRuntimeBatchedMatMulType && prod1->name == bmm_op) {
      bmm_past_v = prod0;
      bmm_new_v = prod1;
    } else if (prod1->name == kRuntimeBatchedMatMulType &&
               prod0->name == bmm_op) {
      bmm_past_v = prod1;
      bmm_new_v = prod0;
    } else {
      continue;
    }

    if (!ValidateRuntimeBmmAttr(bmm_past_v->attr, /*is_src=*/true)) {
      continue;
    }
    const auto* bmm_new_v_attr =
        std::any_cast<::ml_drift::BatchedMatMulAttributes>(&bmm_new_v->attr);
    if (bmm_new_v_attr == nullptr || bmm_new_v_attr->transpose_left) {
      continue;
    }
    if (bmm_past_v->inputs.size() != 3 || bmm_new_v->inputs.size() != 2) {
      continue;
    }

    const ::ml_drift::ir::IrTensorId probs_past = bmm_past_v->inputs[0];
    const ::ml_drift::ir::IrTensorId value_cache = bmm_past_v->inputs[1];
    const ::ml_drift::ir::IrTensorId param_past = bmm_past_v->inputs[2];
    const ::ml_drift::ir::IrTensorId probs_new = bmm_new_v->inputs[0];
    const ::ml_drift::ir::IrTensorId value_new = bmm_new_v->inputs[1];
    const ::ml_drift::ir::IrTensorId attention = add_node->outputs[0];

    const ::ml_drift::ir::IrTensor* vc_t = model->tensor(value_cache);
    const ::ml_drift::ir::IrTensor* vn_t = model->tensor(value_new);
    const ::ml_drift::ir::IrTensor* attn_t = model->tensor(attention);
    if (vc_t == nullptr || vn_t == nullptr || attn_t == nullptr ||
        vc_t->producer.has_value() ||
        !::ml_drift::IsFloatType(vc_t->desc.GetDataType()) ||
        !::ml_drift::IsFloatType(vn_t->desc.GetDataType()) ||
        !::ml_drift::IsFloatType(attn_t->desc.GetDataType())) {
      continue;
    }

    const ::ml_drift::BHWC vc_shape = vc_t->desc.GetBHWCShape();
    const ::ml_drift::BHWC vn_shape = vn_t->desc.GetBHWCShape();
    const ::ml_drift::BHWC out_shape = attn_t->desc.GetBHWCShape();
    const int heads = vc_shape.h;
    const int head_dim = vc_shape.w;
    const int cache_size = vc_shape.c;
    const int rows = out_shape.w;
    if (vc_shape.b != 1 || out_shape.b != 1 || out_shape.h != heads ||
        out_shape.c != head_dim || heads <= 0 || head_dim <= 0 ||
        cache_size <= 0 || head_dim % 4 != 0 || cache_size % 4 != 0) {
      continue;
    }

    const bool value_new_needs_transpose = !bmm_new_v_attr->transpose_right;
    const int new_len = value_new_needs_transpose ? vn_shape.w : vn_shape.c;
    const ::ml_drift::BHWC expected_vn_shape =
        value_new_needs_transpose
            ? ::ml_drift::BHWC(1, heads, new_len, head_dim)
            : ::ml_drift::BHWC(1, heads, head_dim, new_len);
    if (new_len <= 1 || rows <= 0 || rows % new_len != 0 ||
        vn_shape != expected_vn_shape) {
      continue;
    }
    const int g = rows / new_len;

    // If a transpose slot is needed in-place, verify `bmm_new_v` sits strictly
    // after `value_new`'s producer and strictly before `add_node`.
    if (value_new_needs_transpose) {
      if (bmm_new_v->id >= add_node->id) {
        continue;
      }
      if (vn_t->producer.has_value() && *vn_t->producer >= bmm_new_v->id) {
        continue;
      }
    }

    if (!IsSingleUseInternalIr(model, probs_past, orig_graph_outputs) ||
        !IsSingleUseInternalIr(model, probs_new, orig_graph_outputs)) {
      continue;
    }
    ::ml_drift::ir::IrOp* slice_past = model->FindProducer(probs_past);
    ::ml_drift::ir::IrOp* slice_new = model->FindProducer(probs_new);
    if (slice_past == nullptr || slice_new == nullptr ||
        slice_past->name != slice_op || slice_new->name != slice_op ||
        slice_past->inputs.size() != 1 || slice_new->inputs.size() != 1 ||
        slice_past->inputs[0] != slice_new->inputs[0] ||
        !IsUnitStrideSlice(slice_past->attr, ::ml_drift::BHWC(0, 0, 0, 0),
                           ::ml_drift::BHWC(1, heads, rows, cache_size)) ||
        !IsUnitStrideSlice(
            slice_new->attr, ::ml_drift::BHWC(0, 0, 0, cache_size),
            ::ml_drift::BHWC(1, heads, rows, cache_size + new_len))) {
      continue;
    }
    const ::ml_drift::ir::IrTensorId probs_all = slice_past->inputs[0];
    const ::ml_drift::ir::IrTensor* probs_all_t = model->tensor(probs_all);
    if (probs_all_t == nullptr || orig_graph_outputs.contains(probs_all) ||
        probs_all_t->consumers.size() != 2 ||
        probs_all_t->desc.GetBHWCShape() !=
            ::ml_drift::BHWC(1, heads, rows, cache_size + new_len)) {
      continue;
    }

    ::ml_drift::ir::IrOp* softmax_node = model->FindProducer(probs_all);
    if (softmax_node == nullptr || softmax_node->name != softmax_op ||
        softmax_node->inputs.size() != 1) {
      continue;
    }
    const auto* softmax_attr =
        std::any_cast<::ml_drift::SoftmaxAttributes>(&softmax_node->attr);
    if (softmax_attr == nullptr ||
        softmax_attr->axis != ::ml_drift::Axis::kChannels) {
      continue;
    }

    std::vector<::ml_drift::ir::IrOpId> extra_ops;
    ::ml_drift::ir::IrTensorId cur_tid = softmax_node->inputs[0];

    if (IsSingleUseInternalIr(model, cur_tid, orig_graph_outputs)) {
      ::ml_drift::ir::IrOp* min_node = model->FindProducer(cur_tid);
      if (min_node != nullptr && min_node->name == min_op &&
          min_node->inputs.size() == 1) {
        std::optional<float> min_cap =
            GetScalarFloatFromElementwiseAttr(min_node->attr);
        if (min_cap.has_value() &&
            IsSingleUseInternalIr(model, min_node->inputs[0],
                                  orig_graph_outputs)) {
          ::ml_drift::ir::IrOp* max_node =
              model->FindProducer(min_node->inputs[0]);
          if (max_node != nullptr && max_node->name == max_op &&
              max_node->inputs.size() == 1) {
            std::optional<float> max_cap =
                GetScalarFloatFromElementwiseAttr(max_node->attr);
            if (max_cap.has_value() &&
                IsValidFloatCapPair(*min_cap, *max_cap)) {
              extra_ops.push_back(min_node->id);
              extra_ops.push_back(max_node->id);
              cur_tid = max_node->inputs[0];
            }
          }
        }
      }
    }

    if (!IsSingleUseInternalIr(model, cur_tid, orig_graph_outputs)) {
      continue;
    }
    ::ml_drift::ir::IrOp* select_node = model->FindProducer(cur_tid);
    if (select_node == nullptr || select_node->name != select_op ||
        select_node->inputs.size() != 3) {
      continue;
    }
    const ::ml_drift::ir::IrTensorId mask_all = select_node->inputs[0];
    const ::ml_drift::ir::IrTensorId logits_pre_mask = select_node->inputs[1];
    const ::ml_drift::ir::IrTensorId fill_id = select_node->inputs[2];

    ::ml_drift::ir::IrOp* fill_prod = model->FindProducer(fill_id);
    if (fill_prod == nullptr || fill_prod->name != const_op) {
      continue;
    }
    std::optional<float> fill_scalar =
        GetScalarFloatFromConstAttr(fill_prod->attr);
    if (!fill_scalar.has_value() || !IsMaskFillNeg10000(*fill_scalar)) {
      continue;
    }

    std::optional<::ml_drift::ir::IrTensorId> mask_raw =
        MatchOriginalMaskIr(model, mask_all, g, new_len, cache_size);
    if (!mask_raw.has_value()) {
      continue;
    }

    std::optional<float> softcap;
    ::ml_drift::ir::IrTensorId logits_all = logits_pre_mask;
    TryPeelSoftcapIr(model, logits_all, softcap, extra_ops, orig_graph_outputs);

    if (!IsSingleUseInternalIr(model, logits_all, orig_graph_outputs)) {
      continue;
    }
    ::ml_drift::ir::IrOp* concat_logits = model->FindProducer(logits_all);
    if (concat_logits == nullptr || concat_logits->name != concat_op ||
        !IsConcatOnAxis(concat_logits->attr, ::ml_drift::Axis::kChannels) ||
        concat_logits->inputs.size() != 2 ||
        !IsSingleUseInternalIr(model, concat_logits->inputs[0],
                               orig_graph_outputs) ||
        !IsSingleUseInternalIr(model, concat_logits->inputs[1],
                               orig_graph_outputs)) {
      continue;
    }
    const ::ml_drift::ir::IrTensorId logits_past = concat_logits->inputs[0];
    const ::ml_drift::ir::IrTensorId logits_new = concat_logits->inputs[1];
    const ::ml_drift::ir::IrTensor* lp_t = model->tensor(logits_past);
    const ::ml_drift::ir::IrTensor* ln_t = model->tensor(logits_new);
    if (lp_t == nullptr || ln_t == nullptr ||
        lp_t->desc.GetBHWCShape() !=
            ::ml_drift::BHWC(1, heads, rows, cache_size) ||
        ln_t->desc.GetBHWCShape() !=
            ::ml_drift::BHWC(1, heads, rows, new_len)) {
      continue;
    }

    ::ml_drift::ir::IrOp* bmm_past_k = model->FindProducer(logits_past);
    ::ml_drift::ir::IrOp* bmm_new_k = model->FindProducer(logits_new);
    if (bmm_past_k == nullptr || bmm_new_k == nullptr ||
        bmm_past_k->name != kRuntimeBatchedMatMulType ||
        bmm_new_k->name != bmm_op || bmm_past_k->inputs.size() != 3 ||
        bmm_new_k->inputs.size() != 2) {
      continue;
    }
    if (!ValidateRuntimeBmmAttr(bmm_past_k->attr, /*is_src=*/false)) {
      continue;
    }
    const auto* bmm_new_k_attr =
        std::any_cast<::ml_drift::BatchedMatMulAttributes>(&bmm_new_k->attr);
    if (bmm_new_k_attr == nullptr || bmm_new_k_attr->transpose_left ||
        !bmm_new_k_attr->transpose_right) {
      continue;
    }

    const ::ml_drift::ir::IrTensorId query = bmm_past_k->inputs[0];
    const ::ml_drift::ir::IrTensorId key_cache = bmm_past_k->inputs[1];
    const ::ml_drift::ir::IrTensorId param_past_k = bmm_past_k->inputs[2];
    const ::ml_drift::ir::IrTensorId key_new = bmm_new_k->inputs[1];

    const ::ml_drift::ir::IrTensor* q_t = model->tensor(query);
    const ::ml_drift::ir::IrTensor* kc_t = model->tensor(key_cache);
    const ::ml_drift::ir::IrTensor* kn_t = model->tensor(key_new);
    if (bmm_new_k->inputs[0] != query || param_past_k != param_past ||
        q_t == nullptr || kc_t == nullptr || kn_t == nullptr ||
        kc_t->producer.has_value() ||
        kc_t->desc.GetDataType() != vc_t->desc.GetDataType() ||
        !::ml_drift::IsFloatType(q_t->desc.GetDataType()) ||
        !::ml_drift::IsFloatType(kn_t->desc.GetDataType()) ||
        q_t->desc.GetBHWCShape() !=
            ::ml_drift::BHWC(1, heads, rows, head_dim) ||
        kc_t->desc.GetBHWCShape() !=
            ::ml_drift::BHWC(1, heads, cache_size, head_dim) ||
        kn_t->desc.GetBHWCShape() !=
            ::ml_drift::BHWC(1, heads, new_len, head_dim)) {
      continue;
    }

    std::optional<MatchedParamIr> matched_param =
        MatchOriginalParamIr(model, param_past, cache_size);
    if (!matched_param.has_value()) {
      continue;
    }

    // Verify all 7 inputs are already produced strictly before `add_node->id`.
    const ::ml_drift::ir::IrTensorId input_tids[7] = {
        query,
        key_cache,
        value_cache,
        key_new,
        value_new,
        *mask_raw,
        matched_param->param_raw,
    };
    bool inputs_precede_add = true;
    for (::ml_drift::ir::IrTensorId tid : input_tids) {
      const ::ml_drift::ir::IrTensor* t = model->tensor(tid);
      if (t == nullptr ||
          (t->producer.has_value() && *t->producer >= add_node->id)) {
        inputs_precede_add = false;
        break;
      }
    }
    if (!inputs_precede_add) {
      continue;
    }

    ::ml_drift::ir::IrOp* cache_update_op = nullptr;
    std::optional<::ml_drift::ir::IrTensorId> key_cache_out;
    std::optional<::ml_drift::ir::IrTensorId> value_cache_out;
    std::optional<::ml_drift::ir::IrTensorId> param_upd;
    std::optional<::ml_drift::ir::IrTensorId> cu_v_in;
    bool has_incompatible_cache_update = false;

    for (::ml_drift::ir::IrOp* consumer : model->FindConsumers(key_new)) {
      if (consumer == nullptr || consumer->name != kAddValuesToCacheType) {
        continue;
      }
      if (cache_update_op != nullptr) {
        has_incompatible_cache_update = true;
        break;
      }
      if (consumer->inputs.size() != 3 || consumer->outputs.size() != 2 ||
          consumer->inputs[0] != key_new) {
        has_incompatible_cache_update = true;
        break;
      }
      int expected_v_update_ts_idx = -1;
      if (consumer->inputs[1] == value_new) {
        expected_v_update_ts_idx = value_new_needs_transpose ? 2 : 3;
      } else {
        ::ml_drift::ir::IrOp* cu_v_prod =
            model->FindProducer(consumer->inputs[1]);
        if (cu_v_prod != nullptr && cu_v_prod->name == transpose_op &&
            cu_v_prod->inputs.size() == 1 &&
            cu_v_prod->inputs[0] == value_new &&
            IsSwapLastTwoDimsTranspose(cu_v_prod->attr)) {
          expected_v_update_ts_idx = value_new_needs_transpose ? 3 : 2;
        }
      }
      if (expected_v_update_ts_idx < 0 || new_len > cache_size ||
          !ValidateRingCacheUpdateAttr(consumer->attr, heads, cache_size,
                                       head_dim, expected_v_update_ts_idx) ||
          !ValidateUpdateParamIr(model, consumer->inputs[2], *matched_param,
                                 cache_size)) {
        ABSL_VLOG(1) << "ir::FuseSplitSdpaCacheUpdate: rejecting add_node="
                     << add_node->id << " due to cache_update attr/param.";
        has_incompatible_cache_update = true;
        break;
      }
      const ::ml_drift::ir::IrTensor* kco_t =
          model->tensor(consumer->outputs[0]);
      const ::ml_drift::ir::IrTensor* vco_t =
          model->tensor(consumer->outputs[1]);
      if (kco_t == nullptr || vco_t == nullptr || kc_t->consumers.size() != 1 ||
          vc_t->consumers.size() != 1 ||
          !orig_graph_outputs.contains(consumer->outputs[0]) ||
          !orig_graph_outputs.contains(consumer->outputs[1]) ||
          !kco_t->consumers.empty() || !vco_t->consumers.empty() ||
          kco_t->desc.GetBHWCShape() != kc_t->desc.GetBHWCShape() ||
          vco_t->desc.GetBHWCShape() != vc_t->desc.GetBHWCShape() ||
          kco_t->desc.GetDataType() != kc_t->desc.GetDataType() ||
          vco_t->desc.GetDataType() != vc_t->desc.GetDataType()) {
        has_incompatible_cache_update = true;
        break;
      }
      cache_update_op = consumer;
      key_cache_out = consumer->outputs[0];
      value_cache_out = consumer->outputs[1];
      param_upd = consumer->inputs[2];
      cu_v_in = consumer->inputs[1];
    }
    if (has_incompatible_cache_update) {
      continue;
    }

    std::vector<::ml_drift::ir::IrOpId> all_matched = {
        add_node->id,   bmm_new_v->id,    bmm_past_v->id,  slice_new->id,
        slice_past->id, softmax_node->id, select_node->id, concat_logits->id,
        bmm_new_k->id,  bmm_past_k->id,
    };
    all_matched.insert(all_matched.end(), extra_ops.begin(), extra_ops.end());
    if (cache_update_op != nullptr) {
      all_matched.push_back(cache_update_op->id);
    }

    bool overlaps = false;
    for (::ml_drift::ir::IrOpId oid : all_matched) {
      if (matched_op_ids.contains(oid)) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) {
      continue;
    }
    for (::ml_drift::ir::IrOpId oid : all_matched) {
      matched_op_ids.insert(oid);
    }

    IrSplitSdpaMatch match;
    match.query = query;
    match.key_cache = key_cache;
    match.value_cache = value_cache;
    match.key_new = key_new;
    match.value_new = value_new;
    match.mask_raw = *mask_raw;
    match.param_raw = matched_param->param_raw;
    match.attention = attention;
    match.key_cache_out = key_cache_out;
    match.value_cache_out = value_cache_out;
    if (cache_update_op != nullptr) {
      match.cache_update_op_id = cache_update_op->id;
    }
    match.softcap = softcap;
    match.value_new_needs_transpose = value_new_needs_transpose;
    match.heads = heads;
    match.head_dim = head_dim;
    match.new_len = new_len;
    match.transpose_slot_id = bmm_new_v->id;
    match.fused_slot_id = add_node->id;
    for (::ml_drift::ir::IrOpId oid : all_matched) {
      if (oid == match.fused_slot_id) {
        continue;
      }
      if (match.value_new_needs_transpose && oid == match.transpose_slot_id) {
        continue;
      }
      match.ops_to_delete.push_back(oid);
    }
    match.dead_tensor_candidates = {mask_all, fill_id, param_past};
    if (param_upd.has_value()) {
      match.dead_tensor_candidates.push_back(*param_upd);
    }
    if (cu_v_in.has_value() && *cu_v_in != value_new) {
      match.dead_tensor_candidates.push_back(*cu_v_in);
    }
    matches.push_back(std::move(match));
  }

  if (matches.empty()) {
    return absl::OkStatus();
  }

  ABSL_VLOG(1) << "ir::FuseSplitSdpaCacheUpdate: fusing " << matches.size()
               << " sliding-window SDPA subgraph(s) into "
               << kFusedSdpaCacheUpdateType << ".";

  std::vector<::ml_drift::ir::IrTensorId> dead_tensor_worklist;
  for (IrSplitSdpaMatch& match : matches) {
    ::ml_drift::ir::IrTensorId value_new_for_sdpa = match.value_new;
    if (match.value_new_needs_transpose) {
      ::ml_drift::ir::IrOp* transpose_op_ptr =
          model->ops()[match.transpose_slot_id].get();
      ClearIrOpInPlace(model, transpose_op_ptr);
      transpose_op_ptr->name = transpose_op;
      ::ml_drift::TransposeAttributes t_attr;
      t_attr.perm = ::ml_drift::BHWC(0, 1, 3, 2);
      transpose_op_ptr->attr = t_attr;
      const ::ml_drift::DataType v_dtype =
          model->tensor(match.value_new)->desc.GetDataType();
      ::ml_drift::ir::IrTensor* v_transposed = model->add_tensor(
          v_dtype,
          ::ml_drift::BHWC(1, match.heads, match.head_dim, match.new_len));
      model->AddConsumer(match.value_new, transpose_op_ptr->id);
      model->SetProducer(v_transposed->id, transpose_op_ptr->id);
      value_new_for_sdpa = v_transposed->id;
    }

    // Detach cache outputs from `cache_update` before transferring ownership
    // to `fused_op` so `RemoveOp(cache_update)` does not clear their producer.
    if (match.cache_update_op_id.has_value()) {
      ::ml_drift::ir::IrOp* cu_op =
          model->ops()[*match.cache_update_op_id].get();
      if (cu_op != nullptr) {
        for (::ml_drift::ir::IrTensorId out_id : cu_op->outputs) {
          if (::ml_drift::ir::IrTensor* t = model->GetMutableTensor(out_id);
              t != nullptr && t->producer == cu_op->id) {
            t->producer.reset();
          }
        }
        cu_op->outputs.clear();
      }
    }

    ::ml_drift::ir::IrOp* fused_op = model->ops()[match.fused_slot_id].get();
    ClearIrOpInPlace(model, fused_op);
    fused_op->name = kFusedSdpaCacheUpdateType;
    FusedSdpaCacheUpdateAttributes fused_attr;
    fused_attr.softcap = match.softcap;
    fused_attr.update_cache = match.key_cache_out.has_value();
    fused_op->attr = fused_attr;

    model->AddConsumer(match.query, fused_op->id);
    model->AddConsumer(match.key_cache, fused_op->id);
    model->AddConsumer(match.value_cache, fused_op->id);
    model->AddConsumer(match.key_new, fused_op->id);
    model->AddConsumer(value_new_for_sdpa, fused_op->id);
    model->AddConsumer(match.mask_raw, fused_op->id);
    model->AddConsumer(match.param_raw, fused_op->id);

    model->SetProducer(match.attention, fused_op->id);
    if (match.key_cache_out.has_value() && match.value_cache_out.has_value()) {
      model->SetProducer(*match.key_cache_out, fused_op->id);
      model->SetProducer(*match.value_cache_out, fused_op->id);
    }

    // Remove remaining matched ops in descending ID order (consumers before
    // producers) so intermediate tensors are cleanly orphaned and removed.
    std::sort(match.ops_to_delete.begin(), match.ops_to_delete.end(),
              std::greater<::ml_drift::ir::IrOpId>());
    for (::ml_drift::ir::IrOpId oid : match.ops_to_delete) {
      ABSL_RETURN_IF_ERROR(model->RemoveOp(oid));
    }

    dead_tensor_worklist.insert(dead_tensor_worklist.end(),
                                match.dead_tensor_candidates.begin(),
                                match.dead_tensor_candidates.end());
  }

  return PruneDeadOpsIr(model, std::move(dead_tensor_worklist),
                        orig_graph_outputs);
}

}  // namespace ir
}  // namespace litert::ml_drift
