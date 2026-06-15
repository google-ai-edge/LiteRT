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

#include "litert/vendors/intel_openvino/compiler/npu_optimizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace litert {
namespace openvino {
namespace {

// The Intel NPU SDPA kernel requires the merged key/value sequence dimension
// to be a multiple of this alignment.
constexpr int64_t kSdpaKvSeqAlignment = 16;

// Additive-mask sentinel used for padded key positions. Value taken from
// Gemma models.
constexpr float kSdpaPadMaskBias = -100.0f;

// Returns true if |output| feeds exactly one consumer input.
bool HasSingleConsumer(const ov::Output<ov::Node>& output) {
  return output.get_target_inputs().size() == 1;
}

// Pads |input| at the end of the (possibly negative) |axis| by |pad_amount|
// elements, filling with |pad_value|. |rank| is the static rank of |input|.
// Returns the original output unchanged when |pad_amount| is zero.
ov::Output<ov::Node> PadEndOfAxis(const ov::Output<ov::Node>& input,
                                  int64_t rank, int64_t axis,
                                  int64_t pad_amount, float pad_value) {
  if (pad_amount == 0) {
    return input;
  }
  const int64_t norm_axis = axis < 0 ? axis + rank : axis;
  std::vector<int64_t> begin(rank, 0);
  std::vector<int64_t> end(rank, 0);
  end[norm_axis] = pad_amount;
  auto pads_begin = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{static_cast<size_t>(rank)}, begin);
  auto pads_end = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{static_cast<size_t>(rank)}, end);
  auto value = ov::op::v0::Constant::create(input.get_element_type(),
                                            ov::Shape{}, {pad_value});
  return std::make_shared<ov::op::v1::Pad>(input, pads_begin, pads_end, value,
                                           ov::op::PadMode::CONSTANT)
      ->output(0);
}

}  // namespace

EliminateMatMulFakeQuantize::EliminateMatMulFakeQuantize() {
  namespace pattern = ov::pass::pattern;
  auto matmul_pattern = pattern::wrap_type<ov::op::v0::MatMul>(
      {pattern::any_input(), pattern::any_input()},
      pattern::consumers_count(1));
  auto fq_pattern = pattern::wrap_type<ov::op::v0::FakeQuantize>(
      {matmul_pattern, pattern::any_input(), pattern::any_input(),
       pattern::any_input(), pattern::any_input()});

  ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
    auto pattern_map = m.get_pattern_value_map();
    auto matmul = pattern_map[matmul_pattern];
    auto fq = pattern_map[fq_pattern].get_node_shared_ptr();

    ov::copy_runtime_info(fq, matmul.get_node_shared_ptr());
    ov::replace_node(fq, matmul.get_node_shared_ptr());
    return true;
  };

  auto m = std::make_shared<pattern::Matcher>(fq_pattern,
                                              "EliminateMatMulFakeQuantize");
  register_matcher(m, callback);
}

CastIntegerSignToFloat::CastIntegerSignToFloat() {
  namespace pattern = ov::pass::pattern;
  auto sign_pattern = pattern::wrap_type<ov::op::v0::Sign>(
      {pattern::any_input()}, [](const ov::Output<ov::Node>& output) {
        const auto node = output.get_node_shared_ptr();
        // Match Sign nodes whose input element type is a non-real (integer)
        // type. Float types are already supported by the NPU plugin.
        return !node->get_input_element_type(0).is_real();
      });

  ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
    auto pattern_map = m.get_pattern_value_map();
    auto sign_node = std::dynamic_pointer_cast<ov::op::v0::Sign>(
        pattern_map[sign_pattern].get_node_shared_ptr());
    if (!sign_node) {
      return false;
    }

    const auto original_int_type = sign_node->get_output_element_type(0);
    auto to_fp32 = std::make_shared<ov::op::v0::Convert>(
        sign_node->input_value(0), ov::element::f32);
    auto sign_fp32 = std::make_shared<ov::op::v0::Sign>(to_fp32);
    auto cast_back =
        std::make_shared<ov::op::v0::Convert>(sign_fp32, original_int_type);

    cast_back->set_friendly_name(sign_node->get_friendly_name());
    ov::copy_runtime_info(sign_node, {to_fp32, sign_fp32, cast_back});
    ov::replace_node(sign_node, cast_back);
    return true;
  };

  auto m = std::make_shared<pattern::Matcher>(sign_pattern,
                                              "CastIntegerSignToFloat");
  register_matcher(m, callback);
}

FuseSplitAttentionToSDPA::FuseSplitAttentionToSDPA(bool pad_kv_to_alignment) {
  namespace pattern = ov::pass::pattern;

  auto q_input = pattern::any_input();
  auto k_cache_input = pattern::any_input();
  auto k_new_input = pattern::any_input();
  auto v_cache_input = pattern::any_input();
  auto v_new_input = pattern::any_input();
  auto mask_input = pattern::any_input();

  auto qk_cache = pattern::wrap_type<ov::op::v0::MatMul>(
      {q_input, k_cache_input}, pattern::consumers_count(1));
  auto qk_new = pattern::wrap_type<ov::op::v0::MatMul>(
      {q_input, k_new_input}, pattern::consumers_count(1));

  auto scores_concat = pattern::wrap_type<ov::op::v0::Concat>(
      {qk_cache, qk_new}, pattern::consumers_count(1));
  auto masked_scores = pattern::wrap_type<ov::op::v1::Add>(
      {scores_concat, mask_input}, pattern::consumers_count(1));
  auto softmax = pattern::wrap_type<ov::op::v8::Softmax>(
      {masked_scores}, pattern::consumers_count(1));

  auto attn_cache = pattern::wrap_type<ov::op::v0::MatMul>(
      {pattern::any_input(), v_cache_input});
  auto attn_new = pattern::wrap_type<ov::op::v0::MatMul>(
      {pattern::any_input(), v_new_input});
  auto output_add = pattern::wrap_type<ov::op::v1::Add>({attn_cache, attn_new});

  ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
    const std::string root_name = m.get_match_root()->get_friendly_name();
    auto add_node =
        std::dynamic_pointer_cast<ov::op::v1::Add>(m.get_match_root());
    if (!add_node) {
      LITERT_LOG(LITERT_ERROR,
                 "FuseSplitAttentionToSDPA[%s]: reject: root is not v1::Add",
                 root_name.c_str());
      return false;
    }

    auto v_matmul_cache = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
        add_node->input_value(0).get_node_shared_ptr());
    auto v_matmul_new = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
        add_node->input_value(1).get_node_shared_ptr());
    if (!v_matmul_cache || !v_matmul_new) {
      return false;
    }

    // Both V matmuls' first input must be a Slice / StridedSlice over a
    // common Softmax.
    auto cache_src_node = v_matmul_cache->input_value(0).get_node_shared_ptr();
    auto new_src_node = v_matmul_new->input_value(0).get_node_shared_ptr();
    const bool cache_is_slice =
        ov::is_type<ov::op::v1::StridedSlice>(cache_src_node) ||
        ov::is_type<ov::op::v8::Slice>(cache_src_node);
    const bool new_is_slice =
        ov::is_type<ov::op::v1::StridedSlice>(new_src_node) ||
        ov::is_type<ov::op::v8::Slice>(new_src_node);
    if (!cache_is_slice || !new_is_slice) {
      return false;
    }
    auto sm_cache = std::dynamic_pointer_cast<ov::op::v8::Softmax>(
        cache_src_node->input_value(0).get_node_shared_ptr());
    auto sm_new = std::dynamic_pointer_cast<ov::op::v8::Softmax>(
        new_src_node->input_value(0).get_node_shared_ptr());
    if (!sm_cache || sm_cache != sm_new) {
      LITERT_LOG(LITERT_DEBUG,
                 "FuseSplitAttentionToSDPA[%s]: reject: slices do not share a "
                 "common v8::Softmax source (cache_sm='%s', new_sm='%s')",
                 root_name.c_str(), cache_src_node->get_type_name(),
                 new_src_node->get_type_name());
      return false;
    }
    auto softmax_node = sm_cache;

    // Softmax must be on the last axis.
    auto sm_rank = softmax_node->get_output_partial_shape(0).rank();
    if (sm_rank.is_dynamic()) {
      return false;
    }
    int64_t sm_axis = static_cast<int64_t>(softmax_node->get_axis());
    if (sm_axis < 0) sm_axis += sm_rank.get_length();
    if (sm_axis != sm_rank.get_length() - 1) {
      return false;
    }

    auto mask_add_node = std::dynamic_pointer_cast<ov::op::v1::Add>(
        softmax_node->input_value(0).get_node_shared_ptr());
    if (!mask_add_node) {
      return false;
    }

    auto concat_node = std::dynamic_pointer_cast<ov::op::v0::Concat>(
        mask_add_node->input_value(0).get_node_shared_ptr());
    if (!concat_node || concat_node->get_input_size() != 2) {
      return false;
    }
    auto concat_rank = concat_node->get_output_partial_shape(0).rank();
    if (concat_rank.is_dynamic()) {
      return false;
    }
    int64_t concat_axis = concat_node->get_axis();
    if (concat_axis < 0) concat_axis += concat_rank.get_length();
    if (concat_axis != concat_rank.get_length() - 1) {
      return false;
    }

    auto qk_cache_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
        concat_node->input_value(0).get_node_shared_ptr());
    auto qk_new_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
        concat_node->input_value(1).get_node_shared_ptr());
    if (!qk_cache_node || !qk_new_node) {
      return false;
    }

    // Both QK matmuls must share Q and agree on transpose flags.
    if (qk_cache_node->input_value(0) != qk_new_node->input_value(0)) {
      return false;
    }
    if (qk_cache_node->get_transpose_a() || qk_new_node->get_transpose_a()) {
      return false;
    }
    if (qk_cache_node->get_transpose_b() != qk_new_node->get_transpose_b()) {
      LITERT_LOG(LITERT_DEBUG,
                 "FuseSplitAttentionToSDPA[%s]: reject: QK MatMul transpose_b "
                 "flags disagree (cache=%d, new=%d)",
                 root_name.c_str(), qk_cache_node->get_transpose_b(),
                 qk_new_node->get_transpose_b());
      return false;
    }
    if (v_matmul_cache->get_transpose_a() || v_matmul_new->get_transpose_a()) {
      return false;
    }
    if (v_matmul_cache->get_transpose_b() != v_matmul_new->get_transpose_b()) {
      LITERT_LOG(LITERT_DEBUG,
                 "FuseSplitAttentionToSDPA[%s]: reject: V MatMul transpose_b "
                 "flags disagree (cache=%d, new=%d)",
                 root_name.c_str(), v_matmul_cache->get_transpose_b(),
                 v_matmul_new->get_transpose_b());
      return false;
    }

    auto q = qk_cache_node->input_value(0);
    auto k_cache = qk_cache_node->input_value(1);
    auto k_new = qk_new_node->input_value(1);
    auto v_cache = v_matmul_cache->input_value(1);
    auto v_new = v_matmul_new->input_value(1);
    auto mask_value = mask_add_node->input_value(1);

    // KV-cache sharing guard: K_cache / V_cache must each feed exactly one
    // consumer (the QK / attn*V MatMul we are about to fuse). If the same KV
    // tensor is consumed elsewhere — e.g. multiple attention layers sharing
    // the same KV cache — rewriting it as part of this fusion would alter the
    // graph seen by the other consumer. Skip such layers.
    if (!HasSingleConsumer(k_cache) || !HasSingleConsumer(v_cache)) {
      LITERT_LOG(LITERT_DEBUG,
                 "FuseSplitAttentionToSDPA[%s]: reject: K_cache or V_cache is "
                 "shared with other consumers (K=%zu, V=%zu)",
                 root_name.c_str(), k_cache.get_target_inputs().size(),
                 v_cache.get_target_inputs().size());
      return false;
    }

    // Require static 4-D ranks for predictable layout reasoning.
    const char* const in_names[] = {"Q", "K_cache", "K_new", "V_cache",
                                    "V_new"};
    const ov::Output<ov::Node> ins[] = {q, k_cache, k_new, v_cache, v_new};
    for (size_t i = 0; i < 5; ++i) {
      const auto& ps = ins[i].get_partial_shape();
      if (ps.rank().is_dynamic() || ps.rank().get_length() != 4) {
        LITERT_LOG(LITERT_DEBUG,
                   "FuseSplitAttentionToSDPA[%s]: reject: %s does not have "
                   "static 4-D rank (rank=%s)",
                   root_name.c_str(), in_names[i],
                   ps.rank().is_dynamic()
                       ? "dynamic"
                       : std::to_string(ps.rank().get_length()).c_str());
        return false;
      }
    }

    // Interpret MatMul transpose flags as a physical-layout hint:
    //   qk transpose_b == true  -> K stored as [B,H,S,D] (standard).
    //   qk transpose_b == false -> K stored as [B,H,D,S] (already K^T).
    //   attn transpose_b == true  -> V stored as [B,H,D,S].
    //   attn transpose_b == false -> V stored as [B,H,S,D] (standard).
    const bool k_is_transposed = !qk_cache_node->get_transpose_b();
    const bool v_is_transposed = v_matmul_cache->get_transpose_b();

    const int64_t k_concat_axis = k_is_transposed ? 3 : 2;
    const int64_t v_concat_axis = v_is_transposed ? 3 : 2;

    // The merged KV sequence length determines whether padding is needed.
    // After Concat, the (logical) S_kv dim is K_cache.S + K_new.S; the NPU
    // SDPA kernel requires this be a multiple of kSdpaKvSeqAlignment.
    const auto& k_cache_ps = k_cache.get_partial_shape();
    const auto& k_new_ps = k_new.get_partial_shape();
    // The S dimension of K is at axis 2 if standard, axis 3 if pre-transposed.
    const int64_t k_seq_axis = k_concat_axis;
    const bool kv_len_static =
        k_cache_ps[k_seq_axis].is_static() && k_new_ps[k_seq_axis].is_static();
    int64_t kv_len = 0;
    int64_t kv_pad = 0;
    if (kv_len_static) {
      kv_len = k_cache_ps[k_seq_axis].get_length() +
               k_new_ps[k_seq_axis].get_length();
      const int64_t aligned =
          ((kv_len + kSdpaKvSeqAlignment - 1) / kSdpaKvSeqAlignment) *
          kSdpaKvSeqAlignment;
      kv_pad = aligned - kv_len;
    }
    if ((!pad_kv_to_alignment && (!kv_len_static || kv_pad != 0))) {
      LITERT_LOG(LITERT_DEBUG,
                 "FuseSplitAttentionToSDPA[%s]: reject: merged KV length is "
                 "not %lld-aligned and pad_kv_to_alignment is off "
                 "(kv_len_static=%d, kv_len=%lld, kv_pad=%lld)",
                 root_name.c_str(), static_cast<long long>(kSdpaKvSeqAlignment),
                 static_cast<int>(kv_len_static),
                 static_cast<long long>(kv_len),
                 static_cast<long long>(kv_pad));
      return false;
    }

    auto k_concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{k_cache, k_new}, k_concat_axis);
    auto v_concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{v_cache, v_new}, v_concat_axis);

    std::shared_ptr<ov::Node> k_input = k_concat;
    std::shared_ptr<ov::Node> v_input = v_concat;

    ov::NodeVector new_nodes{k_concat, v_concat};
    auto add_transpose = [&new_nodes](std::shared_ptr<ov::Node>& target) {
      auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4},
                                               {0, 1, 3, 2});
      auto transpose = std::make_shared<ov::op::v1::Transpose>(target, perm);
      new_nodes.push_back(perm);
      new_nodes.push_back(transpose);
      target = transpose;
    };
    if (k_is_transposed) add_transpose(k_input);
    if (v_is_transposed) add_transpose(v_input);

    // After Concat (+ optional Transpose), K and V are in standard
    // [B,H,S_kv,D] layout. Pad the S_kv dim (axis -2) up to the alignment.
    // PadEndOfAxis short-circuits to the original output when kv_pad == 0,
    // so the explicit guard is unnecessary.
    ov::Output<ov::Node> key_out =
        PadEndOfAxis(k_input->output(0), /*rank=*/4, /*axis=*/-2, kv_pad,
                     /*pad_value=*/0.0f);
    ov::Output<ov::Node> val_out =
        PadEndOfAxis(v_input->output(0), /*rank=*/4, /*axis=*/-2, kv_pad,
                     /*pad_value=*/0.0f);
    if (kv_pad > 0) {
      new_nodes.push_back(key_out.get_node_shared_ptr());
      new_nodes.push_back(val_out.get_node_shared_ptr());
    }

    // Mask. Its KV axis (-1) must match the (possibly padded) KV length, so
    // pad it by the same amount with a large finite negative bias to mask
    // the padded positions in softmax (kSdpaPadMaskBias is -3e4: small enough
    // that exp underflows, large enough that fp16 stays finite, avoiding
    // NaN in flash-attention tile rescaling).
    const int64_t mask_rank =
        mask_value.get_partial_shape().rank().is_static()
            ? mask_value.get_partial_shape().rank().get_length()
            : 4;
    ov::Output<ov::Node> attn_mask = PadEndOfAxis(
        mask_value, mask_rank, /*axis=*/-1, kv_pad, kSdpaPadMaskBias);
    if (kv_pad > 0) {
      new_nodes.push_back(attn_mask.get_node_shared_ptr());
    }

    // Scale = 1.0: any required scaling is assumed pre-applied to Q.
    auto scale_const = ov::op::v0::Constant::create(
        q.get_element_type(), ov::Shape{}, std::vector<float>{1.0f});
    new_nodes.push_back(scale_const);

    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
        q, key_out, val_out, attn_mask, scale_const, /*is_causal=*/false);
    new_nodes.push_back(sdpa);

    sdpa->set_friendly_name(add_node->get_friendly_name());
    ov::copy_runtime_info(m.get_matched_nodes(), new_nodes);
    ov::replace_node(add_node, sdpa);
    LITERT_LOG(LITERT_DEBUG,
               "FuseSplitAttentionToSDPA: fused split-attention into "
               "v13::ScaledDotProductAttention '%s' "
               "(k_pre_transposed=%d, v_pre_transposed=%d, "
               "kv_len=%lld, kv_pad=%lld)",
               sdpa->get_friendly_name().c_str(),
               static_cast<int>(k_is_transposed),
               static_cast<int>(v_is_transposed),
               static_cast<long long>(kv_len), static_cast<long long>(kv_pad));
    return true;
  };

  auto m = std::make_shared<pattern::Matcher>(output_add,
                                              "FuseSplitAttentionToSDPA");

  register_matcher(m, callback);
}

void NpuOptimizer::Run(const std::shared_ptr<ov::Model>& model) const {
  ov::pass::Manager pass_manager;
  if (cast_integer_sign_to_float_) {
    pass_manager.register_pass<CastIntegerSignToFloat>();
  }
  if (fuse_split_attention_to_sdpa_) {
    pass_manager.register_pass<FuseSplitAttentionToSDPA>(
        sdpa_pad_kv_to_alignment_);
  }
  if (eliminate_matmul_fq_) {
    pass_manager.register_pass<EliminateMatMulFakeQuantize>();
  }
  pass_manager.run_passes(model);
}

}  // namespace openvino
}  // namespace litert
