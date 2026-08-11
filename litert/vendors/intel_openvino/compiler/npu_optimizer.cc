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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/pass/constant_folding.hpp"
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

namespace {

// One expert's GEGLU branch:
//   hidden -> up/gate MatMul(W_up) -> Slice/Gelu/Mul (GEGLU) ->
//   down MatMul(W_down) -> Multiply(x router score) -> Add (into accum chain)
struct ExpertBranch {
  int64_t expert_id = -1;                           // read from the Equal const
  std::shared_ptr<ov::op::v0::MatMul> up_matmul;    // gate/up projection
  std::shared_ptr<ov::op::v0::MatMul> down_matmul;  // down projection
  ov::Output<ov::Node> w_up_src;  // weight source to stack (post i4 dequant)
  ov::Output<ov::Node> w_down_src;
  // Raw pre-dequant source, when the up/down weight matches the
  // Multiply(Convert(Constant<u4/i4>), scale) pattern: the packed Constant,
  // its dequant scale, and the Convert's target type. Populated by
  // FindDequantSource(); w_up_has_dequant/w_down_has_dequant are set to
  // false when the pattern is not found.
  ov::Output<ov::Node> w_up_packed;
  ov::Output<ov::Node> w_up_scale;
  ov::element::Type w_up_dequant_type;
  bool w_up_has_dequant = false;
  ov::Output<ov::Node> w_down_packed;
  ov::Output<ov::Node> w_down_scale;
  ov::element::Type w_down_dequant_type;
  bool w_down_has_dequant = false;
  ov::Output<ov::Node> score;           // per-expert router score (mask input)
  ov::Output<ov::Node> router_weights;  // shared [1,K] routing-weight vector
  ov::Output<ov::Node> branch_output;   // masked output feeding the Add chain
  std::shared_ptr<ov::Node> chain_add;  // this branch's link in the Add chain
};

struct MoELayer {
  std::shared_ptr<ov::op::v11::TopK> topk;
  ov::Output<ov::Node> topk_indices;  // TopK output(1)
  int64_t k = 0;                      // active experts (top-K)
  size_t num_experts = 0;             // total experts (=128 for Gemma4-26B)
  std::vector<ExpertBranch> experts;  // filled + sorted by expert_id
  // The router's per-position routing-weight vector [1,K] that dense multiplies
  // each expert by (shared across all experts). Gemma4's router is
  // SoftMax(128) -> TopK -> divide-by-sum renormalize; the vector fed to each
  // expert's Equal-masked ReduceSum is that fully-normalized result, NOT the
  // raw TopK value. Captured in CollectExpertBranch.
  ov::Output<ov::Node> router_weights;
};

// Reads the constant expert-id an Equal(topk_indices, const) compares against.
// Returns nullopt if input(1) is not a 1-element Constant.
std::optional<int64_t> ExtractExpertIdFromEqual(
    const std::shared_ptr<ov::Node>& equal) {
  for (size_t i = 0; i < 2; ++i) {
    if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            equal->input_value(i).get_node_shared_ptr())) {
      if (ov::shape_size(c->get_shape()) == 1) {
        return c->cast_vector<int64_t>().front();
      }
    }
  }
  return std::nullopt;
}

// Walks forward from |out| through single-consumer nodes (max |max_depth|
// hops) looking for the first node matching |pred|. Returns nullptr if the
// chain forks (a consumer count != 1) or |pred| isn't matched in time.
std::shared_ptr<ov::Node> FindDescendant(
    const ov::Output<ov::Node>& out,
    const std::function<bool(const std::shared_ptr<ov::Node>&)>& pred,
    int max_depth) {
  ov::Output<ov::Node> cur = out;
  for (int i = 0; i < max_depth; ++i) {
    if (cur.get_target_inputs().size() != 1) return nullptr;
    auto n = cur.get_target_inputs().begin()->get_node()->shared_from_this();
    if (pred(n)) return n;
    cur = n->output(0);
  }
  return nullptr;
}

// Walks backward from |out| along input(0) (max |max_depth| hops) looking for
// the first ov::op::v0::MatMul. This follows the §0.1 GEGLU chain (Multiply
// -> Gelu -> Slice -> MatMul, or MatMul -> ... -> MatMul) since each of those
// ops keeps the "main" data path on input(0).
std::shared_ptr<ov::op::v0::MatMul> FindAncestorMatmul(
    const ov::Output<ov::Node>& out, int max_depth) {
  std::shared_ptr<ov::Node> n = out.get_node_shared_ptr();
  for (int i = 0; i < max_depth; ++i) {
    if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(n)) return mm;
    if (n->get_input_size() == 0) return nullptr;
    n = n->input_value(0).get_node_shared_ptr();
  }
  return nullptr;
}

// Detects the weight-compression pattern feeding a MatMul weight input:
//   Multiply(Convert(Constant<u4/i4>), scale)
// (operand order may be either way). On match, returns the raw packed
// Constant, the dequant scale, and the Convert's target element type;
// returns false if |weight_src| isn't that exact shape (e.g. weights are
// stored uncompressed), in which case the caller must stack/gather
// |weight_src| directly instead.
bool FindDequantSource(const ov::Output<ov::Node>& weight_src,
                       ov::Output<ov::Node>& packed_out,
                       ov::Output<ov::Node>& scale_out,
                       ov::element::Type& dequant_type_out) {
  auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(
      weight_src.get_node_shared_ptr());
  if (!mul) return false;
  for (size_t i = 0; i < 2; ++i) {
    auto convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(
        mul->input_value(i).get_node_shared_ptr());
    if (!convert) continue;
    auto packed = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        convert->input_value(0).get_node_shared_ptr());
    if (!packed) continue;
    const auto& packed_type = packed->get_element_type();
    if (packed_type != ov::element::u4 && packed_type != ov::element::i4)
      continue;
    packed_out = convert->input_value(0);
    scale_out = mul->input_value(1 - i);
    dequant_type_out = convert->get_element_type();
    return true;
  }
  return false;
}

// Matches one expert's GEGLU branch starting from its Equal mask node
//   Equal(indices==expert_id) -> ReduceSum(score) -> Multiply(mask) -> Add
// and, from the Multiply's non-score input, backs up through
//   down_matmul <- GEGLU (Multiply/Gelu/Slice) <- up_matmul
// Returns false (leaving the graph untouched) if any step doesn't match —
// callers must treat that expert as "not found" rather than guessing.
bool CollectExpertBranch(const std::shared_ptr<ov::Node>& equal_node,
                         ExpertBranch& out) {
  auto id = ExtractExpertIdFromEqual(equal_node);
  if (!id) return false;
  out.expert_id = *id;

  auto reduce = FindDescendant(
      equal_node->output(0),
      [](const std::shared_ptr<ov::Node>& n) {
        return ov::is_type<ov::op::v1::ReduceSum>(n);
      },
      /*max_depth=*/3);
  if (!reduce) return false;
  out.score = reduce->output(0);

  // Capture the router's per-position routing-weight vector.
  auto premul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(
      reduce->input_value(0).get_node_shared_ptr());
  if (!premul) return false;
  auto traces_to_equal = [&](const ov::Output<ov::Node>& o) {
    ov::Node* n = o.get_node();
    for (int i = 0; i < 4 && n; ++i) {
      if (n == equal_node.get()) return true;
      if (n->get_input_size() == 0) break;
      n = n->input_value(0).get_node();
    }
    return false;
  };
  const bool in0_is_mask = traces_to_equal(premul->input_value(0));
  const bool in1_is_mask = traces_to_equal(premul->input_value(1));
  if (in0_is_mask == in1_is_mask) return false;  // can't disambiguate -> bail
  out.router_weights =
      in0_is_mask ? premul->input_value(1) : premul->input_value(0);

  auto mult = FindDescendant(
      reduce->output(0),
      [](const std::shared_ptr<ov::Node>& n) {
        return ov::is_type<ov::op::v1::Multiply>(n);
      },
      /*max_depth=*/2);
  if (!mult) return false;

  auto add = FindDescendant(
      mult->output(0),
      [](const std::shared_ptr<ov::Node>& n) {
        return ov::is_type<ov::op::v1::Add>(n);
      },
      /*max_depth=*/2);
  if (!add) return false;
  out.branch_output = mult->output(0);
  out.chain_add = add;

  // The Multiply's other input is the pre-mask branch (down-proj) output.
  const bool score_is_input0 = mult->input_value(0) == reduce->output(0);
  ov::Output<ov::Node> branch_val =
      score_is_input0 ? mult->input_value(1) : mult->input_value(0);

  auto down_mm = FindAncestorMatmul(branch_val, /*max_depth=*/4);
  if (!down_mm) return false;
  out.down_matmul = down_mm;
  out.w_down_src = down_mm->input_value(1);
  out.w_down_has_dequant =
      FindDequantSource(out.w_down_src, out.w_down_packed, out.w_down_scale,
                        out.w_down_dequant_type);

  auto up_mm = FindAncestorMatmul(down_mm->input_value(0), /*max_depth=*/4);
  if (!up_mm || up_mm == down_mm) return false;
  out.up_matmul = up_mm;
  out.w_up_src = up_mm->input_value(1);
  out.w_up_has_dequant = FindDequantSource(
      out.w_up_src, out.w_up_packed, out.w_up_scale, out.w_up_dequant_type);

  return true;
}

// Stacks per-expert raw Constants (sorted by expert_id) into one grouped
// [N, ...] Constant by directly concatenating their raw byte buffers,
// bypassing the generic Concat/Unsqueeze ops entirely.
//
// This exists specifically for sub-byte packed weight types (u4/i4): OV's
// Concat/Unsqueeze constant-folding evaluate() decompresses such types to a
// wider element type when it materializes the folded result (observed:
// u4/i4 -> a wider type), which would silently multiply the grouped weight
// constant's on-disk size by 2-8x. Building the merged Constant's data
// buffer ourselves guarantees the packed representation survives unchanged.
std::shared_ptr<ov::op::v0::Constant> StackConstantsRaw(
    const ov::OutputVector& per_expert /*sorted*/) {
  if (per_expert.empty()) return nullptr;

  std::vector<std::shared_ptr<ov::op::v0::Constant>> consts;
  consts.reserve(per_expert.size());
  for (const auto& o : per_expert) {
    auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        o.get_node_shared_ptr());
    if (!c) return nullptr;
    consts.push_back(c);
  }

  const ov::element::Type type = consts.front()->get_element_type();
  const ov::Shape per_shape = consts.front()->get_shape();
  const size_t per_elems = ov::shape_size(per_shape);

  // Verify that all constants have the exact same data type and identical
  // shape. Using shape inequality (c->get_shape() != per_shape) is safer than
  // just checking total elements, avoiding silent corruption if dimensions are
  // swapped.
  for (const auto& c : consts) {
    if (c->get_element_type() != type || c->get_shape() != per_shape) {
      return nullptr;
    }
  }

  // Ensure byte-alignment. Sub-byte types (like i4/u4) must perfectly fit into
  // complete bytes to allow safe memcpy operations.
  const size_t bits = type.bitwidth();
  if ((per_elems * bits) % 8 != 0) return nullptr;
  const size_t per_bytes = (per_elems * bits) / 8;

  ov::Shape new_shape;
  new_shape.reserve(per_shape.size() + 1);
  new_shape.push_back(consts.size());
  new_shape.insert(new_shape.end(), per_shape.begin(), per_shape.end());

  std::vector<uint8_t> buffer(per_bytes * consts.size());
  for (size_t i = 0; i < consts.size(); ++i) {
    std::memcpy(buffer.data() + i * per_bytes, consts[i]->get_data_ptr(),
                per_bytes);
  }

  // The ov::op::v0::Constant constructor taking a void* data pointer
  // will internally copy the data from the buffer.
  return std::make_shared<ov::op::v0::Constant>(type, new_shape, buffer.data());
}

// Expands per-expert scalar indices |idx| ([K], i64/i32) into flat row
// indices selecting whole rows out of a [N*rows, cols] flattened table
// (expert e's rows land at [e*rows, (e+1)*rows)).
ov::Output<ov::Node> ExpandExpertRowIndices(const ov::Output<ov::Node>& idx,
                                            int64_t k, int64_t rows) {
  auto col_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2},
                                                std::vector<int64_t>{k, 1});
  auto idx_col = std::make_shared<ov::op::v1::Reshape>(idx, col_shape, false);
  // TopK indices are i32; match idx's type so Multiply/Add don't mismatch.
  const auto idx_type = idx.get_element_type();
  auto rows_const = ov::op::v0::Constant::create(idx_type, ov::Shape{1, 1},
                                                 std::vector<int64_t>{rows});
  auto base = std::make_shared<ov::op::v1::Multiply>(idx_col, rows_const);
  std::vector<int64_t> row_iota(rows);
  std::iota(row_iota.begin(), row_iota.end(), 0);
  auto offsets = ov::op::v0::Constant::create(
      idx_type, ov::Shape{1, static_cast<size_t>(rows)}, row_iota);
  auto expanded = std::make_shared<ov::op::v1::Add>(base, offsets);
  auto flat_shape =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {k * rows});
  return std::make_shared<ov::op::v1::Reshape>(expanded, flat_shape, false)
      ->output(0);
}

// Reshapes a per-expert 3-D packed weight Constant [N, rows, cols] to 2-D
// [N*rows, cols], Gathers whole rows for the K selected experts, then
// reshapes the result back to [K, rows, cols]. The NPU backend's Gather
// kernel is better optimized for 2-D row-gather than for 3-D batched gather
// on sub-byte (i4/u4) tensors, so flattening gets better hardware utilization.
// The (already f32) scale Gathers directly in 3-D and needs none of this.
// Falls back to a direct 3-D Gather if grouped_packed isn't 3-D (shouldn't
// happen in practice, since StackConstantsRaw always produces a 3-D
// constant).
ov::Output<ov::Node> GatherPackedRowsViaFlatten(
    const std::shared_ptr<ov::op::v0::Constant>& grouped_packed,
    const ov::Output<ov::Node>& idx, int64_t k) {
  const ov::Shape& shape = grouped_packed->get_shape();
  auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
  if (shape.size() != 3) {
    LITERT_LOG(
        LITERT_DEBUG,
        "GatherPackedRowsViaFlatten: grouped_packed shape is not 3-D got %zuD"
        ", falling back to direct Gather",
        shape.size());
    return std::make_shared<ov::op::v8::Gather>(grouped_packed, idx, axis0)
        ->output(0);
  }
  const int64_t rows = static_cast<int64_t>(shape[1]);
  const int64_t cols = static_cast<int64_t>(shape[2]);
  auto flat2d_shape = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, cols});
  auto flat2d = std::make_shared<ov::op::v1::Reshape>(grouped_packed,
                                                      flat2d_shape, false);
  auto flat_idx = ExpandExpertRowIndices(idx, k, rows);
  auto gathered2d =
      std::make_shared<ov::op::v8::Gather>(flat2d, flat_idx, axis0);
  auto out3d_shape = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{3}, std::vector<int64_t>{k, rows, cols});
  return std::make_shared<ov::op::v1::Reshape>(gathered2d, out3d_shape, false)
      ->output(0);
}

// Rewrites one MoE layer into gather-K form:
//   1. Validate that ALL experts strictly share uniform quantization patterns
//      and byte-aligned plain Constant weights/scales.
//   2. Stack per-expert weights (sorted by expert_id) into grouped [N,...]
//      constants and Gather the K selected by the router.
//   3. Recompute the up-proj with the gathered weights (batched MatMul of
//      hidden [1,H] against gathered_up_weights [K,gate_up,H] -> [K,1,gate_up]
//      -> [K,gate_up]), then rebuild the GEGLU EXPLICITLY (feature-axis slices
//      + Gelu-TANH) and the down-proj as a batched MatMul. The GEGLU is
//      rebuilt rather than cloned because the model's GEGLU Slice carries a
//      batch-1-baked size that would truncate the K experts back to 1.
//   4. Weight each of the K down-proj outputs by layer.router_weights (the
//      dense graph's actual per-position routing weight [1,K], divide-by-sum
//      renormalized -- NOT the raw TopK value) and ReduceSum over the K axis.
//   5. Replace the final node of the 128-way Add accumulation chain with that
//   sum.
// Returns false (leaving the graph untouched) if any expert lacks valid packed
// weights, shapes/types are non-uniform, or the accumulation chain shape fails.
bool RegroupAndRewrite(const MoELayer& layer) {
  const std::string name = layer.topk->get_friendly_name();
  if (layer.experts.size() != layer.num_experts || layer.k <= 0) {
    LITERT_LOG(LITERT_INFO,
               "[MoEGather] rewrite[%s]: abort: experts.size()=%zu "
               "num_experts=%zu k=%lld",
               name.c_str(), layer.experts.size(), layer.num_experts,
               static_cast<long long>(layer.k));
    return false;
  }
  if (!layer.router_weights.get_node_shared_ptr()) {
    LITERT_LOG(LITERT_INFO,
               "[MoEGather] rewrite[%s]: abort: router weight vector not "
               "captured (unexpected router topology)",
               name.c_str());
    return false;
  }
  // Gather indexes the grouped weight table by expert_id value while rows are
  // stacked in sorted-expert_id order; only correct when the sorted expert_ids
  // are exactly 0..N-1 (contiguous). Confirmed for Gemma4 (branch p tests
  // topk==p and uses weight_p); bail to dense otherwise.
  for (size_t i = 0; i < layer.experts.size(); ++i) {
    if (layer.experts[i].expert_id != static_cast<int64_t>(i)) {
      LITERT_LOG(LITERT_INFO,
                 "[MoEGather] rewrite[%s]: abort: expert_ids not contiguous "
                 "0..N-1 (experts[%zu].expert_id=%lld)",
                 name.c_str(), i,
                 static_cast<long long>(layer.experts[i].expert_id));
      return false;
    }
  }
  // The fused gate+up projection width must be statically known and even (it is
  // sliced in half: gate | up) to reproduce the original two-Slice GEGLU.
  const auto up_ps =
      layer.experts.front().up_matmul->get_output_partial_shape(0);
  const int64_t up_rank =
      up_ps.rank().is_static() ? up_ps.rank().get_length() : 0;
  if (up_rank < 1 || !up_ps[up_rank - 1].is_static() ||
      up_ps[up_rank - 1].get_length() % 2 != 0) {
    LITERT_LOG(LITERT_INFO,
               "[MoEGather] rewrite[%s]: abort: gate+up width not static/even",
               name.c_str());
    return false;
  }
  const int64_t half = up_ps[up_rank - 1].get_length() / 2;

  std::set<ov::Node*> chain_adds;
  for (auto& e : layer.experts) {
    chain_adds.insert(e.chain_add.get());
  }
  std::shared_ptr<ov::Node> final_add;
  int root_count = 0;
  for (auto& e : layer.experts) {
    bool feeds_another_chain_add = false;
    for (const auto& ti : e.chain_add->output(0).get_target_inputs()) {
      if (chain_adds.count(ti.get_node())) {
        // This expert's Add feeds another expert's Add, so it is not a root.
        feeds_another_chain_add = true;
        break;
      }
    }
    if (!feeds_another_chain_add) {
      ++root_count;
      final_add = e.chain_add;
    }
  }
  if (root_count != 1) {
    LITERT_LOG(LITERT_INFO,
               "[MoEGather] rewrite[%s]: abort: found %d chain roots "
               "(expected exactly 1)",
               name.c_str(), root_count);
    return false;
  }

  auto k_shape =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {layer.k});
  auto idx =
      std::make_shared<ov::op::v1::Reshape>(layer.topk_indices, k_shape, false);
  auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

  // Ensure ALL experts share the exact same quantization standard.
  auto& ref = layer.experts.front();
  for (const auto& e : layer.experts) {
    if (!e.w_up_has_dequant || !e.w_down_has_dequant) {
      LITERT_LOG(LITERT_INFO,
                 "[MoEGather] rewrite[%s]: abort: missing i4/u4 packed "
                 "pattern for some experts.",
                 name.c_str());
      return false;
    }
    if (e.w_up_dequant_type != ref.w_up_dequant_type ||
        e.w_down_dequant_type != ref.w_down_dequant_type) {
      LITERT_LOG(
          LITERT_INFO,
          "[MoEGather] rewrite[%s]: abort: mixed dequant types among experts.",
          name.c_str());
      return false;
    }
  }

  // Helper lambda: Extracts constants, stacks them, and builds the Gather
  // subgraph. Returns an empty Output if strict stacking fails.
  auto build_gathered_weights = [&](bool is_up_proj) -> ov::Output<ov::Node> {
    ov::OutputVector packed, scales;
    packed.reserve(layer.experts.size());
    scales.reserve(layer.experts.size());

    for (const auto& e : layer.experts) {
      packed.push_back(is_up_proj ? e.w_up_packed : e.w_down_packed);
      scales.push_back(is_up_proj ? e.w_up_scale : e.w_down_scale);
    }
    auto grouped_packed = StackConstantsRaw(packed);
    auto grouped_scale = StackConstantsRaw(scales);

    // Strict Stacking check: abort if not uniform constants or byte-aligned.
    if (!grouped_packed || !grouped_scale) {
      LITERT_LOG(LITERT_INFO,
                 "[MoEGather] rewrite[%s]: abort: %s weights or scales are not "
                 "uniform plain constants or not byte-aligned.",
                 name.c_str(), is_up_proj ? "up" : "down");
      return ov::Output<ov::Node>();  // Return empty output to signal failure
    }
    auto dequant_type =
        is_up_proj ? ref.w_up_dequant_type : ref.w_down_dequant_type;
    // Flattening to 2-D before Gather gets better hardware utilization on the
    // NPU backend than a direct 3-D batched gather (see
    // GatherPackedRowsViaFlatten). The (already f32) scale Gathers directly in
    // 3-D either way.
    ov::Output<ov::Node> g_packed =
        GatherPackedRowsViaFlatten(grouped_packed, idx, layer.k);
    auto g_scale =
        std::make_shared<ov::op::v8::Gather>(grouped_scale, idx, axis0);
    auto g_dequant =
        std::make_shared<ov::op::v0::Convert>(g_packed, dequant_type);

    return std::make_shared<ov::op::v1::Multiply>(g_dequant, g_scale);
  };

  // Build the strict gather subgraphs
  ov::Output<ov::Node> gathered_up_weights = build_gathered_weights(true);
  if (!gathered_up_weights.get_node_shared_ptr()) return false;
  ov::Output<ov::Node> gathered_down_weights = build_gathered_weights(false);
  if (!gathered_down_weights.get_node_shared_ptr()) return false;

  // Batched up/gate projection. Keep hidden as [1,H] (do NOT squeeze to 1-D --
  // 1-D MatMul lowers poorly on the NPU). MatMul([1,H],
  // gathered_up_weights[K,gate_up,H], transpose_b) broadcasts to [K,1,gate_up];
  // squeeze the size-1 middle dim.
  auto new_up_bcast = ref.up_matmul->clone_with_new_inputs(
      {ref.up_matmul->input_value(0), gathered_up_weights});
  auto sq_axis1 =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto up_2d = std::make_shared<ov::op::v0::Squeeze>(new_up_bcast, sq_axis1);

  // Reproduce the original GEGLU faithfully: two Slice ops (one per consumer
  // chain -- gate half and up half), a Gelu, and a Multiply. The ONLY deviation
  // forced by the weights-gather is the slice axis: the original Slice carries
  // a batch-1-baked size ([1,half]) that would truncate our K experts (axis 0)
  // back to 1, so we slice only the feature axis (axes=[1]) and leave axis 0
  // (the K experts) untouched. gate=[:,0:half] (Gelu-TANH), up=[:,half:2*half].
  auto slice_step =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto slice_axis =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto gate_start =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
  auto gate_stop =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {half});
  auto gate = std::make_shared<ov::op::v8::Slice>(up_2d, gate_start, gate_stop,
                                                  slice_step, slice_axis);
  auto up_start =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {half});
  auto up_stop =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2 * half});
  auto up_half = std::make_shared<ov::op::v8::Slice>(up_2d, up_start, up_stop,
                                                     slice_step, slice_axis);
  auto gate_act = std::make_shared<ov::op::v7::Gelu>(
      gate, ov::op::GeluApproximationMode::TANH);
  auto geglu = std::make_shared<ov::op::v1::Multiply>(gate_act, up_half);

  // Batched down projection: [K,half] -> [K,1,half] to batch-matmul against
  // gathered_down_weights[K,H,half] (transpose_b) -> [K,1,H].
  auto un_axis1 =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto geglu_3d = std::make_shared<ov::op::v0::Unsqueeze>(geglu, un_axis1);
  auto new_down =
      ref.down_matmul->clone_with_new_inputs({geglu_3d, gathered_down_weights});

  // Weight each expert by the dense graph's actual routing-weight vector
  // (renormalized [1,K], captured as layer.router_weights) -- NOT
  // TopK.output(0) which lacks Gemma4's divide-by-sum renormalization -- then
  // sum over K.
  auto weights = std::make_shared<ov::op::v1::Reshape>(layer.router_weights,
                                                       k_shape, false);
  auto weights_bshape = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{3}, std::vector<int64_t>{layer.k, 1, 1});
  auto weights_b =
      std::make_shared<ov::op::v1::Reshape>(weights, weights_bshape, false);
  auto weighted = std::make_shared<ov::op::v1::Multiply>(new_down, weights_b);
  auto reduce_axis =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
  auto summed =
      std::make_shared<ov::op::v1::ReduceSum>(weighted, reduce_axis, false);

  summed->set_friendly_name(final_add->get_friendly_name());
  ov::copy_runtime_info(
      final_add, {new_up_bcast, up_2d, gate, up_half, gate_act, geglu, geglu_3d,
                  new_down, gathered_up_weights.get_node_shared_ptr(),
                  gathered_down_weights.get_node_shared_ptr(), weights,
                  weights_b, weighted, summed});
  ov::replace_node(final_add, summed);
  LITERT_LOG(LITERT_INFO,
             "[MoEGather] rewrite[%s]: OK, replaced chain root '%s' with "
             "gather-K subgraph",
             name.c_str(), summed->get_friendly_name().c_str());
  return true;
}

// Discovers all MoE layers: each router TopK, its expert branches (via the
// Equal consumers of TopK.output(1)), and K.
std::vector<MoELayer> FindMoeLayers(const std::shared_ptr<ov::Model>& model) {
  std::vector<MoELayer> layers;
  for (const auto& node : model->get_ordered_ops()) {
    auto topk = std::dynamic_pointer_cast<ov::op::v11::TopK>(node);
    if (!topk) continue;

    MoELayer layer;
    layer.topk = topk;
    layer.topk_indices = topk->output(1);

    if (auto kc = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            topk->input_value(1).get_node_shared_ptr())) {
      if (ov::shape_size(kc->get_shape()) >= 1)
        layer.k = kc->cast_vector<int64_t>().front();
    }
    if (layer.k <= 0) {
      continue;
    }

    // The indices (output 1) drive one per-expert Equal mask each.
    std::vector<std::shared_ptr<ov::Node>> equals;
    for (const auto& ti : layer.topk_indices.get_target_inputs()) {
      auto n = ti.get_node()->shared_from_this();
      if (std::dynamic_pointer_cast<ov::op::v1::Equal>(n)) equals.push_back(n);
    }
    if (equals.size() < 2) continue;  // not a router-style TopK

    for (const auto& eq : equals) {
      ExpertBranch br;
      if (CollectExpertBranch(eq, br)) layer.experts.push_back(std::move(br));
    }
    if (layer.experts.size() != equals.size()) {
      LITERT_LOG(LITERT_DEBUG,
                 "[MoE] layer TopK='%s' K=%lld experts=%zu: some experts "
                 "failed to match GEGLU branch",
                 layer.topk->get_friendly_name().c_str(),
                 static_cast<long long>(layer.k), layer.experts.size());
      continue;
    }
    layer.num_experts = layer.experts.size();

    // Decode/generate only: the gather form assumes a single token. At this
    // point (right after the TFLite frontend, before shape resolution) the
    // batch dim may still be dynamic, so only reject when it is STATICALLY not
    // 1 — matching NPUW's DeviceRoutedMoETransform guard. The concrete K used
    // for the gather comes from the TopK 'k' constant, not from this shape.
    const auto ishape = layer.topk_indices.get_partial_shape();
    bool decode_ok = true;
    if (ishape.rank().is_static() && ishape.rank().get_length() >= 1) {
      if (ishape[0].is_static() && ishape[0].get_length() != 1) {
        decode_ok = false;
      }
    }
    std::ostringstream shp;
    shp << ishape;
    LITERT_LOG(LITERT_DEBUG,
               "[MoE] layer TopK='%s' K=%lld experts=%zu indices=%s%s",
               layer.topk->get_friendly_name().c_str(),
               static_cast<long long>(layer.k), layer.experts.size(),
               shp.str().c_str(), decode_ok ? "" : " [skip: batch != 1]");
    if (!decode_ok) continue;

    std::sort(layer.experts.begin(), layer.experts.end(),
              [](const ExpertBranch& a, const ExpertBranch& b) {
                return a.expert_id < b.expert_id;
              });
    // The routing-weight vector is shared across all experts; lift it to layer.
    if (!layer.experts.empty()) {
      layer.router_weights = layer.experts.front().router_weights;
    }
    layers.push_back(std::move(layer));
  }
  return layers;
}

}  // namespace

bool MoEGatherRewrite::run_on_model(const std::shared_ptr<ov::Model>& model) {
  // 1. Find all TopK nodes that look like MoE routers.
  auto layers = FindMoeLayers(model);
  LITERT_LOG(LITERT_INFO, "[MoEGather] discovered %zu candidate MoE layer(s)",
             layers.size());

  // 2. For each candidate, attempt to rewrite it into gather-K form.
  bool changed = false;
  for (auto& layer : layers) {
    if (RegroupAndRewrite(layer)) {
      changed = true;
    }
  }

  return changed;
}

void NpuOptimizer::Run(const std::shared_ptr<ov::Model>& model) const {
  ov::pass::Manager pass_manager;
  // First: fold constants (removes dynamic shapes) before the passes below.
  if (constant_fold_) {
    pass_manager.register_pass<ov::pass::ConstantFolding>();
  }
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
  if (enable_moe_gather_) {
    pass_manager.register_pass<MoEGatherRewrite>();
  }
  pass_manager.run_passes(model);
}

}  // namespace openvino
}  // namespace litert
