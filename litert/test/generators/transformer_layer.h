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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSFORMER_LAYER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSFORMER_LAYER_H_

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

enum class AttentionType { kMHA, kGQA };
enum class NormType { kLayerNorm, kRMSNorm };
enum class FfnType { kStandard, kSwiGLU };

template <typename AttentionT, typename NormT, typename FfnT,
          typename DataTypeT>
class TransformerLayer : public TestGraph {
 private:
  static constexpr AttentionType kAttention = AttentionT::value;
  static constexpr NormType kNorm = NormT::value;
  static constexpr FfnType kFfn = FfnT::value;
  using T = DataTypeT;
  static constexpr ElementType kElementType = GetElementType<T>();

  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  struct Params {
    int batch_size = 1;
    int seq_len = 16;
    int num_heads = 4;
    int head_dim = 32;
    int emb_dim = 128;
    int hidden_dim = 256;
    int n_kv_groups = 2;  // For GQA
  };

 public:
  using Ptr = std::unique_ptr<TransformerLayer>;

  static constexpr absl::string_view Name() { return "TransformerLayer"; }

  template <typename Rng>
  static Expected<TransformerLayer::Ptr> Create(Rng& rng) {
    Params params;
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params, rng));
    return std::make_unique<TransformerLayer>(params, std::move(model));
  }

  bool HasReference() const override {
    return false;
  }  // Fallback to CPU reference

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    std::array<Layout::Dim, 3> shape = {params_.batch_size, params_.seq_len,
                                        params_.emb_dim};
    LITERT_ASSIGN_OR_RETURN(auto input, SimpleBuffer::Create<T>(shape));
    auto constrained = data_builder;
    if (!constrained.IsFloatDummy()) {
      constrained.SetFloatRange(-1.0f, 1.0f);
    }
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T>(constrained, device)));
    inputs.push_back(std::move(input));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return Error(kLiteRtStatusErrorUnsupported);
  }

  TransformerLayer(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(params) {}

 private:
  template <typename Rng>
  static TensorTf MakeWeight(absl::string_view name,
                             const std::vector<int>& shape, Rng& rng) {
    size_t count = 1;
    for (int d : shape) count *= d;
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
      data[i] =
          static_cast<float>(rng()) / static_cast<float>(Rng::max()) * 0.2f -
          0.1f;
    }
    return litert::tensor::Create(
        name, litert::tensor::ApiType<T>::value, shape,
        litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kFP32>(
            data));
  }

  static TensorTf StandardRmsNorm(const TensorTf& input,
                                  const TensorTf& scale) {
    TensorTf x_squared = Mul(input, input);
    const auto& input_shape = input.GetShape();
    int last_axis = static_cast<int>(input_shape.size()) - 1;
    TensorTf mean_squared =
        Mean(x_squared, /*axes=*/{last_axis}, /*keep_dims=*/true);
    TensorTf eps_tensor = litert::tensor::Create(
        "rms_eps", litert::tensor::Type::kFP32, /*shape=*/{1},
        litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kFP32>(
            {1e-6f}));
    TensorTf variance_plus_eps = Add(mean_squared, eps_tensor);
    TensorTf inv_rms = Rsqrt(variance_plus_eps);
    TensorTf x_norm = Mul(input, inv_rms);
    return Mul(x_norm, scale);
  }

  static TensorTf StandardLayerNorm(const TensorTf& input,
                                    const TensorTf& scale,
                                    const TensorTf& bias) {
    const auto& input_shape = input.GetShape();
    int last_axis = static_cast<int>(input_shape.size()) - 1;
    TensorTf mean = Mean(input, /*axes=*/{last_axis}, /*keep_dims=*/true);
    TensorTf x_centered = Sub(input, mean);
    TensorTf x_centered_sq = Mul(x_centered, x_centered);
    TensorTf variance =
        Mean(x_centered_sq, /*axes=*/{last_axis}, /*keep_dims=*/true);
    TensorTf eps_tensor = litert::tensor::Create(
        "layernorm_eps", litert::tensor::Type::kFP32, /*shape=*/{1},
        litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kFP32>(
            {1e-5f}));
    TensorTf variance_plus_eps = Add(variance, eps_tensor);
    TensorTf inv_std = Rsqrt(variance_plus_eps);
    TensorTf x_norm = Mul(x_centered, inv_std);
    TensorTf scaled = Mul(x_norm, scale);
    return Add(scaled, bias);
  }

  template <typename Rng>
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params,
                                                Rng& rng) {
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value,
        /*shape=*/{params.batch_size, params.seq_len, params.emb_dim});

    // 1. Pre-attention Normalization
    TensorTf normed_input;
    if constexpr (kNorm == NormType::kRMSNorm) {
      TensorTf norm1_scale = MakeWeight("norm1_scale", {params.emb_dim}, rng);
      normed_input = StandardRmsNorm(input, norm1_scale);
    } else {
      TensorTf norm1_scale = MakeWeight("norm1_scale", {params.emb_dim}, rng);
      TensorTf norm1_bias = MakeWeight("norm1_bias", {params.emb_dim}, rng);
      normed_input = StandardLayerNorm(input, norm1_scale, norm1_bias);
    }

    // 2. Attention
    int qkv_dim = params.num_heads * params.head_dim;
    int kv_dim = (kAttention == AttentionType::kGQA)
                     ? params.n_kv_groups * params.head_dim
                     : qkv_dim;

    TensorTf q_proj = MakeWeight("q_proj", {qkv_dim, params.emb_dim}, rng);
    TensorTf k_proj = MakeWeight("k_proj", {kv_dim, params.emb_dim}, rng);
    TensorTf v_proj = MakeWeight("v_proj", {kv_dim, params.emb_dim}, rng);
    TensorTf o_proj = MakeWeight("o_proj", {params.emb_dim, qkv_dim}, rng);

    TensorTf q = FullyConnected(normed_input, q_proj);
    TensorTf k = FullyConnected(normed_input, k_proj);
    TensorTf v = FullyConnected(normed_input, v_proj);

    // Reshape for Attention: [batch, seq, heads, head_dim]
    int kv_heads = (kAttention == AttentionType::kGQA) ? params.n_kv_groups
                                                       : params.num_heads;
    const std::vector<int> kPerm = {0, 2, 1, 3};
    q = litert::tensor::Reshape(
        q, /*new_shape=*/{params.batch_size, params.seq_len, params.num_heads,
                          params.head_dim});
    q = Transpose(q, /*perm=*/kPerm);  // [batch, heads, seq, head_dim]

    k = litert::tensor::Reshape(
        k, /*new_shape=*/{params.batch_size, params.seq_len, kv_heads,
                          params.head_dim});
    k = Transpose(k, /*perm=*/kPerm);  // [batch, kv_heads, seq, head_dim]

    v = litert::tensor::Reshape(
        v, /*new_shape=*/{params.batch_size, params.seq_len, kv_heads,
                          params.head_dim});
    v = Transpose(v, /*perm=*/kPerm);  // [batch, kv_heads, seq, head_dim]

    if constexpr (kAttention == AttentionType::kGQA) {
      int num_groups = params.num_heads / params.n_kv_groups;
      if (num_groups > 1) {
        k = Tile(k, /*multipliers=*/{1, num_groups, 1, 1});
        v = Tile(v, /*multipliers=*/{1, num_groups, 1, 1});
      }
    }

    TensorTf scores = BatchMatMul(q, k, /*adj_x=*/false, /*adj_y=*/true);
    TensorTf scale_tensor = litert::tensor::Create(
        "attn_scale", litert::tensor::Type::kFP32, /*shape=*/{1},
        litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kFP32>(
            {1.0f / std::sqrt(static_cast<float>(params.head_dim))}));
    scores = Mul(scores, scale_tensor);
    TensorTf attn_weights = Softmax(scores);
    TensorTf attn_out = BatchMatMul(attn_weights, v);

    attn_out = Transpose(attn_out, /*perm=*/kPerm);
    attn_out = litert::tensor::Reshape(
        attn_out, /*new_shape=*/{params.batch_size, params.seq_len, qkv_dim});
    TensorTf proj_out = FullyConnected(attn_out, o_proj);

    // 3. Residual 1
    TensorTf residual_1 = Add(input, proj_out);

    // 4. Pre-FFN Normalization
    TensorTf normed_residual;
    if constexpr (kNorm == NormType::kRMSNorm) {
      TensorTf norm2_scale = MakeWeight("norm2_scale", {params.emb_dim}, rng);
      normed_residual = StandardRmsNorm(residual_1, norm2_scale);
    } else {
      TensorTf norm2_scale = MakeWeight("norm2_scale", {params.emb_dim}, rng);
      TensorTf norm2_bias = MakeWeight("norm2_bias", {params.emb_dim}, rng);
      normed_residual = StandardLayerNorm(residual_1, norm2_scale, norm2_bias);
    }

    // 5. FFN
    TensorTf ffn_out;
    if constexpr (kFfn == FfnType::kSwiGLU) {
      TensorTf ffn_gate =
          MakeWeight("ffn_gate", {params.hidden_dim, params.emb_dim}, rng);
      TensorTf ffn_up =
          MakeWeight("ffn_up", {params.hidden_dim, params.emb_dim}, rng);
      TensorTf ffn_down =
          MakeWeight("ffn_down", {params.emb_dim, params.hidden_dim}, rng);

      TensorTf fc_gate = FullyConnected(normed_residual, ffn_gate);
      TensorTf gate = Mul(fc_gate, Logistic(fc_gate));
      TensorTf up = FullyConnected(normed_residual, ffn_up);
      ffn_out = FullyConnected(Mul(gate, up), ffn_down);
    } else {
      TensorTf ffn_proj1 =
          MakeWeight("ffn_proj1", {params.hidden_dim, params.emb_dim}, rng);
      TensorTf ffn_proj2 =
          MakeWeight("ffn_proj2", {params.emb_dim, params.hidden_dim}, rng);
      TensorTf hidden = Relu(FullyConnected(normed_residual, ffn_proj1));
      ffn_out = FullyConnected(hidden, ffn_proj2);
    }

    // 6. Final Residual
    TensorTf output = Add(residual_1, ffn_out);
    output.SetName("output");

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSFORMER_LAYER_H_
