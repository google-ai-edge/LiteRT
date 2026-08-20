# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Reference Python implementation for Gemma 4 Transformer Layer.

Computes the expected reference values for Transformer Layer unit tests.
"""

import numpy as np

from tensor.examples.gemma4.helpers.reference import attention
from tensor.examples.gemma4.helpers.reference import feed_forward_network
from tensor.examples.gemma4.helpers.reference import rmsnorm
from tensor.examples.gemma4.helpers.reference import utils


def transformer_layer(
    x: np.ndarray,
    attention_mask: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    weights: dict[str, np.ndarray],
    key_cache: np.ndarray | None = None,
    value_cache: np.ndarray | None = None,
    per_layer_input: np.ndarray | None = None,
    shared_key: np.ndarray | None = None,
    shared_value: np.ndarray | None = None,
    layer_idx: int = 0,
    num_heads: int = 2,
    num_kv_heads: int = 1,
    head_dim: int = 4,
    per_layer_input_dim: int = 0,
    use_post_attn_norm: bool = True,
    use_post_ffw_norm: bool = True,
    is_global: bool = False,
    global_key_size: int = 4,
    rms_norm_eps: float = 1e-6,
    attn_logits_soft_cap: float | None = None,
    return_all: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
  """Computes Transformer layer output and auxiliary KV outputs.

  Args:
    x: Input tensor of shape [batch_size, seq_len, embed_dim].
    attention_mask: Attention mask tensor.
    cos: Cosine positional embedding tensor.
    sin: Sine positional embedding tensor.
    weights: Dictionary mapping weight names to numpy arrays.
    key_cache: Optional key cache tensor.
    value_cache: Optional value cache tensor.
    per_layer_input: Optional per-layer input tensor of shape [batch_size,
      seq_len, per_layer_input_dim].
    shared_key: Optional shared key tensor.
    shared_value: Optional shared value tensor.
    layer_idx: Index of the transformer layer.
    num_heads: Number of attention query heads.
    num_kv_heads: Number of key/value heads.
    head_dim: Dimension of each attention head.
    per_layer_input_dim: Dimension of per-layer input if enabled.
    use_post_attn_norm: Whether to apply post-attention RMSNorm.
    use_post_ffw_norm: Whether to apply post-FFN RMSNorm.
    is_global: Whether layer is global layer.
    global_key_size: Dimension of key for global layer.
    rms_norm_eps: Small constant for RMSNorm stability.
    attn_logits_soft_cap: Soft capping threshold for attention logits.
    return_all: Whether to return all intermediate KV outputs.

  Returns:
    Output tensor of shape [batch_size, seq_len, embed_dim], or tuple with KV.
  """
  layer_prefix = f"model.layers.{layer_idx}"

  # 1. Pre-attention RMSNorm
  pre_attn_norm_scale = weights[f"{layer_prefix}.input_layernorm.weight"]
  _, normed_input = rmsnorm.rms_norm(
      x, scale=pre_attn_norm_scale, eps=rms_norm_eps
  )

  # 2. Attention
  q_proj = weights[f"{layer_prefix}.self_attn.q_proj.weight"]
  k_proj = weights[f"{layer_prefix}.self_attn.k_proj.weight"]
  v_proj = weights[f"{layer_prefix}.self_attn.v_proj.weight"]
  o_proj = weights[f"{layer_prefix}.self_attn.o_proj.weight"]
  q_norm = weights[f"{layer_prefix}.self_attn.q_norm.weight"]
  k_norm = weights[f"{layer_prefix}.self_attn.k_norm.weight"]

  effective_head_dim = global_key_size if is_global else head_dim

  attn_out, updated_k, updated_v, k_untiled, v_untiled = attention.attention(
      x=normed_input,
      q_proj=q_proj,
      k_proj=k_proj,
      v_proj=v_proj,
      o_proj=o_proj,
      q_norm=q_norm,
      k_norm=k_norm,
      cos=cos,
      sin=sin,
      attention_mask=attention_mask,
      key_cache=key_cache,
      value_cache=value_cache,
      shared_key=shared_key,
      shared_value=shared_value,
      num_heads=num_heads,
      num_kv_heads=num_kv_heads,
      head_dim=effective_head_dim,
      rms_norm_eps=rms_norm_eps,
      attn_logits_soft_cap=attn_logits_soft_cap,
      return_all=True,
  )

  # 3. Post-attention RMSNorm (optional)
  if use_post_attn_norm:
    post_attn_norm_scale = weights[
        f"{layer_prefix}.post_attention_layernorm.weight"
    ]
    _, attn_out = rmsnorm.rms_norm(
        attn_out, scale=post_attn_norm_scale, eps=rms_norm_eps
    )

  # 4. Residual connection 1
  attn_residual = attn_out + x

  # 5. Pre-FFN RMSNorm
  pre_ffn_norm_scale = weights[
      f"{layer_prefix}.pre_feedforward_layernorm.weight"
  ]
  _, normed_attn_output = rmsnorm.rms_norm(
      attn_residual, scale=pre_ffn_norm_scale, eps=rms_norm_eps
  )

  # 6. FFN (SwiGLU)
  gate_proj = weights[f"{layer_prefix}.mlp.gate_proj.weight"]
  up_proj = weights[f"{layer_prefix}.mlp.up_proj.weight"]
  down_proj = weights[f"{layer_prefix}.mlp.down_proj.weight"]

  raw_ffn_output = feed_forward_network.feed_forward_network(
      normed_attn_output, gate_proj, up_proj, down_proj
  )
  ffn_output = raw_ffn_output

  # 7. Post-FFN RMSNorm (optional)
  if use_post_ffw_norm:
    post_ffn_norm_scale = weights[
        f"{layer_prefix}.post_feedforward_layernorm.weight"
    ]
    _, ffn_output = rmsnorm.rms_norm(
        raw_ffn_output, scale=post_ffn_norm_scale, eps=rms_norm_eps
    )

  # 8. Residual connection 2
  ffn_residual = ffn_output + attn_residual

  # 9. Per-layer input integration (optional)
  if (
      per_layer_input_dim > 0
      and per_layer_input is not None
      and per_layer_input.size > 0
  ):
    per_layer_input_gate = weights[
        f"{layer_prefix}.per_layer_input_gate.weight"
    ]
    per_layer_projection = weights[
        f"{layer_prefix}.per_layer_projection.weight"
    ]
    post_per_layer_input_norm = weights[
        f"{layer_prefix}.post_per_layer_input_norm.weight"
    ]

    gate_val = np.matmul(ffn_residual, per_layer_input_gate.T)
    gated = feed_forward_network.gelu_approx(gate_val) * per_layer_input
    projected = np.matmul(gated, per_layer_projection.T)
    _, normed_projected = rmsnorm.rms_norm(
        projected, scale=post_per_layer_input_norm, eps=rms_norm_eps
    )
    ffn_residual = ffn_residual + normed_projected

  # 10. Layer Scalar
  layer_scalar = weights[f"{layer_prefix}.layer_scalar"]
  output = ffn_residual * layer_scalar

  if return_all:
    return output, updated_k, updated_v, k_untiled, v_untiled
  return output


def get_default_test_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
  """Constructs standard inputs and weights for transformer tests."""
  x = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32)
  attention_mask = np.array([[[[0.0, -1e9], [0.0, 0.0]]]], dtype=np.float32)

  base_angles = utils.generate_angles(head_dim=4)
  angles = np.reshape(
      np.stack([base_angles, base_angles * 1.5], axis=0).astype(np.float32),
      (1, 1, 2, 4),
  )
  cos = np.cos(angles)
  sin = np.sin(angles)

  weights = {
      "model.layers.0.self_attn.q_proj.weight": np.array(
          [
              [1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
          ],
          dtype=np.float32,
      ),
      "model.layers.0.self_attn.k_proj.weight": np.array(
          [
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
              [1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
          ],
          dtype=np.float32,
      ),
      "model.layers.0.self_attn.v_proj.weight": np.array(
          [
              [0.5, 0.0, 0.0, 0.0],
              [0.0, 0.5, 0.0, 0.0],
              [0.0, 0.0, 0.5, 0.0],
              [0.0, 0.0, 0.0, 0.5],
          ],
          dtype=np.float32,
      ),
      "model.layers.0.self_attn.o_proj.weight": np.array(
          [
              [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
          ],
          dtype=np.float32,
      ),
      "model.layers.0.self_attn.q_norm.weight": np.ones((4,), dtype=np.float32),
      "model.layers.0.self_attn.k_norm.weight": np.ones((4,), dtype=np.float32),
      "model.layers.0.input_layernorm.weight": np.ones((4,), dtype=np.float32),
      "model.layers.0.post_attention_layernorm.weight": np.ones(
          (4,), dtype=np.float32
      ),
      "model.layers.0.pre_feedforward_layernorm.weight": np.ones(
          (4,), dtype=np.float32
      ),
      "model.layers.0.post_feedforward_layernorm.weight": np.ones(
          (4,), dtype=np.float32
      ),
      "model.layers.0.mlp.gate_proj.weight": np.array(
          [
              [1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
              [1.0, -1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, -1.0],
          ],
          dtype=np.float32,
      ),
      "model.layers.0.mlp.up_proj.weight": np.array(
          [
              [0.5, 0.5, 0.0, 0.0],
              [0.0, 0.5, 0.5, 0.0],
              [0.0, 0.0, 0.5, 0.5],
              [0.5, 0.0, 0.0, 0.5],
              [1.0, 0.0, -1.0, 0.0],
              [0.0, 1.0, 0.0, -1.0],
          ],
          dtype=np.float32,
      ),
      "model.layers.0.mlp.down_proj.weight": np.array(
          [
              [1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
              [0.0, 0.0, 1.0, 0.0, -0.5, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, -0.5],
          ],
          dtype=np.float32,
      ),
      "model.layers.0.layer_scalar": np.array([0.5], dtype=np.float32),
  }
  return x, attention_mask, cos, sin, weights


def test_default_transformer() -> None:
  """Tests standard transformer layer computation."""
  print("=== Default Transformer Layer ===")
  x, mask, cos, sin, weights = get_default_test_inputs()
  output, updated_k, updated_v, _, _ = transformer_layer(
      x,
      mask,
      cos,
      sin,
      weights,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      return_all=True,
  )
  utils.cpp_print("Output:", output)
  utils.cpp_print("Key Cache:", updated_k)
  utils.cpp_print("Value Cache:", updated_v)


def test_disabled_post_norms() -> None:
  """Tests transformer layer with post-attention and post-FFN norm disabled."""
  print("\n=== Disabled Post-Norms ===")
  x, mask, cos, sin, weights = get_default_test_inputs()
  output, _, _, _, _ = transformer_layer(
      x,
      mask,
      cos,
      sin,
      weights,
      use_post_attn_norm=False,
      use_post_ffw_norm=False,
      return_all=True,
  )
  utils.cpp_print("Output (no post-norms):", output)


def test_per_layer_input() -> None:
  """Tests transformer layer with per-layer input integration."""
  print("\n=== Per-Layer Input Integration ===")
  x, mask, cos, sin, weights = get_default_test_inputs()
  per_layer_input = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)

  weights["model.layers.0.per_layer_input_gate.weight"] = np.array(
      [
          [0.5, 0.0, 0.0, 0.0],
          [0.0, 0.5, 0.0, 0.0],
      ],
      dtype=np.float32,
  )
  weights["model.layers.0.per_layer_projection.weight"] = np.array(
      [
          [1.0, 0.0],
          [0.0, 1.0],
          [0.5, 0.0],
          [0.0, 0.5],
      ],
      dtype=np.float32,
  )
  weights["model.layers.0.post_per_layer_input_norm.weight"] = np.ones(
      (4,), dtype=np.float32
  )

  output, _, _, _, _ = transformer_layer(
      x,
      mask,
      cos,
      sin,
      weights,
      per_layer_input=per_layer_input,
      per_layer_input_dim=2,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      return_all=True,
  )
  utils.cpp_print("Output (per-layer input):", output)


def test_kv_cache_provided() -> None:
  """Tests transformer layer with pre-populated KV cache."""
  print("\n=== Pre-populated KV Cache ===")
  x, _, cos, sin, weights = get_default_test_inputs()
  key_cache_in = np.array(
      [[[[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]]]], dtype=np.float32
  )
  val_cache_in = np.array(
      [[[[0.2, 0.2, 0.2, 0.2], [0.4, 0.4, 0.4, 0.4]]]], dtype=np.float32
  )

  kv_mask = np.array(
      [[[[0.0, 0.0, 0.0, -1e9], [0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32
  )

  output, updated_k, updated_v, _, _ = transformer_layer(
      x,
      kv_mask,
      cos,
      sin,
      weights,
      key_cache=key_cache_in,
      value_cache=val_cache_in,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      return_all=True,
  )
  utils.cpp_print("Output (with KV cache):", output)
  utils.cpp_print("Updated Key Cache:", updated_k)
  utils.cpp_print("Updated Value Cache:", updated_v)


def test_shared_kv_provided() -> None:
  """Tests transformer layer with shared KV tensors."""
  print("\n=== Shared KV Tensors ===")
  x, mask, cos, sin, weights = get_default_test_inputs()
  shared_k_in = np.array(
      [[[[0.3, 0.3, 0.3, 0.3], [0.6, 0.6, 0.6, 0.6]]]], dtype=np.float32
  )
  shared_v_in = np.array(
      [[[[0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8]]]], dtype=np.float32
  )

  output, updated_k, updated_v, _, _ = transformer_layer(
      x,
      mask,
      cos,
      sin,
      weights,
      shared_key=shared_k_in,
      shared_value=shared_v_in,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      return_all=True,
  )
  utils.cpp_print("Output (with shared KV):", output)
  utils.cpp_print("Updated Key Cache:", updated_k)
  utils.cpp_print("Updated Value Cache:", updated_v)


def test_soft_capping() -> None:
  """Tests transformer layer with attention logits soft capping."""
  print("\n=== Soft Capping ===")
  x, mask, cos, sin, weights = get_default_test_inputs()
  output, updated_k, updated_v, _, _ = transformer_layer(
      x,
      mask,
      cos,
      sin,
      weights,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      attn_logits_soft_cap=1.0,
      return_all=True,
  )
  utils.cpp_print("Output (soft capping):", output)
  utils.cpp_print("Key Cache:", updated_k)
  utils.cpp_print("Value Cache:", updated_v)


def test_global_layer() -> None:
  """Tests transformer layer configured as a global layer."""
  print("\n=== Global Layer ===")
  x, mask, cos, sin, weights = get_default_test_inputs()
  # Layer idx 0 with is_global=True
  output, updated_k, updated_v, _, _ = transformer_layer(
      x,
      mask,
      cos,
      sin,
      weights,
      layer_idx=0,
      is_global=True,
      global_key_size=4,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      return_all=True,
  )
  utils.cpp_print("Output (global layer):", output)
  utils.cpp_print("Key Cache:", updated_k)
  utils.cpp_print("Value Cache:", updated_v)


def test_multi_kv_heads_gqa() -> None:
  """Tests transformer layer with multi-KV-head GQA (num_heads=4, num_kv_heads=2)."""
  print("\n=== Multi-KV Heads GQA ===")
  x = np.array(
      [[
          [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0],
      ]],
      dtype=np.float32,
  )
  mask = np.array([[[[0.0, -1e9], [0.0, 0.0]]]], dtype=np.float32)
  base_angles = utils.generate_angles(head_dim=4)
  angles = np.reshape(
      np.stack([base_angles, base_angles * 1.5], axis=0).astype(np.float32),
      (1, 1, 2, 4),
  )
  cos = np.cos(angles)
  sin = np.sin(angles)

  weights = {
      "model.layers.0.self_attn.q_proj.weight": np.tile(
          np.eye(4, dtype=np.float32), (4, 2)
      ),
      "model.layers.0.self_attn.k_proj.weight": np.tile(
          np.eye(4, dtype=np.float32), (2, 2)
      ),
      "model.layers.0.self_attn.v_proj.weight": np.tile(
          np.eye(4, dtype=np.float32) * 0.5, (2, 2)
      ),
      "model.layers.0.self_attn.o_proj.weight": np.tile(
          np.eye(4, dtype=np.float32) * 0.5, (2, 4)
      ),
      "model.layers.0.self_attn.q_norm.weight": np.ones((4,), dtype=np.float32),
      "model.layers.0.self_attn.k_norm.weight": np.ones((4,), dtype=np.float32),
      "model.layers.0.input_layernorm.weight": np.ones((8,), dtype=np.float32),
      "model.layers.0.post_attention_layernorm.weight": np.ones(
          (8,), dtype=np.float32
      ),
      "model.layers.0.pre_feedforward_layernorm.weight": np.ones(
          (8,), dtype=np.float32
      ),
      "model.layers.0.post_feedforward_layernorm.weight": np.ones(
          (8,), dtype=np.float32
      ),
      "model.layers.0.mlp.gate_proj.weight": np.tile(
          np.eye(4, dtype=np.float32), (3, 2)
      ),
      "model.layers.0.mlp.up_proj.weight": np.tile(
          np.eye(4, dtype=np.float32) * 0.5, (3, 2)
      ),
      "model.layers.0.mlp.down_proj.weight": np.tile(
          np.eye(4, dtype=np.float32) * 0.5, (2, 3)
      ),
      "model.layers.0.layer_scalar": np.array([0.5], dtype=np.float32),
  }

  output, updated_k, updated_v, _, _ = transformer_layer(
      x,
      mask,
      cos,
      sin,
      weights,
      num_heads=4,
      num_kv_heads=2,
      head_dim=4,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      return_all=True,
  )
  utils.cpp_print("Output (multi-kv heads GQA):", output)
  utils.cpp_print("Key Cache:", updated_k)
  utils.cpp_print("Value Cache:", updated_v)


def main() -> None:
  test_default_transformer()
  test_disabled_post_norms()
  test_per_layer_input()
  test_kv_cache_provided()
  test_shared_kv_provided()
  test_soft_capping()
  test_global_layer()
  test_multi_kv_heads_gqa()


if __name__ == "__main__":
  main()
