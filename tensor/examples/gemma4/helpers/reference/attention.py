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

"""Reference Python implementation for Gemma 4 Attention.

Computes the expected reference values for Attention unit tests.
"""

import numpy as np

from tensor.examples.gemma4.helpers.reference import rmsnorm
from tensor.examples.gemma4.helpers.reference import rope
from tensor.examples.gemma4.helpers.reference import utils


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  """Computes numerically stable softmax along specified axis.

  Args:
    x: Input tensor.
    axis: Axis along which softmax is computed.

  Returns:
    Output tensor after softmax.
  """
  e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
  return e_x / np.sum(e_x, axis=axis, keepdims=True)


def attention(
    x: np.ndarray,
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    o_proj: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    attention_mask: np.ndarray,
    key_cache: np.ndarray | None = None,
    value_cache: np.ndarray | None = None,
    shared_key: np.ndarray | None = None,
    shared_value: np.ndarray | None = None,
    num_heads: int = 2,
    num_kv_heads: int = 1,
    head_dim: int = 4,
    rms_norm_eps: float = 1e-6,
    attn_logits_soft_cap: float | None = None,
    return_all: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
  """Computes Attention layer output and auxiliary KV outputs.

  Args:
    x: Input tensor of shape [batch_size, seq_len, embed_dim].
    q_proj: Query projection matrix of shape [q_out_dim, embed_dim].
    k_proj: Key projection matrix of shape [kv_out_dim, embed_dim].
    v_proj: Value projection matrix of shape [kv_out_dim, embed_dim].
    o_proj: Output projection matrix of shape [embed_dim, q_out_dim].
    q_norm: Query normalization scale vector of shape [head_dim].
    k_norm: Key normalization scale vector of shape [head_dim].
    cos: Cosine positional embedding tensor.
    sin: Sine positional embedding tensor.
    attention_mask: Attention mask tensor of shape [batch_size, 1, seq_len,
      seq_len].
    key_cache: Optional cached key tensor of shape [batch_size, num_kv_heads,
      cache_len, head_dim].
    value_cache: Optional cached value tensor of shape [batch_size,
      num_kv_heads, cache_len, head_dim].
    shared_key: Optional shared key tensor of shape [batch_size, num_kv_heads,
      seq_len, head_dim].
    shared_value: Optional shared value tensor of shape [batch_size,
      num_kv_heads, seq_len, head_dim].
    num_heads: Number of attention query heads.
    num_kv_heads: Number of key/value heads.
    head_dim: Dimension of each head.
    rms_norm_eps: Small constant for RMSNorm numerical stability.
    attn_logits_soft_cap: Optional float threshold for logits soft capping.
    return_all: Whether to return all intermediate KV tensors matching
      AttentionOutput.

  Returns:
    Output tensor of shape [batch_size, seq_len, embed_dim], or tuple of all
    outputs if return_all=True.
  """
  batch_size, seq_len, _ = x.shape

  # Project Q
  q = np.matmul(x, q_proj.T)
  q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
  _, q = rmsnorm.rms_norm(q, scale=q_norm, eps=rms_norm_eps)
  q = rope.rope(q, cos, sin)

  has_valid_shared_kv = (
      shared_key is not None
      and shared_value is not None
      and shared_key.ndim == 4
      and shared_value.ndim == 4
      and shared_key.shape[0] == batch_size
      and shared_key.shape[1] == num_kv_heads
      and shared_key.shape[3] == head_dim
      and shared_value.shape[0] == batch_size
      and shared_value.shape[1] == num_kv_heads
      and shared_value.shape[3] == head_dim
      and shared_key.shape[2] == shared_value.shape[2]
  )

  if has_valid_shared_kv:
    k_for_attn = shared_key
    v_for_attn = shared_value
    updated_key_cache = shared_key
    updated_value_cache = shared_value
  else:
    k = np.matmul(x, k_proj.T)
    v = np.matmul(x, v_proj.T)

    k = k.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(
        0, 2, 1, 3
    )
    v = v.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(
        0, 2, 1, 3
    )

    _, k = rmsnorm.rms_norm(k, scale=k_norm, eps=rms_norm_eps)
    _, v = rmsnorm.rms_norm(v, scale=None, eps=rms_norm_eps)

    k = rope.rope(k, cos, sin)

    k_for_attn = k
    v_for_attn = v

    if (
        key_cache is not None
        and value_cache is not None
        and key_cache.ndim == 4
        and key_cache.shape[2] > 0
    ):
      k_for_attn = np.concatenate([key_cache, k], axis=2)
      v_for_attn = np.concatenate([value_cache, v], axis=2)

    updated_key_cache = k
    updated_value_cache = v

  # Untiled KV references
  k_for_attn_untiled = k_for_attn
  v_for_attn_untiled = v_for_attn

  # GQA Tiling
  num_groups = num_heads // num_kv_heads
  if num_groups > 1:
    k_for_attn = np.repeat(k_for_attn, num_groups, axis=1)
    v_for_attn = np.repeat(v_for_attn, num_groups, axis=1)

  # Attention Scores
  scores = np.matmul(q, k_for_attn.transpose(0, 1, 3, 2))

  if attn_logits_soft_cap is not None:
    scores = np.tanh(scores / attn_logits_soft_cap) * attn_logits_soft_cap

  # Masking & Softmax
  scores = scores + attention_mask
  probs = softmax(scores, axis=-1)

  # Context Vector
  context = np.matmul(probs, v_for_attn)
  context = context.transpose(0, 2, 1, 3).reshape(
      batch_size, seq_len, num_heads * head_dim
  )

  # Output Projection
  output = np.matmul(context, o_proj.T)
  if return_all:
    return (
        output,
        updated_key_cache,
        updated_value_cache,
        k_for_attn_untiled,
        v_for_attn_untiled,
    )
  return output


def generate_weights(
    m: int, n: int, scale: float = 0.01, start: float = 0.1
) -> np.ndarray:
  """Generates a matrix for testing.

  Args:
    m: Number of rows.
    n: Number of columns.
    scale: Step increment per element.
    start: Initial element value.

  Returns:
    Numpy array of shape [m, n].
  """
  return (start + np.arange(m * n, dtype=np.float32) * scale).reshape(m, n)


def test_default_attention(
    x: np.ndarray,
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    o_proj: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    attention_mask: np.ndarray,
) -> None:
  """Tests standard single-KV head attention computation."""
  print("=== Default Attention ===")
  output, updated_k, updated_v, k_untiled, v_untiled = attention(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
      num_heads=2,
      num_kv_heads=1,
      head_dim=4,
      return_all=True,
  )
  utils.cpp_print("Output:", output)
  utils.cpp_print("Updated Key Cache:", updated_k)
  utils.cpp_print("Updated Value Cache:", updated_v)
  utils.cpp_print("Key Untiled:", k_untiled)
  utils.cpp_print("Value Untiled:", v_untiled)


def test_multi_kv_heads_gqa(
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    attention_mask: np.ndarray,
) -> None:
  """Tests Grouped-Query Attention (GQA) with num_kv_heads > 1."""
  print("\n=== Multi-KV Head GQA (num_heads=4, num_kv_heads=2) ===")
  x8 = np.array(
      [[
          [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0],
      ]],
      dtype=np.float32,
  )
  q_proj16 = generate_weights(16, 8, scale=0.005, start=0.01)
  k_proj8 = generate_weights(8, 8, scale=0.01, start=0.02)
  v_proj8 = generate_weights(8, 8, scale=0.008, start=0.01)
  o_proj8_16 = generate_weights(8, 16, scale=0.005, start=0.01)

  out_gqa, gqa_k, gqa_v, gqa_k_untiled, gqa_v_untiled = attention(
      x8,
      q_proj16,
      k_proj8,
      v_proj8,
      o_proj8_16,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
      num_heads=4,
      num_kv_heads=2,
      head_dim=4,
      return_all=True,
  )
  utils.cpp_print("Output Multi-KV GQA:", out_gqa)
  utils.cpp_print("Key Cache Multi-KV GQA:", gqa_k)
  utils.cpp_print("Value Cache Multi-KV GQA:", gqa_v)
  utils.cpp_print("Key Untiled Multi-KV GQA:", gqa_k_untiled)
  utils.cpp_print("Value Untiled Multi-KV GQA:", gqa_v_untiled)


def test_multi_head_attention(
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    attention_mask: np.ndarray,
) -> None:
  """Tests Multi-Head Attention (MHA) with num_heads == num_kv_heads == 2."""
  print("\n=== Multi-Head Attention (num_heads=2, num_kv_heads=2) ===")
  x = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32)
  q_proj = generate_weights(8, 4, scale=0.01, start=0.1)
  k_proj = generate_weights(8, 4, scale=0.02, start=0.05)
  v_proj = generate_weights(8, 4, scale=0.015, start=0.02)
  o_proj = generate_weights(4, 8, scale=0.01, start=0.05)

  out_mha, mha_k, mha_v, mha_k_untiled, mha_v_untiled = attention(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
      num_heads=2,
      num_kv_heads=2,
      head_dim=4,
      return_all=True,
  )
  utils.cpp_print("Output MHA:", out_mha)
  utils.cpp_print("Key Cache MHA:", mha_k)
  utils.cpp_print("Value Cache MHA:", mha_v)
  utils.cpp_print("Key Untiled MHA:", mha_k_untiled)
  utils.cpp_print("Value Untiled MHA:", mha_v_untiled)


def test_soft_capping(
    x: np.ndarray,
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    o_proj: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    attention_mask: np.ndarray,
) -> None:
  """Tests attention logits soft-capping behavior."""
  print("\n=== Soft Capping (cap=1.0) ===")
  out_cap, cap_k, cap_v, cap_k_untiled, cap_v_untiled = attention(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
      num_heads=2,
      num_kv_heads=1,
      head_dim=4,
      attn_logits_soft_cap=1.0,
      return_all=True,
  )
  utils.cpp_print("Output Soft Cap:", out_cap)
  utils.cpp_print("Key Cache Soft Cap:", cap_k)
  utils.cpp_print("Value Cache Soft Cap:", cap_v)
  utils.cpp_print("Key Untiled Soft Cap:", cap_k_untiled)
  utils.cpp_print("Value Untiled Soft Cap:", cap_v_untiled)


def test_kv_cache_provided(
    x: np.ndarray,
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    o_proj: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> None:
  """Tests attention evaluation when pre-populated KV cache is provided."""
  print("\n=== Key/Value Cache Provided ===")
  key_cache_in = np.array(
      [[[[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]]]], dtype=np.float32
  )
  val_cache_in = np.array(
      [[[[0.2, 0.2, 0.2, 0.2], [0.4, 0.4, 0.4, 0.4]]]], dtype=np.float32
  )
  kv_attention_mask = np.array(
      [[[[0.0, 0.0, 0.0, -1e9], [0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32
  )
  out_cache, cache_k, cache_v, cache_k_untiled, cache_v_untiled = attention(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      kv_attention_mask,
      key_cache=key_cache_in,
      value_cache=val_cache_in,
      num_heads=2,
      num_kv_heads=1,
      head_dim=4,
      return_all=True,
  )
  utils.cpp_print("Output with KV Cache:", out_cache)
  utils.cpp_print("Key Cache Output:", cache_k)
  utils.cpp_print("Value Cache Output:", cache_v)
  utils.cpp_print("Key Untiled with KV Cache:", cache_k_untiled)
  utils.cpp_print("Value Untiled with KV Cache:", cache_v_untiled)


def test_empty_kv_cache(
    x: np.ndarray,
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    o_proj: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    attention_mask: np.ndarray,
) -> None:
  """Tests attention when provided KV cache has sequence length 0."""
  print("\n=== Empty Key/Value Cache Provided ===")
  key_cache_empty = np.zeros((1, 1, 0, 4), dtype=np.float32)
  val_cache_empty = np.zeros((1, 1, 0, 4), dtype=np.float32)
  out_empty, empty_k, empty_v, empty_k_untiled, empty_v_untiled = attention(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
      key_cache=key_cache_empty,
      value_cache=val_cache_empty,
      num_heads=2,
      num_kv_heads=1,
      head_dim=4,
      return_all=True,
  )
  utils.cpp_print("Output Empty KV Cache:", out_empty)
  utils.cpp_print("Key Cache Empty:", empty_k)
  utils.cpp_print("Value Cache Empty:", empty_v)
  utils.cpp_print("Key Untiled Empty KV Cache:", empty_k_untiled)
  utils.cpp_print("Value Untiled Empty KV Cache:", empty_v_untiled)


def test_shared_kv_provided(
    x: np.ndarray,
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    o_proj: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    attention_mask: np.ndarray,
) -> None:
  """Tests attention evaluation when shared KV tensors are provided."""
  print("\n=== Shared Key/Value Provided ===")
  shared_k_in = np.array(
      [[[[0.3, 0.3, 0.3, 0.3], [0.6, 0.6, 0.6, 0.6]]]], dtype=np.float32
  )
  shared_v_in = np.array(
      [[[[0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8]]]], dtype=np.float32
  )
  out_shared, shared_k_ret, shared_v_ret, shared_k_untiled, shared_v_untiled = (
      attention(
          x,
          q_proj,
          k_proj,
          v_proj,
          o_proj,
          q_norm,
          k_norm,
          cos,
          sin,
          attention_mask,
          shared_key=shared_k_in,
          shared_value=shared_v_in,
          num_heads=2,
          num_kv_heads=1,
          head_dim=4,
          return_all=True,
      )
  )
  utils.cpp_print("Output with Shared KV:", out_shared)
  utils.cpp_print("Shared Key Cache Ret:", shared_k_ret)
  utils.cpp_print("Shared Value Cache Ret:", shared_v_ret)
  utils.cpp_print("Key Untiled Shared KV:", shared_k_untiled)
  utils.cpp_print("Value Untiled Shared KV:", shared_v_untiled)


def test_mismatched_shared_kv(
    x: np.ndarray,
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    o_proj: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    attention_mask: np.ndarray,
) -> None:
  """Tests attention fallback when shared KV shapes mismatch."""
  print("\n=== Mismatched Shared Key/Value Fallback ===")
  shared_k_mismatched = np.array(
      [[[[0.3, 0.3, 0.3, 0.3], [0.6, 0.6, 0.6, 0.6]]]],
      dtype=np.float32,
  )
  shared_v_mismatched = np.array(
      [[[[0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8], [0.5, 0.5, 0.5, 0.5]]]],
      dtype=np.float32,
  )
  out_mis, mis_k, mis_v, mis_k_untiled, mis_v_untiled = attention(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
      shared_key=shared_k_mismatched,
      shared_value=shared_v_mismatched,
      num_heads=2,
      num_kv_heads=1,
      head_dim=4,
      return_all=True,
  )
  utils.cpp_print("Output Mismatched Shared KV:", out_mis)
  utils.cpp_print("Key Cache Mismatched Shared KV:", mis_k)
  utils.cpp_print("Value Cache Mismatched Shared KV:", mis_v)
  utils.cpp_print("Key Untiled Mismatched Shared KV:", mis_k_untiled)
  utils.cpp_print("Value Untiled Mismatched Shared KV:", mis_v_untiled)


def test_multi_kv_heads_gqa_kv_cache() -> None:
  """Tests multi-KV heads GQA with pre-populated KV cache."""
  print("\n=== Multi-KV Head GQA with KV Cache ===")
  x = np.array([[[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
  attention_mask = np.zeros((1, 1, 1, 4), dtype=np.float32)
  cos = np.ones((1, 1, 1, 4), dtype=np.float32)
  sin = np.zeros((1, 1, 1, 4), dtype=np.float32)

  q_proj = generate_weights(16, 8, scale=0.005, start=0.01)
  k_proj = generate_weights(8, 8, scale=0.01, start=0.02)
  v_proj = generate_weights(8, 8, scale=0.008, start=0.01)
  o_proj = generate_weights(8, 16, scale=0.005, start=0.01)

  q_norm = np.ones(4, dtype=np.float32)
  k_norm = np.ones(4, dtype=np.float32)

  key_cache = np.full((1, 2, 3, 4), 0.5, dtype=np.float32)
  value_cache = np.full((1, 2, 3, 4), 0.2, dtype=np.float32)

  out_cache, cache_k, cache_v, _, _ = attention(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
      key_cache=key_cache,
      value_cache=value_cache,
      num_heads=4,
      num_kv_heads=2,
      head_dim=4,
      return_all=True,
  )
  utils.cpp_print("Output Multi-KV GQA KV Cache:", out_cache)
  utils.cpp_print("Key Cache Multi-KV GQA KV Cache:", cache_k)
  utils.cpp_print("Value Cache Multi-KV GQA KV Cache:", cache_v)


def main() -> None:
  x = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32)
  attention_mask = np.array([[[[0.0, -1e9], [0.0, 0.0]]]], dtype=np.float32)
  base_angles = utils.generate_angles(head_dim=4)
  angles = np.reshape(
      np.stack([base_angles, base_angles * 1.5], axis=0).astype(np.float32),
      (1, 1, 2, 4),
  )
  cos = np.cos(angles)
  sin = np.sin(angles)

  q_proj = generate_weights(8, 4, scale=0.01, start=0.1)
  k_proj = generate_weights(4, 4, scale=0.02, start=0.05)
  v_proj = generate_weights(4, 4, scale=0.015, start=0.02)
  o_proj = generate_weights(4, 8, scale=0.01, start=0.05)

  q_norm = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
  k_norm = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

  test_default_attention(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
  )
  test_multi_kv_heads_gqa(q_norm, k_norm, cos, sin, attention_mask)
  test_multi_head_attention(q_norm, k_norm, cos, sin, attention_mask)
  test_soft_capping(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
  )
  test_kv_cache_provided(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
  )
  test_empty_kv_cache(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
  )
  test_shared_kv_provided(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
  )
  test_mismatched_shared_kv(
      x,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      cos,
      sin,
      attention_mask,
  )
  test_multi_kv_heads_gqa_kv_cache()


if __name__ == "__main__":
  main()
