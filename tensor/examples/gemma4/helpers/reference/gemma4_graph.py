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

"""Reference Python implementation for Gemma 4 Graph.

Computes expected reference values for the C++ Gemma 4 Graph unit tests.

This module is expected to be run manually and updated to reflect the test cases
that are written in the corresponding C++ implementation of the Gemma 4 graph.
It is not intended to be used in any other way and should not be relied upon for
anything other than generating specific test expectations.
"""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np

from tensor.examples.gemma4.helpers.reference import rmsnorm
from tensor.examples.gemma4.helpers.reference import transformer
from tensor.examples.gemma4.helpers.reference import utils


def _get_kv_cache_sharing_patterns(
    num_layers: int,
    frac_shared_layers: float,
    share_local: bool,
    share_global: bool,
    attention_pattern_size: int,
) -> Sequence[int]:
  """Returns KV cache sharing layer indices for each layer.

  Args:
    num_layers: The number of transformer layers in the model.
    frac_shared_layers: The fraction of layers that share KV projections with
      earlier layers. These layers are counted from the last layer up.
    share_local: Enables sharing for sliding window attention layers.
    share_global: Enables sharing for global attention layers.
    attention_pattern_size: The size of the attention pattern. Every
      `attention_pattern_size`_th layer is global. The others are local.

  Returns:
    A sequence holding the index of the transformer layer that the KV cache
    should be shared with.
      - `seq[i] == i` means that the KV cache belongs to the layer.
      - `seq[i] != i` means that the KV cache belongs to another layer.
  """
  patterns = list(range(num_layers))
  num_unshared_layers = int(num_layers - frac_shared_layers * num_layers)

  for i in range(num_unshared_layers, num_layers):
    is_global = ((i + 1) % attention_pattern_size) == 0
    if share_local and not is_global:
      patterns[i] = num_unshared_layers - 2 if num_unshared_layers > 2 else 0
    elif share_global and is_global:
      patterns[i] = num_unshared_layers - 1 if num_unshared_layers > 1 else 0
  return patterns


# The output tensor list for a Gemma 4 graph.
# - The logits computed by the graph.
# - The list of key caches updated during the graph computation.
# - The list of value caches updated during the graph computation.
_Gemma4GraphOutputs: TypeAlias = tuple[
    np.ndarray, list[np.ndarray], list[np.ndarray]
]


def _gemma4_graph(
    *,
    embedded_input: np.ndarray,
    weights: dict[str, np.ndarray],
    global_attention_mask: np.ndarray | None = None,
    sliding_attention_mask: np.ndarray | None = None,
    rope_global_cos: np.ndarray | None = None,
    rope_global_sin: np.ndarray | None = None,
    rope_local_cos: np.ndarray | None = None,
    rope_local_sin: np.ndarray | None = None,
    key_caches: list[np.ndarray] | None = None,
    value_caches: list[np.ndarray] | None = None,
    per_layer_token_embeddings: list[np.ndarray] | None = None,
    num_layers: int = 2,
    embed_dim: int = 4,
    head_dim: int = 4,
    num_heads: int = 2,
    num_kv_heads: int = 1,
    per_layer_input_dim: int = 0,
    use_post_attn_norm: bool = True,
    use_post_ffw_norm: bool = True,
    final_logit_softcap: float = 10.0,
    rms_norm_eps: float = 1e-6,
    frac_shared_layers: float = 0.0,
    share_global: bool = False,
    share_local: bool = False,
    attention_pattern_size: int = 6,
    global_key_size: int = 4,
    attn_logits_soft_cap: float | None = None,
) -> _Gemma4GraphOutputs:
  """Computes full Gemma 4 model graph outputs.

  Args:
    embedded_input: Input token embeddings of shape `(batch_size, seq_len,
      embed_dim)`.
    weights: Dictionary mapping weight parameter names to NumPy arrays for all
      projections, layer norms, and embeddings.
    global_attention_mask: Attention mask array applied to global attention
      layers.
    sliding_attention_mask: Attention mask array applied to local sliding-window
      attention layers.
    rope_global_cos: Cosine positional embeddings for global attention layers.
    rope_global_sin: Sine positional embeddings for global attention layers.
    rope_local_cos: Cosine positional embeddings for local attention layers.
    rope_local_sin: Sine positional embeddings for local attention layers.
    key_caches: List of input key cache arrays for each layer.
    value_caches: List of input value cache arrays for each layer.
    per_layer_token_embeddings: List of per-layer token embedding arrays for
      each layer.
    num_layers: Total number of transformer layers in the model.
    embed_dim: Dimensionality of the model embedding and hidden states.
    head_dim: Dimensionality of each attention head.
    num_heads: Number of query attention heads.
    num_kv_heads: Number of key/value attention heads.
    per_layer_input_dim: Dimensionality of per-layer inputs, if used.
    use_post_attn_norm: Whether to apply RMS normalization after self-attention.
    use_post_ffw_norm: Whether to apply RMS normalization after feed-forward
      layers.
    final_logit_softcap: Threshold value for soft-capping output logits.
      Soft-capping is disabled if set to <= 0.0.
    rms_norm_eps: Epsilon value for numerical stability in RMS normalization.
    frac_shared_layers: Fraction of total layers at the end of the model that
      share KV projections with earlier unshared layers.
    share_global: Whether global attention layers in the shared block reuse KV
      projections from earlier unshared global layers.
    share_local: Whether local attention layers in the shared block reuse KV
      projections from earlier unshared local layers.
    attention_pattern_size: Period/frequency of global attention layers (every
      Nth layer is a global layer).
    global_key_size: Key dimension size for global attention layers.
    attn_logits_soft_cap: Threshold value for soft-capping attention logits.

  Returns:
    A tuple of `(logits, updated_key_caches, updated_value_caches)` containing:
      - logits: Output prediction logits of shape `(batch_size, seq_len,
        vocab_size)`.
      - updated_key_caches: List of updated key cache arrays per layer after
        execution.
      - updated_value_caches: List of updated value cache arrays per layer after
        execution.
  """
  emb_scale = np.sqrt(float(embed_dim))
  hidden_states = embedded_input * emb_scale

  sharing_patterns = _get_kv_cache_sharing_patterns(
      num_layers,
      frac_shared_layers,
      share_local,
      share_global,
      attention_pattern_size,
  )

  computed_keys = [None] * num_layers
  computed_values = [None] * num_layers

  updated_key_caches = []
  updated_value_caches = []

  for layer_idx in range(num_layers):
    shared_idx = sharing_patterns[layer_idx]
    is_shared = shared_idx != layer_idx

    shared_k = computed_keys[shared_idx] if is_shared else None
    shared_v = computed_values[shared_idx] if is_shared else None

    is_global = ((layer_idx + 1) % attention_pattern_size) == 0

    mask = global_attention_mask if is_global else sliding_attention_mask
    cos = rope_global_cos if is_global else rope_local_cos
    sin = rope_global_sin if is_global else rope_local_sin

    kc = (
        key_caches[layer_idx]
        if key_caches is not None and layer_idx < len(key_caches)
        else None
    )
    vc = (
        value_caches[layer_idx]
        if value_caches is not None and layer_idx < len(value_caches)
        else None
    )
    pli = None
    if per_layer_input_dim > 0:
      assert (
          per_layer_token_embeddings is not None
          and layer_idx < len(per_layer_token_embeddings)
          and per_layer_token_embeddings[layer_idx] is not None
      ), (
          f"per_layer_token_embeddings required for layer {layer_idx} when"
          " per_layer_input_dim > 0"
      )
      proj_w = weights[
          f"model.layers.{layer_idx}.per_layer_model_projection.weight"
      ]
      proj_out = np.matmul(embedded_input, proj_w.T)

      norm_w = weights["model.per_layer_projection_norm.weight"]
      _, normed_proj = rmsnorm.rms_norm(
          proj_out, scale=norm_w, eps=rms_norm_eps
      )

      sqrt_per_layer_dim = np.sqrt(float(per_layer_input_dim))
      scaled_tok_emb = (
          per_layer_token_embeddings[layer_idx] * sqrt_per_layer_dim
      )

      sum_emb = normed_proj + scaled_tok_emb
      rsqrt_2 = 1.0 / np.sqrt(2.0)
      pli = sum_emb * rsqrt_2

    layer_out, updated_k, updated_v, k_untiled, v_untiled = (
        transformer.transformer_layer(
            x=hidden_states,
            attention_mask=mask,
            cos=cos,
            sin=sin,
            weights=weights,
            key_cache=kc,
            value_cache=vc,
            per_layer_input=pli,
            shared_key=shared_k,
            shared_value=shared_v,
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            per_layer_input_dim=per_layer_input_dim,
            use_post_attn_norm=use_post_attn_norm,
            use_post_ffw_norm=use_post_ffw_norm,
            is_global=is_global,
            global_key_size=global_key_size,
            rms_norm_eps=rms_norm_eps,
            attn_logits_soft_cap=attn_logits_soft_cap,
            return_all=True,
        )
    )

    hidden_states = layer_out
    computed_keys[layer_idx] = k_untiled
    computed_values[layer_idx] = v_untiled
    updated_key_caches.append(updated_k)
    updated_value_caches.append(updated_v)

  final_norm_scale = weights["model.norm.weight"]
  _, final_output = rmsnorm.rms_norm(
      hidden_states, scale=final_norm_scale, eps=rms_norm_eps
  )

  embedding_table = weights["model.embed_tokens.weight"]
  logits = np.matmul(final_output, embedding_table.T)

  if final_logit_softcap > 0.0:
    logits = np.tanh(logits / final_logit_softcap) * final_logit_softcap

  return logits, updated_key_caches, updated_value_caches


def _create_test_weights(num_layers: int = 2) -> Mapping[str, np.ndarray]:
  """Creates dummy test weights for Gemma 4 graph tests.

  Args:
    num_layers: The number of layers in the graph.

  Returns:
    A mapping from the weight names to their tensor data.
  """
  weights = {}
  for l in range(num_layers):
    prefix = f"model.layers.{l}"
    weights[f"{prefix}.self_attn.q_proj.weight"] = np.array(
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
    )
    weights[f"{prefix}.self_attn.k_proj.weight"] = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    weights[f"{prefix}.self_attn.v_proj.weight"] = np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    weights[f"{prefix}.self_attn.o_proj.weight"] = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    weights[f"{prefix}.self_attn.q_norm.weight"] = np.ones(
        (4,), dtype=np.float32
    )
    weights[f"{prefix}.self_attn.k_norm.weight"] = np.ones(
        (4,), dtype=np.float32
    )
    weights[f"{prefix}.input_layernorm.weight"] = np.ones(
        (4,), dtype=np.float32
    )
    weights[f"{prefix}.post_attention_layernorm.weight"] = np.ones(
        (4,), dtype=np.float32
    )
    weights[f"{prefix}.pre_feedforward_layernorm.weight"] = np.ones(
        (4,), dtype=np.float32
    )
    weights[f"{prefix}.post_feedforward_layernorm.weight"] = np.ones(
        (4,), dtype=np.float32
    )
    weights[f"{prefix}.mlp.gate_proj.weight"] = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ],
        dtype=np.float32,
    )
    weights[f"{prefix}.mlp.up_proj.weight"] = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0, 0.5],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    weights[f"{prefix}.mlp.down_proj.weight"] = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0, -0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, -0.5],
        ],
        dtype=np.float32,
    )
    weights[f"{prefix}.layer_scalar"] = np.array([0.5], dtype=np.float32)

  weights["model.norm.weight"] = np.ones((4,), dtype=np.float32)
  embed_table = np.zeros((10, 4), dtype=np.float32)
  for i in range(10):
    for j in range(4):
      embed_table[i, j] = i * 0.1 + j * 0.01
  weights["model.embed_tokens.weight"] = embed_table

  return weights


def _generate_expected_values_for_default_gemma4_graph_test() -> None:
  """Generates the expected results for a default Gemma 4 graph execution."""
  print("=== Default Gemma 4 Graph ===")
  embedded_input = np.array(
      [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32
  )
  sliding_mask = np.array([[[[0.0, -1e9], [0.0, 0.0]]]], dtype=np.float32)
  cos = np.array(
      [[[[0.8660254, 0.5, 0.8660254, 0.5], [0.7071068, 0.0, 0.7071068, 0.0]]]],
      dtype=np.float32,
  )
  sin = np.array(
      [[[[0.5, 0.8660254, 0.5, 0.8660254], [0.7071068, 1.0, 0.7071068, 1.0]]]],
      dtype=np.float32,
  )
  weights = _create_test_weights(num_layers=2)

  logits, kc, vc = _gemma4_graph(
      embedded_input=embedded_input,
      weights=weights,
      sliding_attention_mask=sliding_mask,
      rope_local_cos=cos,
      rope_local_sin=sin,
      num_layers=2,
  )
  utils.cpp_print("Logits:", logits)
  utils.cpp_print("Key Cache Layer 0:", kc[0])
  utils.cpp_print("Value Cache Layer 0:", vc[0])
  utils.cpp_print("Key Cache Layer 1:", kc[1])
  utils.cpp_print("Value Cache Layer 1:", vc[1])


def _generate_expected_values_for_global_layer_graph_test() -> None:
  """Generates results for global layer & hybrid attention pattern graph."""
  print("\n=== Global Layer & Hybrid Attention Pattern Graph ===")
  embedded_input = np.array(
      [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32
  )
  sliding_mask = np.array([[[[0.0, -1e9], [0.0, 0.0]]]], dtype=np.float32)
  global_mask = np.array([[[[0.0, 0.0], [0.0, 0.0]]]], dtype=np.float32)
  cos_local = np.array(
      [[[[0.8660254, 0.5, 0.8660254, 0.5], [0.7071068, 0.0, 0.7071068, 0.0]]]],
      dtype=np.float32,
  )
  sin_local = np.array(
      [[[[0.5, 0.8660254, 0.5, 0.8660254], [0.7071068, 1.0, 0.7071068, 1.0]]]],
      dtype=np.float32,
  )
  cos_global = np.array(
      [[[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]]], dtype=np.float32
  )
  sin_global = np.array(
      [[[[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]]], dtype=np.float32
  )
  weights = _create_test_weights(num_layers=2)

  logits, _, _ = _gemma4_graph(
      embedded_input=embedded_input,
      weights=weights,
      global_attention_mask=global_mask,
      sliding_attention_mask=sliding_mask,
      rope_global_cos=cos_global,
      rope_global_sin=sin_global,
      rope_local_cos=cos_local,
      rope_local_sin=sin_local,
      num_layers=2,
      attention_pattern_size=2,  # Layer 0 is local, layer 1 is global.
  )
  utils.cpp_print("Logits (hybrid layers):", logits)


def _generate_expected_values_for_kv_cache_sharing_graph_test() -> None:
  """Generates the expected results for KV cache sharing graph execution."""
  print("\n=== KV Cache Sharing Graph ===")
  embedded_input = np.array(
      [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32
  )
  sliding_mask = np.array([[[[0.0, -1e9], [0.0, 0.0]]]], dtype=np.float32)
  cos = np.array(
      [[[[0.8660254, 0.5, 0.8660254, 0.5], [0.7071068, 0.0, 0.7071068, 0.0]]]],
      dtype=np.float32,
  )
  sin = np.array(
      [[[[0.5, 0.8660254, 0.5, 0.8660254], [0.7071068, 1.0, 0.7071068, 1.0]]]],
      dtype=np.float32,
  )
  weights = _create_test_weights(num_layers=3)

  logits, kc, _ = _gemma4_graph(
      embedded_input=embedded_input,
      weights=weights,
      sliding_attention_mask=sliding_mask,
      rope_local_cos=cos,
      rope_local_sin=sin,
      # num_layers = 3, frac_shared_layers = 1/3 -> layer 2 is shared.
      num_layers=3,
      frac_shared_layers=1.0 / 3.0,
      share_local=True,
      share_global=False,
      attention_pattern_size=6,
  )
  utils.cpp_print("Logits (KV cache sharing):", logits)
  utils.cpp_print("Key Cache Layer 2 (Shared):", kc[2])


def _generate_expected_values_for_per_layer_inputs_graph_test() -> None:
  """Generates the expected results for per-layer inputs graph execution."""
  print("\n=== Per-Layer Inputs Graph ===")
  embedded_input = np.array(
      [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32
  )
  sliding_mask = np.array([[[[0.0, -1e9], [0.0, 0.0]]]], dtype=np.float32)
  cos = np.array(
      [[[[0.8660254, 0.5, 0.8660254, 0.5], [0.7071068, 0.0, 0.7071068, 0.0]]]],
      dtype=np.float32,
  )
  sin = np.array(
      [[[[0.5, 0.8660254, 0.5, 0.8660254], [0.7071068, 1.0, 0.7071068, 1.0]]]],
      dtype=np.float32,
  )
  weights = _create_test_weights(num_layers=2)
  weights["model.layers.0.per_layer_model_projection.weight"] = np.array(
      [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32
  )
  weights["model.layers.1.per_layer_model_projection.weight"] = np.array(
      [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32
  )
  weights["model.per_layer_projection_norm.weight"] = np.ones(
      (2,), dtype=np.float32
  )

  for l in range(2):
    prefix = f"model.layers.{l}"
    weights[f"{prefix}.per_layer_input_gate.weight"] = np.array(
        [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]], dtype=np.float32
    )
    weights[f"{prefix}.per_layer_projection.weight"] = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5]], dtype=np.float32
    )
    weights[f"{prefix}.post_per_layer_input_norm.weight"] = np.ones(
        (4,), dtype=np.float32
    )

  per_layer_token_embeddings = [
      np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
      np.array([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32),
  ]

  logits, _, _ = _gemma4_graph(
      embedded_input=embedded_input,
      weights=weights,
      sliding_attention_mask=sliding_mask,
      rope_local_cos=cos,
      rope_local_sin=sin,
      per_layer_token_embeddings=per_layer_token_embeddings,
      per_layer_input_dim=2,
      num_layers=2,
  )
  utils.cpp_print("Logits (per-layer inputs):", logits)


def _generate_expected_values_for_no_softcap_graph_test() -> None:
  """Generates the expected results for graph execution with logit softcapping disabled."""
  print("\n=== No Logit Softcapping Graph ===")
  embedded_input = np.array(
      [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32
  )
  sliding_mask = np.array([[[[0.0, -1e9], [0.0, 0.0]]]], dtype=np.float32)
  cos = np.array(
      [[[[0.8660254, 0.5, 0.8660254, 0.5], [0.7071068, 0.0, 0.7071068, 0.0]]]],
      dtype=np.float32,
  )
  sin = np.array(
      [[[[0.5, 0.8660254, 0.5, 0.8660254], [0.7071068, 1.0, 0.7071068, 1.0]]]],
      dtype=np.float32,
  )
  weights = _create_test_weights(num_layers=2)

  logits, _, _ = _gemma4_graph(
      embedded_input=embedded_input,
      weights=weights,
      sliding_attention_mask=sliding_mask,
      rope_local_cos=cos,
      rope_local_sin=sin,
      num_layers=2,
      final_logit_softcap=0.0,
  )
  utils.cpp_print("Logits (no softcap):", logits)


def _generate_expected_values_for_per_layer_inputs_with_projection_graph_test() -> (
    None
):
  """Generates the expected results for per-layer inputs with projection graph execution."""
  print("\n=== Per-Layer Inputs With Projection Graph ===")
  embedded_input = np.array(
      [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32
  )
  sliding_mask = np.array([[[[0.0, -1e9], [0.0, 0.0]]]], dtype=np.float32)
  cos = np.array(
      [[[[0.8660254, 0.5, 0.8660254, 0.5], [0.7071068, 0.0, 0.7071068, 0.0]]]],
      dtype=np.float32,
  )
  sin = np.array(
      [[[[0.5, 0.8660254, 0.5, 0.8660254], [0.7071068, 1.0, 0.7071068, 1.0]]]],
      dtype=np.float32,
  )

  proj_w_data = np.array(
      [
          [0.1, 0.2, 0.3, 0.4],
          [0.5, 0.6, 0.7, 0.8],
          [0.1, 0.0, 0.1, 0.0],
          [0.2, 0.1, 0.2, 0.1],
      ],
      dtype=np.float32,
  )
  norm_w_data = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

  weights = _create_test_weights(num_layers=2)
  weights["model.layers.0.per_layer_model_projection.weight"] = proj_w_data
  weights["model.layers.1.per_layer_model_projection.weight"] = proj_w_data
  weights["model.per_layer_projection_norm.weight"] = norm_w_data

  for l in range(2):
    prefix = f"model.layers.{l}"
    weights[f"{prefix}.per_layer_input_gate.weight"] = np.zeros(
        (4, 4), dtype=np.float32
    )
    weights[f"{prefix}.per_layer_projection.weight"] = np.zeros(
        (4, 4), dtype=np.float32
    )
    weights[f"{prefix}.post_per_layer_input_norm.weight"] = np.zeros(
        (4,), dtype=np.float32
    )

  layer_emb_data = [
      np.full((1, 2, 4), 0.1, dtype=np.float32),
      np.full((1, 2, 4), 0.1, dtype=np.float32),
  ]

  logits, _, _ = _gemma4_graph(
      embedded_input=embedded_input,
      weights=weights,
      sliding_attention_mask=sliding_mask,
      rope_local_cos=cos,
      rope_local_sin=sin,
      per_layer_token_embeddings=layer_emb_data,
      per_layer_input_dim=4,
      num_layers=2,
      final_logit_softcap=30.0,
  )
  utils.cpp_print("Logits (per-layer inputs with projection):", logits)


def main() -> None:
  _generate_expected_values_for_default_gemma4_graph_test()
  _generate_expected_values_for_global_layer_graph_test()
  _generate_expected_values_for_kv_cache_sharing_graph_test()
  _generate_expected_values_for_per_layer_inputs_graph_test()
  _generate_expected_values_for_per_layer_inputs_with_projection_graph_test()
  _generate_expected_values_for_no_softcap_graph_test()


if __name__ == "__main__":
  main()
