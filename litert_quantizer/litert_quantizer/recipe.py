# Copyright 2024 Google LLC
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

"""Quantization recipe module.

Existing recipes include:

  1. Dynamic quantization recipes:
    - dynamic_wi8_afp32
    - dynamic_wi4_afp32
  2. Weight-only quantization recipes:
    - weight_only_wi8_afp32
    - weight_only_wi4_afp32
  3. Static quantization recipes:
    - static_wi8_ai8
    - static_wi8_ai16
  4. LiteRT-LM recipes:
    - gemma4_mixed48
    - gemma4_mixed48_hr
    - gemma4_mixed48_b32
    - gemma4_mixed48_b64

Naming convention decoder for recipes:
  - 'dynamic': dynamic range quantization (weights quantized statically,
    activations dynamically at runtime).
  - 'wi[N]': weight integer N-bit (e.g., wi8: 8-bit weights).
  - 'c' or 'b[M]': Granularity. 'c' for channelwise, 'b[M]' for blockwise with
    block size M (e.g., b32: 32-blockwise).
  - 'hr' (optional): Hadamard rotations are used for the quantization
    parameters estimation. (typically for better quality at lower bits, see
    algorithms/uniform_quantize/hadamard_rotation.py).
  - 'afp32': activations remain float32 in the model (may be dynamically
    quantized at runtime by setting compute_precision=INTEGER,
    explicit_dequantize=False).
"""

from litert_quantizer import algorithm_manager
from litert_quantizer import qtyping
from litert_quantizer import recipe_manager

AlgorithmName = algorithm_manager.AlgorithmName
QuantGranularity = qtyping.QuantGranularity
TFLOperationName = qtyping.TFLOperationName


def _dynamic_wix_afp32(
    num_bits: int,
    regex: str = '.*',
    operation_name: TFLOperationName = TFLOperationName.ALL_SUPPORTED,
    **kwargs
) -> qtyping.ModelQuantizationRecipe:
  """Returns a dynamic quantization recipe with int weights and float32 activation.

  All supported ops will be quantized with `num_bits`-bit weights and
  float32 activations, which will be dynamically quantized to `num_bits`-bit
  values during inference to enable integer compute. The model quality may
  suffer due to the on-the-fly quantization. If quality is a concern, consider
  using weight-only quantization.

  Args:
    num_bits: The number of bits to quantize to.
    regex: Optional regular expression for layer name matching.
    operation_name: Target TFLite operation.
    **kwargs: Additional arguments passed to `RecipeManager.add_dynamic_config`.

  Returns:
    A dynamic quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_dynamic_config(
      regex=regex, operation_name=operation_name, num_bits=num_bits, **kwargs
  )
  return rp_manager.get_quantization_recipe()


def dynamic_wi8_afp32(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
) -> qtyping.ModelQuantizationRecipe:
  """Returns a dynamic quantization recipe with channelwise quantized int8 weights and float32 activation.

  All supported ops will be quantized with int8 weights and float32 activations,
  which will be dynamically quantized to int8 during inference to enable int8
  compute. The model quality may suffer due to the on-the-fly quantization. If
  quality is a concern, consider using weight-only quantization.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A dynamic quantization recipe.
  """
  return dynamic_wi8c_afp32(algorithm_key=algorithm_key)


def dynamic_wi4_afp32(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
) -> qtyping.ModelQuantizationRecipe:
  """Returns a dynamic quantization recipe with channelwise quantized int4 weights and float32 activation.

  All supported ops will be quantized with int4 weights and float32 activations,
  which will be dynamically quantized to int4 during inference to enable int4
  compute.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A dynamic quantization recipe.
  """
  return dynamic_wi4c_afp32(algorithm_key=algorithm_key)


def weight_only_wi8_afp32(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
) -> qtyping.ModelQuantizationRecipe:
  """Returns a weight-only quantization recipe with int8 weights and float32 activation.

  All supported ops will be quantized with int8 weights and float32 activations.
  The weights will be explicitly dequantized before being fed into the op to
  enable float compute thus retain model quality. If latency is a concern,
  consider using dynamic range quantization.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A weight-only quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_weight_only_config(
      regex='.*',
      operation_name=TFLOperationName.ALL_SUPPORTED,
      num_bits=8,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def weight_only_wi4_afp32(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
) -> qtyping.ModelQuantizationRecipe:
  """Returns a weight-only quantization recipe with int4 weights and float32 activation.

  All supported ops will be quantized with int4 weights and float32 activations.
  The weights will be explicitly dequantized before being fed into the op to
  enable float compute thus retain model quality.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A weight-only quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_weight_only_config(
      regex='.*',
      operation_name=TFLOperationName.ALL_SUPPORTED,
      num_bits=4,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def static_wi8_ai8(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
) -> qtyping.ModelQuantizationRecipe:
  """Returns a static quantization recipe with int8 weights and int8 activations.

  All supported ops will be quantized with int8 weights and int8 activations.
  Calibration is needed to use this recipe.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A static quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_static_config(
      regex='.*',
      operation_name=TFLOperationName.ALL_SUPPORTED,
      activation_num_bits=8,
      weight_num_bits=8,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def static_wi8_ai16(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
) -> qtyping.ModelQuantizationRecipe:
  """Returns a static quantization recipe with int8 weights and int16 activations.

  All supported ops will be quantized with int8 weights and int16 activations.
  Calibration is needed to use this recipe.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A static quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_static_config(
      regex='.*',
      operation_name=TFLOperationName.ALL_SUPPORTED,
      activation_num_bits=16,
      weight_num_bits=8,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def dynamic_legacy_wi8_afp32():
  """Returns a dynamic quantization legacy recipe with int8 weights and float32 activation.

  The difference between this and dynamic_wi8_afp32 is that this recipe sets
  min_weight_elements to 1024 to match the old quantizer behavior. Only layers
  with at least 1024 weight elements will be quantized. This 'legacy' recipe is
  intended for models that were quantized with older versions of TFLite
  quantization tools.
  """
  return [
      dict({
          'regex': '.*',
          'operation': '*',
          'algorithm_key': 'min_max_uniform_quantize',
          'op_config': {
              'weight_tensor_config': {
                  'num_bits': 8,
                  'symmetric': True,
                  'granularity': 'CHANNELWISE',
                  'dtype': 'INT',
                  'block_size': 0,
              },
              'compute_precision': 'INTEGER',
              'explicit_dequantize': False,
              'skip_checks': False,
              'min_weight_elements': 1024,
          },
      })
  ]


# Recipe aliases, i.e. basic recipes only, to be used as building blocks for
# more complex recipes.

# Naming convention decoder for aliases:
# - 'dynamic': dynamic range quantization (weights quantized statically,
#   activations dynamically at runtime)
# - 'wi[N]': weight integer N-bit (e.g., wi8: 8-bit weights)
# - 'c' or 'b[M]': Granularity. 'c' for channelwise, 'b[M]' for blockwise with
#   block size M (e.g., b32: 32-blockwise).
# - 'hr' (optional): Hadamard rotations are used for the quantization
#   parameters estimation. (typically for better quality at lower bits, see
#   algorithms/uniform_quantize/hadamard_rotation.py).
# - 'afp32': activations remain float32 in the model (may be dynamically
#   quantized at runtime by setting compute_precision=INTEGER,
#   explicit_dequantize=False).

# Dynamic quantization recipe with channelwise quantized 8/4-bit int
# weights and float32 activations.
dynamic_wi8c_afp32 = lambda **kwargs: _dynamic_wix_afp32(num_bits=8, **kwargs)
dynamic_wi4c_afp32 = lambda **kwargs: _dynamic_wix_afp32(num_bits=4, **kwargs)
dynamic_wi2c_afp32 = lambda **kwargs: _dynamic_wix_afp32(num_bits=2, **kwargs)


# Dynamic quantization recipe with 32/64-blockwisse quantized 8/4-bit int
# weights and float32 activations for everything.
dynamic_wi8b32_afp32 = lambda **kwargs: _dynamic_wix_afp32(
    num_bits=8, granularity=QuantGranularity.BLOCKWISE_32, **kwargs
)
dynamic_wi4b32_afp32 = lambda **kwargs: _dynamic_wix_afp32(
    num_bits=4, granularity=QuantGranularity.BLOCKWISE_32, **kwargs
)
dynamic_wi2b32_afp32 = lambda **kwargs: _dynamic_wix_afp32(
    num_bits=2, granularity=QuantGranularity.BLOCKWISE_32, **kwargs
)
dynamic_wi8b64_afp32 = lambda **kwargs: _dynamic_wix_afp32(
    num_bits=8, granularity=QuantGranularity.BLOCKWISE_64, **kwargs
)
dynamic_wi4b64_afp32 = lambda **kwargs: _dynamic_wix_afp32(
    num_bits=4, granularity=QuantGranularity.BLOCKWISE_64, **kwargs
)
dynamic_wi2b64_afp32 = lambda **kwargs: _dynamic_wix_afp32(
    num_bits=2, granularity=QuantGranularity.BLOCKWISE_64, **kwargs
)

# Dynamic quantization recipe with channelwise quantized 8/4-bit int
# weights with Hadamard rotations (`hr`) and float32 activations.
dynamic_wi8c_hr_afp32 = lambda **kwargs: dynamic_wi8c_afp32(
    algorithm_key=AlgorithmName.DECOMPOSED_HADAMARD_ROTATION, **kwargs
)
dynamic_wi4c_hr_afp32 = lambda **kwargs: dynamic_wi4c_afp32(
    algorithm_key=AlgorithmName.DECOMPOSED_HADAMARD_ROTATION, **kwargs
)
dynamic_wi2c_hr_afp32 = lambda **kwargs: dynamic_wi2c_afp32(
    algorithm_key=AlgorithmName.DECOMPOSED_HADAMARD_ROTATION, **kwargs
)

# LiteRT-LM Recipes for specific model families, build from the above recipes.
#
# These recipes typically define mixed-precision configurations. For example,
# using lower precision (4-bit) for most weights to save space, but keeping
# sensitive layers (like embeddings, projections, etc) at higher precision
# (8-bit) to preserve model quality.

# Gemma-4 mixed 4/8-bit channelwise quantization:
gemma4_mixed48 = lambda: {
    'tf_lite_embedder': dynamic_wi4c_afp32(
        operation_name=TFLOperationName.EMBEDDING_LOOKUP,
    ),
    'tf_lite_per_layer_embedder': dynamic_wi4c_afp32(
        operation_name=TFLOperationName.EMBEDDING_LOOKUP,
    ),
    'tf_lite_prefill_decode': (
        dynamic_wi4c_afp32(
            operation_name=TFLOperationName.FULLY_CONNECTED,
        )
        # Per-layer embeddings need 8 bits.
        + dynamic_wi8c_afp32(
            regex='per_layer',
            operation_name=TFLOperationName.FULLY_CONNECTED,
        )
    ),
}

# Gemma-4 mixed 4/8-bit channelwise quantization with Hadamard rotations for the
# 4-bit ops:
gemma4_mixed48_hr = lambda: {
    'tf_lite_embedder': dynamic_wi4c_hr_afp32(
        operation_name=TFLOperationName.EMBEDDING_LOOKUP,
    ),
    'tf_lite_per_layer_embedder': dynamic_wi4c_hr_afp32(
        operation_name=TFLOperationName.EMBEDDING_LOOKUP,
    ),
    'tf_lite_prefill_decode': (
        dynamic_wi4c_hr_afp32(
            operation_name=TFLOperationName.FULLY_CONNECTED,
        )
        # Per-layer embeddings need 8 bits.
        + dynamic_wi8c_afp32(
            regex='per_layer',
            operation_name=TFLOperationName.FULLY_CONNECTED,
        )
    ),
}

# Gemma-4 mixed 4-bit blockwise quantization, using block sizes of 32 and 64.
gemma4_mixed48_b32 = lambda: {
    'tf_lite_embedder': dynamic_wi4b32_afp32(
        operation_name=TFLOperationName.EMBEDDING_LOOKUP,
    ),
    'tf_lite_per_layer_embedder': dynamic_wi4b32_afp32(
        operation_name=TFLOperationName.EMBEDDING_LOOKUP,
    ),
    'tf_lite_prefill_decode': (
        dynamic_wi4b32_afp32(
            operation_name=TFLOperationName.FULLY_CONNECTED,
        )
        # Per-layer embeddings need 8 bits.
        + dynamic_wi8c_afp32(
            regex='per_layer',
            operation_name=TFLOperationName.FULLY_CONNECTED,
        )
    ),
}
gemma4_mixed48_b64 = lambda: {
    'tf_lite_embedder': dynamic_wi4b64_afp32(
        operation_name=TFLOperationName.EMBEDDING_LOOKUP,
    ),
    'tf_lite_per_layer_embedder': dynamic_wi4b64_afp32(
        operation_name=TFLOperationName.EMBEDDING_LOOKUP,
    ),
    'tf_lite_prefill_decode': (
        dynamic_wi4b64_afp32(
            operation_name=TFLOperationName.FULLY_CONNECTED,
        )
        # Per-layer embeddings need 8 bits.
        + dynamic_wi8c_afp32(
            regex='per_layer',
            operation_name=TFLOperationName.FULLY_CONNECTED,
        )
    ),
}
