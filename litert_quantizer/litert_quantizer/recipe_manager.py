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

"""Manages model quantization recipe (configuration) for the quantizer."""

import collections
import dataclasses
import re
from typing import Mapping, Optional
from absl import logging
from litert_quantizer import algorithm_manager
from litert_quantizer import qtyping

# A collection of quantization configuration.
# Key: scope regex.
# Value: list of OpQuantizationRecipe in dictionary format.
ModelQuantizationRecipe = qtyping.ModelQuantizationRecipe

# Expose algorithm names to users.
AlgorithmName = algorithm_manager.AlgorithmName

# Internal types.
_TFLOpName = qtyping.TFLOperationName
_OpQuantizationConfig = qtyping.OpQuantizationConfig
_TensorQuantizationConfig = qtyping.TensorQuantizationConfig


@dataclasses.dataclass
class OpQuantizationRecipe:
  """Dataclass for quantization configuration under a scope.

  This class is the main entry point to a recipe schema. There could be a single
  instance of this associated with a model (with `regex=.*` if the full model is
  to be quantized with the same spec), or multiple instances targeting different
  `regex` or `operation`.

  Attributes:
    regex: Regular expression for scope name matching. Any op that matches
      `regex` will be quantized according to this instance. The narrowest scope
      would be the full output tensor name of an op. The widest scope would be
      '.*' which applies to the full model.
    operation: Target TFL operation. * for any supported TFLite operation.
    algorithm_key: Algorithm key to be applied. This can be any one of the
      strings as enumerated in `AlgorithmName`.
    op_config: Quantization configuration to be applied for the op.
  """

  regex: str
  operation: _TFLOpName
  algorithm_key: str
  op_config: _OpQuantizationConfig = dataclasses.field(
      default_factory=_OpQuantizationConfig
  )


class RecipeManager:
  """Sets the quantization recipe for target model.

  This class is internal to Quantizer to help manage loading recipes and
  resolving conflicts between recipes input by the user.
  """

  def __init__(self):
    """Scope name config.

    Key: scope regex. ".*" for all scopes.
    Value: list of operator quantization settings under the scope.
    The priority between rules are determined by the order they entered: later
    one has higher priority.
    """
    self._scope_configs: collections.OrderedDict[
        str, list[OpQuantizationRecipe]
    ] = collections.OrderedDict()

  def add_quantization_config(
      self,
      regex: str,
      operation_name: _TFLOpName,
      op_config: Optional[_OpQuantizationConfig] = None,
      algorithm_key: str = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
  ) -> None:
    """Adds a quantization configuration.

    Conflict arises when we are trying to set an operation under a certain regex
    which is already existed in the config dictionary. Under such circumstance,
    the new config is used to replace the previous one.

    We also have special treatment for _TFLOperationKey.ALL. If the new config
    is on _TFLOperationKey.ALL and there are existing op configs inside the same
    scope, we clear the previous configs and use _TFLOperationKey.ALL.

    Args:
      regex: Regular expression for layer name matching.
      operation_name: Target TFLite operation. * for all supported TFLite
        operation.
      op_config: Quantization configuration which will be used to update the
        default configuration. None or empty dict means the default
        configuration will be used.
      algorithm_key: Algorithm key to be applied.
    """
    try:
      algorithm_manager.AlgorithmName(algorithm_key)
    except ValueError as e:
      raise ValueError(f'Unsupported algorithm key: {algorithm_key}.') from e

    if op_config is None:
      op_config = _OpQuantizationConfig()

    config = OpQuantizationRecipe(
        regex, operation_name, algorithm_key, op_config
    )
    # Special care if trying to set all ops to some config.
    if config.operation == _TFLOpName.ALL_SUPPORTED:
      self._scope_configs[regex] = [config]
      return

    if algorithm_key != AlgorithmName.NO_QUANTIZE:
      algorithm_manager.check_op_quantization_config(
          algorithm_key, operation_name, op_config
      )

    if regex not in self._scope_configs:
      self._scope_configs[regex] = [config]
    else:
      # Reiterate configs to avoid duplication on op settings.
      configs = []
      is_new_op = True
      for existing_config in self._scope_configs[regex]:
        if existing_config.operation == config.operation:
          is_new_op = False
          op_config = config  # pyrefly: ignore[bad-assignment]
          logging.warning(
              'Overwrite operation %s config under scope_regex %s with %s.',
              existing_config.operation,
              regex,
              config,
          )
        else:
          op_config = existing_config  # pyrefly: ignore[bad-assignment]
        configs.append(op_config)
      if is_new_op:
        configs.append(config)
      self._scope_configs[regex] = configs

  def get_quantization_configs(
      self,
      target_op_name: _TFLOpName,
      scope_name: str,
  ) -> tuple[str, _OpQuantizationConfig]:
    """Gets the algorithm key and quantization configuration for an op.

    We respect the latest valid config and fall back to no quantization.
    Specifically, we search the quantization configuration in the order of the
    scope configs. If there are two or more matching settings, the latest one
    will be used.


    Args:
      target_op_name: Target TFLite operation. * for all supported TFLite
        operation.
      scope_name: Name of the target scope.

    Returns:
       A tuple of quantization algorithm, and quantization configuration.
    """
    result_key, result_config = (
        AlgorithmName.NO_QUANTIZE,
        _OpQuantizationConfig(),
    )
    for scope_regex, recipes in self._scope_configs.items():
      if re.search(scope_regex, scope_name):
        for recipe in recipes:
          if (
              recipe.operation != _TFLOpName.ALL_SUPPORTED
              and recipe.operation != target_op_name
          ):
            continue
          selected_recipe = recipe
          if selected_recipe.algorithm_key != AlgorithmName.NO_QUANTIZE:
            # The selected recipe must contain a supported config.
            try:
              algorithm_manager.check_op_quantization_config(
                  recipe.algorithm_key, target_op_name, recipe.op_config
              )
            except ValueError:
              continue  # Skip the recipe if it is not supported.
          result_config = selected_recipe.op_config
          result_key = selected_recipe.algorithm_key

    return result_key, result_config

  def get_quantization_recipe(self) -> qtyping.ModelQuantizationRecipe:
    """Gets the full quantization recipe from the manager.

    Returns:
      A list of quantization configs in the recipe.
    """
    recipe = []
    for _, op_recipes in self._scope_configs.items():
      for op_recipe in op_recipes:
        recipe_dict = dataclasses.asdict(
            op_recipe,
            dict_factory=lambda x: {  # pylint: disable=g-long-lambda
                k: (
                    dict(v)  # pylint: disable=g-long-ternary
                    if isinstance(v, Mapping) and not isinstance(v, dict)
                    else v
                )
                for (k, v) in x
                # Skip None and empty dict values.
                if v is not None
                and not (isinstance(v, (dict, Mapping)) and not v)
            },
        )
        recipe.append(recipe_dict)
    return recipe

  def load_quantization_recipe(
      self, quantization_recipe: qtyping.ModelQuantizationRecipe
  ) -> None:
    """Loads the quantization recipe to the manager.

    Args:
      quantization_recipe: A configuration dictionary which is generated by
        get_full_config.
    """
    self._scope_configs = collections.OrderedDict()
    for config in quantization_recipe:
      self.add_quantization_config(
          config['regex'],
          config['operation'],
          _OpQuantizationConfig.from_dict(config['op_config'])
          if config['algorithm_key'] != AlgorithmName.NO_QUANTIZE
          else None,
          config['algorithm_key'],
      )

  def need_calibration(self) -> bool:
    """Check if the recipe requires calibration."""
    # At the moment, only SRQ requires calibration.
    for op_quant_config in self.get_quantization_recipe():
      if (
          op_quant_config['op_config']['compute_precision']
          == qtyping.ComputePrecision.INTEGER
          and 'activation_tensor_config' in op_quant_config['op_config']
      ):
        return True
      if op_quant_config['algorithm_key'] == AlgorithmName.GPTQ:
        return True
    return False

  def add_dynamic_config(
      self,
      regex: str,
      operation_name: _TFLOpName,
      num_bits: int,
      granularity: qtyping.QuantGranularity = qtyping.QuantGranularity.CHANNELWISE,
      algorithm_key: str = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
  ):
    """Adds a dynamic quantization configuration to the recipe.

    During dynamic quantization, activations are not processed by AEQ and
    remain in float format. The runtime kernel is expected to quantize these
    activations on-the-fly, as indicated by compute_precision=Integer and
    explicit_dequantize=False.

    The model quality may suffer due to the on-the-fly quantization. If quality
    is a concern, consider using weight-only quantization.

    Args:
      regex: Regular expression for layer name matching.
      operation_name: Target TFLite operation.
      num_bits: Number of bits for quantization.
      granularity: Granularity of quantization.
      algorithm_key: Algorithm key to be applied.
    """
    weight_config = qtyping.TensorQuantizationConfig(
        num_bits=num_bits,
        symmetric=True,  # LiteRT kernels only support symmetric quantized
        # weights.
        granularity=granularity,
    )
    self.add_quantization_config(
        regex,
        operation_name,
        op_config=_OpQuantizationConfig(
            weight_tensor_config=weight_config,
            compute_precision=qtyping.ComputePrecision.INTEGER,
            explicit_dequantize=False,
        ),
        algorithm_key=algorithm_key,
    )

  def add_weight_only_config(
      self,
      regex: str,
      operation_name: _TFLOpName,
      num_bits: int,
      granularity: qtyping.QuantGranularity = qtyping.QuantGranularity.CHANNELWISE,
      algorithm_key: str = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
  ):
    """Adds a weight only quantization configuration to the recipe.

    In weight-only quantization, weights are quantized, but the actual operation
    (op) computation remains in float. The quantized weight is explicitly
    dequantized before being fed into the op. This is achieved by inserting a
    dequantize op between the quantized weight and the consuming op. To enable
    this, both compute_precision will be set to Float and explicit_dequantize to
    True.

    Weight-only quantization is useful for reducing model size but may
    not decrease latency due to float computation. However, quantized model
    generally has better quality than other quantization options (e.g., dynamic
    range quantization) due to no loss of precision on activations. If latency
    is a concern, consider using dynamic quantization.

    Args:
      regex: Regular expression for layer name matching.
      operation_name: Target TFLite operation.
      num_bits: Number of bits for quantization.
      granularity: Granularity of quantization.
      algorithm_key: Algorithm key to be applied.
    """
    # Default to integer quantization but allow float quantization for
    # FLOAT_CASTING algorithm. This is to support weight-only quantization with
    # fp16 weights.
    weight_dtype = qtyping.TensorDataType.INT
    if algorithm_key == AlgorithmName.FLOAT_CASTING:
      weight_dtype = qtyping.TensorDataType.FLOAT

    weight_config = qtyping.TensorQuantizationConfig(
        num_bits=num_bits,
        symmetric=True,  # TFL kernels only support symmetric quantized weights.
        granularity=granularity,
        dtype=weight_dtype,
    )
    self.add_quantization_config(
        regex,
        operation_name,
        op_config=_OpQuantizationConfig(
            weight_tensor_config=weight_config,
            compute_precision=qtyping.ComputePrecision.FLOAT,
            explicit_dequantize=True,
        ),
        algorithm_key=algorithm_key,
    )

  def add_static_config(
      self,
      regex: str,
      operation_name: _TFLOpName,
      activation_num_bits: int,
      weight_num_bits: int,
      weight_granularity: qtyping.QuantGranularity = qtyping.QuantGranularity.CHANNELWISE,
      algorithm_key: str = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
  ):
    """Adds a static range quantization configuration to the recipe.

    In static quantization, both weights and activations are quantized. This
    requires a calibration step to determine the quantization parameters (e.g.,
    min/max ranges) for activations. The quantized model uses integer arithmetic
    for computations, which can lead to significant latency reductions.

    However, calibration is needed to determine the quantization parameters for
    activations, which requires sample data and may lead to quality loss. If
    there is no hardware requirement for full integer quantization, consider
    using dynamic quantization for simplicity.

    Args:
      regex: Regular expression for layer name matching.
      operation_name: Target TFLite operation.
      activation_num_bits: Number of bits for activation quantization.
      weight_num_bits: Number of bits for weight quantization.
      weight_granularity: Granularity of weight quantization.
      algorithm_key: Algorithm key to be applied.
    """
    if activation_num_bits not in [16, 8]:
      raise ValueError(
          'Activation quantization is only supported for 16 or 8 bits.'
      )
    # INT16 is symmetric and INT8 is asymmetric due to LiteRT kernel
    # limitations.
    activation_symmetric = activation_num_bits == 16
    activation_config = qtyping.TensorQuantizationConfig(
        num_bits=activation_num_bits, symmetric=activation_symmetric
    )
    weight_config = qtyping.TensorQuantizationConfig(
        num_bits=weight_num_bits,
        symmetric=True,  # TFL kernels only support symmetric quantized weights.
        granularity=weight_granularity,
    )
    self.add_quantization_config(
        regex,
        operation_name,
        op_config=_OpQuantizationConfig(
            activation_tensor_config=activation_config,
            weight_tensor_config=weight_config,
            compute_precision=qtyping.ComputePrecision.INTEGER,
            explicit_dequantize=False,
        ),
        algorithm_key=algorithm_key,
    )
