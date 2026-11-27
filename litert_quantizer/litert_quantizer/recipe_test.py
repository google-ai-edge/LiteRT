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

import pathlib
import unittest  # pylint: disable=unused-import, required for OSS.

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import quantizer
from litert_quantizer import recipe
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils
from litert_quantizer.utils import tfl_interpreter_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('')


class RecipeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Weights has < 1024 elements so legacy recipe will not quantize it.
    self._small_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH)
        / 'tests/models/single_conv2d_transpose_bias.tflite',
    )
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH)
        / 'tests/models/conv_fc_mnist.tflite',
    )

  def _quantize_with_recipe_func(self, recipe_func, test_model_path):
    qt = quantizer.Quantizer(test_model_path)
    qt.load_quantization_recipe(recipe_func())
    self.assertIsNone(qt._result.quantized_model)
    if qt.need_calibration:
      calibration_data = tfl_interpreter_utils.create_random_normal_input_data(
          qt._float_model_buffer,  # pyrefly: ignore[bad-argument-type]
          num_samples=1,
      )
      calibration_result = qt.calibrate(calibration_data)  # pyrefly: ignore[bad-argument-type]
      quantization_result = qt.quantize(calibration_result)
    else:
      quantization_result = qt.quantize()
    self.assertIsNotNone(quantization_result.quantized_model)
    return quantization_result

  @unittest.skip('skipping due to b/438971945')
  def test_quantization_from_dynamic_wi8_afp32_func_succeeds(self):
    quant_result = self._quantize_with_recipe_func(
        recipe.dynamic_wi8_afp32, self._test_model_path
    )
    self.assertLess(
        len(quant_result.quantized_model),
        pathlib.Path(self._test_model_path).stat().st_size,
    )

  @unittest.skip('skipping due to b/438971945')
  def test_quantization_from_dynamic_wi4_afp32_func_succeeds(self):
    quant_result = self._quantize_with_recipe_func(
        recipe.dynamic_wi4_afp32, self._test_model_path
    )
    self.assertLess(
        len(quant_result.quantized_model),
        pathlib.Path(self._test_model_path).stat().st_size,
    )

  def _verify_quantized_model(
      self,
      quant_result: quantizer.QuantizationResult,
      float_model_path: str,
      min_max_range: tuple[int | float, int | float] | None = None,
  ) -> None:
    quantized_model = quant_result.quantized_model
    if quantized_model is None:
      self.fail('Expected non-None quantized model.')
    self.assertFalse(
        tfl_flatbuffer_utils.is_float_model(
            tfl_flatbuffer_utils.read_model(quantized_model)
        )
    )
    self.assertLess(
        len(quantized_model),
        pathlib.Path(float_model_path).stat().st_size,
    )
    model_bytes = bytes(quantized_model)
    interpreter = tfl_interpreter_utils.create_tfl_interpreter(model_bytes)
    input_data = tfl_interpreter_utils.create_random_normal_input_data(
        model_bytes,
        num_samples=1,
        min_max_range=min_max_range,
    )
    signature_key, samples = next(iter(input_data.items()))
    tfl_interpreter_utils.invoke_interpreter_signature(
        interpreter, samples[0], signature_key=signature_key
    )
    output_tensor = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index']
    )
    self.assertTrue(np.all(np.isfinite(output_tensor)))

  def _create_blockwise_embedding_model(self) -> str:
    # `tests/models/embedding_lookup.tflite` has a [16, 8] table; expand the
    # embedding dimension to 32 in memory so it is divisible by block size 32.
    model_path = (
        pathlib.Path(_TEST_DATA_PREFIX_PATH)
        / 'tests/models/embedding_lookup.tflite'
    )
    model = tfl_flatbuffer_utils.read_model(model_path)
    (embedding_op,) = model.subgraphs[0].operators
    table_tensor = model.subgraphs[0].tensors[embedding_op.inputs[1]]
    output_tensor = model.subgraphs[0].tensors[embedding_op.outputs[0]]
    table = tfl_flatbuffer_utils.get_tensor_data(table_tensor, model.buffers)
    if table is None:
      self.fail('Expected embedding table tensor data.')
    expanded_table = np.tile(table, (1, 4)).astype(np.float32)
    table_tensor.shape = list(expanded_table.shape)
    output_tensor.shape = [1, expanded_table.shape[1]]
    output_tensor.shapeSignature = None
    model.buffers[table_tensor.buffer].data = np.frombuffer(
        expanded_table.tobytes(), dtype=np.uint8
    )
    expanded_model_path = self.create_tempfile().full_path
    tfl_flatbuffer_utils.write_model(model, expanded_model_path)
    return expanded_model_path

  def test_quantization_from_dynamic_wi2c_afp32_func_succeeds(self):
    quant_result = self._quantize_with_recipe_func(
        recipe.dynamic_wi2c_afp32, self._test_model_path
    )
    self._verify_quantized_model(quant_result, self._test_model_path)

  @parameterized.named_parameters(
      dict(
          testcase_name='channelwise',
          granularity=recipe.QuantGranularity.CHANNELWISE,
      ),
      dict(
          testcase_name='tensorwise',
          granularity=recipe.QuantGranularity.TENSORWISE,
      ),
  )
  def test_quantization_from_weight_only_wi2_config_succeeds(self, granularity):
    qt = quantizer.Quantizer(self._test_model_path)
    qt.add_weight_only_config(
        regex='.*',
        operation_name=recipe.TFLOperationName.FULLY_CONNECTED,
        num_bits=2,
        granularity=granularity,
    )
    self._verify_quantized_model(qt.quantize(), self._test_model_path)

  def test_quantization_from_dynamic_wi2b32_afp32_embedding_lookup_succeeds(
      self,
  ):
    model_path = self._create_blockwise_embedding_model()
    qt = quantizer.Quantizer(model_path)
    qt.load_quantization_recipe(
        recipe.dynamic_wi2b32_afp32(
            operation_name=recipe.TFLOperationName.EMBEDDING_LOOKUP
        )
    )
    # `embedding_lookup.tflite` has 16 rows, so lookup indices must be in
    # [0, 16).
    self._verify_quantized_model(
        qt.quantize(), model_path, min_max_range=(0, 16)
    )

  def test_conv2d_int2_config_rejected_by_policy(self):
    qt = quantizer.Quantizer(self._test_model_path)
    with self.assertRaisesRegex(ValueError, 'Unsupported op'):
      qt.add_dynamic_config(
          regex='.*',
          operation_name=recipe.TFLOperationName.CONV_2D,
          num_bits=2,
      )
    with self.assertRaisesRegex(ValueError, 'Unsupported op'):
      qt.add_weight_only_config(
          regex='.*',
          operation_name=recipe.TFLOperationName.CONV_2D,
          num_bits=2,
      )

  @unittest.skip('skipping due to b/438971945')
  def test_quantization_from_weight_only_wi8_afp32_func_succeeds(self):
    quant_result = self._quantize_with_recipe_func(
        recipe.weight_only_wi8_afp32, self._test_model_path
    )
    self.assertLess(
        len(quant_result.quantized_model),
        pathlib.Path(self._test_model_path).stat().st_size,
    )

  @unittest.skip('skipping due to b/438971945')
  def test_quantization_from_weight_only_wi4_afp32_func_succeeds(self):
    quant_result = self._quantize_with_recipe_func(
        recipe.weight_only_wi4_afp32, self._test_model_path
    )
    self.assertLess(
        len(quant_result.quantized_model),
        pathlib.Path(self._test_model_path).stat().st_size,
    )

  def test_quantization_from_dynamic_legacy_wi8_afp32_func_succeeds(self):
    quant_result = self._quantize_with_recipe_func(
        recipe.dynamic_legacy_wi8_afp32,
        self._small_model_path,
    )
    self.assertLen(
        quant_result.quantized_model,
        pathlib.Path(self._small_model_path).stat().st_size,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='dynamic_wi8_afp32',
          recipe_name='dynamic_wi8_afp32',
          recipe_func=recipe.dynamic_wi8_afp32,
      ),
      dict(
          testcase_name='weight_only_wi8_afp32',
          recipe_name='default_af32w8float',
          recipe_func=recipe.weight_only_wi8_afp32,
      ),
      dict(
          testcase_name='weight_only_wi4_afp32',
          recipe_name='default_af32w4float',
          recipe_func=recipe.weight_only_wi4_afp32,
      ),
      dict(
          testcase_name='dynamic_legacy_wi8_afp32',
          recipe_name='dynamic_legacy_wi8_afp32',
          recipe_func=recipe.dynamic_legacy_wi8_afp32,
      ),
      dict(
          testcase_name='a8w8',
          recipe_name='default_a8w8',
          recipe_func=recipe.static_wi8_ai8,
      ),
      dict(
          testcase_name='a16w8',
          recipe_name='default_a16w8',
          recipe_func=recipe.static_wi8_ai16,
      ),
  )
  @unittest.skip('skipping due to b/438971945')
  def test_recipe_func_and_json_matches(self, recipe_name, recipe_func):
    # Quantize with recipe from function in recipe module.
    quant_result_from_func = self._quantize_with_recipe_func(
        recipe_func, self._test_model_path
    )

    # Quantize with recipe from json file.
    qt_json = quantizer.Quantizer(self._test_model_path)
    qt_json.load_quantization_recipe(recipe_name)
    if qt_json.need_calibration:
      calibration_data = tfl_interpreter_utils.create_random_normal_input_data(
          qt_json._float_model_buffer,  # pyrefly: ignore[bad-argument-type]
          num_samples=1,
      )
      calibration_result = qt_json.calibrate(calibration_data)  # pyrefly: ignore[bad-argument-type]
      quant_result_from_json = qt_json.quantize(calibration_result)
    else:
      quant_result_from_json = qt_json.quantize()
    self.assertIsNotNone(quant_result_from_json.quantized_model)

    # Check if the quantized models match.
    self.assertEqual(
        len(quant_result_from_func.quantized_model),
        len(quant_result_from_json.quantized_model),
    )


if __name__ == '__main__':
  absltest.main()
