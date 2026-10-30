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

from litert_quantizer import quantizer
from litert_quantizer import recipe
from litert_quantizer.utils import test_utils
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

  def test_quantization_from_dynamic_wi2c_afp32_func_succeeds(self):
    quant_result = self._quantize_with_recipe_func(
        recipe.dynamic_wi2c_afp32, self._test_model_path
    )
    self.assertLess(
        len(quant_result.quantized_model),
        pathlib.Path(self._test_model_path).stat().st_size,
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
