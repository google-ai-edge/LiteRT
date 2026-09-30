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

import copy
import dataclasses

from absl.testing import absltest
import numpy as np

from litert_quantizer import qtyping


def _create_dummy_uniform_quant_params(**kwargs) -> qtyping.UniformQuantParams:
  default_params = dict(
      num_bits=8,
      quantized_dimension=1,
      scale=np.array([[1.0]], dtype=np.float32),
      zero_point=np.array([[0]], dtype=np.int64),
      symmetric=False,
      quantized_data=np.random.uniform(-127, 127, size=[10, 10]).astype(
          np.uint8
      ),
      block_size=0,
      hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
          random_binary_vector=np.random.uniform(0, 2, size=[10]).astype(
              np.int32
          ),
          hadamard_size=10,
      ),
  )
  default_params.update(kwargs)
  return qtyping.UniformQuantParams(**default_params)


class QtypingTest(absltest.TestCase):

  def test_uniform_quant_params_eq(self):
    # Create some random params.
    quant_params = _create_dummy_uniform_quant_params()
    assert quant_params.quantized_data is not None
    assert quant_params.hadamard is not None

    # Trivial equality check.
    self.assertEqual(quant_params, quant_params)
    self.assertEqual(quant_params, copy.copy(quant_params))
    self.assertEqual(quant_params, copy.deepcopy(quant_params))

    # Check whether `nd.nparray.view` works.
    other = dataclasses.replace(
        quant_params, quantized_data=quant_params.quantized_data.view(np.uint8)
    )
    self.assertEqual(quant_params, other)

    # Check whether changes in any of the parameters will fail.
    other = dataclasses.replace(quant_params, num_bits=4)
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(quant_params, quantized_dimension=0)
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(
        quant_params, scale=np.array([[10.0]], dtype=np.float32)
    )
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(
        quant_params, zero_point=np.array([[1]], dtype=np.int64)
    )
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(quant_params, symmetric=True)
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(
        quant_params,
        quantized_data=np.random.uniform(-127, 127, size=[10, 10]).astype(
            np.uint8
        ),
    )
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(quant_params, block_size=32)
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(
        quant_params,
        hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
            random_binary_vector=quant_params.hadamard.random_binary_vector + 1,
            hadamard_size=10,
        ),
    )
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(
        quant_params,
        hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
            random_binary_vector=quant_params.hadamard.random_binary_vector,
            hadamard_size=16,
        ),
    )
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(quant_params, signed=False)
    self.assertNotEqual(quant_params, other)

  def test_from_tfl_tensor_details_uint8(self):
    tensor_detail = {
        "dtype": np.uint8,
        "quantization_parameters": {
            "quantized_dimension": 0,
            "scales": np.array([0.1], dtype=np.float32),
            "zero_points": np.array([128], dtype=np.int64),
            "block_size": 0,
        },
    }
    params = qtyping.UniformQuantParams.from_tfl_tensor_details(tensor_detail)
    self.assertEqual(params.num_bits, 8)
    self.assertFalse(params.signed)
    self.assertFalse(params.symmetric)
    self.assertEqual(params.block_size, 0)
    self.assertTrue(
        np.array_equal(params.scale, np.array([0.1], dtype=np.float32))
    )
    self.assertTrue(
        np.array_equal(params.zero_point, np.array([128], dtype=np.int64))
    )

  def test_compare_custom_algorithm_param_when_same_object_is_equal(self):
    quant_params = _create_dummy_uniform_quant_params(
        custom_algorithm_param={
            "multiplier": np.array([1.0, 2.0]),
            "str_key": "val",
        }
    )
    self.assertEqual(quant_params, quant_params)

  def test_compare_custom_algorithm_param_when_shallow_copy_is_equal(self):
    quant_params = _create_dummy_uniform_quant_params(
        custom_algorithm_param={
            "multiplier": np.array([1.0, 2.0]),
            "str_key": "val",
        }
    )
    self.assertEqual(quant_params, copy.copy(quant_params))

  def test_compare_custom_algorithm_param_when_deep_copy_is_equal(self):
    quant_params = _create_dummy_uniform_quant_params(
        custom_algorithm_param={
            "multiplier": np.array([1.0, 2.0]),
            "str_key": "val",
        }
    )
    self.assertEqual(quant_params, copy.deepcopy(quant_params))

  def test_compare_custom_algorithm_param_when_none_vs_dict_is_not_equal(self):
    params_none = _create_dummy_uniform_quant_params(
        custom_algorithm_param=None
    )
    params_dict = _create_dummy_uniform_quant_params(
        custom_algorithm_param={"multiplier": np.array([1.0, 2.0])}
    )
    self.assertNotEqual(params_none, params_dict)

  def test_compare_custom_algorithm_param_when_different_values_is_not_equal(
      self,
  ):
    params1 = _create_dummy_uniform_quant_params(
        custom_algorithm_param={
            "multiplier": np.array([1.0, 2.0]),
            "str_key": "val",
        }
    )
    params2 = _create_dummy_uniform_quant_params(
        custom_algorithm_param={
            "multiplier": np.array([1.0, 3.0]),
            "str_key": "val",
        }
    )
    self.assertNotEqual(params1, params2)

  def test_compare_custom_algorithm_param_when_different_keys_is_not_equal(
      self,
  ):
    params1 = _create_dummy_uniform_quant_params(
        custom_algorithm_param={"key_a": np.array([1.0, 2.0])}
    )
    params2 = _create_dummy_uniform_quant_params(
        custom_algorithm_param={"key_b": np.array([1.0, 2.0])}
    )
    self.assertNotEqual(params1, params2)

  def test_compare_custom_algorithm_param_when_different_types_is_not_equal(
      self,
  ):
    params1 = _create_dummy_uniform_quant_params(
        custom_algorithm_param={"val": np.array([1.0, 2.0])}
    )
    params2 = _create_dummy_uniform_quant_params(
        custom_algorithm_param={"val": "not_an_array"}
    )
    self.assertNotEqual(params1, params2)

  def test_non_linear_quant_params_eq(self):
    # Create some random params.
    quant_params = qtyping.NonLinearQuantParams(
        num_bits=8,
        quantized_data=np.random.uniform(-127, 127, size=[10, 10]).astype(
            np.uint8
        ),
        data_type=qtyping.TensorDataType.FLOAT,
    )

    # Trivial equality check.
    self.assertEqual(quant_params, quant_params)
    self.assertEqual(quant_params, copy.copy(quant_params))
    self.assertEqual(quant_params, copy.deepcopy(quant_params))

    # Check whether changes in any of the parameters will fail.
    other = dataclasses.replace(quant_params, num_bits=4)
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(
        quant_params,
        quantized_data=np.random.uniform(-127, 127, size=[10, 10]).astype(
            np.uint8
        ),
    )
    self.assertNotEqual(quant_params, other)
    other = dataclasses.replace(
        quant_params, data_type=qtyping.TensorDataType.INT
    )
    self.assertNotEqual(quant_params, other)


if __name__ == "__main__":
  absltest.main()
