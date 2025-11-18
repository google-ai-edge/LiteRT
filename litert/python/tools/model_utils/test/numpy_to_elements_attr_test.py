# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for conversion between NumPy arrays and ElementsAttr/DenseElementsAttr."""

import ml_dtypes
import numpy as np
from absl.testing import absltest as googletest
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import mlir


class NumpyToElementsAttrTest(testing.ModelUtilsTestCase):

  def _test_roundtrip_spalt(self, dtype, shape=(3, 3, 3)):
    data = np.array([1 for _ in range(np.prod(shape))])
    return self._test_roundtrip(dtype, shape, data)

  def _test_roundtrip(self, dtype, shape=(3, 3, 3), data=None):
    dtype = np.dtype(dtype)
    numel = np.prod(shape)

    if data is None:
      data = np.arange(numel)
    data = data.astype(dtype).reshape(shape)

    attr = mlir.DenseElementsAttr(data)
    recon_data = attr.numpy()
    self.assertTrue(np.array_equal(data, recon_data))

  def test_bool(self):
    self._test_roundtrip(np.bool_)

  def test_bool_splat(self):
    self._test_roundtrip_spalt(np.bool_)

  def test_float16(self):
    self._test_roundtrip(np.float16)

  def test_f32(self):
    self._test_roundtrip(np.float32)

  def test_f32_splat(self):
    self._test_roundtrip_spalt(np.float32)

  def test_f64(self):
    self._test_roundtrip(np.float64)

  def test_i32(self):
    self._test_roundtrip(np.int32)

  def test_i32_splat(self):
    self._test_roundtrip_spalt(np.int32)

  def test_i64(self):
    self._test_roundtrip(np.int64)

  def test_i8(self):
    self._test_roundtrip(np.int8)

  def test_i4(self):
    self._test_roundtrip(ml_dtypes.int4)

  def test_i2(self):
    self._test_roundtrip(ml_dtypes.int2)


if __name__ == "__main__":
  googletest.main()
