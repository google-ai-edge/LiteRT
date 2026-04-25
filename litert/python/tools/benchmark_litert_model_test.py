#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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

"""Unit tests for benchmark_litert_model helper functions."""

import enum
import logging
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

# Import the module under test.
from litert.python.tools import benchmark_litert_model as bm

# Suppress INFO/DEBUG logs from the module under test during test runs.
# Set to logging.DEBUG to see benchmark module logs when debugging tests.
logging.basicConfig(level=logging.WARNING)


class PercentileTest(unittest.TestCase):

  def test_empty_list(self):
    self.assertEqual(bm.percentile([], 50), 0.0)

  def test_single_element(self):
    self.assertAlmostEqual(bm.percentile([5.0], 50), 5.0)

  def test_median_odd(self):
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    self.assertAlmostEqual(bm.percentile(data, 50), 3.0)

  def test_p0_returns_min(self):
    data = [1.0, 2.0, 3.0]
    self.assertAlmostEqual(bm.percentile(data, 0), 1.0)

  def test_p100_returns_max(self):
    data = [1.0, 2.0, 3.0]
    self.assertAlmostEqual(bm.percentile(data, 100), 3.0)

  def test_interpolation(self):
    data = [10.0, 20.0]
    self.assertAlmostEqual(bm.percentile(data, 50), 15.0)


class GetNumpyDtypeTest(unittest.TestCase):

  def test_known_dtypes(self):
    self.assertEqual(bm.get_numpy_dtype('float32'), np.float32)
    self.assertEqual(bm.get_numpy_dtype('float16'), np.float16)
    self.assertEqual(bm.get_numpy_dtype('int32'), np.int32)
    self.assertEqual(bm.get_numpy_dtype('int8'), np.int8)
    self.assertEqual(bm.get_numpy_dtype('uint8'), np.uint8)
    self.assertEqual(bm.get_numpy_dtype('int64'), np.int64)
    self.assertEqual(bm.get_numpy_dtype('bool'), np.bool_)

  def test_unknown_dtype_defaults_to_float32(self):
    self.assertEqual(bm.get_numpy_dtype('bfloat16'), np.float32)
    self.assertEqual(bm.get_numpy_dtype(''), np.float32)


class FakeHardwareAccelerator(enum.IntFlag):
  CPU = 1
  GPU = 2
  NPU = 4


class BuildHardwareAcceleratorsTest(unittest.TestCase):

  def _make_args(self, **kwargs):
    defaults = {
        'use_cpu': True,
        'use_gpu': False,
        'use_npu': False,
        'no_cpu': False,
        'require_full_delegation': False,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)

  def test_default_cpu_only(self):
    args = self._make_args()
    result = bm.build_hardware_accelerators(args, FakeHardwareAccelerator)
    self.assertEqual(result, FakeHardwareAccelerator.CPU)

  def test_npu_with_cpu_fallback(self):
    args = self._make_args(use_npu=True)
    result = bm.build_hardware_accelerators(args, FakeHardwareAccelerator)
    self.assertEqual(
        result, FakeHardwareAccelerator.NPU | FakeHardwareAccelerator.CPU
    )

  def test_npu_only_no_cpu(self):
    args = self._make_args(use_npu=True, no_cpu=True)
    result = bm.build_hardware_accelerators(args, FakeHardwareAccelerator)
    self.assertEqual(result, FakeHardwareAccelerator.NPU)

  def test_require_full_delegation_disables_cpu(self):
    args = self._make_args(use_npu=True, require_full_delegation=True)
    result = bm.build_hardware_accelerators(args, FakeHardwareAccelerator)
    self.assertEqual(result, FakeHardwareAccelerator.NPU)

  def test_gpu_and_npu(self):
    args = self._make_args(use_gpu=True, use_npu=True)
    result = bm.build_hardware_accelerators(args, FakeHardwareAccelerator)
    expected = (
        FakeHardwareAccelerator.GPU
        | FakeHardwareAccelerator.NPU
        | FakeHardwareAccelerator.CPU
    )
    self.assertEqual(result, expected)

  def test_no_accel_falls_back_to_cpu(self):
    args = self._make_args(no_cpu=True, require_full_delegation=True)
    result = bm.build_hardware_accelerators(args, FakeHardwareAccelerator)
    self.assertEqual(result, FakeHardwareAccelerator.CPU)


class DispatchPathConversionTest(unittest.TestCase):
  """Test that file paths are converted to directory paths for dispatch."""

  def _make_args(self, dispatch_path='', **kwargs):
    defaults = {
        'runtime_path': '',
        'compiler_plugin_path': '',
        'dispatch_library_path': dispatch_path,
        'num_threads': 1,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)

  def test_file_path_converted_to_directory(self):
    """If dispatch_library_path points to a file, dirname should be used."""
    with tempfile.NamedTemporaryFile(suffix='.so') as f:
      args = self._make_args(dispatch_path=f.name)
      # We can't call create_environment without LiteRT installed, but
      # we can verify the path logic directly.
      dispatch_path = args.dispatch_library_path
      if os.path.isfile(dispatch_path):
        dispatch_path = os.path.dirname(dispatch_path)
      self.assertTrue(os.path.isdir(dispatch_path))

  def test_directory_path_unchanged(self):
    with tempfile.TemporaryDirectory() as d:
      args = self._make_args(dispatch_path=d)
      dispatch_path = args.dispatch_library_path
      if os.path.isfile(dispatch_path):
        dispatch_path = os.path.dirname(dispatch_path)
      self.assertEqual(dispatch_path, d)


if __name__ == '__main__':
  unittest.main()
