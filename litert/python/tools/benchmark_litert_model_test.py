# Copyright 2026 Google LLC.
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
import sys
import tempfile
import types
import unittest
from unittest import mock

import ml_dtypes
import numpy as np

from litert.python.litert_wrapper.compiled_model_wrapper import (
    cpu_options,
)
from litert.python.litert_wrapper.compiled_model_wrapper import (
    gpu_options,
)
from litert.python.litert_wrapper.compiled_model_wrapper import (
    intel_openvino_options,
)
from litert.python.litert_wrapper.compiled_model_wrapper import (
    options as options_lib,
)
from litert.python.litert_wrapper.compiled_model_wrapper import (
    qualcomm_options,
)

# Import the module under test.
from litert.python.tools import benchmark_litert_model as bm

QualcommOptions = options_lib.QualcommOptions

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
    self.assertEqual(bm.get_numpy_dtype('bfloat16'), ml_dtypes.bfloat16)
    self.assertEqual(
        bm.get_numpy_dtype('float8_e4m3fn'), ml_dtypes.float8_e4m3fn
    )
    self.assertEqual(bm.get_numpy_dtype('float8_e5m2'), ml_dtypes.float8_e5m2)
    self.assertEqual(bm.get_numpy_dtype('float32'), np.float32)
    self.assertEqual(bm.get_numpy_dtype('float16'), np.float16)
    self.assertEqual(bm.get_numpy_dtype('int32'), np.int32)
    self.assertEqual(bm.get_numpy_dtype('int8'), np.int8)
    self.assertEqual(bm.get_numpy_dtype('uint8'), np.uint8)
    self.assertEqual(bm.get_numpy_dtype('int64'), np.int64)
    self.assertEqual(bm.get_numpy_dtype('bool'), np.bool_)

  def test_unknown_dtype_defaults_to_float32(self):
    self.assertEqual(bm.get_numpy_dtype(''), np.float32)


class FakeHardwareAccelerator(enum.IntFlag):
  CPU = 1
  GPU = 2
  NPU = 4


class FakeEnvironmentOptions:

  def __init__(
      self,
      runtime_path='',
      compiler_plugin_path='',
      dispatch_library_path='',
      compiler_cache_dir='',
  ):
    self.runtime_path = runtime_path
    self.compiler_plugin_path = compiler_plugin_path
    self.dispatch_library_path = dispatch_library_path
    self.compiler_cache_dir = compiler_cache_dir


class FakeEnvironment:

  created_options = None

  @classmethod
  def create(cls, options):
    cls.created_options = options
    return options


class FakeCpuOptions:

  def __init__(self, num_threads=0):
    self.num_threads = num_threads


class FakeOptions:

  def __init__(self, hardware_accelerators, cpu_options):
    self.hardware_accelerators = hardware_accelerators
    self.cpu_options = cpu_options


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


class EnvironmentCreationTest(unittest.TestCase):

  def _make_args(self, **kwargs):
    defaults = {
        'runtime_path': '',
        'compiler_plugin_path': '',
        'dispatch_library_path': '',
        'num_threads': 3,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)

  def test_create_environment_uses_grouped_environment_options(self):
    with tempfile.TemporaryDirectory() as dispatch_dir:
      args = self._make_args(dispatch_library_path=dispatch_dir)

      env = bm.create_environment(args, FakeEnvironment, FakeEnvironmentOptions)

      self.assertIs(env, FakeEnvironment.created_options)
      self.assertEqual(env.dispatch_library_path, dispatch_dir)
      self.assertEqual(env.compiler_plugin_path, '')

  def test_create_environment_converts_file_paths_to_directories(self):
    with tempfile.NamedTemporaryFile(suffix='.so') as dispatch_file:
      args = self._make_args(dispatch_library_path=dispatch_file.name)

      env = bm.create_environment(args, FakeEnvironment, FakeEnvironmentOptions)

      self.assertEqual(
          env.dispatch_library_path, os.path.dirname(dispatch_file.name)
      )


class ModelOptionsCreationTest(unittest.TestCase):

  def test_create_model_options_groups_cpu_options(self):
    args = types.SimpleNamespace(num_threads=4)
    hw_accel = FakeHardwareAccelerator.GPU | FakeHardwareAccelerator.CPU

    options = bm.create_model_options(
        args, FakeOptions, FakeCpuOptions, hw_accel
    )

    self.assertEqual(options.hardware_accelerators, hw_accel)
    self.assertIsInstance(options.cpu_options, FakeCpuOptions)
    self.assertEqual(options.cpu_options.num_threads, 4)

  def test_create_model_options_populates_qualcomm_options(self):
    args = types.SimpleNamespace(
        num_threads=2,
        qualcomm_enable_just_in_time=True,
        qualcomm_htp_performance_mode='burst',
        qualcomm_log_level='off',
        qualcomm_qnn_lib_dir='/opt/qairt/lib/aarch64-oe-linux-gcc11.2',
        qualcomm_dsp_skel_dir='/opt/qairt/lib/hexagon-v75/unsigned',
    )
    hw_accel = FakeHardwareAccelerator.NPU | FakeHardwareAccelerator.CPU

    options = bm.create_model_options(
        args, options_lib.Options, options_lib.CpuOptions, hw_accel
    )

    self.assertTrue(options.qualcomm_options.enable_just_in_time)
    self.assertEqual(
        options.qualcomm_options.htp_performance_mode,
        options_lib.QualcommOptions.HTP_PERFORMANCE_MODE.BURST,
    )
    self.assertEqual(
        options.qualcomm_options.log_level,
        options_lib.QualcommOptions.LOG_LEVEL.OFF,
    )
    self.assertEqual(
        options.qualcomm_options.qnn_lib_dir,
        '/opt/qairt/lib/aarch64-oe-linux-gcc11.2',
    )
    self.assertEqual(
        options.qualcomm_options.dsp_skel_dir,
        '/opt/qairt/lib/hexagon-v75/unsigned',
    )


class VendorOptionsFlatteningTest(unittest.TestCase):

  def test_options_facade_reexports_split_modules(self):
    self.assertIs(options_lib.CpuOptions, cpu_options.CpuOptions)
    self.assertIs(options_lib.GpuOptions, gpu_options.GpuOptions)
    self.assertIs(
        options_lib.QualcommOptions,
        qualcomm_options.QualcommOptions,
    )
    self.assertIs(
        options_lib.IntelOpenVinoOptions,
        intel_openvino_options.IntelOpenVinoOptions,
    )

  def test_qualcomm_options_flatten_to_pybind_kwargs(self):
    options = options_lib.Options.create()
    qualcomm_options = options.qualcomm_options
    qualcomm_options.log_level = options_lib.QualcommOptions.LOG_LEVEL.ERROR
    qualcomm_options.htp_performance_mode = (
        options_lib.QualcommOptions.HTP_PERFORMANCE_MODE.BURST
    )
    qualcomm_options.use_int64_bias_as_int32 = False
    qualcomm_options.enable_weight_sharing = True
    qualcomm_options.dump_tensor_ids = [1, 3, 5]
    qualcomm_options.backend = options_lib.QualcommOptions.BACKEND.HTP
    qualcomm_options.graph_io_tensor_mem_type = (
        options_lib.QualcommOptions.GRAPH_IO_TENSOR_MEM_TYPE.RAW
    )
    qualcomm_options.enable_just_in_time = True
    qualcomm_options.qnn_lib_dir = '/opt/qairt/lib/aarch64-oe-linux-gcc11.2'
    qualcomm_options.dsp_skel_dir = '/opt/qairt/lib/hexagon-v75/unsigned'

    kwargs = options._as_flat_kwargs()

    self.assertEqual(kwargs['qualcomm_log_level'], 1)
    self.assertEqual(kwargs['qualcomm_htp_performance_mode'], 2)
    self.assertEqual(kwargs['qualcomm_use_int64_bias_as_int32'], 0)
    self.assertEqual(kwargs['qualcomm_enable_weight_sharing'], 1)
    self.assertEqual(kwargs['qualcomm_enable_just_in_time'], 1)
    self.assertTrue(kwargs['qualcomm_has_dump_tensor_ids'])
    self.assertEqual(kwargs['qualcomm_dump_tensor_ids'], [1, 3, 5])
    self.assertEqual(kwargs['qualcomm_backend'], 2)
    self.assertEqual(kwargs['qualcomm_graph_io_tensor_mem_type'], 0)
    self.assertEqual(
        kwargs['qualcomm_qnn_lib_dir'],
        '/opt/qairt/lib/aarch64-oe-linux-gcc11.2',
    )
    self.assertEqual(
        kwargs['qualcomm_dsp_skel_dir'],
        '/opt/qairt/lib/hexagon-v75/unsigned',
    )

  def test_intel_openvino_options_flatten_to_pybind_kwargs(self):
    options = options_lib.Options.create()
    intel_options = options.intel_openvino_options
    intel_options.graph_backend = (
        options_lib.IntelOpenVinoOptions.GRAPH_BACKEND.NPU
    )
    intel_options.performance_mode = (
        options_lib.IntelOpenVinoOptions.PERFORMANCE_MODE.THROUGHPUT
    )
    intel_options.configs_map['CACHE_DIR'] = '/tmp/openvino-cache'
    intel_options.enable_weight_sharing = False

    kwargs = options._as_flat_kwargs()

    self.assertEqual(kwargs['intel_openvino_graph_backend'], 2)
    self.assertEqual(kwargs['intel_openvino_performance_mode'], 1)
    self.assertEqual(
        kwargs['intel_openvino_configs_map'],
        {'CACHE_DIR': '/tmp/openvino-cache'},
    )
    self.assertEqual(kwargs['intel_openvino_enable_weight_sharing'], 0)

  def test_intel_openvino_weight_sharing_defaults_to_unset(self):
    options = options_lib.Options.create()
    # Touch the OpenVINO options so they are flattened into the kwargs.
    options.intel_openvino_options.performance_mode = (
        options_lib.IntelOpenVinoOptions.PERFORMANCE_MODE.LATENCY
    )

    kwargs = options._as_flat_kwargs()

    # -1 means "not set": the plugin keeps its own default (sharing enabled).
    self.assertEqual(kwargs['intel_openvino_enable_weight_sharing'], -1)


def _parse(*flags):
  with mock.patch.object(
      sys, 'argv', ['benchmark_litert_model', '--model=/m.tflite', *flags]
  ):
    return bm.parse_args()


class NewFlagParsingTest(unittest.TestCase):

  def test_defaults(self):
    args = _parse()
    self.assertEqual(args.compiler_cache_dir, '')
    self.assertFalse(args.qualcomm_enable_just_in_time)
    self.assertEqual(args.qualcomm_htp_performance_mode, '')
    self.assertEqual(args.qualcomm_log_level, '')
    self.assertEqual(args.qualcomm_qnn_lib_dir, '')
    self.assertEqual(args.qualcomm_dsp_skel_dir, '')

  def test_all_flags_set(self):
    args = _parse(
        '--compiler_cache_dir=/tmp/cache',
        '--qualcomm_enable_just_in_time',
        '--qualcomm_htp_performance_mode=burst',
        '--qualcomm_log_level=verbose',
        '--qualcomm_qnn_lib_dir=/qnn/lib',
        '--qualcomm_dsp_skel_dir=/qnn/skel',
    )
    self.assertEqual(args.compiler_cache_dir, '/tmp/cache')
    self.assertTrue(args.qualcomm_enable_just_in_time)
    self.assertEqual(args.qualcomm_htp_performance_mode, 'burst')
    self.assertEqual(args.qualcomm_log_level, 'verbose')
    self.assertEqual(args.qualcomm_qnn_lib_dir, '/qnn/lib')
    self.assertEqual(args.qualcomm_dsp_skel_dir, '/qnn/skel')

  def test_parsed_args_round_trip_into_model_options(self):
    args = _parse(
        '--num_threads=2',
        '--qualcomm_enable_just_in_time',
        '--qualcomm_htp_performance_mode=high_performance',
    )
    opts = bm.create_model_options(
        args,
        options_lib.Options,
        options_lib.CpuOptions,
        options_lib.HardwareAccelerator.NPU,
    )
    kwargs = opts._as_flat_kwargs()
    self.assertEqual(kwargs['qualcomm_enable_just_in_time'], 1)
    self.assertEqual(
        kwargs['qualcomm_htp_performance_mode'],
        int(QualcommOptions.HTP_PERFORMANCE_MODE.HIGH_PERFORMANCE),
    )


class CompilerCacheDirForwardingTest(unittest.TestCase):

  def _args(self, **kwargs):
    defaults = {
        'runtime_path': '',
        'compiler_plugin_path': '',
        'dispatch_library_path': '',
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)

  def test_compiler_cache_dir_forwarded(self):
    env_opts = bm.create_environment(
        self._args(compiler_cache_dir='/tmp/cache'),
        FakeEnvironment,
        FakeEnvironmentOptions,
    )
    self.assertEqual(env_opts.compiler_cache_dir, '/tmp/cache')

  def test_empty_compiler_cache_dir_not_forwarded(self):
    with mock.patch.object(
        FakeEnvironmentOptions, '__init__', autospec=True, return_value=None
    ) as mock_init:
      bm.create_environment(
          self._args(compiler_cache_dir=''),
          FakeEnvironment,
          FakeEnvironmentOptions,
      )
    self.assertNotIn('compiler_cache_dir', mock_init.call_args.kwargs)

  def test_missing_attribute_is_tolerated(self):
    # Callers that build a Namespace without the new flag must keep working.
    env_opts = bm.create_environment(
        self._args(), FakeEnvironment, FakeEnvironmentOptions
    )
    self.assertEqual(env_opts.compiler_cache_dir, '')


class ParseEnumFlagTest(unittest.TestCase):

  def test_valid_values_are_case_insensitive_and_stripped(self):
    for value in ('burst', 'BURST', 'Burst', '  burst\t'):
      with self.subTest(value=value):
        self.assertIs(
            bm._parse_enum_flag(
                QualcommOptions.HTP_PERFORMANCE_MODE, value, '--flag'
            ),
            QualcommOptions.HTP_PERFORMANCE_MODE.BURST,
        )

  def test_invalid_value_lists_valid_values(self):
    with self.assertRaises(ValueError) as cm:
      bm._parse_enum_flag(QualcommOptions.LOG_LEVEL, 'trace', '--my_flag')
    message = str(cm.exception)
    self.assertIn("Unsupported --my_flag: 'trace'", message)
    self.assertIn(
        'Valid values: off, error, warn, info, verbose, debug.', message
    )

  def test_unknown_names_are_rejected(self):
    for value in ('name', 'value', 'mro', '_member_map_', ''):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          bm._parse_enum_flag(QualcommOptions.LOG_LEVEL, value, '--flag')

  def test_non_member_class_attributes_are_rejected(self):
    # getattr() finds the uppercase non-member attribute; it must still be
    # rejected because it is not an enum member.

    class _Mode(enum.Enum):
      FAST = 1

      @classmethod
      def LOOKUP(cls):  # pylint: disable=invalid-name
        return cls.FAST

    self.assertIs(bm._parse_enum_flag(_Mode, 'fast', '--flag'), _Mode.FAST)
    with self.assertRaisesRegex(ValueError, 'Valid values: fast.'):
      bm._parse_enum_flag(_Mode, 'lookup', '--flag')

  def test_numeric_value_is_rejected(self):
    with self.assertRaises(ValueError):
      bm._parse_enum_flag(QualcommOptions.LOG_LEVEL, '3', '--flag')


class QualcommModelOptionsTest(unittest.TestCase):

  def _args(self, **kwargs):
    defaults = {
        'num_threads': 1,
        'qualcomm_enable_just_in_time': False,
        'qualcomm_htp_performance_mode': '',
        'qualcomm_log_level': '',
        'qualcomm_qnn_lib_dir': '',
        'qualcomm_dsp_skel_dir': '',
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)

  def _create(self, args):
    return bm.create_model_options(
        args,
        options_lib.Options,
        options_lib.CpuOptions,
        options_lib.HardwareAccelerator.CPU,
    )

  def test_no_qualcomm_flags_does_not_materialize_qualcomm_options(self):
    opts = self._create(self._args())
    self.assertIsNone(opts._qualcomm_options)
    kwargs = opts._as_flat_kwargs()
    self.assertEqual(kwargs['qualcomm_enable_just_in_time'], -1)
    self.assertEqual(kwargs['qualcomm_qnn_lib_dir'], '')

  def test_jit_false_leaves_tristate_unset(self):
    # store_true default False must map to "unset" (-1), not "disabled" (0).
    opts = self._create(self._args(qualcomm_qnn_lib_dir='/qnn/lib'))
    self.assertIsNone(opts.qualcomm_options.enable_just_in_time)
    self.assertEqual(
        opts._as_flat_kwargs()['qualcomm_enable_just_in_time'], -1
    )

  def test_each_flag_individually_materializes(self):
    for flag, value in (
        ('qualcomm_enable_just_in_time', True),
        ('qualcomm_htp_performance_mode', 'burst'),
        ('qualcomm_log_level', 'off'),
        ('qualcomm_qnn_lib_dir', '/qnn/lib'),
        ('qualcomm_dsp_skel_dir', '/qnn/skel'),
    ):
      with self.subTest(flag=flag):
        opts = self._create(self._args(**{flag: value}))
        self.assertIsNotNone(opts._qualcomm_options)

  def test_htp_performance_mode_all_members_case_insensitive(self):
    for member in QualcommOptions.HTP_PERFORMANCE_MODE:
      for spelling in (member.name.lower(), f'  {member.name.title()} '):
        with self.subTest(spelling=spelling):
          opts = self._create(
              self._args(qualcomm_htp_performance_mode=spelling)
          )
          self.assertEqual(opts.qualcomm_options.htp_performance_mode, member)

  def test_log_level_all_members_case_insensitive(self):
    for member in QualcommOptions.LOG_LEVEL:
      with self.subTest(member=member):
        opts = self._create(self._args(qualcomm_log_level=member.name.lower()))
        self.assertEqual(opts.qualcomm_options.log_level, member)

  def test_invalid_htp_performance_mode_raises(self):
    for bad in ('turbo', '2', 'burst_mode'):
      with self.subTest(bad=bad):
        with self.assertRaisesRegex(
            ValueError, 'Unsupported --qualcomm_htp_performance_mode'
        ):
          self._create(self._args(qualcomm_htp_performance_mode=bad))

  def test_invalid_log_level_raises(self):
    for bad in ('warning', 'trace', '3'):
      with self.subTest(bad=bad):
        with self.assertRaisesRegex(
            ValueError, 'Unsupported --qualcomm_log_level'
        ):
          self._create(self._args(qualcomm_log_level=bad))

  def test_whitespace_only_enum_value_raises(self):
    # '  ' is truthy, so it is parsed and must be rejected, not silently
    # ignored.
    with self.assertRaises(ValueError):
      self._create(self._args(qualcomm_log_level='  '))

  def test_dirs_are_passed_through_verbatim(self):
    opts = self._create(
        self._args(
            qualcomm_qnn_lib_dir='/qnn/lib',
            qualcomm_dsp_skel_dir='/qnn/skel',
        )
    )
    kwargs = opts._as_flat_kwargs()
    self.assertEqual(kwargs['qualcomm_qnn_lib_dir'], '/qnn/lib')
    self.assertEqual(kwargs['qualcomm_dsp_skel_dir'], '/qnn/skel')


class RunBenchmarkTest(unittest.TestCase):

  def test_invalid_enum_flag_fails_before_environment_creation(self):
    mock_env_cls = mock.Mock()
    mock_model_cls = mock.Mock()
    with tempfile.NamedTemporaryFile(suffix='.tflite') as model_file:
      args = _parse(
          f'--model={model_file.name}', '--qualcomm_log_level=bogus'
      )
      with mock.patch.object(
          bm,
          '_import_litert',
          return_value=(
              mock_model_cls,
              options_lib.HardwareAccelerator,
              mock_env_cls,
              FakeEnvironmentOptions,
              options_lib.Options,
              options_lib.CpuOptions,
          ),
      ):
        with self.assertRaisesRegex(
            ValueError, 'Unsupported --qualcomm_log_level'
        ):
          bm.run_benchmark(args)
    mock_env_cls.create.assert_not_called()
    mock_model_cls.from_file.assert_not_called()


if __name__ == '__main__':
  unittest.main()
