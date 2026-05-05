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
r"""Python benchmark tool for LiteRT models with CPU, GPU, and NPU support.

This script mirrors the C++ benchmark_litert_model tool, providing inference
benchmarking via the LiteRT Python API. It supports all hardware accelerators
(CPU, GPU, NPU) and can be used to validate pip wheel installations on target
devices without compiling C++ code.

Usage:
  # CPU only (default)
  ```
  python benchmark_litert_model.py --model=model.tflite
  ```

  # NPU with CPU fallback (Intel OpenVINO)
  ```
  python benchmark_litert_model.py --model=model.tflite --use_npu \
      --dispatch_library_path=/path/to/libLiteRtDispatch_IntelOpenvino.so
  ```

  # GPU with CPU fallback
  ```
  python benchmark_litert_model.py --model=model.tflite --use_gpu
  ```

  # NPU only, require full acceleration
  ```
  python benchmark_litert_model.py --model=model.tflite --use_npu \
      --dispatch_library_path=/path/to/dispatch.so --require_full_delegation
  ```
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time

import numpy as np

import os # import gfile

logger = logging.getLogger('benchmark_litert_model')


def _import_litert():
  """Import LiteRT modules for both litert and ai_edge_litert."""
  try:
    # pylint: disable=g-import-not-at-top
    # pytype: disable=import-error
    from litert.python.litert_wrapper.compiled_model_wrapper import (
        compiled_model as _cm,
    )
    from litert.python.litert_wrapper.compiled_model_wrapper import (
        hardware_accelerator as _ha,
    )
    from litert.python.litert_wrapper.compiled_model_wrapper import (
        options as _options,
    )
    from litert.python.litert_wrapper.environment_wrapper import (
        environment as _env,
    )
    # pytype: enable=import-error
    # pylint: enable=g-import-not-at-top
    return (
        _cm.CompiledModel,
        _ha.HardwareAccelerator,
        _env.Environment,
        _env.EnvironmentOptions,
        _options.Options,
        _options.CpuOptions,
    )
  except ImportError:
    pass
  try:
    # pylint: disable=g-import-not-at-top
    # pytype: disable=import-error
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    from ai_edge_litert.environment import Environment
    from ai_edge_litert.environment import EnvironmentOptions
    from ai_edge_litert.options import CpuOptions
    from ai_edge_litert.options import Options
    # pytype: enable=import-error
    # pylint: enable=g-import-not-at-top
    return (
        CompiledModel,
        HardwareAccelerator,
        Environment,
        EnvironmentOptions,
        Options,
        CpuOptions,
    )
  except ImportError:
    logger.error(
        'Could not import LiteRT. Install with:\n'
        '  pip install ai-edge-litert\n'
        'or run from the LiteRT source tree.'
    )
    sys.exit(1)


def parse_args():
  """Parses command line arguments for the benchmark script.

  Returns:
    An argparse.Namespace object containing the parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description='Benchmark a LiteRT model on CPU, GPU, or NPU.',
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      '--model', required=True, help='Path to the .tflite model file.'
  )
  parser.add_argument(
      '--signature', default='', help='Signature key to run (default: first).'
  )

  parser.add_argument(
      '--use_cpu',
      action='store_true',
      default=True,
      help='Enable CPU accelerator (default: True).',
  )
  parser.add_argument(
      '--no_cpu',
      action='store_true',
      help='Disable CPU fallback.',
  )
  parser.add_argument(
      '--use_gpu', action='store_true', help='Enable GPU accelerator.'
  )
  parser.add_argument(
      '--use_npu', action='store_true', help='Enable NPU accelerator.'
  )
  parser.add_argument(
      '--require_full_delegation',
      action='store_true',
      help='Fail if model is not fully accelerated.',
  )

  parser.add_argument(
      '--dispatch_library_path',
      default='',
      help=(
          'Path to vendor dispatch library (e.g., '
          'libLiteRtDispatch_IntelOpenvino.so).'
      ),
  )
  parser.add_argument(
      '--compiler_plugin_path',
      default='',
      help='Path to compiler plugin library directory.',
  )
  parser.add_argument(
      '--runtime_path',
      default='',
      help='Path to LiteRT runtime library directory.',
  )

  parser.add_argument(
      '--num_runs', type=int, default=50, help='Number of inference runs.'
  )
  parser.add_argument(
      '--warmup_runs', type=int, default=5, help='Number of warmup runs.'
  )
  parser.add_argument(
      '--num_threads',
      type=int,
      default=1,
      help='Number of CPU threads.',
  )

  parser.add_argument(
      '--result_json',
      default='',
      help='Path to save results as JSON.',
  )
  parser.add_argument(
      '--verbose', action='store_true', help='Enable verbose output.'
  )

  return parser.parse_args()


def build_hardware_accelerators(args, hardware_accelerator_cls):
  """Build the HardwareAccelerator bitmask from CLI flags."""
  accel = 0
  if args.use_npu:
    accel |= hardware_accelerator_cls.NPU
  if args.use_gpu:
    accel |= hardware_accelerator_cls.GPU
  if not args.no_cpu and not args.require_full_delegation:
    accel |= hardware_accelerator_cls.CPU
  if accel == 0:
    accel = hardware_accelerator_cls.CPU
  return hardware_accelerator_cls(accel)


def create_environment(args, environment, environment_options):
  """Create a LiteRT Environment with the requested paths."""
  kwargs = {}
  if args.runtime_path:
    kwargs['runtime_path'] = args.runtime_path

  compiler_plugin_path = args.compiler_plugin_path
  if compiler_plugin_path:
    if os.path.isfile(compiler_plugin_path):
      compiler_plugin_path = os.path.dirname(compiler_plugin_path)
    kwargs['compiler_plugin_path'] = compiler_plugin_path

  dispatch_path = args.dispatch_library_path
  if dispatch_path:
    if os.path.isfile(dispatch_path):
      dispatch_path = os.path.dirname(dispatch_path)
    kwargs['dispatch_library_path'] = dispatch_path

  return environment.create(environment_options(**kwargs))


def create_model_options(args, options, cpu_options, hardware_accel):
  """Create grouped LiteRT model options for CompiledModel creation."""
  return options(
      hardware_accelerators=hardware_accel,
      cpu_options=cpu_options(num_threads=args.num_threads),
  )


def get_numpy_dtype(dtype_str):
  """Map LiteRT dtype string to numpy dtype."""
  mapping = {
      'float32': np.float32,
      'float16': np.float16,
      'int32': np.int32,
      'int8': np.int8,
      'uint8': np.uint8,
      'int64': np.int64,
      'bool': np.bool_,
  }
  return mapping.get(dtype_str, np.float32)


def fill_random_input(tensor_buffer, details):
  """Fill a tensor buffer with random data appropriate for its dtype."""
  dtype = get_numpy_dtype(details.get('dtype', 'float32'))
  shape = details.get('shape', [1])
  if np.issubdtype(dtype, np.floating):
    data = np.random.uniform(-1.0, 1.0, shape).astype(dtype)
  elif np.issubdtype(dtype, np.integer):
    info = np.iinfo(dtype)
    low = max(info.min, -128)
    high = min(info.max, 127)
    data = np.random.randint(low, high + 1, shape, dtype=dtype)
  else:
    data = np.zeros(shape, dtype=dtype)
  tensor_buffer.write(data)


def percentile(sorted_data, p):
  """Compute the p-th percentile of sorted data."""
  if not sorted_data:
    return 0.0
  k = (len(sorted_data) - 1) * p / 100.0
  f = int(k)
  c = f + 1
  if c >= len(sorted_data):
    return sorted_data[-1]
  return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_benchmark(args):
  """Run the benchmark and return results dict."""
  (
      compiled_model_cls,
      hardware_accelerator_cls,
      environment_cls,
      environment_options_cls,
      options_cls,
      cpu_options_cls,
  ) = _import_litert()

  model_path = args.model
  if not os.path.isfile(model_path):
    logger.error('Model file not found: %s', model_path)
    sys.exit(1)

  model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
  hw_accel = build_hardware_accelerators(args, hardware_accelerator_cls)

  accel_names = []
  if hw_accel & hardware_accelerator_cls.NPU:
    accel_names.append('NPU')
  if hw_accel & hardware_accelerator_cls.GPU:
    accel_names.append('GPU')
  if hw_accel & hardware_accelerator_cls.CPU:
    accel_names.append('CPU')

  logger.info('Model:           %s', model_path)
  logger.info('Model size:      %.2f MB', model_size_mb)
  logger.info('Accelerators:    %s', ' | '.join(accel_names))
  logger.info('Num threads:     %d', args.num_threads)
  logger.info('Warmup runs:     %d', args.warmup_runs)
  logger.info('Benchmark runs:  %d', args.num_runs)
  if args.dispatch_library_path:
    logger.info('Dispatch lib:    %s', args.dispatch_library_path)
  if args.compiler_plugin_path:
    logger.info('Compiler plugin: %s', args.compiler_plugin_path)
  logger.info('Creating environment...')
  t0 = time.perf_counter()
  env = create_environment(args, environment_cls, environment_options_cls)
  env_time_ms = (time.perf_counter() - t0) * 1000
  logger.info('Environment created (%.1f ms)', env_time_ms)

  logger.info('Loading model...')
  t0 = time.perf_counter()
  model_options = create_model_options(
      args, options_cls, cpu_options_cls, hw_accel
  )
  model = compiled_model_cls.from_file(
      model_path,
      environment=env,
      options=model_options,
  )
  init_time_ms = (time.perf_counter() - t0) * 1000
  logger.info('Model loaded (%.1f ms)', init_time_ms)

  sig_list = model.get_signature_list()
  num_signatures = model.get_num_signatures()
  logger.info('Signatures:      %d', num_signatures)

  if args.signature:
    sig_key = args.signature
  else:
    sig_key = list(sig_list.keys())[0] if sig_list else 'serving_default'
  logger.info('Using signature: %s', sig_key)

  sig_idx = model.get_signature_index(sig_key)
  if sig_idx < 0:
    logger.error('Signature "%s" not found.', sig_key)
    sys.exit(1)

  is_fully_accel = model.is_fully_accelerated()
  logger.info('Fully accelerated: %s', is_fully_accel)
  if args.require_full_delegation and not is_fully_accel:
    logger.error(
        'Model is not fully accelerated but --require_full_delegation was set.'
    )
    sys.exit(1)

  input_details = model.get_input_tensor_details(sig_key)
  output_details = model.get_output_tensor_details(sig_key)

  if args.verbose:
    logger.debug('Input tensors:')
    for name, detail in input_details.items():
      logger.debug(
          '  %s: dtype=%s, shape=%s',
          name,
          detail.get('dtype'),
          detail.get('shape'),
      )
    logger.debug('Output tensors:')
    for name, detail in output_details.items():
      logger.debug(
          '  %s: dtype=%s, shape=%s',
          name,
          detail.get('dtype'),
          detail.get('shape'),
      )

  input_names = list(input_details.keys())
  output_names = list(output_details.keys())

  input_buffers = {
      name: model.create_input_buffer_by_name(sig_key, name)
      for name in input_names
  }
  output_buffers = {
      name: model.create_output_buffer_by_name(sig_key, name)
      for name in output_names
  }

  for name, buf in input_buffers.items():
    fill_random_input(buf, input_details[name])

  total_input_bytes = 0
  for name, detail in input_details.items():
    shape = detail.get('shape', [1])
    dtype = get_numpy_dtype(detail.get('dtype', 'float32'))
    total_input_bytes += int(np.prod(shape)) * np.dtype(dtype).itemsize
  total_input_mb = total_input_bytes / (1024 * 1024)

  logger.info('Running %d warmup iterations...', args.warmup_runs)
  warmup_times = []
  for i in range(args.warmup_runs):
    t0 = time.perf_counter()
    model.run_by_name(sig_key, input_buffers, output_buffers)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    warmup_times.append(elapsed_ms)
    if args.verbose:
      logger.debug(
          '  Warmup %d/%d: %.2f ms',
          i + 1,
          args.warmup_runs,
          elapsed_ms,
      )

  logger.info('Running %d benchmark iterations...', args.num_runs)
  inference_times = []
  for i in range(args.num_runs):
    t0 = time.perf_counter()
    model.run_by_name(sig_key, input_buffers, output_buffers)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    inference_times.append(elapsed_ms)
    if args.verbose and (i + 1) % 10 == 0:
      logger.debug('  Run %d/%d: %.2f ms', i + 1, args.num_runs, elapsed_ms)

  sorted_times = sorted(inference_times)
  avg_ms = statistics.mean(inference_times)
  min_ms = min(inference_times)
  max_ms = max(inference_times)
  std_ms = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
  median_ms = statistics.median(inference_times)
  p5_ms = percentile(sorted_times, 5)
  p95_ms = percentile(sorted_times, 95)

  warmup_avg_ms = statistics.mean(warmup_times) if warmup_times else 0
  warmup_first_ms = warmup_times[0] if warmup_times else 0

  throughput_mb_s = (total_input_mb / (avg_ms / 1000)) if avg_ms > 0 else 0

  # Printed to stdout (not logger) so results are always visible and parseable.

  print()
  print('=' * 40)
  print(' BENCHMARK RESULTS')
  print('=' * 40)
  print(f'Model initialization:  {init_time_ms:.2f} ms')
  print(f'Warmup (first):        {warmup_first_ms:.2f} ms')
  warmup_n = len(warmup_times)
  print(f'Warmup (avg):          {warmup_avg_ms:.2f} ms ({warmup_n} runs)')
  print(f'Inference (avg):       {avg_ms:.2f} ms ({len(inference_times)} runs)')
  print(f'Inference (min):       {min_ms:.2f} ms')
  print(f'Inference (max):       {max_ms:.2f} ms')
  print(f'Inference (median):    {median_ms:.2f} ms')
  print(f'Inference (std):       {std_ms:.2f} ms')
  print(f'Inference (p5):        {p5_ms:.2f} ms')
  print(f'Inference (p95):       {p95_ms:.2f} ms')
  if throughput_mb_s > 0:
    print(f'Throughput:            {throughput_mb_s:.2f} MB/s')
  print(f'Fully accelerated:     {is_fully_accel}')
  print('=' * 40)

  results = {
      'model': model_path,
      'model_size_mb': round(model_size_mb, 2),
      'accelerators': accel_names,
      'signature': sig_key,
      'fully_accelerated': is_fully_accel,
      'num_threads': args.num_threads,
      'latency': {
          'init_ms': round(init_time_ms, 2),
          'warmup_first_ms': round(warmup_first_ms, 2),
          'warmup_avg_ms': round(warmup_avg_ms, 2),
          'num_warmup_runs': len(warmup_times),
          'avg_ms': round(avg_ms, 2),
          'min_ms': round(min_ms, 2),
          'max_ms': round(max_ms, 2),
          'median_ms': round(median_ms, 2),
          'std_ms': round(std_ms, 2),
          'p5_ms': round(p5_ms, 2),
          'p95_ms': round(p95_ms, 2),
          'num_runs': len(inference_times),
      },
      'throughput_mb_s': round(throughput_mb_s, 2),
  }

  if args.result_json:
    try:
      with open(args.result_json, 'w') as f:
        json.dump(results, f, indent=2)
      logger.info('Results saved to: %s', args.result_json)
    except OSError as e:
      logger.error('Failed to write results to %s: %s', args.result_json, e)

  return results


def main():
  args = parse_args()
  logging.basicConfig(
      level=logging.DEBUG if args.verbose else logging.INFO,
      format='%(asctime)s %(levelname)s %(name)s: %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
  )
  run_benchmark(args)


if __name__ == '__main__':
  main()
