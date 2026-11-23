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

"""End-to-end histogram calibration and A8W8 static quantization example."""

from collections.abc import Iterable, Sequence
import dataclasses
import math
import pathlib
import time
from typing import Any
import urllib.error
import urllib.request

from absl import app
from absl import flags
import numpy as np
from PIL import Image

import os
import io
from litert_quantizer import calibrator
from litert_quantizer import quantizer
from litert_quantizer import recipe
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils

RangeMethod = naive_min_max_quantize.ActivationRangeMethod
CalibrationMode = calibrator.CalibrationMode

_CALIBRATION_MODE_NAMES = (
    'CALIBRATION_PROFILER_BASED',
    'CALIBRATION_PRESERVE_ALL_TENSORS',
)

LOCAL_SAMPLE_DIR = test_utils.get_path_to_datafile(
    '../../../litert/samples/async_segmentation'
)
LOCAL_MODEL_PATH = str(
    pathlib.Path(LOCAL_SAMPLE_DIR)
    / 'models'
    / 'selfie_multiclass_256x256.tflite'
)
LOCAL_IMAGE_PATH = str(
    pathlib.Path(LOCAL_SAMPLE_DIR) / 'test_images' / 'image.jpeg'
)

GITHUB_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/image_segmenter/'
    'selfie_multiclass_256x256/float32/1/'
    'selfie_multiclass_256x256.tflite'
)
GITHUB_IMAGE_URL = (
    'https://storage.googleapis.com/mediapipe-assets/portrait.jpg'
)
_DOWNLOAD_TIMEOUT_SEC = 60

_DEFAULT_CONFIGS: tuple[tuple[str, RangeMethod, CalibrationMode], ...] = (
    (
        'selfie_a8w8_minmax_profiler',
        RangeMethod.MIN_MAX,
        CalibrationMode.CALIBRATION_PROFILER_BASED,
    ),
    (
        'selfie_a8w8_hist_pct_profiler',
        RangeMethod.HISTOGRAM_PERCENTILE,
        CalibrationMode.CALIBRATION_PROFILER_BASED,
    ),
    (
        'selfie_a8w8_hist_mse_profiler',
        RangeMethod.HISTOGRAM_MSE,
        CalibrationMode.CALIBRATION_PROFILER_BASED,
    ),
    (
        'selfie_a8w8_hist_mse_preserve',
        RangeMethod.HISTOGRAM_MSE,
        CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
    ),
)

_FLOAT_MODEL_PATH = flags.DEFINE_string(
    'float_model_path',
    '',
    'Path to the floating-point TFLite segmentation model. Uses the default'
    ' sample model when empty.',
)
_IMAGE_PATHS = flags.DEFINE_list(
    'image_paths',
    [],
    'Comma-separated calibration and evaluation image paths. Uses the default'
    ' sample image when empty.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp',
    'Directory to save downloaded sample assets, quantized models, and'
    ' recipes.',
)
_NUM_CALIB_SAMPLES = flags.DEFINE_integer(
    'num_calib_samples',
    32,
    'Number of augmented image samples to use for calibration.',
)
_RANGE_METHOD = flags.DEFINE_enum(
    'range_method',
    None,
    [m.value for m in RangeMethod],
    'Optional activation range method to run. If None, runs all four default'
    ' configurations.',
)
_CALIBRATION_METHOD = flags.DEFINE_enum(
    'calibration_method',
    'CALIBRATION_PROFILER_BASED',
    list(_CALIBRATION_MODE_NAMES),
    'Calibration method used when --range_method is specified.',
)


@dataclasses.dataclass(frozen=True)
class SegmentationQualityReport:
  """Numerical and task-level quality metrics for a quantized model.

  Attributes:
    model_name: Quantized model configuration name.
    calibration_mode: Quantizer calibration execution mode.
    calibration_time_sec: Elapsed seconds spent in `Quantizer.calibrate()`.
    model_size_mb: Quantized model size in MiB.
    output_mse: Mean squared error between float and quantized output logits.
    output_sqnr_db: Signal-to-quantization-noise ratio in dB.
    cosine_similarity: Cosine similarity between flattened output logits.
    kl_divergence: Mean per-pixel `D_KL(P_float || P_quant)` over softmax
      probabilities.
    pixel_top1_agreement: Fraction of pixels matching the float argmax class.
    mean_iou: Mean intersection-over-union against the float argmax mask.
  """

  model_name: str
  calibration_mode: CalibrationMode
  calibration_time_sec: float
  model_size_mb: float
  output_mse: float
  output_sqnr_db: float
  cosine_similarity: float
  kl_divergence: float
  pixel_top1_agreement: float
  mean_iou: float


def get_model_input_spec(model_path: str) -> tuple[str, tuple[int, ...]]:
  """Returns `(input_name, input_shape)` for the default signature."""
  interp = tfl_interpreter_utils.create_tfl_interpreter(
      model_path, allocate_tensors=False
  )
  sig_defs = interp.get_signature_list()
  input_details = interp.get_input_details()
  if sig_defs and tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY in sig_defs:
    sig_inputs = sig_defs[tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY]['inputs']
    input_name = sig_inputs[0]
  else:
    input_name = input_details[0]['name'].split(':')[0]
  shape = tuple(int(d) for d in input_details[0]['shape'])
  return input_name, shape


def load_and_preprocess_images(
    image_paths: Sequence[str],
    target_shape: tuple[int, ...],
    num_augmented_samples: int = 32,
    seed: int = 42,
) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
  """Loads images and generates `(calibration_arrays, eval_arrays)`.

  The evaluation arrays are the resized images. The calibration arrays are the
  same images plus random crops, flips, and contrast/brightness changes of them,
  up to `num_augmented_samples` arrays.

  Args:
    image_paths: Paths of the images to load.
    target_shape: The model input shape, `(1, height, width, channels)`.
    num_augmented_samples: Number of calibration arrays to return.
    seed: Seed for the random augmentations.

  Returns:
    A tuple of the calibration arrays and the evaluation arrays, each of shape
    `target_shape` with values in [0, 1].

  Raises:
    ValueError: If `image_paths` is empty.
  """
  if not image_paths:
    raise ValueError('No image paths provided.')
  _, height, width, channels = target_shape
  base_images: list[np.ndarray] = []
  eval_arrays: list[np.ndarray] = []
  mode = 'L' if channels == 1 else 'RGB'

  for img_path in image_paths:
    with open(img_path, 'rb') as f:
      pil_img = Image.open(f).convert(mode)
      base_images.append(np.asarray(pil_img, dtype=np.float32) / 255.0)
      resized = pil_img.resize((width, height), Image.Resampling.BILINEAR)
      arr = np.asarray(resized, dtype=np.float32) / 255.0
      eval_arrays.append(arr.reshape((1, height, width, channels)))

  calib_arrays: list[np.ndarray] = list(eval_arrays)
  rng = np.random.default_rng(seed)

  while len(calib_arrays) < num_augmented_samples:
    src = base_images[len(calib_arrays) % len(base_images)]
    src_h, src_w = src.shape[0], src.shape[1]

    crop_scale = float(rng.uniform(0.70, 1.0))
    crop_h = max(1, int(src_h * crop_scale))
    crop_w = max(1, int(src_w * crop_scale))
    top = int(rng.integers(0, max(1, src_h - crop_h + 1)))
    left = int(rng.integers(0, max(1, src_w - crop_w + 1)))
    cropped = src[top : top + crop_h, left : left + crop_w]

    uint8_crop = np.clip(cropped * 255.0, 0.0, 255.0).astype(np.uint8)
    pil_crop = Image.fromarray(uint8_crop, mode=mode).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    aug = np.asarray(pil_crop, dtype=np.float32) / 255.0
    if rng.random() < 0.5:
      aug = np.flip(aug, axis=1)

    contrast = float(rng.uniform(0.85, 1.15))
    brightness = float(rng.uniform(-0.08, 0.08))
    mean_val = float(np.mean(aug))
    aug = np.clip((aug - mean_val) * contrast + mean_val + brightness, 0.0, 1.0)
    calib_arrays.append(
        aug.astype(np.float32).reshape((1, height, width, channels))
    )

  return calib_arrays[:num_augmented_samples], eval_arrays


def build_calibration_dataset(
    input_name: str, samples: Sequence[np.ndarray]
) -> dict[str, Iterable[dict[str, Any]]]:
  """Formats input arrays into an AEQ signature-keyed calibration dataset."""
  dataset: dict[str, Iterable[dict[str, Any]]] = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: [
          {input_name: sample} for sample in samples
      ]
  }
  return dataset


def run_model_on_samples(
    model_source: str | bytes, samples: Sequence[np.ndarray]
) -> Sequence[np.ndarray]:
  """Runs TFLite inference on `samples` and returns dequantized float outputs."""
  interp = tfl_interpreter_utils.create_tfl_interpreter(model_source)
  output_details = interp.get_output_details()[0]

  outputs: list[np.ndarray] = []
  for sample in samples:
    tfl_interpreter_utils.invoke_interpreter_once(interp, [sample])
    out = tfl_interpreter_utils.get_tensor_data(
        interp, output_details, dequantize=True
    )
    outputs.append(np.asarray(out, dtype=np.float32).copy())
  return outputs


def _softmax(logits: np.ndarray, eps: float = 1e-7) -> np.ndarray:
  """Returns numerically stable class probabilities clipped below at `eps`."""
  exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
  return np.clip(exp / np.sum(exp, axis=-1, keepdims=True), eps, 1.0)


def evaluate_segmentation_quality(
    float_outputs: Sequence[np.ndarray],
    quant_model_bytes: bytes,
    eval_samples: Sequence[np.ndarray],
    model_name: str,
    calibration_mode: CalibrationMode,
    calibration_time_sec: float,
) -> SegmentationQualityReport:
  """Evaluates numerical and segmentation quality of a quantized model."""
  quant_outputs = run_model_on_samples(quant_model_bytes, eval_samples)

  ref_all = np.concatenate(float_outputs, axis=0).astype(np.float64)
  q_all = np.concatenate(quant_outputs, axis=0).astype(np.float64)
  diff = q_all - ref_all

  mse = float(np.mean(np.square(diff)))
  signal_power = float(np.mean(np.square(ref_all)))
  sqnr_db = 10.0 * math.log10(signal_power / max(mse, 1e-15))

  ref_flat = ref_all.ravel()
  q_flat = q_all.ravel()
  denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(q_flat))
  cosine_sim = float(np.dot(ref_flat, q_flat) / max(denom, 1e-15))

  p_ref = _softmax(ref_all)
  p_q = _softmax(q_all)
  kl_div = float(np.mean(np.sum(p_ref * np.log(p_ref / p_q), axis=-1)))

  ref_mask = np.argmax(ref_all, axis=-1)
  q_mask = np.argmax(q_all, axis=-1)
  pixel_agreement = float(np.mean(ref_mask == q_mask))

  ious: list[float] = []
  for c in range(ref_all.shape[-1]):
    ref_c = ref_mask == c
    q_c = q_mask == c
    union = int(np.sum(ref_c | q_c))
    if union > 0:
      ious.append(float(np.sum(ref_c & q_c)) / float(union))
  mean_iou = float(np.mean(ious))

  return SegmentationQualityReport(
      model_name=model_name,
      calibration_mode=calibration_mode,
      calibration_time_sec=calibration_time_sec,
      model_size_mb=len(quant_model_bytes) / (1024.0 * 1024.0),
      output_mse=mse,
      output_sqnr_db=sqnr_db,
      cosine_similarity=cosine_sim,
      kl_divergence=kl_div,
      pixel_top1_agreement=pixel_agreement,
      mean_iou=mean_iou,
  )


def calibrate_and_quantize_a8w8(
    float_model_path: str,
    calibration_data: dict[str, Iterable[dict[str, Any]]],
    eval_samples: Sequence[np.ndarray],
    float_outputs: Sequence[np.ndarray],
    range_method: RangeMethod,
    calibration_mode: CalibrationMode,
    output_dir: str,
    model_name: str,
) -> SegmentationQualityReport:
  """Calibrates, quantizes to A8W8, saves artifacts, and evaluates quality.

  Args:
    float_model_path: Path to the floating-point TFLite model.
    calibration_data: Signature-keyed calibration dataset.
    eval_samples: Resized input arrays for evaluation.
    float_outputs: Reference FP32 model outputs on `eval_samples`.
    range_method: Activation range calibration method.
    calibration_mode: Quantizer calibration execution mode.
    output_dir: Directory to save the quantized model and recipe.
    model_name: Output filename prefix for the saved artifacts.

  Returns:
    Quality metrics for the quantized model.

  Raises:
    ValueError: If quantization produces no model bytes.
  """
  qt = quantizer.Quantizer(float_model_path)
  qt.load_quantization_recipe(
      recipe.static_wi8_ai8(
          activation_algorithm_params={'range_method': range_method.value}
      )
  )

  t0 = time.perf_counter()
  calibration_result = qt.calibrate(calibration_data, mode=calibration_mode)
  calibration_time = time.perf_counter() - t0

  quant_result = qt.quantize(calibration_result)
  quant_result.save(output_dir, model_name=model_name, overwrite=True)

  raw_bytes = quant_result.quantized_model
  if raw_bytes is None:
    raise ValueError(f'Quantization produced no model bytes for {model_name}.')
  quant_bytes = (
      bytes(raw_bytes) if isinstance(raw_bytes, bytearray) else raw_bytes
  )

  return evaluate_segmentation_quality(
      float_outputs=float_outputs,
      quant_model_bytes=quant_bytes,
      eval_samples=eval_samples,
      model_name=model_name,
      calibration_mode=calibration_mode,
      calibration_time_sec=calibration_time,
  )


def print_quality_reports(
    reports: Sequence[SegmentationQualityReport],
) -> None:
  """Prints a formatted segmentation quality table."""
  print(
      f'\n{"Model / Algorithm":<36} {"Mode":<10} {"Calib(s)":>8} '
      f'{"Size(MB)":>8} {"SQNR(dB)":>9} {"MSE":>10} {"CosSim":>8} '
      f'{"KL Div":>9} {"Top1 Acc":>9} {"mIoU":>8}'
  )
  for r in reports:
    mode_short = (
        'PROFILER'
        if r.calibration_mode == CalibrationMode.CALIBRATION_PROFILER_BASED
        else 'PRESERVE'
    )
    print(
        f'{r.model_name:<36} {mode_short:<10} '
        f'{r.calibration_time_sec:>8.1f} {r.model_size_mb:>8.2f} '
        f'{r.output_sqnr_db:>9.2f} {r.output_mse:>10.2e} '
        f'{r.cosine_similarity:>8.5f} {r.kl_divergence:>9.5f} '
        f'{100.0 * r.pixel_top1_agreement:>8.2f}% '
        f'{100.0 * r.mean_iou:>7.2f}%'
    )


def resolve_inputs(
    float_model_path: str,
    image_paths: Sequence[str],
    output_dir: str,
) -> tuple[str, Sequence[str]]:
  """Resolves `(model_path, image_paths)` from flags, local sample, or GitHub.

  Args:
    float_model_path: Explicit model path, or empty string for defaults.
    image_paths: Explicit image paths, or empty sequence for defaults.
    output_dir: Directory where GitHub defaults are downloaded.

  Returns:
    Tuple of `(model_path, image_paths)`.

  Raises:
    ValueError: If any resolved input file does not exist.
    RuntimeError: If downloading default GitHub assets fails.
  """
  if os.path.exists(LOCAL_SAMPLE_DIR):
    model_path = float_model_path or LOCAL_MODEL_PATH
    resolved_images = tuple(image_paths) or (LOCAL_IMAGE_PATH,)
  else:
    out_root = pathlib.Path(output_dir)
    model_path = float_model_path or str(
        out_root / GITHUB_MODEL_URL.rsplit('/', 1)[-1]
    )
    resolved_images = tuple(image_paths) or (
        str(out_root / GITHUB_IMAGE_URL.rsplit('/', 1)[-1]),
    )
    for url, dest_path, needed in (
        (GITHUB_MODEL_URL, model_path, not float_model_path),
        (GITHUB_IMAGE_URL, resolved_images[0], not image_paths),
    ):
      if not needed:
        continue
      print(f'Downloading {url} to {dest_path} ...')
      try:
        with (
            urllib.request.build_opener().open(
                url, timeout=_DOWNLOAD_TIMEOUT_SEC
            ) as response,
            open(dest_path, 'wb') as f,
        ):
          f.write(response.read())
      except (urllib.error.URLError, OSError) as err:
        raise RuntimeError(f'Failed to download {url}: {err}') from err

  for path in (model_path, *resolved_images):
    if not os.path.exists(path):
      raise ValueError(f'Input file does not exist: {path}')
  return model_path, resolved_images


def run_quantization_sweep(
    *,
    float_model_path: str,
    image_paths: Sequence[str],
    num_calib_samples: int,
    output_dir: str,
    range_method: str | None = None,
    calibration_method: str = 'CALIBRATION_PROFILER_BASED',
) -> Sequence[SegmentationQualityReport]:
  """Runs calibration, quantization, and evaluation for selected configs.

  Args:
    float_model_path: Explicit float model path, or empty string for defaults.
    image_paths: Explicit image paths, or empty sequence for defaults.
    num_calib_samples: Number of augmented calibration samples to generate.
    output_dir: Directory for downloaded assets, quantized models, and recipes.
    range_method: Activation range method name, or `None` to run all four
      default configurations.
    calibration_method: Calibration mode name used with `range_method`.

  Returns:
    Quality reports for each evaluated configuration.

  Raises:
    ValueError: If `output_dir` is empty, an input file is missing, or
      `range_method` / `calibration_method` is invalid.
    RuntimeError: If downloading default GitHub assets fails.
  """
  if not output_dir:
    raise ValueError('output_dir must not be empty.')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  model_path, resolved_images = resolve_inputs(
      float_model_path, image_paths, output_dir
  )
  input_name, input_shape = get_model_input_spec(model_path)
  calib_samples, eval_samples = load_and_preprocess_images(
      resolved_images,
      target_shape=input_shape,
      num_augmented_samples=num_calib_samples,
  )
  calib_dataset = build_calibration_dataset(input_name, calib_samples)
  float_outputs = run_model_on_samples(model_path, eval_samples)

  if range_method is not None:
    method = RangeMethod(range_method)
    try:
      mode = CalibrationMode[calibration_method]
    except KeyError as err:
      raise ValueError(
          f'Invalid calibration method: {calibration_method!r}'
      ) from err
    configs: Sequence[tuple[str, RangeMethod, CalibrationMode]] = (
        (f'selfie_a8w8_{method.value}', method, mode),
    )
  else:
    configs = _DEFAULT_CONFIGS

  reports = [
      calibrate_and_quantize_a8w8(
          float_model_path=model_path,
          calibration_data=calib_dataset,
          eval_samples=eval_samples,
          float_outputs=float_outputs,
          range_method=method,
          calibration_mode=mode,
          output_dir=output_dir,
          model_name=name,
      )
      for name, method, mode in configs
  ]

  print(f'Saved quantized models and recipes to: {output_dir}')
  return reports


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  reports = run_quantization_sweep(
      float_model_path=_FLOAT_MODEL_PATH.value,
      image_paths=_IMAGE_PATHS.value,
      num_calib_samples=_NUM_CALIB_SAMPLES.value,
      output_dir=_OUTPUT_DIR.value,
      range_method=_RANGE_METHOD.value,
      calibration_method=_CALIBRATION_METHOD.value,
  )
  print_quality_reports(reports)


if __name__ == '__main__':
  app.run(main)
