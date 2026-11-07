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

"""End-to-end histogram calibration and A8W8 static quantization example."""

from collections.abc import Mapping, Sequence
import dataclasses
import math
import pathlib
import time
from typing import Any
import urllib.request

from absl import app
from absl import flags
import numpy as np
from PIL import Image

import os
import io
from litert_quantizer import calibrator
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer import recipe
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.utils import histogram_range_utils
from litert_quantizer.utils import histogram_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils

RangeMethod = naive_min_max_quantize.ActivationRangeMethod
CalibrationMode = calibrator.CalibrationMode

SELFIE_CLASS_NAMES = (
    'background',
    'hair',
    'body_skin',
    'face_skin',
    'clothes',
    'others',
)

_CALIBRATION_MODES = {
    'CALIBRATION_PROFILER_BASED': CalibrationMode.CALIBRATION_PROFILER_BASED,
    'CALIBRATION_PRESERVE_ALL_TENSORS': (
        CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS
    ),
}

_DEFAULT_MODEL_PATH = test_utils.get_path_to_datafile(
    '../../../../litert/litert/samples/async_segmentation/models/'
    'selfie_multiclass_256x256.tflite'
)
_SELFIE_MODEL_GCS_URL = (
    'https://storage.googleapis.com/mediapipe-models/image_segmenter/'
    'selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
)

_FLOAT_MODEL_PATH = flags.DEFINE_string(
    'float_model_path',
    _DEFAULT_MODEL_PATH,
    'Path to the floating-point TFLite segmentation model.',
)
_IMAGE_DIR = flags.DEFINE_string(
    'image_dir',
    test_utils.get_path_to_datafile('data'),
    'Directory containing calibration and evaluation images.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/selfie_a8w8_quantization',
    'Directory to save quantized models and recipes.',
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
    list(_CALIBRATION_MODES.keys()),
    'Calibration method used when --range_method is specified.',
)
_OUTPUT_TFLITE_PATH = flags.DEFINE_string(
    'output_tflite_path',
    None,
    'Optional path where the single quantized TFLite model should be written.',
)


@dataclasses.dataclass(frozen=True)
class TensorHistogramAnalysis:
  """Histogram clipping statistics for a single calibrated tensor."""

  tensor_name: str
  raw_min: float
  raw_max: float
  clipped_min: float
  clipped_max: float
  range_ratio: float
  effective_bit_gain: float
  tail_clip_fraction: float


@dataclasses.dataclass(frozen=True)
class CalibrationSummary:
  """Aggregated histogram clipping summary across all calibrated tensors."""

  range_method: str
  calibration_mode: CalibrationMode
  wall_time_sec: float
  num_histogram_tensors: int
  num_shrunk_tensors: int
  mean_range_ratio: float
  min_range_ratio: float
  mean_bit_gain: float
  max_bit_gain: float
  top_outlier_tensors: list[TensorHistogramAnalysis]


@dataclasses.dataclass(frozen=True)
class SegmentationQualityReport:
  """Numerical and task-level quality metrics for a quantized model."""

  model_name: str
  range_method: str
  calibration_mode: CalibrationMode
  model_size_mb: float
  compression_ratio: float
  output_mse: float
  output_sqnr_db: float
  cosine_similarity: float
  kl_divergence: float
  pixel_top1_agreement: float
  mean_iou: float
  per_class_iou: dict[str, float]


def get_model_input_spec(model_path: str) -> tuple[str, tuple[int, ...], Any]:
  """Returns `(input_name, input_shape, input_dtype)` for the default signature."""
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
  return input_name, shape, input_details[0]['dtype']


def load_and_preprocess_images(
    image_dir: str,
    target_shape: tuple[int, ...],
    num_augmented_samples: int = 32,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Loads images and generates `(calibration_arrays, eval_arrays)`."""
  _, height, width, channels = target_shape
  dir_path = pathlib.Path(image_dir)
  image_paths: list[pathlib.Path] = []
  for ext in ('*.jpg', '*.jpeg', '*.png'):
    image_paths.extend(dir_path.glob(ext))
  image_paths.sort()
  if not image_paths:
    raise ValueError(f'No images found in {image_dir}.')

  base_images: list[np.ndarray] = []
  eval_arrays: list[np.ndarray] = []
  mode = 'L' if channels == 1 else 'RGB'

  for img_path in image_paths:
    with open(str(img_path), 'rb') as f:
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
) -> dict[str, list[dict[str, np.ndarray]]]:
  """Formats input arrays into an AEQ signature-keyed calibration dataset."""
  return {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: [
          {input_name: sample} for sample in samples
      ]
  }


def analyze_histogram_qsvs(
    model_qsvs: Mapping[str, qtyping.QSV],
    range_method: str,
    calibration_mode: CalibrationMode,
    wall_time_sec: float,
    num_bits: int = 8,
    symmetric: bool = False,
) -> CalibrationSummary:
  """Summarizes range compression and tail clipping across histogram QSVs."""
  analyses: list[TensorHistogramAnalysis] = []

  for tensor_name, qsv in model_qsvs.items():
    if not qsv.get('histogram'):
      continue
    hist = histogram_utils.DynamicHistogram.from_dict(qsv['histogram'])
    if not hist.initialized or hist.bin_width is None:
      continue
    counts = np.asarray(hist.counts, dtype=np.float64).ravel()
    total_samples = int(np.sum(counts))
    raw_min = float(np.asarray(hist.global_min).ravel()[0])
    raw_max = float(np.asarray(hist.global_max).ravel()[0])

    mids = histogram_range_utils.clamp_and_snap_zero_bin(
        histogram_range_utils.get_hist_mids(
            len(counts), float(hist.lower_bound), float(hist.bin_width)
        ),
        float(hist.bin_width),
        raw_min,
        raw_max,
    )

    if range_method == RangeMethod.HISTOGRAM_PERCENTILE:
      clip_min, clip_max = (
          histogram_range_utils.percentile_range_from_histogram(
              hist, fallback_min=raw_min, fallback_max=raw_max
          )
      )
    else:
      clip_min, clip_max = histogram_range_utils.mse_range_from_histogram(
          hist,
          num_bits=num_bits,
          symmetric=symmetric,
          fallback_min=raw_min,
          fallback_max=raw_max,
      )

    # TFLite asymmetric quantization always extends the range to include 0.0.
    eff_raw_min = min(raw_min, 0.0)
    eff_raw_max = max(raw_max, 0.0)
    eff_clip_min = min(clip_min, 0.0)
    eff_clip_max = max(clip_max, 0.0)

    raw_span = max(eff_raw_max - eff_raw_min, 1e-12)
    clip_span = max(eff_clip_max - eff_clip_min, 1e-12)
    range_ratio = min(clip_span / raw_span, 1.0)
    effective_bit_gain = max(0.0, -math.log2(max(range_ratio, 1e-6)))

    tail_mask = (mids < eff_clip_min) | (mids > eff_clip_max)
    tail_clip_fraction = float(np.sum(counts[tail_mask])) / float(total_samples)

    analyses.append(
        TensorHistogramAnalysis(
            tensor_name=tensor_name,
            raw_min=raw_min,
            raw_max=raw_max,
            clipped_min=clip_min,
            clipped_max=clip_max,
            range_ratio=range_ratio,
            effective_bit_gain=effective_bit_gain,
            tail_clip_fraction=tail_clip_fraction,
        )
    )

  if not analyses:
    return CalibrationSummary(
        range_method=range_method,
        calibration_mode=calibration_mode,
        wall_time_sec=wall_time_sec,
        num_histogram_tensors=0,
        num_shrunk_tensors=0,
        mean_range_ratio=1.0,
        min_range_ratio=1.0,
        mean_bit_gain=0.0,
        max_bit_gain=0.0,
        top_outlier_tensors=[],
    )

  ratios = [a.range_ratio for a in analyses]
  bit_gains = [a.effective_bit_gain for a in analyses]
  top_outliers = sorted(analyses, key=lambda a: a.range_ratio)[:10]

  return CalibrationSummary(
      range_method=range_method,
      calibration_mode=calibration_mode,
      wall_time_sec=wall_time_sec,
      num_histogram_tensors=len(analyses),
      num_shrunk_tensors=sum(1 for r in ratios if r < 0.99),
      mean_range_ratio=float(np.mean(ratios)),
      min_range_ratio=float(np.min(ratios)),
      mean_bit_gain=float(np.mean(bit_gains)),
      max_bit_gain=float(np.max(bit_gains)),
      top_outlier_tensors=top_outliers,
  )


def run_model_on_samples(
    model_source: str | bytes, samples: Sequence[np.ndarray]
) -> list[np.ndarray]:
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


def evaluate_segmentation_quality(
    float_outputs: Sequence[np.ndarray],
    quant_model_bytes: bytes,
    eval_samples: Sequence[np.ndarray],
    model_name: str,
    range_method: str,
    calibration_mode: CalibrationMode,
    float_model_size_bytes: int,
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

  eps = 1e-7

  def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return np.clip(exp / np.sum(exp, axis=-1, keepdims=True), eps, 1.0)

  p_ref = softmax(ref_all)
  p_q = softmax(q_all)
  kl_div = float(np.mean(np.sum(p_ref * np.log(p_ref / p_q), axis=-1)))

  ref_mask = np.argmax(ref_all, axis=-1)
  q_mask = np.argmax(q_all, axis=-1)
  pixel_agreement = float(np.mean(ref_mask == q_mask))

  per_class_iou: dict[str, float] = {}
  for c in range(ref_all.shape[-1]):
    class_name = (
        SELFIE_CLASS_NAMES[c] if c < len(SELFIE_CLASS_NAMES) else f'class_{c}'
    )
    ref_c = ref_mask == c
    q_c = q_mask == c
    union = int(np.sum(ref_c | q_c))
    if union == 0:
      continue
    per_class_iou[class_name] = float(np.sum(ref_c & q_c)) / float(union)
  mean_iou = float(np.mean(list(per_class_iou.values())))

  quant_size = len(quant_model_bytes)
  return SegmentationQualityReport(
      model_name=model_name,
      range_method=range_method,
      calibration_mode=calibration_mode,
      model_size_mb=quant_size / (1024.0 * 1024.0),
      compression_ratio=float(float_model_size_bytes) / max(quant_size, 1),
      output_mse=mse,
      output_sqnr_db=sqnr_db,
      cosine_similarity=cosine_sim,
      kl_divergence=kl_div,
      pixel_top1_agreement=pixel_agreement,
      mean_iou=mean_iou,
      per_class_iou=per_class_iou,
  )


def calibrate_and_quantize_a8w8(
    float_model_path: str,
    calibration_data: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    eval_samples: Sequence[np.ndarray],
    float_outputs: Sequence[np.ndarray],
    range_method: RangeMethod,
    calibration_mode: CalibrationMode,
    output_dir: str,
    model_name: str,
) -> tuple[bytes, CalibrationSummary, SegmentationQualityReport]:
  """Calibrates, analyzes QSVs, quantizes to A8W8, and evaluates quality."""
  qt = quantizer.Quantizer(float_model_path)
  qt.load_quantization_recipe(
      recipe.static_wi8_ai8(
          activation_algorithm_params={'range_method': range_method.value}
      )
  )

  t0 = time.perf_counter()
  calibration_result = qt.calibrate(
      calibration_data,  # pyrefly: ignore[bad-argument-type]
      mode=calibration_mode,
  )
  wall_time = time.perf_counter() - t0

  calib_summary = analyze_histogram_qsvs(
      calibration_result,
      range_method=range_method.value,
      calibration_mode=calibration_mode,
      wall_time_sec=wall_time,
  )

  quant_result = qt.quantize(calibration_result)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  quant_result.save(output_dir, model_name=model_name, overwrite=True)

  raw_bytes = quant_result.quantized_model
  quant_bytes = (
      bytes(raw_bytes) if isinstance(raw_bytes, bytearray) else raw_bytes
  )
  assert quant_bytes is not None

  float_size = os.path.getsize(float_model_path)
  quality_report = evaluate_segmentation_quality(
      float_outputs=float_outputs,
      quant_model_bytes=quant_bytes,
      eval_samples=eval_samples,
      model_name=model_name,
      range_method=range_method.value,
      calibration_mode=calibration_mode,
      float_model_size_bytes=float_size,
  )
  return quant_bytes, calib_summary, quality_report


def print_analysis_and_quality_reports(
    summaries: Sequence[CalibrationSummary],
    reports: Sequence[SegmentationQualityReport],
) -> None:
  """Prints formatted histogram calibration and segmentation quality tables."""
  for s in summaries:
    print(
        f'\n[{s.range_method} | {s.calibration_mode.name}] '
        f'Wall time: {s.wall_time_sec:.3f}s, '
        f'hist_tensors={s.num_histogram_tensors} '
        f'(shrunk={s.num_shrunk_tensors}), '
        f'mean_ratio={s.mean_range_ratio:.4f}, '
        f'mean_bit_gain=+{s.mean_bit_gain:.3f}b'
    )
  print(
      f'\n{"Model / Algorithm":<36} {"Mode":<10} {"Size(MB)":>8} '
      f'{"SQNR(dB)":>9} {"MSE":>10} {"CosSim":>8} {"KL Div":>9} '
      f'{"Top1 Acc":>9} {"mIoU":>8}'
  )
  for r in reports:
    mode_short = (
        'PROFILER'
        if r.calibration_mode == CalibrationMode.CALIBRATION_PROFILER_BASED
        else 'PRESERVE'
    )
    print(
        f'{r.model_name:<36} {mode_short:<10} {r.model_size_mb:>8.2f} '
        f'{r.output_sqnr_db:>9.2f} {r.output_mse:>10.2e} '
        f'{r.cosine_similarity:>8.5f} {r.kl_divergence:>9.5f} '
        f'{100.0 * r.pixel_top1_agreement:>8.2f}% '
        f'{100.0 * r.mean_iou:>7.2f}%'
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_path = _FLOAT_MODEL_PATH.value
  image_dir = _IMAGE_DIR.value
  output_dir = _OUTPUT_DIR.value
  assert model_path is not None
  assert image_dir is not None
  assert output_dir is not None

  if not os.path.exists(model_path):
    if model_path == _DEFAULT_MODEL_PATH:
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      downloaded_path = str(
          pathlib.Path(output_dir) / 'selfie_multiclass_256x256.tflite'
      )
      if not os.path.exists(downloaded_path):
        print(f'Downloading selfie model from {_SELFIE_MODEL_GCS_URL} ...')
        urllib.request.urlretrieve(_SELFIE_MODEL_GCS_URL, downloaded_path)
      model_path = downloaded_path
    else:
      raise ValueError(f'Model file does not exist: {model_path}')

  input_name, input_shape, _ = get_model_input_spec(model_path)
  calib_samples, eval_samples = load_and_preprocess_images(
      image_dir,
      target_shape=input_shape,
      num_augmented_samples=_NUM_CALIB_SAMPLES.value,
  )
  calib_dataset = build_calibration_dataset(input_name, calib_samples)
  float_outputs = run_model_on_samples(model_path, eval_samples)

  if _RANGE_METHOD.value is not None:
    method = RangeMethod(_RANGE_METHOD.value)
    mode = _CALIBRATION_MODES[_CALIBRATION_METHOD.value]
    configs = [(f'selfie_a8w8_{method.value}', method, mode)]
  else:
    configs = [
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
    ]

  summaries: list[CalibrationSummary] = []
  reports: list[SegmentationQualityReport] = []
  for name, method, mode in configs:
    quant_bytes, summary, report = calibrate_and_quantize_a8w8(
        float_model_path=model_path,
        calibration_data=calib_dataset,
        eval_samples=eval_samples,
        float_outputs=float_outputs,
        range_method=method,
        calibration_mode=mode,
        output_dir=output_dir,
        model_name=name,
    )
    summaries.append(summary)
    reports.append(report)
    if _OUTPUT_TFLITE_PATH.value is not None and len(configs) == 1:
      out_tflite = pathlib.Path(_OUTPUT_TFLITE_PATH.value)
      if not os.path.exists(str(out_tflite.parent)):
        os.makedirs(str(out_tflite.parent))
      with open(str(out_tflite), 'wb') as f:
        f.write(quant_bytes)

  print_analysis_and_quality_reports(summaries, reports)


if __name__ == '__main__':
  app.run(main)
