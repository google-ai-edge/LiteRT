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

"""End-to-end histogram calibration and A8W8 static quantization example.

This script demonstrates calibrating and quantizing a vision segmentation model
(such as `selfie_multiclass_256x256.tflite`) to static 8-bit activations and
8-bit weights (`a8w8`) using the profiler-based calibrator with histogram
calibration (`HISTOGRAM_MSE` and `HISTOGRAM_PERCENTILE`), performing a
first-principles analysis of the collected activation histograms, and
evaluating post-quantization quality against the FP32 baseline.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math
import pathlib
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
from PIL import Image

import os
import io
from litert_quantizer import algorithm_manager
from litert_quantizer import calibrator
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer import recipe
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.utils import histogram_range_utils
from litert_quantizer.utils import histogram_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import

AlgorithmName = algorithm_manager.AlgorithmName
CalibrationMode = calibrator.CalibrationMode

SELFIE_CLASS_NAMES = (
    'background',
    'hair',
    'body_skin',
    'face_skin',
    'clothes',
    'others',
)

_FLOAT_MODEL_PATH = flags.DEFINE_string(
    'float_model_path',
    test_utils.get_path_to_datafile(
        '../../../../litert/litert/samples/async_segmentation/models/'
        'selfie_multiclass_256x256.tflite'
    ),
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
    'Directory to save quantized models, recipes, and calibration QSVs.',
)
_NUM_CALIB_SAMPLES = flags.DEFINE_integer(
    'num_calib_samples',
    32,
    'Number of augmented image samples to use for calibration.',
)

flags.DEFINE_alias('model_path', 'float_model_path')
flags.DEFINE_alias('dataset_dir', 'image_dir')
flags.DEFINE_alias('num_calibration_steps', 'num_calib_samples')

_CALIBRATION_ALGORITHM = flags.DEFINE_string(
    'calibration_algorithm',
    None,
    'Optional calibration algorithm to run (HISTOGRAM_MSE,'
    ' HISTOGRAM_PERCENTILE, or MIN_MAX_UNIFORM_QUANT). If None, runs all'
    ' algorithms and prints a comparison table.',
)
_CALIBRATION_METHOD = flags.DEFINE_string(
    'calibration_method',
    None,
    'Optional calibration method (CALIBRATION_PROFILER_BASED or'
    ' CALIBRATION_PRESERVE_ALL_TENSORS) for --calibration_algorithm. Defaults'
    ' to CALIBRATION_PROFILER_BASED. Ignored if --calibration_algorithm is'
    ' unset.',
)
_OUTPUT_TFLITE_PATH = flags.DEFINE_string(
    'output_tflite_path',
    None,
    'Optional path where the single quantized TFLite model should be written'
    ' when running a specific algorithm.',
)


@dataclasses.dataclass(frozen=True)
class TensorHistogramAnalysis:
  """First-principles analysis metrics for a single calibrated tensor."""

  tensor_name: str
  total_samples: int
  num_bins: int
  raw_min: float
  raw_max: float
  clipped_min: float
  clipped_max: float
  range_ratio: float
  effective_bit_gain: float
  tail_clip_fraction: float
  crest_factor: float
  minmax_granular_mse: float
  hist_granular_mse: float
  hist_overload_mse: float
  minmax_total_mse: float
  hist_total_mse: float
  mse_reduction_ratio: float


@dataclasses.dataclass(frozen=True)
class CalibrationSummary:
  """Aggregated first-principles summary across all calibrated tensors."""

  algorithm_key: str
  calibration_mode: CalibrationMode
  wall_time_sec: float
  num_histogram_tensors: int
  num_shrunk_tensors: int
  mean_range_ratio: float
  min_range_ratio: float
  mean_bit_gain: float
  max_bit_gain: float
  mean_mse_reduction_ratio: float
  median_mse_reduction_ratio: float
  top_outlier_tensors: list[TensorHistogramAnalysis]
  model_qsvs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class SegmentationQualityReport:
  """Numerical and task-level quality metrics for a quantized model."""

  model_name: str
  algorithm_key: str
  calibration_mode: CalibrationMode
  model_size_mb: float
  compression_ratio: float
  output_mse: float
  output_rmse: float
  output_mae: float
  output_max_abs_err: float
  output_sqnr_db: float
  cosine_similarity: float
  kl_divergence: float
  pixel_top1_agreement: float
  mean_iou: float
  per_class_iou: dict[str, float]


def get_model_input_spec(model_path: str) -> tuple[str, tuple[int, ...], Any]:
  """Returns (input_name, input_shape, input_dtype) for the default signature."""
  interp = tfl_interpreter.Interpreter(model_path=model_path)
  interp.allocate_tensors()
  sig_defs = interp.get_signature_list()
  input_details = interp.get_input_details()
  if sig_defs and tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY in sig_defs:
    sig_inputs = sig_defs[tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY]['inputs']
    input_name = sig_inputs[0]
  else:
    input_name = input_details[0]['name'].split(':')[0]
  shape = tuple(int(d) for d in input_details[0]['shape'])
  dtype = input_details[0]['dtype']
  return input_name, shape, dtype


def load_and_preprocess_images(
    image_dir: str,
    target_shape: tuple[int, ...],
    num_augmented_samples: int = 32,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Loads real images and generates calibration and evaluation sets.

  Args:
    image_dir: Directory containing `.jpg` or `.png` images.
    target_shape: Model input tensor shape `(1, H, W, C)`.
    num_augmented_samples: Total number of calibration samples to produce.
    seed: Random seed for deterministic spatial/photometric augmentations.

  Returns:
    Tuple of `(calibration_arrays, eval_arrays)`, where each element has shape
    `target_shape` and dtype `float32` normalized to `[0.0, 1.0]`.
  """
  _, height, width, channels = target_shape
  dir_path = pathlib.Path(image_dir)
  image_paths = sorted(
      list(dir_path.glob('*.jpg'))
      + list(dir_path.glob('*.jpeg'))
      + list(dir_path.glob('*.png'))
  )
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
      arr = arr.reshape((1, height, width, channels))
      eval_arrays.append(arr)

  calib_arrays: list[np.ndarray] = list(eval_arrays)
  rng = np.random.default_rng(seed)

  while len(calib_arrays) < num_augmented_samples:
    src = base_images[len(calib_arrays) % len(base_images)]
    src_h, src_w = src.shape[0], src.shape[1]

    # Random crop covering 70% to 100% of the image.
    crop_scale = float(rng.uniform(0.70, 1.0))
    crop_h = max(1, int(src_h * crop_scale))
    crop_w = max(1, int(src_w * crop_scale))
    top = int(rng.integers(0, max(1, src_h - crop_h + 1)))
    left = int(rng.integers(0, max(1, src_w - crop_w + 1)))
    cropped = src[top : top + crop_h, left : left + crop_w]

    # Convert back to PIL for bilinear resize to target (W, H).
    uint8_crop = np.clip(cropped * 255.0, 0.0, 255.0).astype(np.uint8)
    pil_crop = Image.fromarray(uint8_crop, mode=mode).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    aug = np.asarray(pil_crop, dtype=np.float32) / 255.0

    # Random horizontal flip.
    if rng.random() < 0.5:
      aug = np.flip(aug, axis=1)

    # Mild brightness and contrast jitter within [0, 1].
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


def _compute_uniform_quant_mse_decomposition(
    mids: np.ndarray,
    counts: np.ndarray,
    clip_min: float,
    clip_max: float,
    num_bits: int = 8,
    symmetric: bool = False,
    signed: bool = True,
) -> tuple[float, float, float]:
  """Decomposes expected uniform quantization MSE into granular and overload MSE.

  From first principles, a uniform quantizer with clipping bounds
  `[clip_min, clip_max]` has two orthogonal error components over a distribution
  with bin centers `mids` and frequencies `counts`:
  1. Granular (step-size) noise on in-range bins
     (`mids` in `[clip_min, clip_max]`): controlled by the quantization step
     size `scale = (clip_max - clip_min) / (2^b - 1)`.
  2. Overload (saturation/clipping) distortion on tail bins
     (`mids` outside `[clip_min, clip_max]`): caused by clamping outliers to
     the grid endpoints.

  Args:
    mids: 1D array of histogram bin centers.
    counts: 1D array of histogram bin counts.
    clip_min: Lower clipping bound.
    clip_max: Upper clipping bound.
    num_bits: Quantization bit width.
    symmetric: Whether quantization is symmetric.
    signed: Whether quantization is signed.

  Returns:
    Tuple of `(granular_mse, overload_mse, total_mse)`.
  """
  total_count = float(np.sum(counts))
  qtype = uniform_quantize_tensor.IntType(num_bits, signed=signed)
  qmin, qmax = uniform_quantize_tensor.get_quantized_range(qtype)
  zp_arr, scale_arr = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
      np.array([clip_min], dtype=np.float64),
      np.array([clip_max], dtype=np.float64),
      num_bits=num_bits,
      symmetric=symmetric,
      granularity=qtyping.QuantGranularity.TENSORWISE,
      signed=signed,
  )
  scale = float(np.asarray(scale_arr, dtype=np.float64).ravel()[0])
  zp = float(np.asarray(zp_arr, dtype=np.float64).ravel()[0])

  dequant_min = (qmin - zp) * scale
  dequant_max = (qmax - zp) * scale
  in_range_mask = (mids >= dequant_min) & (mids <= dequant_max)
  clamped_mids = np.clip(mids, dequant_min, dequant_max)
  overload_sq_err = np.square(mids - clamped_mids)

  granular_mse = (
      float(np.sum(counts[in_range_mask]))
      * ((scale**2) / 12.0)
      / total_count
  )
  overload_mse = (
      float(np.sum(counts[~in_range_mask] * overload_sq_err[~in_range_mask]))
      / total_count
  )
  return granular_mse, overload_mse, granular_mse + overload_mse


def analyze_histogram_qsvs(
    model_qsvs: Mapping[str, qtyping.QSV],
    algorithm_key: str,
    calibration_mode: CalibrationMode,
    wall_time_sec: float,
    num_bits: int = 8,
    symmetric: bool = False,
) -> CalibrationSummary:
  """Performs first-principles analysis on calibrated histogram QSVs.

  Args:
    model_qsvs: Dictionary mapping tensor names to calibrated QSV dictionaries.
    algorithm_key: Calibration algorithm key (`HISTOGRAM_PERCENTILE` selects
      percentile clipping; any other key uses MSE clipping).
    calibration_mode: Calibration mode used.
    wall_time_sec: Calibration wall time in seconds.
    num_bits: Activation bit width (default 8).
    symmetric: Whether activation quantization is symmetric (default False).

  Returns:
    A `CalibrationSummary` summarizing range compression, bit gain, tail
    clipping, and granular vs. overload quantization MSE across all tensors.
  """
  analyses: list[TensorHistogramAnalysis] = []

  for tensor_name, qsv in model_qsvs.items():
    if not qsv.get('histogram'):
      continue
    hist = histogram_utils.DynamicHistogram.from_dict(qsv['histogram'])
    if not hist.initialized or hist.bin_width is None:
      continue
    counts = np.asarray(hist.counts, dtype=np.float64).ravel()
    total_samples = int(np.sum(counts))

    lower_bound = float(hist.lower_bound)
    bin_width = float(hist.bin_width)
    num_bins = len(counts)
    raw_min = float(np.asarray(hist.global_min).ravel()[0])
    raw_max = float(np.asarray(hist.global_max).ravel()[0])

    mids = histogram_range_utils.clamp_and_snap_zero_bin(
        histogram_range_utils.get_hist_mids(num_bins, lower_bound, bin_width),
        bin_width,
        raw_min,
        raw_max,
    )

    if algorithm_key == AlgorithmName.HISTOGRAM_PERCENTILE:
      clip_min, clip_max = (
          histogram_range_utils.percentile_range_from_histogram(
              hist,
              fallback_min=raw_min,
              fallback_max=raw_max,
          )
      )
    else:
      clip_min, clip_max = histogram_range_utils.mse_range_from_histogram(
          hist,
          num_bits=num_bits,
          symmetric=symmetric,
          signed=True,
          fallback_min=raw_min,
          fallback_max=raw_max,
      )

    # Include zero in range as TFLite asymmetric uniform quantization requires
    # zero to be representable.
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

    mean_val = float(np.sum(counts * mids)) / float(total_samples)
    variance = float(np.sum(counts * np.square(mids - mean_val))) / float(
        total_samples
    )
    std_val = math.sqrt(max(variance, 1e-12))
    crest_factor = max(abs(raw_min), abs(raw_max)) / std_val

    minmax_gran_mse, _, minmax_total_mse = (
        _compute_uniform_quant_mse_decomposition(
            mids,
            counts,
            raw_min,
            raw_max,
            num_bits=num_bits,
            symmetric=symmetric,
        )
    )
    hist_gran_mse, hist_over_mse, hist_total_mse = (
        _compute_uniform_quant_mse_decomposition(
            mids,
            counts,
            clip_min,
            clip_max,
            num_bits=num_bits,
            symmetric=symmetric,
        )
    )
    mse_reduction_ratio = (
        hist_total_mse / minmax_total_mse if minmax_total_mse > 0.0 else 1.0
    )

    analyses.append(
        TensorHistogramAnalysis(
            tensor_name=tensor_name,
            total_samples=total_samples,
            num_bins=num_bins,
            raw_min=raw_min,
            raw_max=raw_max,
            clipped_min=clip_min,
            clipped_max=clip_max,
            range_ratio=range_ratio,
            effective_bit_gain=effective_bit_gain,
            tail_clip_fraction=tail_clip_fraction,
            crest_factor=crest_factor,
            minmax_granular_mse=minmax_gran_mse,
            hist_granular_mse=hist_gran_mse,
            hist_overload_mse=hist_over_mse,
            minmax_total_mse=minmax_total_mse,
            hist_total_mse=hist_total_mse,
            mse_reduction_ratio=mse_reduction_ratio,
        )
    )

  if not analyses:
    return CalibrationSummary(
        algorithm_key=algorithm_key,
        calibration_mode=calibration_mode,
        wall_time_sec=wall_time_sec,
        num_histogram_tensors=0,
        num_shrunk_tensors=0,
        mean_range_ratio=1.0,
        min_range_ratio=1.0,
        mean_bit_gain=0.0,
        max_bit_gain=0.0,
        mean_mse_reduction_ratio=1.0,
        median_mse_reduction_ratio=1.0,
        top_outlier_tensors=[],
        model_qsvs=model_qsvs,
    )

  ratios = [a.range_ratio for a in analyses]
  bit_gains = [a.effective_bit_gain for a in analyses]
  mse_ratios = [a.mse_reduction_ratio for a in analyses]
  top_outliers = sorted(analyses, key=lambda a: a.mse_reduction_ratio)[:10]

  return CalibrationSummary(
      algorithm_key=algorithm_key,
      calibration_mode=calibration_mode,
      wall_time_sec=wall_time_sec,
      num_histogram_tensors=len(analyses),
      num_shrunk_tensors=sum(1 for r in ratios if r < 0.99),
      mean_range_ratio=float(np.mean(ratios)),
      min_range_ratio=float(np.min(ratios)),
      mean_bit_gain=float(np.mean(bit_gains)),
      max_bit_gain=float(np.max(bit_gains)),
      mean_mse_reduction_ratio=float(np.mean(mse_ratios)),
      median_mse_reduction_ratio=float(np.median(mse_ratios)),
      top_outlier_tensors=top_outliers,
      model_qsvs=model_qsvs,
  )


def _run_model_on_samples(
    model_source: str | bytes, samples: Sequence[np.ndarray]
) -> list[np.ndarray]:
  """Runs TFLite inference on `samples` and returns dequantized float outputs."""
  if isinstance(model_source, str):
    interp = tfl_interpreter.Interpreter(model_path=model_source)
  else:
    interp = tfl_interpreter.Interpreter(model_content=model_source)
  interp.allocate_tensors()
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
    algorithm_key: str,
    calibration_mode: CalibrationMode,
    float_model_size_bytes: int,
) -> SegmentationQualityReport:
  """Evaluates numerical and segmentation quality of a quantized model."""
  quant_outputs = _run_model_on_samples(quant_model_bytes, eval_samples)

  ref_all = np.concatenate(float_outputs, axis=0).astype(np.float64)
  q_all = np.concatenate(quant_outputs, axis=0).astype(np.float64)
  diff = q_all - ref_all

  mse = float(np.mean(np.square(diff)))
  rmse = math.sqrt(mse)
  mae = float(np.mean(np.abs(diff)))
  max_abs_err = float(np.max(np.abs(diff)))

  signal_power = float(np.mean(np.square(ref_all)))
  sqnr_db = 10.0 * math.log10(signal_power / max(mse, 1e-15))

  ref_flat = ref_all.ravel()
  q_flat = q_all.ravel()
  denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(q_flat))
  cosine_sim = float(np.dot(ref_flat, q_flat) / max(denom, 1e-15))

  # KL divergence D_KL(P_float || P_quant) across per-pixel class
  # probabilities. The selfie model outputs class logits, so both outputs are
  # normalized with a softmax.
  eps = 1e-7

  def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return np.clip(exp / np.sum(exp, axis=-1, keepdims=True), eps, 1.0)

  p_ref = softmax(ref_all)
  p_q = softmax(q_all)
  kl_div = float(np.mean(np.sum(p_ref * np.log(p_ref / p_q), axis=-1)))

  # Task metrics: Top-1 class agreement and per-class IoU over argmax masks.
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
    intersection = int(np.sum(ref_c & q_c))
    per_class_iou[class_name] = float(intersection) / float(union)
  mean_iou = float(np.mean(list(per_class_iou.values())))

  quant_size = len(quant_model_bytes)
  return SegmentationQualityReport(
      model_name=model_name,
      algorithm_key=algorithm_key,
      calibration_mode=calibration_mode,
      model_size_mb=quant_size / (1024.0 * 1024.0),
      compression_ratio=float(float_model_size_bytes) / max(quant_size, 1),
      output_mse=mse,
      output_rmse=rmse,
      output_mae=mae,
      output_max_abs_err=max_abs_err,
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
    algorithm_key: AlgorithmName,
    calibration_mode: CalibrationMode,
    output_dir: str,
    model_name: str,
) -> tuple[bytes, CalibrationSummary, SegmentationQualityReport]:
  """Calibrates, analyzes QSVs, quantizes to A8W8, and evaluates quality."""
  qt = quantizer.Quantizer(float_model_path)
  qt.load_quantization_recipe(
      recipe.static_wi8_ai8(algorithm_key=algorithm_key)
  )

  t0 = time.perf_counter()
  calibration_result = qt.calibrate(
      calibration_data,  # pyrefly: ignore[bad-argument-type]
      mode=calibration_mode,
  )
  wall_time = time.perf_counter() - t0

  calib_summary = analyze_histogram_qsvs(
      calibration_result,
      algorithm_key=str(algorithm_key.value),
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
      algorithm_key=str(algorithm_key.value),
      calibration_mode=calibration_mode,
      float_model_size_bytes=float_size,
  )
  return quant_bytes, calib_summary, quality_report


def print_analysis_and_quality_reports(
    summaries: Sequence[CalibrationSummary],
    reports: Sequence[SegmentationQualityReport],
) -> None:
  """Prints formatted first-principles calibration and quality tables."""
  print('\n' + '=' * 88)
  print('1. FIRST-PRINCIPLES HISTOGRAM CALIBRATION ANALYSIS')
  print('=' * 88)
  for s in summaries:
    if s.num_histogram_tensors == 0:
      print(
          f'\n[{s.algorithm_key} | {s.calibration_mode.name}] '
          f'Wall time: {s.wall_time_sec:.3f}s (Min/Max only - no histograms)'
      )
      continue
    print(
        f'\n[{s.algorithm_key} | {s.calibration_mode.name}] '
        f'Wall time: {s.wall_time_sec:.3f}s'
    )
    print(
        f'  Calibrated activation tensors : {s.num_histogram_tensors} '
        f'({s.num_shrunk_tensors} clipped tighter than raw min/max)'
    )
    print(
        f'  Mean range ratio (clip/raw)   : {s.mean_range_ratio:.4f} '
        f'(min={s.min_range_ratio:.4f})'
    )
    print(
        f'  Effective int8 resolution gain: mean=+{s.mean_bit_gain:.3f} bits, '
        f'max=+{s.max_bit_gain:.3f} bits'
    )
    print(
        f'  Expected quant MSE vs MinMax  : '
        f'mean={s.mean_mse_reduction_ratio:.4f}x, '
        f'median={s.median_mse_reduction_ratio:.4f}x'
    )
    print(
        '  Top outlier-heavy tensors (where histogram clipping reduces '
        'quantization MSE most):'
    )
    print(
        '    '
        f'{"Tensor":<34} {"Raw [min, max]":<22} {"Clipped [min, max]":<22} '
        f'{"Ratio":>6} {"Gain(b)":>7} {"Tail%":>7} {"MSE(H/MM)":>9}'
    )
    for t in s.top_outlier_tensors[:8]:
      short_name = (
          t.tensor_name
          if len(t.tensor_name) <= 32
          else '...' + t.tensor_name[-29:]
      )
      raw_str = f'[{t.raw_min:.2f}, {t.raw_max:.2f}]'
      clip_str = f'[{t.clipped_min:.2f}, {t.clipped_max:.2f}]'
      print(
          '    '
          f'{short_name:<34} {raw_str:<22} {clip_str:<22} '
          f'{t.range_ratio:>6.3f} {t.effective_bit_gain:>7.3f} '
          f'{100.0 * t.tail_clip_fraction:>6.3f}% '
          f'{t.mse_reduction_ratio:>8.3f}x'
      )

  print('\n' + '=' * 88)
  print('2. A8W8 STATIC QUANTIZATION QUALITY EVALUATION (VS. FP32 BASELINE)')
  print('=' * 88)
  print(
      f'{"Model / Algorithm":<36} {"Mode":<10} {"Size(MB)":>8} '
      f'{"SQNR(dB)":>9} {"MSE":>10} {"CosSim":>8} {"KL Div":>9} '
      f'{"Top1 Acc":>9} {"mIoU":>8}'
  )
  print('-' * 112)
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

  print('\nPer-Class Segmentation IoU (%) vs. FP32 Predictions:')
  for r in reports:
    iou_items = ', '.join(
        f'{cls}={100.0 * val:.2f}%' for cls, val in r.per_class_iou.items()
    )
    print(f'  {r.model_name:<34}: {iou_items}')
  print('=' * 88 + '\n')


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
    raise ValueError(f'Model file does not exist: {model_path}')

  input_name, input_shape, _ = get_model_input_spec(model_path)
  calib_samples, eval_samples = load_and_preprocess_images(
      image_dir,
      target_shape=input_shape,
      num_augmented_samples=_NUM_CALIB_SAMPLES.value,
  )
  calib_dataset = build_calibration_dataset(input_name, calib_samples)
  float_outputs = _run_model_on_samples(model_path, eval_samples)

  if _CALIBRATION_ALGORITHM.value is not None:
    algo_str = _CALIBRATION_ALGORITHM.value.upper()
    algo_map = {
        'HISTOGRAM_MSE': AlgorithmName.HISTOGRAM_MSE,
        'HISTOGRAM_PERCENTILE': AlgorithmName.HISTOGRAM_PERCENTILE,
        'MIN_MAX_UNIFORM_QUANT': AlgorithmName.MIN_MAX_UNIFORM_QUANT,
    }
    if algo_str not in algo_map:
      raise app.UsageError(
          f'Unknown --calibration_algorithm: {_CALIBRATION_ALGORITHM.value}. '
          f'Expected one of: {list(algo_map.keys())}'
      )
    algo = algo_map[algo_str]

    mode_map = {
        'CALIBRATION_PROFILER_BASED': (
            CalibrationMode.CALIBRATION_PROFILER_BASED
        ),
        'CALIBRATION_PRESERVE_ALL_TENSORS': (
            CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS
        ),
    }
    if _CALIBRATION_METHOD.value is not None:
      method_str = _CALIBRATION_METHOD.value.upper()
      if method_str not in mode_map:
        raise app.UsageError(
            f'Unknown --calibration_method: {_CALIBRATION_METHOD.value}. '
            f'Expected one of: {list(mode_map.keys())}'
        )
      mode = mode_map[method_str]
    else:
      mode = CalibrationMode.CALIBRATION_PROFILER_BASED

    configs = [(f'selfie_a8w8_{algo_str.lower()}', algo, mode)]
  else:
    configs = [
        (
            'selfie_a8w8_minmax_profiler',
            AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            CalibrationMode.CALIBRATION_PROFILER_BASED,
        ),
        (
            'selfie_a8w8_hist_pct_profiler',
            AlgorithmName.HISTOGRAM_PERCENTILE,
            CalibrationMode.CALIBRATION_PROFILER_BASED,
        ),
        (
            'selfie_a8w8_hist_mse_profiler',
            AlgorithmName.HISTOGRAM_MSE,
            CalibrationMode.CALIBRATION_PROFILER_BASED,
        ),
        (
            'selfie_a8w8_hist_mse_preserve',
            AlgorithmName.HISTOGRAM_MSE,
            CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
        ),
    ]

  summaries: list[CalibrationSummary] = []
  reports: list[SegmentationQualityReport] = []
  for name, algo, mode in configs:
    quant_bytes, summary, report = calibrate_and_quantize_a8w8(
        float_model_path=model_path,
        calibration_data=calib_dataset,
        eval_samples=eval_samples,
        float_outputs=float_outputs,
        algorithm_key=algo,
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
