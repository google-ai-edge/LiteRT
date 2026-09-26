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

#!/usr/bin/env python3
r"""Standalone Gemma safetensor quantizer for LiteRT per-channel affine INT8.

Example:
  python3 gemma3_safetensor_quantizer.py \
      --input /path/to/model.safetensors \
      --output /path/to/model_litert_qpc.safetensors \
      --strict
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any, Dict, List, Tuple

import numpy as np

# pylint: disable=g-import-not-at-top
try:
  import ml_dtypes
except ImportError:
  ml_dtypes = None

try:
  import torch
except ImportError:
  torch = None

try:
  from safetensors.numpy import load_file as load_safetensors_numpy_file
  from safetensors.numpy import save_file as save_safetensors_numpy_file
except ImportError:
  load_safetensors_numpy_file = None
  save_safetensors_numpy_file = None

try:
  from safetensors.torch import load_file as load_safetensors_torch_file
  from safetensors.torch import save_file as save_safetensors_torch_file
except ImportError:
  load_safetensors_torch_file = None
  save_safetensors_torch_file = None
# pylint: enable=g-import-not-at-top


DEFAULT_COS_THRESHOLD = 0.99
DEFAULT_SNR_DB_THRESHOLD = 18.0
DEFAULT_SATURATION_THRESHOLD = 0.05
GEMMA3_LINEAR_WEIGHT_RE = re.compile(
    r"""
    ^model\.layers\.\d+\.  # Start of the layer block
    (
        self_attn\.         # Self-attention weights
        (q_proj|k_proj|v_proj|o_proj) # Projection types
        |
        mlp\.               # MLP weights
        (gate_proj|up_proj|down_proj) # MLP projection types
    )
    \.weight$             # End with .weight
""",
    re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Generate LiteRT-consumable per-channel affine INT8 safetensors "
          "from a Gemma safetensors checkpoint."
      )
  )
  parser.add_argument(
      "--input",
      required=True,
      help="Path to source safetensors file (for example model.safetensors).",
  )
  parser.add_argument(
      "--output",
      default=None,
      help=(
          "Output safetensors path (default:"
          " <input_stem>_litert_qpc.safetensors in the same directory)."
      ),
  )
  parser.add_argument(
      "--strict",
      action="store_true",
      help="Fail if any quantized tensor violates quality thresholds.",
  )
  parser.add_argument(
      "--cos-threshold",
      type=float,
      default=DEFAULT_COS_THRESHOLD,
      help=f"Minimum cosine similarity (default: {DEFAULT_COS_THRESHOLD}).",
  )
  parser.add_argument(
      "--snr-threshold",
      type=float,
      default=DEFAULT_SNR_DB_THRESHOLD,
      help=f"Minimum SNR in dB (default: {DEFAULT_SNR_DB_THRESHOLD}).",
  )
  parser.add_argument(
      "--saturation-threshold",
      type=float,
      default=DEFAULT_SATURATION_THRESHOLD,
      help=(
          f"Maximum saturation rate (default: {DEFAULT_SATURATION_THRESHOLD})."
      ),
  )
  parser.add_argument(
      "--nonquantized-float-dtype",
      default="BF16",
      choices=["BF16", "FP16", "FP32"],
      help="Storage dtype for non-quantized floating tensors (default: BF16).",
  )
  parser.add_argument(
      "--keep-tied-lm-head",
      action="store_true",
      help="Keep lm_head.weight even if it is tied-identical to embeddings.",
  )
  parser.add_argument(
      "--prefer-numpy-io",
      action="store_true",
      help=(
          "Use safetensors numpy load/save path even when torch is available "
          "(BF16 will be emitted as FP16 in that mode)."
      ),
  )
  return parser.parse_args()


def _as_numpy(tensor: Any) -> np.ndarray:
  if torch is not None and isinstance(tensor, torch.Tensor):
    t = tensor.detach().cpu()
    if t.dtype == torch.bfloat16:
      t = t.float()
    return t.numpy()
  return np.asarray(tensor)


def _tensors_equal(lhs: Any, rhs: Any) -> bool:
  """Checks if two tensors are equal (supports both numpy and torch)."""
  if (
      torch is not None
      and isinstance(lhs, torch.Tensor)
      and isinstance(rhs, torch.Tensor)
  ):
    lhs_cpu = lhs.detach().cpu()
    rhs_cpu = rhs.detach().cpu()
    return bool(
        lhs_cpu.shape == rhs_cpu.shape
        and lhs_cpu.dtype == rhs_cpu.dtype
        and torch.equal(lhs_cpu, rhs_cpu)
    )
  lhs_np = np.asarray(lhs)
  rhs_np = np.asarray(rhs)
  return bool(lhs_np.shape == rhs_np.shape and np.array_equal(lhs_np, rhs_np))


def _load_state_dict(
    input_path: pathlib.Path, prefer_numpy_io: bool
) -> Tuple[Dict[str, Any], str]:
  """Loads state dict from safetensors file."""
  if (
      not prefer_numpy_io
      and torch is not None
      and load_safetensors_torch_file is not None
  ):
    tensors = load_safetensors_torch_file(str(input_path))
    return dict(tensors), "torch"

  if load_safetensors_numpy_file is None:
    raise ImportError(
        "safetensors numpy loader is unavailable. Install with `pip install"
        " safetensors`."
    )
  tensors = load_safetensors_numpy_file(str(input_path))
  return dict(tensors), "numpy"


def _save_state_dict(
    tensors: Dict[str, Any],
    output_path: pathlib.Path,
    prefer_numpy_io: bool,
) -> None:
  """Saves state dict to safetensors file."""
  if (
      not prefer_numpy_io
      and torch is not None
      and save_safetensors_torch_file is not None
  ):
    torch_tensors: Dict[str, "torch.Tensor"] = {}
    for name, value in tensors.items():
      if isinstance(value, torch.Tensor):
        torch_tensors[name] = value.detach().cpu().contiguous()
      else:
        torch_tensors[name] = torch.from_numpy(
            np.ascontiguousarray(np.asarray(value))
        )
    save_safetensors_torch_file(torch_tensors, str(output_path))
    return

  if save_safetensors_numpy_file is None:
    raise ImportError(
        "safetensors numpy saver is unavailable. Install with `pip install"
        " safetensors`."
    )

  numpy_tensors: Dict[str, np.ndarray] = {}
  for name, value in tensors.items():
    if torch is not None and isinstance(value, torch.Tensor):
      t = value.detach().cpu()
      if t.dtype == torch.bfloat16:
        t = t.to(torch.float16)
      numpy_tensors[name] = np.ascontiguousarray(t.numpy())
    else:
      numpy_tensors[name] = np.ascontiguousarray(np.asarray(value))
  save_safetensors_numpy_file(numpy_tensors, str(output_path))


def is_quantizable_linear_weight(name: str, array: np.ndarray) -> bool:
  return array.ndim == 2 and bool(GEMMA3_LINEAR_WEIGHT_RE.match(name))


def quantize_per_channel_affine_int8(
    weights: np.ndarray, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Quantizes weights to per-channel affine INT8.

  Args:
    weights: The float32 weights to quantize, expected to be 2D.
    eps: A small epsilon to prevent division by zero in scale calculation.

  Returns:
    A tuple containing:
      - quantized_weights: The weights quantized to INT8.
      - scales: The float32 per-channel scales.
      - zero_points: The int32 per-channel zero points.
      - dequantized_weights: The weights dequantized back to float32 using
        the computed scales and zero points.
  """
  if weights.ndim != 2:
    raise ValueError("Per-channel affine quantization expects a 2D tensor.")

  w = np.asarray(weights, dtype=np.float32)
  mins = np.min(w, axis=1).astype(np.float32)
  maxs = np.max(w, axis=1).astype(np.float32)
  mins = np.minimum(mins, 0.0)
  maxs = np.maximum(maxs, 0.0)
  qmin = -128.0
  qmax = 127.0

  ranges = maxs - mins
  scales = ranges / (qmax - qmin)
  safe_scales = np.where(scales > eps, scales, 1.0).astype(np.float32)
  zero_points = np.round(qmin - mins / safe_scales)
  zero_points = np.clip(zero_points, qmin, qmax).astype(np.int32)

  q = np.round(w / safe_scales[:, np.newaxis] + zero_points[:, np.newaxis])
  q = np.clip(q, qmin, qmax).astype(np.int8)
  dequantized = (
      q.astype(np.float32) - zero_points[:, np.newaxis].astype(np.float32)
  ) * safe_scales[:, np.newaxis]
  return q, safe_scales, zero_points, dequantized


def compute_quantization_metrics(
    original: np.ndarray, quantized: np.ndarray, dequantized: np.ndarray
) -> Dict[str, float]:
  """Computes quantization metrics (MSE, SNR, cosine similarity, saturation rate)."""
  orig = np.asarray(original, dtype=np.float32)
  deq = np.asarray(dequantized, dtype=np.float32)
  q = np.asarray(quantized)

  mse = float(np.mean((orig - deq) ** 2))
  variance = float(np.var(orig))
  if mse <= 0.0:
    snr_db = float("inf")
  else:
    snr_db = float(10.0 * np.log10(max(variance, 1e-20) / mse))

  orig_flat = orig.reshape(-1).astype(np.float64)
  deq_flat = deq.reshape(-1).astype(np.float64)
  cos_denom = float(np.linalg.norm(orig_flat) * np.linalg.norm(deq_flat))
  cosine = (
      1.0
      if cos_denom <= 0.0
      else float(np.dot(orig_flat, deq_flat) / cos_denom)
  )

  saturation_rate = float(np.mean((q == np.int8(-128)) | (q == np.int8(127))))

  return {
      "mse": mse,
      "snr_db": snr_db,
      "cosine_similarity": cosine,
      "saturation_rate": saturation_rate,
  }


def _convert_nonquantized_tensor(
    source_tensor: Any, nonquantized_float_dtype: str
) -> Any:
  """Converts non-quantized tensor to target dtype."""
  target_dtype = nonquantized_float_dtype.upper()

  if torch is not None and isinstance(source_tensor, torch.Tensor):
    out_tensor = source_tensor.detach().cpu().contiguous()
    if out_tensor.dtype.is_floating_point:
      if target_dtype == "BF16":
        out_tensor = out_tensor.to(torch.bfloat16)
      elif target_dtype == "FP16":
        out_tensor = out_tensor.to(torch.float16)
      elif target_dtype == "FP32":
        out_tensor = out_tensor.to(torch.float32)
      else:
        raise ValueError(
            "nonquantized_float_dtype must be one of BF16/FP16/FP32."
        )
    return out_tensor

  array = np.asarray(source_tensor)
  if array.dtype.kind == "f":
    if target_dtype in ("BF16", "FP16"):
      out_array = array.astype(np.float16)
    elif target_dtype == "FP32":
      out_array = array.astype(np.float32)
    else:
      raise ValueError(
          "nonquantized_float_dtype must be one of BF16/FP16/FP32."
      )
  elif array.dtype == np.bool_:
    out_array = array.astype(np.bool_)
  else:
    out_array = array
  return np.ascontiguousarray(out_array)


def quantize_checkpoint(
    state_dict: Dict[str, Any],
    output_path: pathlib.Path,
    strict: bool,
    thresholds: Dict[str, float],
    nonquantized_float_dtype: str,
    prefer_numpy_io: bool,
    drop_tied_lm_head: bool,
) -> Dict[str, Any]:
  """Quantizes a checkpoint and saves it.

  Args:
    state_dict: A dictionary mapping tensor names to tensors.
    output_path: The file path to save the quantized safetensors.
    strict: If True, a RuntimeError will be raised if any quantized tensor
      violates the quality thresholds.
    thresholds: A dictionary containing the quality thresholds: -
      "cosine_similarity": Minimum allowed cosine similarity. - "snr_db":
      Minimum allowed Signal-to-Noise Ratio in dB. - "saturation_rate": Maximum
      allowed saturation rate.
    nonquantized_float_dtype: The desired float dtype for non-quantized tensors
      ("BF16", "FP16", or "FP32").
    drop_tied_lm_head: If True, the "lm_head.weight" tensor will be skipped if
      it is identical to "model.embed_tokens.weight".

  Returns:
    A dictionary containing a report of the quantization process, including
    aggregate metrics, details of quantized tensors, and any violations.

  Raises:
    RuntimeError: If `strict` is True and one or more quantized tensors fail
      to meet the specified quality `thresholds`.
  """
  output_path.parent.mkdir(parents=True, exist_ok=True)

  tensors: Dict[str, Any] = {}
  quantized_reports: Dict[str, Dict[str, float]] = {}
  violations: List[Dict[str, Any]] = []
  skipped_tensors: List[str] = []

  skip_lm_head = False
  if drop_tied_lm_head:
    embed_key = "model.embed_tokens.weight"
    lm_head_key = "lm_head.weight"
    if embed_key in state_dict and lm_head_key in state_dict:
      if _tensors_equal(state_dict[embed_key], state_dict[lm_head_key]):
        skip_lm_head = True

  for name in sorted(state_dict.keys()):
    if skip_lm_head and name == "lm_head.weight":
      skipped_tensors.append(name)
      continue

    source_tensor = state_dict[name]
    array = _as_numpy(source_tensor)

    if is_quantizable_linear_weight(name, array):
      quantized, scales, zero_points, dequantized = (
          quantize_per_channel_affine_int8(array)
      )
      metrics = compute_quantization_metrics(array, quantized, dequantized)
      quantized_reports[name] = metrics

      if (
          metrics["cosine_similarity"] < thresholds["cosine_similarity"]
          or metrics["snr_db"] < thresholds["snr_db"]
          or metrics["saturation_rate"] > thresholds["saturation_rate"]
      ):
        violations.append({"tensor": name, "metrics": metrics})

      tensors[name] = np.ascontiguousarray(quantized)
      tensors[f"{name}.scale"] = np.ascontiguousarray(scales.astype(np.float32))
      tensors[f"{name}.zero_point"] = np.ascontiguousarray(
          zero_points.astype(np.int32)
      )
    else:
      tensors[name] = _convert_nonquantized_tensor(
          source_tensor, nonquantized_float_dtype
      )

  _save_state_dict(tensors, output_path, prefer_numpy_io)

  aggregate: Dict[str, float] = {}
  if quantized_reports:
    for key in ["mse", "snr_db", "cosine_similarity", "saturation_rate"]:
      values = [quantized_reports[n][key] for n in quantized_reports]
      aggregate[f"{key}_mean"] = float(np.mean(values))
      aggregate[f"{key}_min"] = float(np.min(values))
      aggregate[f"{key}_max"] = float(np.max(values))

  report = {
      "output_path": str(output_path),
      "strict": strict,
      "thresholds": thresholds,
      "num_source_tensors": len(state_dict),
      "num_export_tensors": len(tensors),
      "num_quantized_tensors": len(quantized_reports),
      "skipped_tensors": skipped_tensors,
      "aggregate_metrics": aggregate,
      "quantized_tensors": quantized_reports,
      "violations": violations,
  }

  report_path = output_path.with_suffix(".quant_report.json")
  with open(report_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, sort_keys=True, allow_nan=True)

  print(f"LiteRT safetensor export written: {output_path}")
  print(f"Quantization report written: {report_path}")
  print(
      "LiteRT quantized tensors: "
      f"{len(quantized_reports)} / {len(state_dict)} source tensors"
  )
  if skipped_tensors:
    print(f"LiteRT skipped tensors: {', '.join(skipped_tensors)}")
  if quantized_reports:
    print(
        "LiteRT aggregate metrics: "
        f"cos={aggregate.get('cosine_similarity_mean', float('nan')):.6f}, "
        f"snr={aggregate.get('snr_db_mean', float('nan')):.2f} dB, "
        f"sat={aggregate.get('saturation_rate_mean', float('nan')):.4f}"
    )

  if violations:
    print(
        f"Warning: {len(violations)} quantized tensors violate thresholds "
        f"(cos >= {thresholds['cosine_similarity']}, "
        f"snr_db >= {thresholds['snr_db']}, "
        f"sat <= {thresholds['saturation_rate']})."
    )
    if strict:
      raise RuntimeError(
          "LiteRT quantization quality check failed in strict mode. "
          f"See report: {report_path}"
      )

  return report


def main() -> None:
  args = parse_args()

  input_path = pathlib.Path(args.input)
  if not input_path.exists():
    raise FileNotFoundError(f"Input safetensors does not exist: {input_path}")

  if args.output:
    output_path = pathlib.Path(args.output)
  else:
    output_path = input_path.with_name(
        f"{input_path.stem}_litert_qpc.safetensors"
    )

  thresholds = {
      "cosine_similarity": float(args.cos_threshold),
      "snr_db": float(args.snr_threshold),
      "saturation_rate": float(args.saturation_threshold),
  }

  state_dict, io_mode = _load_state_dict(
      input_path, prefer_numpy_io=args.prefer_numpy_io
  )
  print(
      f"Loaded {len(state_dict)} tensors from {input_path} using {io_mode} I/O"
      " mode."
  )

  if io_mode == "numpy" and args.nonquantized_float_dtype.upper() == "BF16":
    print(
        "Note: numpy safetensors mode does not preserve BF16 exactly; "
        "non-quantized floats are stored as FP16."
    )

  quantize_checkpoint(
      state_dict=state_dict,
      output_path=output_path,
      strict=bool(args.strict),
      thresholds=thresholds,
      nonquantized_float_dtype=args.nonquantized_float_dtype,
      prefer_numpy_io=args.prefer_numpy_io,
      drop_tied_lm_head=not bool(args.keep_tied_lm_head),
  )


if __name__ == "__main__":
  main()
