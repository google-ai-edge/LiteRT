# Copyright 2026 Google LLC.
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
"""Test oracle for PR #11544 protocols; never used by the evaluator."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def resize_bilinear(values: np.ndarray, size: Sequence[int]) -> np.ndarray:
  """Resizes HW or CHW values with half-pixel, edge-clamped bilinear sampling.

  Args:
    values: A 2D (HW) or 3D (CHW) array of values to resize.
    size: Target `(height, width)` dimensions.

  Returns:
    A float32 array resized to `(*values.shape[:-2], height, width)`, or a copy
    of `values` if the spatial dimensions already match `size`.

  Raises:
    ValueError: If `values` is not 2D or 3D, or any spatial dimension is
      non-positive.
  """
  values = np.asarray(values)
  height, width = size
  if values.ndim not in (2, 3) or min(height, width, *values.shape[-2:]) <= 0:
    raise ValueError('Expected nonempty HW/CHW values and positive output size')
  if tuple(values.shape[-2:]) == tuple(size):
    return values.copy()
  y = np.maximum((np.arange(height) + 0.5) * values.shape[-2] / height - 0.5, 0)
  x = np.maximum((np.arange(width) + 0.5) * values.shape[-1] / width - 0.5, 0)
  y0, x0 = y.astype(int), x.astype(int)
  y1 = np.minimum(y0 + 1, values.shape[-2] - 1)
  x1 = np.minimum(x0 + 1, values.shape[-1] - 1)
  # Process channels separately to avoid several full-resolution logit copies.
  output = np.empty((*values.shape[:-2], height, width), dtype=np.float32)
  planes = values[None] if values.ndim == 2 else values
  targets = output[None] if output.ndim == 2 else output
  for src, dst in zip(planes, targets, strict=True):
    top = src[y0[:, None], x0] * (1 - (x - x0)) + src[y0[:, None], x1] * (
        x - x0
    )
    bottom = src[y1[:, None], x0] * (1 - (x - x0)) + src[y1[:, None], x1] * (
        x - x0
    )
    dst[:] = top * (1 - (y - y0))[:, None] + bottom * (y - y0)[:, None]
  return output


def require_finite(values: np.ndarray) -> None:
  """Raises ValueError if any element in `values` is NaN or infinite."""
  if not np.isfinite(values).all():
    raise ValueError('Predictions contain NaN or infinity')


class SegmentationMetrics:
  """Global confusion matrix; absent classes are excluded from mean IoU.

  Attributes:
    classes: Number of semantic segmentation classes.
    ignore_label: Label value excluded from confusion matrix accumulation.
    confusion: Accumulated `[classes, classes]` integer confusion matrix.
    images: Number of accumulated images.
  """

  def __init__(self, classes: int = 150, ignore_label: int = 255) -> None:
    """Initializes the instance.

    Args:
      classes: Positive number of segmentation classes.
      ignore_label: Target label value to ignore during scoring.

    Raises:
      ValueError: If `classes` is not positive.
    """
    if classes <= 0:
      raise ValueError('classes must be positive')
    self.classes = classes
    self.ignore_label = ignore_label
    self.confusion = np.zeros((classes, classes), dtype=np.int64)
    self.images = 0

  def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
    """Accumulates one image's predictions and targets into the matrix.

    Args:
      prediction: CHW float logits or HW integer class labels.
      target: HW integer ground-truth class labels.

    Raises:
      ValueError: If shapes, dtypes, finiteness, or label ranges are invalid.
    """
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.ndim != 2 or not np.issubdtype(target.dtype, np.integer):
      raise ValueError('Segmentation targets must be integer HW arrays')
    require_finite(prediction)
    if prediction.ndim == 3:
      if prediction.shape[0] != self.classes:
        raise ValueError('Logit channel count differs from configured classes')
      # Keep memory proportional to image size, not classes * image size.
      # Strict comparison preserves argmax's lowest-index tie breaking.
      labels = np.zeros(target.shape, dtype=np.int64)
      best = np.full(target.shape, -np.inf, dtype=np.float32)
      for index, plane in enumerate(prediction):
        resized = resize_bilinear(plane, target.shape)
        better = resized > best
        labels[better] = index
        best[better] = resized[better]
      prediction = labels
    if prediction.shape != target.shape or not np.issubdtype(
        prediction.dtype, np.integer
    ):
      raise ValueError('Expected CHW logits or matching integer HW labels')
    valid = target != self.ignore_label
    if np.any((target[valid] < 0) | (target[valid] >= self.classes)):
      raise ValueError(
          'Target labels out of range; configure label mapping first'
      )
    if np.any((prediction[valid] < 0) | (prediction[valid] >= self.classes)):
      raise ValueError('Predicted labels out of range')
    self.confusion += np.bincount(
        self.classes * target[valid].astype(np.int64)
        + prediction[valid].astype(np.int64),
        minlength=self.classes**2,
    ).reshape(self.classes, self.classes)
    self.images += 1

  def result(self) -> dict[str, Any]:
    """Returns mIoU and confusion matrix metrics across accumulated images.

    Returns:
      A dictionary containing `images`, `valid_pixels`, `miou` (`None` if no
      valid pixels were scored), `evaluated_classes`, and `confusion`.
    """
    intersection = np.diag(self.confusion)
    union = self.confusion.sum(0) + self.confusion.sum(1) - intersection
    valid = union > 0
    return {
        'images': self.images,
        'valid_pixels': int(self.confusion.sum()),
        'miou': (
            float(np.mean(intersection[valid] / union[valid]))
            if valid.any()
            else None
        ),
        'evaluated_classes': int(valid.sum()),
        'confusion': self.confusion.tolist(),
    }


class DepthMetrics:
  """Equal weight per image, valid target pixels in (min_depth, max_depth].

  Attributes:
    min_depth: Exclusive lower depth bound in meters.
    max_depth: Inclusive upper depth bound in meters.
    rows: Per-image depth metric dictionaries for non-empty images.
    empty_images: Count of images with no valid target depth pixels.
  """

  def __init__(self, min_depth: float = 0.001, max_depth: float = 10.0) -> None:
    """Initializes the instance.

    Args:
      min_depth: Exclusive lower bound on valid depth values.
      max_depth: Inclusive upper bound on valid depth values.

    Raises:
      ValueError: If `0 < min_depth < max_depth` does not hold.
    """
    if not 0 < min_depth < max_depth:
      raise ValueError('Expected 0 < min_depth < max_depth')
    self.min_depth, self.max_depth = min_depth, max_depth
    self.rows: list[dict[str, float | int]] = []
    self.empty_images = 0

  def update(
      self,
      prediction: np.ndarray,
      target: np.ndarray,
      mask: np.ndarray | None = None,
  ) -> None:
    """Records per-image depth metrics over valid target pixels.

    Args:
      prediction: 2D (HW) predicted depth array.
      target: 2D (HW) ground-truth depth array.
      mask: Optional 2D boolean mask of valid target pixels, or `None` to score
        all finite target pixels in `(min_depth, max_depth]`.

    Raises:
      ValueError: If `prediction` or `target` is not 2D, `prediction` contains
        non-finite values, or `mask` does not match `target.shape`.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction)
    if prediction.ndim != 2 or target.ndim != 2:
      raise ValueError('Depth arrays must be HW')
    require_finite(prediction)
    prediction = resize_bilinear(prediction, target.shape).astype(np.float64)
    valid = (
        np.isfinite(target)
        & (target > self.min_depth)
        & (target <= self.max_depth)
    )
    if mask is not None:
      if np.shape(mask) != target.shape:
        raise ValueError('Depth mask shape mismatch')
      valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
      self.empty_images += 1
      return
    pred = np.clip(prediction[valid], self.min_depth, self.max_depth)
    truth = target[valid]
    self.rows.append({
        'valid_pixels': int(valid.sum()),
        'abs_rel': float(np.mean(np.abs(pred - truth) / truth)),
        'rmse': float(np.sqrt(np.mean((pred - truth) ** 2))),
        'delta1': float(np.mean(np.maximum(pred / truth, truth / pred) < 1.25)),
    })

  def result(self) -> dict[str, Any]:
    """Returns mean depth metrics across all non-empty images.

    Returns:
      A dictionary containing `images`, `empty_images`, `valid_pixels`, and
      per-image means `abs_rel`, `rmse`, and `delta1` (each `None` if no
      non-empty images were scored).
    """
    return {
        'images': len(self.rows),
        'empty_images': self.empty_images,
        'valid_pixels': sum(int(r['valid_pixels']) for r in self.rows),
        **{
            k: float(np.mean([r[k] for r in self.rows])) if self.rows else None
            for k in ('abs_rel', 'rmse', 'delta1')
        },
    }


class NormalMetrics:
  """Pixel-weighted oriented angular errors (deg); zero predictions score 90°.

  Attributes:
    errors: List of 1D float32 arrays of per-pixel angular errors in degrees.
    images: Number of non-empty images scored.
    empty_images: Number of images with no valid target normals.
  """

  def __init__(self) -> None:
    """Initializes the instance."""
    self.errors: list[np.ndarray] = []
    self.images = 0
    self.empty_images = 0

  def update(
      self,
      prediction: np.ndarray,
      target: np.ndarray,
      mask: np.ndarray | None = None,
  ) -> None:
    """Records per-pixel angular errors in degrees over valid target normals.

    Args:
      prediction: `[3, H, W]` predicted surface normal array.
      target: `[3, H, W]` ground-truth surface normal array.
      mask: Optional `[H, W]` boolean mask of valid target pixels, or `None` to
        score all finite non-zero target normal vectors.

    Raises:
      ValueError: If `prediction` or `target` does not have shape `[3, H, W]`,
        `prediction` contains non-finite values, or `mask` does not match
        `target.shape[-2:]`.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction)
    if (
        target.ndim != 3
        or prediction.ndim != 3
        or target.shape[0] != 3
        or prediction.shape[0] != 3
    ):
      raise ValueError('Normals must have shape [3,H,W]')
    require_finite(prediction)
    prediction = resize_bilinear(prediction, target.shape[-2:]).astype(
        np.float64
    )
    target_norm = np.linalg.norm(target, axis=0)
    pred_norm = np.linalg.norm(prediction, axis=0)
    valid = np.isfinite(target).all(0) & (target_norm > 1e-12)
    if mask is not None:
      if np.shape(mask) != target.shape[-2:]:
        raise ValueError('Normal mask shape mismatch')
      valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
      self.empty_images += 1
      return
    dot = np.sum(prediction[:, valid] * target[:, valid], axis=0)
    cosine = dot / np.maximum(pred_norm[valid] * target_norm[valid], 1e-30)
    self.errors.append(
        np.degrees(np.arccos(np.clip(cosine, -1, 1))).astype(np.float32)
    )
    self.images += 1

  def result(self) -> dict[str, Any]:
    """Returns aggregated angular error statistics across all scored pixels.

    Returns:
      A dictionary containing `images`, `empty_images`, `valid_pixels`,
      `mean_degrees`, `median_degrees`, and `within_*_degrees` thresholds (each
      metric is `None` if no valid pixels were scored).
    """
    errors = np.concatenate(self.errors) if self.errors else np.empty(0)
    result: dict[str, Any] = {
        'images': self.images,
        'empty_images': self.empty_images,
        'valid_pixels': int(errors.size),
        'mean_degrees': float(errors.mean()) if errors.size else None,
        'median_degrees': float(np.median(errors)) if errors.size else None,
    }
    result.update({
        f'within_{angle}_degrees': (
            float(np.mean(errors < angle)) if errors.size else None
        )
        for angle in (11.25, 22.5, 30)
    })
    return result
