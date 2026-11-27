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
"""Static memory-hygiene checks for .tflite models, for model CI.

Rules:
1. NO_DUPLICATE_CROSS_SUBGRAPH_WEIGHTS (error): inline weights duplicated
   across subgraphs instead of sharing one buffer index.
2. ORPHAN_SUBGRAPH_CHECK (warning): subgraphs unreachable from subgraph 0,
   any SignatureDef, or a subgraph-referencing op.
3. MAX_ACTIVATION_PEAK_BUDGET (heuristic warning): peak live activation bytes
   from static tensor liveness, ignoring in-place reuse and dynamic shapes.

Constants stored past the 2 GB FlatBuffer limit (`Buffer.offset`) and
externalized constants (`Tensor.external_buffer`) count as weights.
"""

from __future__ import annotations

import argparse
from collections.abc import Container, Iterable, Sequence
import dataclasses
import hashlib
import math
import mmap
import pathlib
import sys

from litert.python import schema_py_generated as schema  # pylint:disable=g-direct-tensorflow-import

_MB = 1 << 20

# Approximate element sizes in bytes. Unknown types default to 4 bytes;
# STRING and sub-byte types are rounded to 1 byte, and handle types (RESOURCE,
# VARIANT) have no arena payload (heuristic).
_TENSOR_TYPE_BYTES: dict[int, int] = {
    schema.TensorType.FLOAT32: 4,
    schema.TensorType.FLOAT16: 2,
    schema.TensorType.INT32: 4,
    schema.TensorType.UINT8: 1,
    schema.TensorType.INT64: 8,
    schema.TensorType.STRING: 1,
    schema.TensorType.BOOL: 1,
    schema.TensorType.INT16: 2,
    schema.TensorType.COMPLEX64: 8,
    schema.TensorType.INT8: 1,
    schema.TensorType.FLOAT64: 8,
    schema.TensorType.COMPLEX128: 16,
    schema.TensorType.UINT64: 8,
    schema.TensorType.RESOURCE: 0,
    schema.TensorType.VARIANT: 0,
    schema.TensorType.UINT32: 4,
    schema.TensorType.UINT16: 2,
    schema.TensorType.INT4: 1,
    schema.TensorType.BFLOAT16: 2,
    schema.TensorType.INT2: 1,
    schema.TensorType.UINT4: 1,
    schema.TensorType.FLOAT8_E4M3FN: 1,
    schema.TensorType.FLOAT8_E5M2: 1,
}


def _callee_subgraph_attrs() -> frozenset[str]:
  """Returns OperatorOptionsT attribute names that reference subgraphs."""
  attrs: set[str] = set()
  for name, cls in vars(schema).items():
    if not (name.endswith("OptionsT") and isinstance(cls, type)):
      continue
    for attr in vars(cls()):
      if (
          "Subgraph" in attr
          or attr == "subgraph"
          or attr.endswith("Computations")
      ):
        attrs.add(attr)
  return frozenset(attrs)


# Every operator option field that references subgraphs (IF, WHILE, CALL_ONCE,
# CALL and StableHLO region ops), derived from the schema so new fields are
# followed automatically. This is intentionally a superset of TFLite's C++
# GetReachableSubgraphIndices: STABLEHLO_* ops have no TFLite kernels, but
# their regions are still live for LiteRT compilers.
_CALLEE_SUBGRAPH_ATTRS: frozenset[str] = _callee_subgraph_attrs()

# ("buffer", index) for inline/offset buffers,
# ("external", group, offset, length) for externalized tensors, or
# ("external_id", id) for an external id with no ExternalBuffer entry. The
# latter is still a constant: its bytes never live in the activation arena,
# but its size is unknown.
_WeightKey = tuple[str | int, ...]


@dataclasses.dataclass(frozen=True)
class LintConfig:
  """Rule thresholds."""

  # Heuristic peak activation budget in MB (warning).
  max_activation_mb: float = 30.0
  # Minimum inline buffer size in bytes checked for cross-subgraph duplicates.
  large_buffer_threshold: int = 100 * 1024


@dataclasses.dataclass
class ModelAnalysisResult:
  """Holds lint diagnostics, memory metrics, and status for a model."""

  errors: list[str] = dataclasses.field(default_factory=list)
  warnings: list[str] = dataclasses.field(default_factory=list)
  total_weight_bytes: int = 0
  peak_activation_bytes: int = 0

  def is_passing(self) -> bool:
    return not self.errors


@dataclasses.dataclass(frozen=True)
class _Weight:
  """A constant tensor payload, identified by where its bytes live."""

  key: _WeightKey
  nbytes: int
  data: memoryview[int] | None = None  # Only available for inline buffers.


def _tensor_weight(
    tensor: schema.TensorT,
    buffers: Sequence[schema.BufferT],
    external_buffers: dict[int, schema.ExternalBufferT],
) -> _Weight | None:
  """Returns the constant payload backing `tensor`, or None if not constant."""
  external_id = int(tensor.externalBuffer)
  if external_id:
    ext = external_buffers.get(external_id)
    if ext is None:
      return _Weight(("external_id", external_id), 0)
    length = int(ext.length)
    return _Weight(
        ("external", int(ext.group), int(ext.offset), length), length
    )
  buf_idx = int(tensor.buffer)
  if 0 < buf_idx < len(buffers) and buffers[buf_idx].data is not None:
    data = memoryview(buffers[buf_idx].data)
    if data.nbytes:
      return _Weight(("buffer", buf_idx), data.nbytes, data)
  return None


def _ints(values: Iterable[int] | None) -> list[int]:
  """Object-API int vectors may be None, lists, or numpy arrays."""
  return [] if values is None else [int(v) for v in values]


def _num_elements(shape: Iterable[int] | None) -> int:
  """Static element count; 0 when any dimension is dynamic (negative)."""
  dims = _ints(shape)
  return 0 if any(d < 0 for d in dims) else math.prod(dims)


def _lint_weight_sharing(
    subgraph_weights: Sequence[Iterable[_Weight]],
    large_buffer_threshold: int,
    result: ModelAnalysisResult,
) -> None:
  """NO_DUPLICATE_CROSS_SUBGRAPH_WEIGHTS: flags cross-subgraph duplicates."""
  nbytes: dict[_WeightKey, int] = {}
  # Contents of externalized weights are not available; the externalizer
  # already deduplicates them by digest, so only large inline buffers are
  # candidates, and only hashed when another subgraph has one of equal size.
  candidates: list[tuple[int, _WeightKey, memoryview[int]]] = []
  size_users: dict[int, set[int]] = {}
  for sg_idx, weights in enumerate(subgraph_weights):
    for w in weights:
      if w.nbytes > 0:
        nbytes[w.key] = w.nbytes
      if w.data is not None and w.nbytes >= large_buffer_threshold:
        candidates.append((sg_idx, w.key, w.data))
        size_users.setdefault(w.nbytes, set()).add(sg_idx)
  result.total_weight_bytes = sum(nbytes.values())

  digests: dict[_WeightKey, bytes] = {}
  # Content digest -> subgraph -> buffer keys with that content.
  by_digest: dict[bytes, dict[int, set[_WeightKey]]] = {}
  for sg_idx, key, data in candidates:
    if len(size_users[len(data)]) < 2:
      continue
    if key not in digests:
      digests[key] = hashlib.sha256(data).digest()
    by_digest.setdefault(digests[key], {}).setdefault(sg_idx, set()).add(key)

  # Two subgraphs duplicate a payload only if they share no buffer index
  # holding it. Duplicates within one subgraph are out of scope.
  flagged: set[_WeightKey] = set()
  for sg_keys in by_digest.values():
    sgs = sorted(sg_keys)
    for i, sg_idx in enumerate(sgs):
      key = min(sg_keys[sg_idx])
      if key in flagged:
        continue
      prev_sg = next(
          (p for p in sgs[:i] if not sg_keys[p] & sg_keys[sg_idx]), None
      )
      if prev_sg is None:
        continue
      flagged.add(key)
      result.errors.append(
          "[NO_DUPLICATE_CROSS_SUBGRAPH_WEIGHTS] Subgraph"
          f" {sg_idx} buffer #{key[1]} ({nbytes[key] / _MB:.2f} MB)"
          f" duplicates the contents of buffer #{min(sg_keys[prev_sg])[1]}"
          f" used by subgraph {prev_sg}. Reference one shared buffer index"
          " so the weights are not loaded and repacked once per subgraph."
      )


def _callee_subgraphs(op: schema.OperatorT) -> list[int]:
  """Returns subgraph indices referenced by an OperatorT's options."""
  callees: list[int] = []
  for opts in (op.builtinOptions, op.builtinOptions2):
    if opts is None:
      continue
    for attr in _CALLEE_SUBGRAPH_ATTRS & vars(opts).keys():
      value = getattr(opts, attr)
      # Scalar fields unpack as int; vectors as None, a list or numpy array.
      callees.extend([value] if isinstance(value, int) else _ints(value))
  return callees


def _reachable_subgraphs(model: schema.ModelT) -> set[int]:
  """Subgraphs reachable from subgraph 0 and SignatureDef roots."""
  subgraphs = model.subgraphs or []
  # Interpreter::Invoke() always runs the primary subgraph.
  roots = [0] + [int(sig.subgraphIndex) for sig in model.signatureDefs or []]
  reachable: set[int] = set()
  pending = [idx for idx in roots if 0 <= idx < len(subgraphs)]
  while pending:
    idx = pending.pop()
    if idx in reachable:
      continue
    reachable.add(idx)
    for op in subgraphs[idx].operators or []:
      pending.extend(
          c for c in _callee_subgraphs(op) if 0 <= c < len(subgraphs)
      )
  return reachable


def _lint_orphan_subgraphs(
    num_subgraphs: int, reachable: set[int], result: ModelAnalysisResult
) -> None:
  """ORPHAN_SUBGRAPH_CHECK."""
  orphans = sorted(set(range(num_subgraphs)) - reachable)
  if orphans:
    orphan_str = ", ".join(f"#{idx}" for idx in orphans)
    result.warnings.append(
        f"[ORPHAN_SUBGRAPH_CHECK] Model contains {len(orphans)} unreachable"
        f" subgraph(s) ({orphan_str}) not referenced by any signature or"
        " control-flow operator. Orphan subgraphs waste delegate compilation"
        " time and repacked weight memory on runtimes that delegate all"
        " subgraphs."
    )


def _peak_activation_bytes(
    subgraph: schema.SubGraphT, constants: Container[int]
) -> int:
  """Static liveness sweep over the op schedule (heuristic)."""
  ops = subgraph.operators or []
  tensors = subgraph.tensors or []
  tensor_sizes: dict[int, int] = {}
  for t_idx, tensor in enumerate(tensors):
    if t_idx not in constants:
      tensor_sizes[t_idx] = _num_elements(
          tensor.shape
      ) * _TENSOR_TYPE_BYTES.get(int(tensor.type), 4)

  first_producer: dict[int, int] = {}
  last_producer: dict[int, int] = {}
  last_consumer: dict[int, int] = {}
  for op_idx, op in enumerate(ops):
    for t in _ints(op.inputs):
      if t >= 0:
        last_consumer[t] = op_idx
    for t in _ints(op.outputs):
      if t >= 0:
        first_producer.setdefault(t, op_idx)
        last_producer[t] = op_idx

  sg_inputs = {i for i in _ints(subgraph.inputs) if i >= 0}
  sg_outputs = {o for o in _ints(subgraph.outputs) if o >= 0}
  num_ops = len(ops)
  deltas = [0] * (num_ops + 1)
  for t_idx, size in tensor_sizes.items():
    if not (
        t_idx in first_producer
        or t_idx in last_consumer
        or t_idx in sg_inputs
        or t_idx in sg_outputs
    ):
      continue  # Dead tensor.
    first_op = 0 if t_idx in sg_inputs else first_producer.get(t_idx, 0)
    if tensors[t_idx].isVariable:
      # Variable tensors persist across invocations.
      first_op, last_op = 0, num_ops - 1
    elif t_idx in sg_outputs:
      last_op = num_ops - 1
    else:
      last_op = max(
          last_consumer.get(t_idx, first_op),
          last_producer.get(t_idx, first_op),
      )
    if first_op <= last_op < num_ops:
      deltas[first_op] += size
      deltas[last_op + 1] -= size

  peak = live = 0
  for delta in deltas[:num_ops]:
    live += delta
    peak = max(peak, live)
  return peak


def lint_model(
    model: schema.ModelT, config: LintConfig = LintConfig()
) -> ModelAnalysisResult:
  """Applies all LiteRT lint rules to an unpacked ModelT.

  `Buffer.offset` payloads must already be loaded into `Buffer.data` (see
  `lint_model_file`).

  Args:
    model: Unpacked schema.ModelT object.
    config: Rule thresholds.

  Returns:
    ModelAnalysisResult with recorded errors, warnings, and metrics.
  """
  result = ModelAnalysisResult()
  subgraphs = model.subgraphs or []
  buffers = model.buffers or []
  external_buffers = {int(ext.id): ext for ext in model.externalBuffers or []}

  # Per subgraph: constant tensor index -> weight.
  subgraph_weights: list[dict[int, _Weight]] = []
  for sg in subgraphs:
    weights = (
        _tensor_weight(t, buffers, external_buffers) for t in sg.tensors or []
    )
    subgraph_weights.append(
        {t_idx: w for t_idx, w in enumerate(weights) if w is not None}
    )

  # 1. NO_DUPLICATE_CROSS_SUBGRAPH_WEIGHTS
  _lint_weight_sharing(
      [w.values() for w in subgraph_weights],
      config.large_buffer_threshold,
      result,
  )

  # 2. ORPHAN_SUBGRAPH_CHECK
  reachable = _reachable_subgraphs(model)
  _lint_orphan_subgraphs(len(subgraphs), reachable, result)

  # 3. MAX_ACTIVATION_PEAK_BUDGET, on reachable subgraphs only.
  peak_sg = 0
  for sg_idx in sorted(reachable):
    peak = _peak_activation_bytes(
        subgraphs[sg_idx], subgraph_weights[sg_idx].keys()
    )
    if peak > result.peak_activation_bytes:
      result.peak_activation_bytes, peak_sg = peak, sg_idx
  if result.peak_activation_bytes > int(config.max_activation_mb * _MB):
    result.warnings.append(
        f"[MAX_ACTIVATION_PEAK_BUDGET] (heuristic) Subgraph {peak_sg}:"
        " estimated peak live activations"
        f" {result.peak_activation_bytes / _MB:.2f} MB exceed the"
        f" {config.max_activation_mb:.2f} MB budget. The activation arena is"
        " private dirty memory."
    )
  return result


def lint_model_file(
    model_path: str, config: LintConfig = LintConfig()
) -> ModelAnalysisResult:
  """Loads a .tflite file (including >2 GB offset buffers) and lints it."""
  path = pathlib.Path(model_path)
  if not path.is_file() or not path.stat().st_size:
    return ModelAnalysisResult(
        errors=[f"Model file not found or empty: {model_path}"]
    )

  # Map the file read-only so large weights are paged in from disk on demand
  # rather than read into anonymous memory. The mapping is released once the
  # unpacked model and its buffer views are garbage collected.
  with path.open("rb") as f:
    view = memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
  model = schema.ModelT.InitFromPackedBuf(view, 0)
  # >2 GB models append buffers after the FlatBuffer. Offsets 0 and 1 are
  # sentinels for "no appended data" (see tflite::Verifier).
  bad_offsets = []
  for buf_idx, buffer in enumerate(model.buffers or []):
    offset, size = int(buffer.offset), int(buffer.size)
    if offset <= 1:
      continue
    if offset + size > len(view):
      bad_offsets.append(
          f"Buffer #{buf_idx} [{offset}, {offset + size}) extends past the"
          f" end of the file ({len(view)} bytes)."
      )
      continue
    buffer.data = view[offset : offset + size]
  result = lint_model(model, config)
  result.errors[:0] = bad_offsets
  return result


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point; returns the process exit code."""
  defaults = LintConfig()
  parser = argparse.ArgumentParser(
      description=(
          "LiteRT FlatBuffer memory linter. Duplicate cross-subgraph weights"
          " and out-of-bounds buffers are errors; other rules emit warnings."
      )
  )
  parser.add_argument(
      "--model", required=True, help="Path to .tflite model file"
  )
  parser.add_argument(
      "--max_activation_mb",
      type=float,
      default=defaults.max_activation_mb,
      help=(
          "Heuristic peak activation budget in MB, from static liveness"
          " (warning only)"
      ),
  )
  parser.add_argument(
      "--large_buffer_kb",
      type=int,
      default=defaults.large_buffer_threshold // 1024,
      help="Minimum inline buffer size in KB checked for duplication",
  )
  args = parser.parse_args(argv)

  result = lint_model_file(
      args.model,
      LintConfig(
          max_activation_mb=args.max_activation_mb,
          large_buffer_threshold=args.large_buffer_kb * 1024,
      ),
  )

  print(f"[litert_lint] Auditing {args.model}...")
  for err in result.errors:
    print(f"ERROR: {err}", file=sys.stderr)
  for warn in result.warnings:
    print(f"WARNING: {warn}", file=sys.stderr)
  print(
      f"[litert_lint] weights={result.total_weight_bytes / _MB:.2f} MB"
      f" peak_activations={result.peak_activation_bytes / _MB:.2f} MB"
  )
  if not result.is_passing():
    return 1
  print("[litert_lint] SUCCESS: no errors.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
