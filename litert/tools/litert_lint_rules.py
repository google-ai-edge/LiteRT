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
"""LiteRT / TFLite Static FlatBuffer Linter (`litert_lint_rules`).

Enforces shift-left memory and architecture hygiene on .tflite models during
Bazel build and CI presubmits.

Rules Enforced:
1. NO_UNSHARED_MULTISIG_WEIGHTS:
   Flags multi-signature models where constant weights across subgraphs are
   duplicated into separate buffer indices rather than shared.
2. MAX_ACTIVATION_PEAK_BUDGET:
   Computes static tensor liveness intervals to calculate peak simultaneously
   live activation bytes, flagging models that exceed the allowed activation
   working set.
3. PARTITION_FRAGMENTATION_CHECK:
   Warns when non-delegated fallback operations break execution into more than
   allowed delegate partitions.
4. ORPHAN_SUBGRAPH_CHECK:
   Warns when a multi-subgraph model contains dead/orphan subgraphs that are
   not reachable from any declared signature or control-flow operator.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from litert.python import schema_py_generated as schema  # pylint:disable=g-direct-tensorflow-import

_DEFAULT_LARGE_BUFFER_THRESHOLD = 100 * 1024  # 100 KB
_DEFAULT_MAX_ACTIVATION_MB = 30.0
_DEFAULT_MAX_PARTITIONS = 3

# Data type size mapping in bytes for TFLite schema types
_TENSOR_TYPE_BYTES: Dict[int, int] = {
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
    schema.TensorType.UINT32: 4,
    schema.TensorType.UINT16: 2,
    schema.TensorType.INT4: 1,
}

# Operations generally not supported by standard mobile delegates (e.g. XNNPACK)
_DELEGATE_UNSUPPORTED_BUILTIN_OPS: Set[int] = {
    schema.BuiltinOperator.CUSTOM,
    schema.BuiltinOperator.DELEGATE,
    schema.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR,
    schema.BuiltinOperator.REVERSE_SEQUENCE,
    schema.BuiltinOperator.DYNAMIC_UPDATE_SLICE,
}


@dataclasses.dataclass
class ModelAnalysisResult:
  """Holds lint diagnostics, memory metrics, and status for a model."""

  model_path: str = ""
  errors: List[str] = dataclasses.field(default_factory=list)
  warnings: List[str] = dataclasses.field(default_factory=list)
  subgraphs_count: int = 0
  orphan_subgraphs_count: int = 0
  total_weight_bytes: int = 0
  shared_weight_bytes: int = 0
  peak_activation_bytes: int = 0
  partition_count: int = 1

  def is_passing(self) -> bool:
    return not self.errors


def _get_buffer_size(buffer: Any) -> int:
  """Returns the byte length of a FlatBuffer BufferT or bytes-like object."""
  if buffer is None:
    return 0
  data = (
      buffer
      if isinstance(buffer, (bytes, bytearray, memoryview))
      else getattr(buffer, "data", None)
  )
  return len(data) if data is not None else 0


def _get_buffer_bytes(buffer: Any) -> Optional[bytes]:
  """Extracts raw bytes from a FlatBuffer BufferT object or bytes/bytearray."""
  if buffer is None:
    return None
  if isinstance(buffer, (bytes, bytearray)):
    return bytes(buffer)
  data = getattr(buffer, "data", None)
  if data is None:
    return None
  try:
    return bytes(memoryview(data))
  except TypeError:
    return bytes(data)


def lint_multisig_weight_sharing(
    subgraphs_tensors: List[List[Dict[str, Any]]],
    buffers: List[Any],
    result: ModelAnalysisResult,
    large_buffer_threshold: int = _DEFAULT_LARGE_BUFFER_THRESHOLD,
) -> None:
  """Checks whether multi-subgraph models share constant weight buffers.

  Args:
    subgraphs_tensors: Tensors grouped by subgraph.
    buffers: List of model FlatBuffer buffers.
    result: Analysis result object to append errors to.
    large_buffer_threshold: Byte threshold above which buffers are analyzed.
  """
  buffer_size_by_idx: Dict[int, int] = {}
  buffer_subgraphs: Dict[int, Set[int]] = {}
  digest_by_buf_idx: Dict[int, str] = {}
  seen_buffer_hashes: Dict[str, Tuple[int, int]] = {}
  flagged_duplicate_buffers: Set[int] = set()
  check_duplicates = len(subgraphs_tensors) > 1

  for sg_idx, tensors in enumerate(subgraphs_tensors):
    sg_buf_indices = sorted({
        t.get("buffer", 0)
        for t in tensors
        if 0 < t.get("buffer", 0) < len(buffers)
    })
    for buf_idx in sg_buf_indices:
      if buf_idx not in buffer_size_by_idx:
        buffer_size_by_idx[buf_idx] = _get_buffer_size(buffers[buf_idx])
      buf_size = buffer_size_by_idx[buf_idx]
      if buf_size <= 0:
        continue
      buffer_subgraphs.setdefault(buf_idx, set()).add(sg_idx)
      if not check_duplicates or buf_size < large_buffer_threshold:
        continue
      if buf_idx not in digest_by_buf_idx:
        buf_data = _get_buffer_bytes(buffers[buf_idx])
        if buf_data is None:
          continue
        digest_by_buf_idx[buf_idx] = hashlib.sha256(buf_data).hexdigest()
      digest = digest_by_buf_idx[buf_idx]
      if digest in seen_buffer_hashes:
        prev_sg, prev_buf = seen_buffer_hashes[digest]
        if prev_buf != buf_idx and buf_idx not in flagged_duplicate_buffers:
          flagged_duplicate_buffers.add(buf_idx)
          buf_mb = buf_size / (1024 * 1024)
          result.errors.append(
              "[NO_UNSHARED_MULTISIG_WEIGHTS] Duplicate unshared weight buffer"
              f" detected! Subgraph {sg_idx} (buffer #{buf_idx}, {buf_mb:.2f}"
              f" MB) duplicates Subgraph {prev_sg} (buffer #{prev_buf}). Model"
              " must share buffer indices to prevent multi-fold weight"
              " repacking in memory!"
          )
      else:
        seen_buffer_hashes[digest] = (sg_idx, buf_idx)

  result.total_weight_bytes = sum(
      buffer_size_by_idx[idx] for idx in buffer_subgraphs
  )
  result.shared_weight_bytes = sum(
      buffer_size_by_idx[idx]
      for idx, sgs in buffer_subgraphs.items()
      if len(sgs) > 1
  )


def lint_activation_arena_budget(
    subgraph_ops: List[Dict[str, Any]],
    subgraph_tensors: List[Dict[str, Any]],
    max_budget_bytes: int,
    result: ModelAnalysisResult,
    subgraph_inputs: Optional[Set[int]] = None,
    subgraph_outputs: Optional[Set[int]] = None,
) -> None:
  """Performs static tensor liveness analysis to compute peak activation arena.

  Args:
    subgraph_ops: Operators within the target subgraph.
    subgraph_tensors: Tensor metadata within the target subgraph.
    max_budget_bytes: Maximum allowed live activation bytes.
    result: Analysis result object to append errors to.
    subgraph_inputs: Optional set of subgraph input tensor indices.
    subgraph_outputs: Optional set of subgraph output tensor indices.
  """
  tensor_sizes: Dict[int, int] = {}
  producer_op: Dict[int, int] = {}
  last_consumer_op: Dict[int, int] = {}

  for t_idx, tensor in enumerate(subgraph_tensors):
    if tensor.get("is_constant", tensor.get("buffer", 0) != 0):
      # Constant weight tensor; resides in file or repacked heap
      continue
    shape = tensor.get("shape") or []
    dtype = tensor.get("type", schema.TensorType.FLOAT32)
    dtype_size = tensor.get("dtype_size", _TENSOR_TYPE_BYTES.get(dtype, 4))
    tensor_sizes[t_idx] = math.prod(max(1, dim) for dim in shape) * dtype_size

  for op_idx, op in enumerate(subgraph_ops):
    for in_t in op.get("inputs") or []:
      if in_t >= 0:
        last_consumer_op[in_t] = op_idx
    for out_t in op.get("outputs") or []:
      if out_t >= 0:
        producer_op[out_t] = op_idx

  sg_inputs = subgraph_inputs or set()
  sg_outputs = subgraph_outputs or set()
  num_ops = len(subgraph_ops)
  deltas = [0] * (num_ops + 1)
  for t_idx, size in tensor_sizes.items():
    is_produced = t_idx in producer_op
    has_consumer = t_idx in last_consumer_op
    if (
        num_ops > 0
        and not is_produced
        and not has_consumer
        and t_idx not in sg_inputs
        and t_idx not in sg_outputs
    ):
      continue
    first_op = producer_op.get(t_idx, 0)
    if t_idx in sg_outputs and num_ops > 0:
      last_op = num_ops - 1
    else:
      last_op = last_consumer_op.get(t_idx, first_op)
    if first_op <= last_op < num_ops:
      deltas[first_op] += size
      deltas[last_op + 1] -= size

  # Sweep through op schedule to find peak simultaneously live bytes
  peak_bytes = 0
  current_live_bytes = 0
  for op_idx in range(num_ops):
    current_live_bytes += deltas[op_idx]
    if current_live_bytes > peak_bytes:
      peak_bytes = current_live_bytes

  result.peak_activation_bytes = max(result.peak_activation_bytes, peak_bytes)
  peak_mb = peak_bytes / (1024 * 1024)
  budget_mb = max_budget_bytes / (1024 * 1024)

  if peak_bytes > max_budget_bytes:
    result.errors.append(
        "[MAX_ACTIVATION_PEAK_BUDGET] Peak live activation arena exceeds"
        f" budget: {peak_mb:.2f} MB > allowed {budget_mb:.2f} MB. High"
        " activation footprint causes excessive Private_Dirty memory bloat."
    )


def lint_partition_fragmentation(
    subgraph_ops: List[Dict[str, Any]],
    max_partitions: int,
    result: ModelAnalysisResult,
) -> None:
  """Checks for excessive delegation fallback partitions.

  Args:
    subgraph_ops: Operators within the target subgraph.
    max_partitions: Maximum allowed fallback partition segments.
    result: Analysis result object to append warnings to.
  """
  partitions = 1
  currently_in_delegate = True

  for op in subgraph_ops:
    builtin_code = op.get("builtin_code")
    is_custom = op.get("is_custom", False)
    supported = not is_custom and (
        builtin_code not in _DELEGATE_UNSUPPORTED_BUILTIN_OPS
    )
    if supported != currently_in_delegate:
      partitions += 1
      currently_in_delegate = supported

  result.partition_count = max(result.partition_count, partitions)
  if partitions > max_partitions:
    result.warnings.append(
        "[PARTITION_FRAGMENTATION_CHECK] Model execution graph splits into"
        f" {partitions} partitions (allowed threshold: {max_partitions})."
        " Fragmentation prevents workspace reuse, forces extra intermediate"
        " tensor copies, and prevents kernel fusion."
    )


_CALLEE_SUBGRAPH_ATTRS = (
    "thenSubgraphIndex",
    "elseSubgraphIndex",
    "condSubgraphIndex",
    "bodySubgraphIndex",
    "initSubgraphIndex",
    "subgraph",
    "decompositionSubgraphIndex",
    "updateComputationSubgraphIndex",
    "computationSubgraphIndex",
    "comparatorSubgraphIndex",
)
_CALLEE_SUBGRAPH_LIST_ATTRS = ("subgraphIndices", "calledSubgraphs")


def _extract_callee_subgraphs(op: Any) -> List[int]:
  """Extracts callee subgraph indices referenced by a FlatBuffer OperatorT."""
  candidates: List[Any] = []
  for opts in (
      getattr(op, "builtinOptions", None),
      getattr(op, "builtinOptions2", None),
  ):
    if opts is None:
      continue
    candidates.extend(
        getattr(opts, attr, None) for attr in _CALLEE_SUBGRAPH_ATTRS
    )
    for list_attr in _CALLEE_SUBGRAPH_LIST_ATTRS:
      candidates.extend(getattr(opts, list_attr, None) or ())
  return [
      v
      for v in candidates
      if isinstance(v, int) and not isinstance(v, bool) and v >= 0
  ]


def lint_orphan_subgraphs(
    num_subgraphs: int,
    root_subgraphs: Set[int],
    subgraph_callees: List[List[int]],
    result: ModelAnalysisResult,
) -> None:
  """Warns when subgraphs are unreachable from any declared signature or root.

  Args:
    num_subgraphs: Total number of subgraphs in the model.
    root_subgraphs: Entry-point subgraph indices from signatureDefs (or {0}).
    subgraph_callees: Callee subgraph indices referenced by each subgraph.
    result: Analysis result object to append warnings to.
  """
  if num_subgraphs <= 1 or not root_subgraphs:
    return
  reachable: Set[int] = set()
  pending = [idx for idx in root_subgraphs if 0 <= idx < num_subgraphs]
  while pending:
    curr = pending.pop()
    if curr in reachable:
      continue
    reachable.add(curr)
    if 0 <= curr < len(subgraph_callees):
      for callee in subgraph_callees[curr]:
        if 0 <= callee < num_subgraphs and callee not in reachable:
          pending.append(callee)

  orphans = sorted(set(range(num_subgraphs)) - reachable)
  result.orphan_subgraphs_count = len(orphans)
  if orphans:
    orphan_str = ", ".join(f"#{idx}" for idx in orphans)
    result.warnings.append(
        f"[ORPHAN_SUBGRAPH_CHECK] Model contains {len(orphans)} unreachable"
        f" subgraph(s) ({orphan_str}) not referenced by any signature or"
        " control-flow operator. Orphan subgraphs waste delegate compilation"
        " time and repacked weight memory on runtimes that delegate all"
        " subgraphs."
    )


def lint_model(
    model: schema.ModelT,
    max_activation_mb: float = _DEFAULT_MAX_ACTIVATION_MB,
    max_partitions: int = _DEFAULT_MAX_PARTITIONS,
    large_buffer_threshold: int = _DEFAULT_LARGE_BUFFER_THRESHOLD,
    model_path: str = "",
) -> ModelAnalysisResult:
  """Applies all LiteRT memory and architecture lint rules to an unpacked ModelT object.

  Args:
    model: Unpacked schema.ModelT object.
    max_activation_mb: Maximum allowed activation memory arena in MB.
    max_partitions: Maximum allowed delegate fallback partitions.
    large_buffer_threshold: Minimum buffer size in bytes to flag duplication.
    model_path: Optional path for reporting.

  Returns:
    ModelAnalysisResult with recorded errors, warnings, and metrics.
  """
  result = ModelAnalysisResult(model_path)
  result.subgraphs_count = len(model.subgraphs or [])

  # 1. NO_UNSHARED_MULTISIG_WEIGHTS
  buffers = model.buffers or []
  subgraphs_tensors: List[List[Dict[str, Any]]] = []
  for sg in model.subgraphs or []:
    tensors_info = []
    for t in sg.tensors or []:
      buf_idx = int(t.buffer)
      tensors_info.append({
          "buffer": buf_idx,
          "is_constant": (
              0 < buf_idx < len(buffers)
              and _get_buffer_size(buffers[buf_idx]) > 0
          ),
          "shape": [int(d) for d in t.shape] if t.shape is not None else [],
          "type": int(t.type),
      })
    subgraphs_tensors.append(tensors_info)

  lint_multisig_weight_sharing(
      subgraphs_tensors,
      buffers,
      result,
      large_buffer_threshold=large_buffer_threshold,
  )

  # Map operator codes to builtincodes
  op_codes = []
  for code in model.operatorCodes or []:
    builtin = int(code.builtinCode)
    if builtin == 0 and int(code.deprecatedBuiltinCode) != 0:
      builtin = int(code.deprecatedBuiltinCode)
    is_custom = builtin == schema.BuiltinOperator.CUSTOM or (
        code.customCode is not None and len(code.customCode) > 0
    )
    op_codes.append({"builtin_code": builtin, "is_custom": is_custom})

  # 2. MAX_ACTIVATION_PEAK_BUDGET, 3. PARTITION_FRAGMENTATION_CHECK, and
  # 4. ORPHAN_SUBGRAPH_CHECK
  max_budget_bytes = int(max_activation_mb * 1024 * 1024)
  subgraph_callees: List[List[int]] = []

  for sg_idx, sg in enumerate(model.subgraphs or []):
    sg_ops = []
    callees: List[int] = []
    for op in sg.operators or []:
      opcode_info = (
          op_codes[op.opcodeIndex]
          if 0 <= op.opcodeIndex < len(op_codes)
          else {"builtin_code": 0, "is_custom": False}
      )
      sg_ops.append({
          "inputs": (
              [int(i) for i in op.inputs] if op.inputs is not None else []
          ),
          "outputs": (
              [int(o) for o in op.outputs] if op.outputs is not None else []
          ),
          "builtin_code": opcode_info["builtin_code"],
          "is_custom": opcode_info["is_custom"],
      })
      callees.extend(_extract_callee_subgraphs(op))
    subgraph_callees.append(callees)

    sg_tensors = subgraphs_tensors[sg_idx]
    sg_inputs = {int(i) for i in sg.inputs or [] if int(i) >= 0}
    sg_outputs = {int(o) for o in sg.outputs or [] if int(o) >= 0}
    lint_activation_arena_budget(
        sg_ops,
        sg_tensors,
        max_budget_bytes,
        result,
        subgraph_inputs=sg_inputs,
        subgraph_outputs=sg_outputs,
    )
    lint_partition_fragmentation(sg_ops, max_partitions, result)

  sig_defs = getattr(model, "signatureDefs", None)
  if sig_defs:
    root_subgraphs = {
        int(sig.subgraphIndex)
        for sig in sig_defs
        if getattr(sig, "subgraphIndex", None) is not None
    }
  else:
    root_subgraphs = {0} if result.subgraphs_count > 0 else set()
  lint_orphan_subgraphs(
      result.subgraphs_count, root_subgraphs, subgraph_callees, result
  )

  return result


def lint_model_file(
    model_path: str,
    max_activation_mb: float = _DEFAULT_MAX_ACTIVATION_MB,
    max_partitions: int = _DEFAULT_MAX_PARTITIONS,
    large_buffer_threshold: int = _DEFAULT_LARGE_BUFFER_THRESHOLD,
) -> ModelAnalysisResult:
  """Loads and audits a .tflite model file against LiteRT lint rules.

  Args:
    model_path: Path to the .tflite model file.
    max_activation_mb: Maximum allowed activation memory arena in MB.
    max_partitions: Maximum allowed delegate fallback partitions.
    large_buffer_threshold: Minimum buffer size in bytes to flag duplication.

  Returns:
    ModelAnalysisResult with recorded errors, warnings, and metrics.
  """
  path = pathlib.Path(model_path)
  if not path.is_file():
    result = ModelAnalysisResult(model_path)
    result.errors.append(f"Model file not found: {model_path}")
    return result

  input_bytes = path.read_bytes()
  model = schema.ModelT.InitFromPackedBuf(input_bytes, 0)
  return lint_model(
      model,
      max_activation_mb=max_activation_mb,
      max_partitions=max_partitions,
      large_buffer_threshold=large_buffer_threshold,
      model_path=model_path,
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="LiteRT FlatBuffer Memory Linter"
  )
  parser.add_argument(
      "--model", required=True, help="Path to .tflite model file"
  )
  parser.add_argument(
      "--max_activation_mb",
      type=float,
      default=_DEFAULT_MAX_ACTIVATION_MB,
      help="Maximum allowed peak activation arena in MB",
  )
  parser.add_argument(
      "--max_partitions",
      type=int,
      default=_DEFAULT_MAX_PARTITIONS,
      help="Maximum allowed delegate fallback partitions",
  )
  args = parser.parse_args()

  result = lint_model_file(
      args.model,
      max_activation_mb=args.max_activation_mb,
      max_partitions=args.max_partitions,
  )

  print(f"[litert_lint] Auditing {args.model}...")
  for err in result.errors:
    print(f"ERROR: {err}", file=sys.stderr)
  for warn in result.warnings:
    print(f"WARNING: {warn}", file=sys.stderr)
  if result.is_passing():
    print(
        "[litert_lint] SUCCESS: All memory and architecture assertions passed."
    )
    sys.exit(0)
  sys.exit(1)


if __name__ == "__main__":
  main()
