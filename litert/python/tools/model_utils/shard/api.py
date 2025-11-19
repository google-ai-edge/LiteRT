# Copyright 2025 Google LLC.
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
"""Sharding APIs."""
import xdsl.ir
from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils.dialect import mlir
from . import _shard

Block = xdsl.ir.Block


def shard(
    x: mlir.SSARankedTensorValue | mu.core.MlirOpBase,
    dim: int,
    parts: int,
    *,
    sharders=None,
    max_shard_depth: int | None = None,
    sharding_allowlist: (
        set[mu.core.MlirOpBase | mlir.SSARankedTensorValue] | None
    ) = None,
    sharding_denylist: (
        set[mu.core.MlirOpBase | mlir.SSARankedTensorValue] | None
    ) = None,
):
  """Shards an operation or tensor value along a specified dimension.

  This function provides a high-level interface for sharding,
  handling the creation of a `BottomUpSharder` instance and invoking
  its `shard` method. It then replaces the original value with
  the result of concatenation of the sharded results. The module where the
  operation is defined will be mutated in-place with the sharded ops.

  Args:
    x: The MLIR operation (tflxx.core.MlirOpBase) or tensor value
      (mlir.SSARankedTensorValue) to shard.
    dim: The dimension along which to shard.
    parts: The number of shards to create.
    sharders: A dictionary mapping operation names (strings) to sharding
      functions. If None, defaults to `self.default_sharders()`.
    max_shard_depth: The maximum depth for recursive sharding. Sharding will
      stop if the depth exceeds this limit.
    sharding_allowlist: If provided, only operations/values in this set will be
      considered for sharding.
    sharding_denylist: If provided, only operations/values in this set will
      never be sharded.

  Returns:
      The concatenated result of the sharded operation or tensor, or the
      original value if no sharding was performed (e.g., due to denylist
      or allowlist constraints).
  """
  op = x
  if isinstance(op, mlir.SSARankedTensorValue):
    op = op.owner

  assert isinstance(op, mu.core.MlirOpBase)

  with mu.core.OpBuildingContext(op):
    results = _shard.BottomUpSharder(
        sharders=sharders,
        max_shard_depth=max_shard_depth,
        sharding_allowlist=sharding_allowlist,
        sharding_denylist=sharding_denylist,
    ).shard(x, dim, parts)
    assert len(results) >= 1
    if len(results) == 1:
      return results[0]

    y = _shard.concat(results, dim)

    mlir.SSARankedTensorValue.get(x).replace_by(y)
    if isinstance(op.parent, Block):
      mu.graph_utils.topological_sort(op.parent)
    return y
