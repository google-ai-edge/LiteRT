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
"""LiteRT ModelUtils is the Python toolkit for creating, inspecting, and rewriting TFLite/LiteRT flatbuffer models."""

import sys
from typing import cast
import xdsl

RECURSION_LIMIT = 100000
# Use higher recursion limit for resursive graph copy.
if sys.getrecursionlimit() < RECURSION_LIMIT:
  sys.setrecursionlimit(RECURSION_LIMIT)

# pylint: disable=g-import-not-at-top
from . import core
from . import dialect
from . import graph_utils
from . import match
from . import model_builder
from . import passes
from . import shard
from . import signature_builder
from . import transform
# pylint: enable=g-import-not-at-top


# Shortcuts
MatchingContext = match.MatchingContext
OpBuildingContext = core.OpBuildingContext
RewritePatternPassBase = core.RewritePatternPassBase
SignatureBuilder = signature_builder.SignatureBuilder

read_flatbuffer = transform.read_flatbuffer
read_mlir = transform.read_mlir
convert_to_mlir = transform.convert_to_mlir
write_flatbuffer = transform.write_flatbuffer
get_ir_context = transform.get_ir_context

SSAValue = xdsl.irdl.SSAValue
SSARankedTensorValue = dialect.mlir.SSARankedTensorValue
TensorValue = SSARankedTensorValue  # Alias of SSARankedTensorValue


def visualize(
    obj,
    host="localhost",
    port=8080,
    colab_height=850,
    reuse_server: bool = False,
    reuse_server_host: str = "localhost",
    reuse_server_port: int | None = None,
) -> None:
  """Visualize the ModelUtils Python object in Model Explorer.

  Args:
    obj: The ModelUtils Python object to visualize.
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
    colab_height: The height of the embedded iFrame when running in colab.
    reuse_server: Whether to reuse the current server/browser tab(s) to
      visualize.
    reuse_server_host: the host of the server to reuse. Default to localhost.
    reuse_server_port: the port of the server to reuse. If unspecified, it will
      try to find a running server from port 8080 to 8099.

  Returns:
    None
  """
  # pylint: disable=g-import-not-at-top
  # Intended for lazy-importing model_explorer related features. This allows
  # ModelUtils core to be used in environments where model_explorer is not
  # installed.
  from litert.python.tools.model_utils import model_explorer_integration
  # pylint: enable=g-import-not-at-top

  return model_explorer_integration.visualize.visualize(
      obj,
      host=host,
      port=port,
      colab_height=colab_height,
      reuse_server=reuse_server,
      reuse_server_host=reuse_server_host,
      reuse_server_port=reuse_server_port,
  )
