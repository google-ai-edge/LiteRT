"""LiteRT ModelUtils is the Python toolkit for creating, inspecting, and rewriting TFLite/LiteRT flatbuffer models."""

import sys
from typing import cast
from google3.third_party.tensorflow.lite.tools import flatbuffer_utils  # pylint: disable=g-direct-tensorflow-import

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
from . import testing
from . import transform
# pylint: enable=g-import-not-at-top

RewritePatternPassBase = core.RewritePatternPassBase
OpBuildingContext = core.OpBuildingContext
MatchingContext = match.MatchingContext

# Shortcuts
read_flatbuffer = transform.read_flatbuffer
read_mlir = transform.read_mlir
convert_to_mlir = transform.convert_to_mlir
write_flatbuffer = transform.write_flatbuffer
get_ir_context = transform.get_ir_context


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
