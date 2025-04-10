import sys
from typing import cast
from google3.third_party.tensorflow.lite.tools import flatbuffer_utils

RECURSION_LIMIT = 100000
# Use higher recursion limit for resursive graph copy.
if sys.getrecursionlimit() < RECURSION_LIMIT:
  sys.setrecursionlimit(RECURSION_LIMIT)

from . import core
from . import dialect
from . import graph_utils
from . import matcher
from . import passes
from . import shard
from . import transform
from . import testing
from . import model_builder

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
):
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
  """
  from litert.python.google.tools.model_utils import model_explorer_integration

  return model_explorer_integration.visualize.visualize(
      obj,
      host=host,
      port=port,
      colab_height=colab_height,
      reuse_server=reuse_server,
      reuse_server_host=reuse_server_host,
      reuse_server_port=reuse_server_port,
  )
