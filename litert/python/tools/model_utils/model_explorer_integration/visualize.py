import json
import os
import pathlib
import sys
import tempfile

import model_explorer
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils import model_explorer_integration as me_integration
from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir


try:
  from IPython import display
except ImportError:
  display = None


def in_colab():
  return (
      display is not None
      and "google.colab" in sys.modules
      or os.getenv("COLAB_RELEASE_TAG")
  )


VISUALIZATION_TYPES = (
    mlir.ModuleOp | func.FuncOp | Region | Block | core.MlirOpBase
)
_TEMP_DIR = None


def visualize(
    obj: VISUALIZATION_TYPES,
    host="localhost",
    port=8080,
    colab_height=850,
    reuse_server: bool = False,
    reuse_server_host: str = "localhost",
    reuse_server_port: Union[int, None] = None,
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
  global _TEMP_DIR
  adapter = _build_adapter(obj)

  if _TEMP_DIR is None:
    _TEMP_DIR = pathlib.Path(tempfile.mkdtemp())

  json_str = adapter.json
  json_filename = (
      _TEMP_DIR / f"model_utils_model_explorer_{hex(hash(json_str))[2:]}.json"
  )

  if not json_filename.exists():
    json_filename.write_text(json_str)

  model_explorer.visualize(
      str(json_filename),
      host=host,
      port=port,
      colab_height=colab_height,
      reuse_server=reuse_server,
      reuse_server_host=reuse_server_host,
      reuse_server_port=reuse_server_port,
  )

  if in_colab():
    ui_state = _build_selected_ui_state(adapter, obj)
    if ui_state is None:
      return
    display.display(display.Javascript("""
      const observer = new MutationObserver(() => {
        const iframe = document.body.getElementsByTagName("iframe")[0];
        if (!iframe) {
          return;
        }

        const searchParams = new URLSearchParams(new URL(iframe.src).search);
        const data = JSON.parse(searchParams.get('data'));

        data.uiState = %UI_STATE%

        searchParams.set("data", JSON.stringify(data))
        const newUrl = new URL(iframe.src);
        newUrl.search = searchParams;
        iframe.src = newUrl.toString();

        observer.disconnect();
      })
      observer.observe(document.body, {childList: true});
      """.replace("%UI_STATE%", json.dumps(ui_state))))


def _build_selected_ui_state(adapter: me_integration.adapter.Adapter, op):
  if not isinstance(op, core.MlirOpBase):
    return None
  if isinstance(op, (mlir.ModuleOp, func.FuncOp)):
    return None

  node = adapter.node_mapping[op]
  graph = next(
      filter(lambda g: node in g.nodes, adapter.graph_collection.graphs)
  )
  ui_state = {
      "paneStates": [{
          "deepestExpandedGroupNodeIds": [],
          "selectedNodeId": node.id,
          "selectedGraphId": graph.id,
          "selectedCollectionLabel": adapter.graph_collection.label,
          "widthFraction": 1,
          "selected": True,
      }]
  }
  return ui_state


def _build_adapter(obj: VISUALIZATION_TYPES):
  if isinstance(obj, SSAValue):
    return _build_adapter(obj.owner)
  elif isinstance(obj, mlir.ModuleOp):
    adapter = me_integration.adapter.Adapter()
    for op in obj.body.ops:
      if isinstance(op, func.FuncOp):
        adapter.add_func_op(op)
    return adapter
  elif isinstance(obj, func.FuncOp):
    adapter = me_integration.adapter.Adapter()
    adapter.add_func_op(obj)
    return adapter
  elif isinstance(obj, Region):
    adapter = me_integration.adapter.Adapter()
    for block in obj.blocks:
      adapter.build_graph(block)
    return adapter
  elif isinstance(obj, Block):
    adapter = me_integration.adapter.Adapter()
    adapter.build_graph(obj)
    return adapter
  elif isinstance(obj, core.MlirOpBase):
    parent = obj.parent
    if not isinstance(parent, Block):
      raise ValueError("Op must be in a block for visualization")
    adapter = me_integration.adapter.Adapter()
    adapter.build_graph(parent)
    return adapter
  raise ValueError(f"Visualizing {type(obj)} is not supported")
