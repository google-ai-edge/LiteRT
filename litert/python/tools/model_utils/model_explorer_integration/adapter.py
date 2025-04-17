import dataclasses
import json
import re
from model_explorer import graph_builder
import numpy as np
from xdsl.irdl import *
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir

CONST_NUMEL_LIMIT = 32


class Adapter:
  """Adapter to build Model Explorer GraphCollection from ModelUtils Python objects."""

  def __init__(self, graph_collection_label="model_utils"):
    self.graph_collection = graph_builder.GraphCollection(
        label=graph_collection_label
    )
    self.node_mapping = {}

  @property
  def json(self):
    return json.dumps(dataclasses.asdict(self.graph_collection))

  def kv(self, k, v):
    return graph_builder.KeyValue(key=str(k), value=str(v))

  def kvs(self, d: dict[str, str]):
    return [self.kv(k, v) for k, v in d.items()]

  def add_func_op(self, op: func.FuncOp):
    assert isinstance(op, func.FuncOp)
    return self.build_graph(op.body.blocks[0], graph_id=op.sym_name)

  def build_attribute_str(self, attr: Attribute):
    if isinstance(attr, mlir.DenseElementsAttr):
      return str(attr)
    elif isinstance(attr, mlir.StringAttr):
      return f'"{attr.data}"'
    elif hasattr(attr, "data"):
      return str(attr.data)
    else:
      return str(attr)

  def build_namespace(self, op: core.MlirOpBase):
    if not isinstance(op, core.MlirOpBase):
      return ""

    loc = str(op.location)
    nameloc = re.search(r'"([^"]+)"', loc)
    namespace = nameloc.group(1) if nameloc else f"{op.name}/{id(op)}"

    def is_trivial(seg):
      seg = seg.lower()
      return not seg or re.match(r"\d+", seg) or "xlacallmodule" in seg

    segments = namespace.split(";")
    for seg in segments:
      if not is_trivial(seg):
        return seg
    return segments[0]

  def build_tensor_name(self, index: int, value: SSAValue):
    owner = value.owner
    if not isinstance(owner, core.MlirOpBase):
      return f"SSAVALUE_{id(value)}"

    loc = str(owner.location)
    nameloc = re.search(r'"([^"]+)"', loc)
    name = nameloc.group(1) if nameloc else f"{owner.name}/{id(owner)}"
    name = f"{name};{index}"
    return name

  def build_tensor_metadata(
      self, index: int, value: SSAValue, tensor_name=None
  ):
    ty = value.type
    if not isinstance(ty, mlir.RankedTensorType):
      return graph_builder.MetadataItem(
          str(index),
          self.kvs({
              "PY_TYPE": str(type(ty)),
              "PY_VALUE": str(ty)[:100],
          }),
      )

    shape_str = ", ".join(map(str, ty.shape))
    elty_str = str(ty.element_type)
    if len(elty_str) > 100:
      elty_str = elty_str[:100]

    if tensor_name is None:
      tensor_name = self.build_tensor_name(index, value)

    return graph_builder.MetadataItem(
        str(index),
        self.kvs({
            "tensor_name": tensor_name,
            "tensor_shape": f"{elty_str}[{shape_str}]",
        }),
    )

  def build_node(self, op: core.MlirOpBase):
    # label
    label = op.name
    if label == "arith.constant":
      # HACK: For better const visualization in Model Explorer .
      label = "tfl.pseudo_const"

    if label == "func.return":
      label = "GraphOutputs"
    elif label.startswith("tfl."):
      label = label[label.find(".") + 1 :]

    # TODO: subgraphIds
    # TODO: inputsMetadata

    # outputsMetadata
    outputs_metadata = []
    for i, result in enumerate(op.results):
      outputs_metadata.append(self.build_tensor_metadata(i, result))

    # incomingEdges
    incoming_edges = []
    for i, operand in enumerate(op.operands):
      operand_index = operand.index if hasattr(operand, "index") else 0
      incoming_edges.append(
          graph_builder.IncomingEdge(
              sourceNodeId=self.node_mapping[operand.owner].id,
              sourceNodeOutputId=str(operand_index),
              targetNodeInputId=str(i),
          )
      )

    # attrs
    attrs = []
    for name, attr in op.attributes.items():
      attrs.append(self.kv(name, self.build_attribute_str(attr)))
    if hasattr(op, "numpy"):
      # const op
      try:
        numel = np.prod(op.results[0].type.shape)
        if numel < CONST_NUMEL_LIMIT:
          value = op.numpy().tolist()
          value = json.dumps(value)
          attrs.append(self.kv("__value", value))
      except:
        pass

    namespace = self.build_namespace(op)
    node = graph_builder.GraphNode(
        id=str(id(op)),
        label=label,
        namespace=namespace,
        outputsMetadata=outputs_metadata,
        incomingEdges=incoming_edges,
        attrs=attrs,
    )
    self.node_mapping[op] = node
    return node

  def build_graph(self, block: Block, graph_id=None):
    assert isinstance(block, Block)
    graph_inputs_node = graph_builder.GraphNode(
        id=str(id(block)),
        label="GraphInputs",
        outputsMetadata=[
            self.build_tensor_metadata(i, arg, tensor_name=f"input_{i}")
            for i, arg in enumerate(block.args)
        ],
    )
    self.node_mapping[block] = graph_inputs_node

    nodes = [graph_inputs_node]
    for i, op in enumerate(block.ops):
      node = self.build_node(op)
      node.id = str(i)
      nodes.append(node)
    if graph_id is None:
      graph_id = f"BLOCK_{id(block)}"
    graph = graph_builder.Graph(id=graph_id, nodes=nodes)
    self.graph_collection.graphs.append(graph)
    return graph
