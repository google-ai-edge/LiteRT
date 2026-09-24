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

"""Trace KV cache operators and subgraphs in a published TFLite bundle."""

from collections.abc import Sequence
import json
import mmap
import pathlib
from typing import Any

from absl import app
from absl import flags
from tflite import BuiltinOperator as tflite_builtin_operator
from tflite import Model as tflite_model
from tflite import StableHLOCompositeOptions as tflite_composite_options
from tflite import TensorType as tflite_tensor_type

_SCHEMA_DIR = flags.DEFINE_string(
    "schema_dir",
    None,
    "Generated TFLite Python package parent (contains tflite/Model.py).",
    required=True,
)
_INVENTORY = flags.DEFINE_string(
    "inventory",
    None,
    "Path to the published bundle inventory JSON file.",
    required=True,
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Path to the output KV graph trace JSON file.",
    required=True,
)

_TENSOR_TYPES = {
    v: k
    for k, v in vars(tflite_tensor_type.TensorType).items()
    if isinstance(v, int)
}
_BUILTIN_OPS = {
    v: k
    for k, v in vars(tflite_builtin_operator.BuiltinOperator).items()
    if isinstance(v, int)
}


def _describe_tensor(subgraph: Any, tensor_index: int) -> dict[str, Any]:
  """Extracts metadata and quantization summary for a subgraph tensor."""
  tensor_obj = subgraph.Tensors(tensor_index)
  quant = tensor_obj.Quantization()
  record: dict[str, Any] = {
      "index": int(tensor_index),
      "name": tensor_obj.Name().decode(),
      "dtype": _TENSOR_TYPES[tensor_obj.Type()],
      "shape": tensor_obj.ShapeAsNumpy().tolist(),
  }
  if quant and quant.ScaleLength():
    record["scales"] = quant.ScaleAsNumpy()[:4].tolist()
    record["zero_points"] = quant.ZeroPointAsNumpy()[:4].tolist()
  return record


def _describe_operator(
    model: Any, subgraph: Any, op_index: int
) -> dict[str, Any]:
  """Extracts operator type, inputs, outputs, and composite attributes."""
  op = subgraph.Operators(op_index)
  opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
  record: dict[str, Any] = {
      "index": op_index,
      "type": _BUILTIN_OPS[opcode],
      "inputs": [
          _describe_tensor(subgraph, int(j))
          for j in op.InputsAsNumpy()
          if j >= 0
      ],
      "outputs": [
          _describe_tensor(subgraph, int(j))
          for j in op.OutputsAsNumpy()
          if j >= 0
      ],
  }
  if record["type"] == "STABLEHLO_COMPOSITE":
    table = op.BuiltinOptions2()
    opts = tflite_composite_options.StableHLOCompositeOptions()
    opts.Init(table.Bytes, table.Pos)
    record["composite_name"] = opts.Name().decode()
    record["decomposition_subgraph"] = opts.DecompositionSubgraphIndex()
  return record


def main(argv: Sequence[str]) -> None:
  """Traces subgraphs and composite decompositions in the TFLite section."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  schema_dir = pathlib.Path(_SCHEMA_DIR.value)
  if not (schema_dir / "tflite/Model.py").is_file():
    raise app.UsageError("--schema_dir must contain generated tflite/Model.py")
  inventory_path = pathlib.Path(_INVENTORY.value)
  output_path = pathlib.Path(_OUTPUT.value)
  if output_path.exists():
    raise FileExistsError(output_path)

  inventory = json.loads(inventory_path.read_text())
  section = next(
      x
      for x in inventory["sections"]
      if x["items"].get("model_type") == "tf_lite_prefill_decode"
  )
  with (
      open(inventory["path"], "rb") as bundle_file,
      mmap.mmap(bundle_file.fileno(), 0, access=mmap.ACCESS_READ) as mm,
  ):
    model = tflite_model.Model.GetRootAsModel(
        memoryview(mm)[section["begin"] : section["end"]], 0
    )
    report: dict[str, Any] = {"bundle": inventory["path"], "graphs": []}
    for sg_index in range(model.SubgraphsLength()):
      subgraph = model.Subgraphs(sg_index)
      name = subgraph.Name().decode()
      if name not in ["decode", "prefill_128", "prefill_1024", "verify"]:
        continue
      graph_record: dict[str, Any] = {
          "index": sg_index,
          "name": name,
          "inputs": [
              _describe_tensor(subgraph, int(i))
              for i in subgraph.InputsAsNumpy()
          ],
          "outputs": [
              _describe_tensor(subgraph, int(i))
              for i in subgraph.OutputsAsNumpy()
          ],
          "operators": [
              _describe_operator(model, subgraph, i)
              for i in range(subgraph.OperatorsLength())
          ],
      }
      report["graphs"].append(graph_record)
      print("GRAPH", sg_index, name)
      for op_entry in graph_record["operators"][:80]:
        ins = ",".join(f"{t['index']}:{t['dtype']}" for t in op_entry["inputs"])
        outs = ",".join(
            f"{t['index']}:{t['dtype']} {t['name'].split('/')[-1]}"
            for t in op_entry["outputs"]
        )
        print(
            op_entry["index"],
            op_entry["type"],
            op_entry.get("composite_name", ""),
            op_entry.get("decomposition_subgraph", ""),
            ins,
            "->",
            outs,
        )
      # Include the body of each referenced composite once in structured output.
      decomps = {
          x["decomposition_subgraph"]
          for x in graph_record["operators"]
          if "decomposition_subgraph" in x
      }
      graph_record["decompositions"] = {
          str(d): {
              "name": model.Subgraphs(d).Name().decode(),
              "inputs": [
                  _describe_tensor(model.Subgraphs(d), int(i))
                  for i in model.Subgraphs(d).InputsAsNumpy()
              ],
              "outputs": [
                  _describe_tensor(model.Subgraphs(d), int(i))
                  for i in model.Subgraphs(d).OutputsAsNumpy()
              ],
              "operators": [
                  _describe_operator(model, model.Subgraphs(d), i)
                  for i in range(model.Subgraphs(d).OperatorsLength())
              ],
          }
          for d in decomps
      }
    output_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
  app.run(main)
