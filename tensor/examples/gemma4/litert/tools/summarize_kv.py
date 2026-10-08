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

"""Summarize published owner sharing and KV quantization from a graph trace."""

from collections.abc import Sequence
import json
import pathlib
import re

from absl import app
from absl import flags

_TRACE = flags.DEFINE_string(
    "trace",
    None,
    "Path to the input KV graph trace JSON file.",
    required=True,
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Path to the output KV placement summary JSON file.",
    required=True,
)


def main(argv: Sequence[str]) -> None:
  """Summarizes owner KV cache updates and attention reads across subgraphs."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  trace_path = pathlib.Path(_TRACE.value)
  output_path = pathlib.Path(_OUTPUT.value)
  if output_path.exists():
    raise FileExistsError(output_path)
  payload = json.loads(trace_path.read_text())
  summary = {"graphs": []}
  for graph in payload["graphs"]:
    # Include full verify graph to check sharing on a multi-token path too.
    cache_map = {}
    updates = []
    reads = []
    for op in graph["operators"]:
      if op.get("composite_name") != "odml.cache_update":
        continue
      key_in = next(t for t in op["inputs"] if "_kv_cache_k_" in t["name"])
      value_in = next(t for t in op["inputs"] if "_kv_cache_v_" in t["name"])
      owner_match = re.search(r"_kv_cache_k_(\d+)", key_in["name"])
      assert owner_match is not None
      owner = int(owner_match.group(1))
      for kind, tensor_entry in zip(["K", "V"], op["outputs"]):
        cache_map[tensor_entry["index"]] = {
            "owner": owner,
            "kind": kind,
            "scales": tensor_entry["scales"],
            "dtype": tensor_entry["dtype"],
        }
      decomp = graph["decompositions"][str(op["decomposition_subgraph"])]
      quant_ops = [o for o in decomp["operators"] if o["type"] == "QUANTIZE"]
      assert len(quant_ops) == 2
      updates.append({
          "owner": owner,
          "operator": op["index"],
          "key_scale": key_in["scales"][0],
          "value_scale": value_in["scales"][0],
          "cache_update_decomposition": op["decomposition_subgraph"],
          "first_two_inputs": list(op["inputs"][:2]),
          "quantize_operators": quant_ops,
      })
    for op in graph["operators"]:
      if op.get("composite_name") != "odml.runtime_bmm":
        continue
      info = cache_map[op["inputs"][1]["index"]]
      layer_match = re.search(r"/layer_(\d+)/", op["outputs"][0]["name"])
      assert layer_match is not None
      layer = int(layer_match.group(1))
      decomp = graph["decompositions"][str(op["decomposition_subgraph"])]
      assert any(
          o["type"] == "DEQUANTIZE" and o["inputs"][0]["dtype"] == "INT8"
          for o in decomp["operators"]
      )
      assert any(
          o["type"] == "BATCH_MATMUL"
          and all(t["dtype"] == "FLOAT32" for t in o["inputs"])
          for o in decomp["operators"]
      )
      reads.append({
          "layer": layer,
          "operator": op["index"],
          **info,
          "dequantized_inside_bmm": True,
      })
    expected_reads = 28 if graph["name"].startswith("prefill") else 70
    assert len(updates) == 15 and len(reads) == expected_reads
    expected_owners = {
        i: i if i < 15 else (14 if i % 5 == 4 else 13) for i in range(35)
    }
    assert all(expected_owners[r["layer"]] == r["owner"] for r in reads)
    summary["graphs"].append({
        "name": graph["name"],
        "index": graph["index"],
        "cache_updates": updates,
        "attention_reads": reads,
        "output_dtypes": [t["dtype"] for t in graph["outputs"]],
        "output_shapes": [t["shape"] for t in graph["outputs"]],
        "fully_connected_count": sum(
            o["type"] == "FULLY_CONNECTED" for o in graph["operators"]
        ),
    })
  output_path.write_text(json.dumps(summary, indent=2) + "\n")
  for graph in summary["graphs"]:
    print(
        graph["name"],
        len(graph["cache_updates"]),
        "owner cache updates,",
        len(graph["attention_reads"]),
        "INT8 cache reads through dequantized FP32 BMM; sharing checked",
    )
  print(
      "scales",
      [
          (x["owner"], x["key_scale"], x["value_scale"])
          for x in summary["graphs"][0]["cache_updates"]
      ],
  )


if __name__ == "__main__":
  app.run(main)
