"""Generate input based on context binary using qnn-context-binary-utility."""

import json
import os
import pathlib
from typing import Sequence
from absl import app
from absl import flags
import numpy as np


JSON_FILE = flags.DEFINE_string(
    "json_file",
    "/tmp/qnn_context_info.json",
    "The path to output the json file",
)

SAVE_PATH = flags.DEFINE_string(
    "save_path",
    "/tmp",
    "The path to save input files",
)

INPUT_SOURCE = flags.DEFINE_string(
    "input_source",
    "zero",
    "The source of input, rand or zero",
)


def gen_input(json_pth: str):
  """Generate input."""

  with open(json_pth, "r", encoding="utf-8") as file:
    json_dict = json.load(file)

  for graph_id, graph in enumerate(json_dict["info"]["graphs"]):
    input_list = []
    for inp in graph["info"]["graphInputs"]:
      dtype = None
      if inp["info"]["dataType"] == "QNN_DATATYPE_INT_32":
        dtype = np.int32
      elif (
          inp["info"]["dataType"] == "QNN_DATATYPE_SFIXED_POINT_16"
          or inp["info"]["dataType"] == "QNN_DATATYPE_INT_16"
      ):
        dtype = np.int16
      elif (
          inp["info"]["dataType"] == "QNN_DATATYPE_SFIXED_POINT_8"
          or inp["info"]["dataType"] == "QNN_DATATYPE_INT_8"
      ):
        dtype = np.int8
      elif inp["info"]["dataType"] == "QNN_DATATYPE_BOOL_8":
        dtype = np.bool_
      elif inp["info"]["dataType"] == "QNN_DATATYPE_FLOAT_32":
        dtype = np.float32
      elif inp["info"]["dataType"] == "QNN_DATATYPE_UFIXED_POINT_16":
        dtype = np.uint16
      else:
        raise ValueError(f"Unsupported data type: {inp['info']['dataType']}")
      if INPUT_SOURCE.value == "rand":
        input_tensor = np.random.rand(*inp["info"]["dimensions"]).astype(dtype)
      else:
        input_tensor = np.zeros(inp["info"]["dimensions"]).astype(dtype)
      input_list.append(input_tensor)
    for input_id, input_tensor in enumerate(input_list):
      graph_input_pth = os.path.join(
          SAVE_PATH.value, f"inputs/graph_{graph_id}"
      )
      pathlib.Path(graph_input_pth).mkdir(parents=True, exist_ok=True)
      input_tensor.tofile(
          os.path.join(graph_input_pth, f"input_{input_id}.raw")
      )
      with open(
          os.path.join(graph_input_pth, "input_list.txt"), "w", encoding="utf-8"
      ) as f:
        for input_id, _ in enumerate(input_list):
          f.write(f"inputs/graph_{graph_id}/input_{input_id}.raw ")


def main(_: Sequence[str]) -> None:
  gen_input(JSON_FILE.value)


if __name__ == "__main__":
  app.run(main)
