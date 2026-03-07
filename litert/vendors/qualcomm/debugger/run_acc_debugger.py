"""Run the accuracy per-layer debugging script."""

import argparse
from pathlib import Path
import os
import logging
import subprocess
import csv
from types import FunctionType
import json
import tensorflow as tf
import numpy as np
import shutil
from tflite.Model import Model
from common_args import add_common_arguments
from utils import (
    setup_logging,
    generate_ctx_bin,
    get_ctx_bin_info,
    run_ctx_bin,
    get_adb_cmd,
    ASSETS_DIR,
    DEVICE_WORKING_DIR,
)


def get_metric(measure: FunctionType, golden_data: dict) -> list:
  metric_table = []
  npu_out_dir = ASSETS_DIR / "output_htp" / "Result_0"
  for fname in os.listdir(npu_out_dir):
    if fname.endswith(".raw"):
      tensor_ind = int(fname.split("_")[3].replace(".raw", ""))
      if tensor_ind not in golden_data:
        raise ValueError("Tensor Index %d not found.", tensor_ind)
      cpu_out = golden_data[tensor_ind]
      npu_out = np.fromfile(
          npu_out_dir / fname,
          dtype=cpu_out.dtype
      )
      metric_table.append(
          {
              "tensor_index": tensor_ind,
              "metric": measure(cpu_out, npu_out)
          }
      )
  if os.path.exists(ASSETS_DIR / "output_htp"):
    shutil.rmtree(ASSETS_DIR / "output_htp")
  return metric_table


def _get_tensor_index(qnn_tensor_name: str) -> int:
  return int(qnn_tensor_name.split("_")[2])


def _set_npu_inputs(ctx_bin_path: Path, ctx_bin_info: dict, cpu_golden: dict):
  graph_name = ctx_bin_info["graphs"][0]["info"]["graphName"]
  input_dirs = Path("inputs") / ctx_bin_path.stem / graph_name
  graph_input_path = ASSETS_DIR / input_dirs
  os.makedirs(graph_input_path, exist_ok=True)
  lines = []
  for input_id, graph_input in enumerate(ctx_bin_info["graphs"][0]["info"]["graphInputs"]):
    qnn_input_name = graph_input["info"]["name"]
    tensor_index = _get_tensor_index(qnn_input_name)
    cpu_golden[tensor_index].tofile(graph_input_path / f"input_{input_id}.raw")
    lines.append(str(input_dirs / f"input_{input_id}.raw"))
  input_list_path = graph_input_path / "input_list.txt"
  input_list_path.write_text(" ".join(lines) + "\n", encoding="utf-8")


def _generate_cpu_outputs(model_path: Path, input_dir:Path) -> dict[int: np.ndarray]:
  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(
      model_path=model_path,
      experimental_preserve_all_tensors=True,
  )
  interpreter.allocate_tensors()

  # Set the inputs of the model
  input_details = interpreter.get_input_details()
  for detail in input_details:
    tensor_index = detail["index"]
    logging.info("Set input tensor %d (%s).", tensor_index, detail["name"])
    input_path = input_dir / f"{tensor_index}.npy"

    # Check if the provided input .npy exists.
    if not os.path.isfile(input_path):
      raise FileNotFoundError(f"Cannot access {input_path}.")

    # Check if input_dir contains readable .npy files.
    inp = np.load(input_path)
    if detail["dtype"] != inp.dtype.type:
      expected_dtype = detail["dtype"]
      raise TypeError(f"Expect dtype is {expected_dtype}, but get {inp.dtype.type}.")
    if not np.array_equal(detail["shape"], inp.shape):
      expected_shape = detail["shape"]
      raise ValueError(f"Expect input shape is {expected_shape}, but get {inp.shape}.")
    interpreter.set_tensor(detail["index"], inp)

  # Run inference.
  logging.info("Invoke the model on CPU.")
  interpreter.invoke()

  # Get the output of each layer.
  return {
      detail["index"]: interpreter.get_tensor(detail["index"])
      for detail in interpreter.get_tensor_details()
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Given a context binary, generate QNN HTP Optrace Profiling."
  )
  add_common_arguments(parser)
  parser.add_argument(
      "--input_dir",
      type=Path,
      required=True,
      help="Specify the directory that contains the input files, including <tensor_index>.npy.",
  )
  args = parser.parse_args()
  setup_logging(args.log_level)

  cpu_golden = _generate_cpu_outputs(args.model, args.input_dir)

  ctx_bin_paths = generate_ctx_bin(
      args.model,
      args.soc_model,
      optrace=False,
      per_layer_dump=True,
  )
  if len(ctx_bin_paths) != 1:
    raise ValueError(
        "Only support one context binary in compiled tflite."
    )
  ctx_bin_info = get_ctx_bin_info(ctx_bin_paths[0], args.qairt_sdk)
  if ctx_bin_info["numGraphs"] != 1:
        raise ValueError(
        "Only support one graph in the context binary of compiled tflite."
    )

  # Set NPU inputs.
  _set_npu_inputs(ctx_bin_paths[0], ctx_bin_info, cpu_golden)
  ADB_CMD = get_adb_cmd(args.hostname, args.serial)

  # Get NPU outputs.
  run_ctx_bin(
      ADB_CMD,
      ctx_bin_paths[0],
      args.htp_arch,
      args.qairt_sdk,
      0,
      ctx_bin_info,
      optrace=False
  )
  cmd = f"{ADB_CMD} pull {DEVICE_WORKING_DIR}/output_htp {ASSETS_DIR}"
  subprocess.run(cmd, check=True, shell=True)
  
  # Compute Cos Similarity.
  def cosine_similarity(vec1, vec2) -> float:
    vec1 = np.array(vec1).astype(np.float32).flatten()
    vec2 = np.array(vec2).astype(np.float32).flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return round(float(similarity), 6)

  data = get_metric(cosine_similarity, cpu_golden)

  # Create the analaysis result in .csv.
  os.makedirs(args.output_dir, exist_ok=True)
  with open(args.output_dir / "output.csv", mode="w", encoding="utf-8") as file:
    fieldnames = ["tensor_index", "metric"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
  logging.info("Output: %s", args.output_dir / "output.csv")

  # Create the analysis result in custom node data format (.json).
  buf = Path(args.model).read_bytes()
  model = Model.GetRootAsModel(buf, 0)
  subgraph = model.Subgraphs(0)
  output_index_map = {}
  for i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(i)
    output_index_map[op.Outputs(0)] = i

  custom_data = {
      "gradient": [
          {
              "stop": 0,
              "bgColor": "#eb0000"
          },
          {
             "stop": 1,
             "bgColor": "#00aa03"
          }
      ],
      "results": {}
  }
  for metric_data in data:
    node_id = output_index_map[metric_data["tensor_index"]]
    custom_data["results"][node_id] = {
        "value": float(metric_data["metric"])
    }
  with open(args.output_dir / "cossim.json", mode="w", encoding="utf-8") as file:
    json.dump(custom_data, file, indent = 4)
    logging.info("Output: %s", args.output_dir / "cossim.json")
