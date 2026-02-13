"""Run the profiling script."""

import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
import numpy as np
from common_args import add_common_arguments
from utils import (
    setup_logging,
    get_ctx_bin_info,
    get_adb_cmd,
    generate_ctx_bin,
    run_ctx_bin,
    ASSETS_DIR,
    DEVICE_WORKING_DIR,
    LITERT_ROOT,
)


def _generate_zero_inputs(ctx_bin_name:str, ctx_bin_info: dict[str, Any], graph_idx: int) -> None:
  """Generates zero-filled input files based on the context binary info.

  This function creates raw binary files for each input tensor defined in the
  context binary, filled with zeros. It also generates an `input_list.txt` file
  listing these generated input files. The files are saved within a
  subdirectory of `ASSETS_DIR / "inputs"`.

  Args:
      ctx_bin_name (str): The name of the context binary.
      ctx_bin_info (dict[str, Any]): A dictionary containing the context binary
        information, typically obtained from `get_ctx_bin_info`.
      graph_idx (int): The index of the graph within the context binary.

  Raises:
      ValueError: If the context binary contains more or less than one graph.
      TypeError: If an unknown QNN datatype is encountered.
  """
  logging.info("Generating input data for graph %d.")
  if len(ctx_bin_info["graphs"]) <= graph_idx:
    raise ValueError("Graph %d is not available in the context binary.")

  graph = ctx_bin_info["graphs"][graph_idx]
  dtype_map = {
      "QNN_DATATYPE_INT_8": np.int8,
      "QNN_DATATYPE_INT_16": np.int16,
      "QNN_DATATYPE_INT_32": np.int32,
      "QNN_DATATYPE_INT_64": np.int64,
      "QNN_DATATYPE_UINT_8": np.uint8,
      "QNN_DATATYPE_UINT_16": np.uint16,
      "QNN_DATATYPE_UINT_32": np.uint32,
      "QNN_DATATYPE_UINT_64": np.uint64,
      "QNN_DATATYPE_FLOAT_16": np.float16,
      "QNN_DATATYPE_FLOAT_32": np.float32,
      "QNN_DATATYPE_FLOAT_64": np.float64,
      "QNN_DATATYPE_SFIXED_POINT_8": np.int8,
      "QNN_DATATYPE_SFIXED_POINT_16": np.int16,
      "QNN_DATATYPE_SFIXED_POINT_32": np.int32,
      "QNN_DATATYPE_UFIXED_POINT_8": np.uint8,
      "QNN_DATATYPE_UFIXED_POINT_16": np.uint16,
      "QNN_DATATYPE_UFIXED_POINT_32": np.uint32,
      "QNN_DATATYPE_BOOL_8": np.uint8,
  }
  input_list = []
  for inp in graph["info"]["graphInputs"]:
    qnn_datatype = inp["info"]["dataType"]
    dtype = dtype_map.get(qnn_datatype, None)
    if dtype is None:
      raise TypeError(f"Unknown datatype {qnn_datatype}")
    input_tensor = np.zeros(inp["info"]["dimensions"]).astype(dtype)
    input_list.append(input_tensor)
  graph_name = graph["info"]["graphName"]
  input_dirs = Path("inputs") / ctx_bin_name / graph_name
  graph_input_path = ASSETS_DIR / input_dirs
  os.makedirs(graph_input_path, exist_ok=True)
  
  lines = []
  for input_id, input_tensor in enumerate(input_list):
    input_tensor.tofile(graph_input_path / f"input_{input_id}.raw")
    lines.append(str(input_dirs / f"input_{input_id}.raw"))
  input_list_path = graph_input_path / "input_list.txt"
  input_list_path.write_text(" ".join(lines) + "\n", encoding="utf-8")


def _generate_profiler_output(
    adb_cmd: str, ctx_bin_name: str, output_dir: str, qairt_sdk: Path
) -> None:
  """Generate profiler output.

  Args:
      adb_cmd (str): The adb command.
      ctx_bin_name (str): The name of the context binary.
      output_dir (str): The path to the output directory.
      qairt_sdk (Path): The path to the QAIRT SDK.
  """
  logging.info("Exectuing qnn-profile-viewer with the outputs...")
  os.makedirs(output_dir, exist_ok=True)
  cmd = f"{adb_cmd} pull {DEVICE_WORKING_DIR}/output_htp {ASSETS_DIR}"
  subprocess.run(cmd, check=True, shell=True)
  log_path = ASSETS_DIR / "output_htp" / "qnn-profiling-data_0.log"
  schematic_path = (
      LITERT_ROOT
      / "bazel-bin"
      / "litert"
      / "tools"
      / "apply_plugin_main.runfiles"
      / "litert"
      / f"{ctx_bin_name}_schematic.bin"
  )
  logging.info("schematic_path %s", schematic_path)
  cmd_lst = [
      qairt_sdk / "bin" / "x86_64-linux-clang" / "qnn-profile-viewer",
      "--input_log",
      log_path,
      "--config",
      ASSETS_DIR / "config_viewer.json",
      "--reader",
      qairt_sdk
      / "lib"
      / "x86_64-linux-clang"
      / "libQnnHtpOptraceProfilingReader.so",
      "--schematic",
      schematic_path,
      "--output",
      Path(output_dir) / "chromeTrace.json",
  ]
  os.makedirs(output_dir, exist_ok=True)
  logging.info(" ".join(str(item) for item in cmd_lst))
  subprocess.run(
      cmd_lst,
      check=True,
  )
  if os.path.exists(ASSETS_DIR / "output_htp"):
    shutil.rmtree(ASSETS_DIR / "output_htp")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Given a context binary, generate QNN HTP Optrace Profiling."
  )
  add_common_arguments(parser)
  parser.add_argument(
      "--ctx_bins",
      nargs='+',
      help='One or more context binary names to process.'
  )
  args = parser.parse_args()
  setup_logging(args.log_level)
  ADB_CMD = get_adb_cmd(args.hostname, args.serial)
  for ctx_bin in generate_ctx_bin(args.model, args.soc_model, optrace=True):
    ctx_bin_name = ctx_bin.stem
    if ctx_bin_name not in args.ctx_bins:
      os.remove(ctx_bin)
      continue
    ctx_bin_info = get_ctx_bin_info(ctx_bin, args.qairt_sdk)
    for index, _ in enumerate(ctx_bin_info["graphs"]):
      _generate_zero_inputs(ctx_bin_name, ctx_bin_info, index)
      run_ctx_bin(
          ADB_CMD,
          ctx_bin,
          args.htp_arch,
          args.qairt_sdk,
          index,
          ctx_bin_info,
          optrace=True,
      )
      graph_name = ctx_bin_info["graphs"][index]["info"]["graphName"]
      profiling_output_dir = Path(args.output_dir) / ctx_bin_name / graph_name
      _generate_profiler_output(
        ADB_CMD,
        f"qnn_partition_{index}",
        profiling_output_dir,
        args.qairt_sdk,
      )
      logging.info("Profiling data for qnn_partition_%d is generated.", index)
    os.remove(ctx_bin)
    logging.info("Success! Profiling data is in %s", args.output_dir)
