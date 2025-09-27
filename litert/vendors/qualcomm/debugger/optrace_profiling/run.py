"""Run the profiling script."""

import argparse
import getpass
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
from typing import Any, Optional
import numpy as np

Path = pathlib.Path
_DEVICE_WORKING_DIR = f"/data/local/tmp/{getpass.getuser()}/litert"
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_LITERT_ROOT = Path(__file__).resolve().parents[5]


def _extract_build_id(build_id_hdr: str) -> Optional[str]:
  """Extract the build ID from the header file.

  Args:
      build_id_hdr (str): Path to the build ID header file.

  Returns:
      str: Build ID if found, None otherwise.
  """
  with open(build_id_hdr, "r", encoding="utf-8") as file:
    content = file.read()

  # Use regex to find the build ID
  match = re.search(r'#define\s+QNN_SDK_BUILD_ID\s+"([^"]+)"', content)
  if match:
    return match.group(1)
  else:
    return None


def _get_ctx_bin_info(ctx_bin_path: str, qairt_sdk: Path) -> dict[str, Any]:
  """Get the context binary information.

  Args:
      ctx_bin_path (str): Path to the context binary.
      qairt_sdk (str): Path to the QNN SDK.

  Returns:
      json_data (dict): Context binary information.
  """
  json_path = _ASSETS_DIR / "tmp.json"
  subprocess.run(
      [
          qairt_sdk
          / "bin"
          / "x86_64-linux-clang"
          / "qnn-context-binary-utility",
          "--context_binary",
          f"{ctx_bin_path}",
          "--json_file",
          json_path,
      ],
      check=True,
  )

  with open(json_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)
  os.remove(json_path)
  bin_id = json_data["info"]["buildId"]
  qairt_id = _extract_build_id(
      qairt_sdk / "include" / "QNN" / "QnnSdkBuildId.h"
  )
  logging.info("Context Binary Build Id: %s", bin_id)

  def version_parse(build_id: str) -> tuple[int, ...]:
    """Parse version string and return a tuple of integers.

    Args:
        build_id (str): Version string.

    Returns:
        tuple[int, ...]: Tuple of integers representing the version number.
    """
    parts = build_id.lstrip("v").split(".")
    return tuple(map(int, parts[:3]))

  if version_parse(bin_id) == version_parse(qairt_id):
    logging.info("Build ID matches.")
  else:
    logging.warning(
        "Ensure the context binary '%s' is built with the same SDK"
        " version '%s'.",
        bin_id,
        qairt_id,
    )
  return json_data["info"]


def _generate_zero_inputs(ctx_bin_info: dict[str, Any]) -> None:
  """Generates zero-filled input files based on the context binary info.

  This function creates raw binary files for each input tensor defined in the
  context binary, filled with zeros. It also generates an `input_list.txt` file
  listing these generated input files. The files are saved within a
  subdirectory of `_ASSETS_DIR / "inputs"`.

  Args:
      ctx_bin_info (dict[str, Any]): A dictionary containing the context binary
        information, typically obtained from `_get_ctx_bin_info`.

  Raises:
      ValueError: If the context binary contains more or less than one graph.
      TypeError: If an unknown QNN datatype is encountered.
  """
  logging.info("Generating input data...")

  if len(ctx_bin_info["graphs"]) != 1:
    raise ValueError(
        "The input generator currently supports only one graph in context"
        " binary."
    )
  graph = ctx_bin_info["graphs"][0]
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
  graph_input_path = _ASSETS_DIR / "inputs" / graph_name
  os.makedirs(graph_input_path, exist_ok=True)
  input_dirs = Path("inputs") / graph_name
  lines = []
  for input_id, input_tensor in enumerate(input_list):
    input_tensor.tofile(graph_input_path / f"input_{input_id}.raw")
    lines.append(str(input_dirs / f"input_{input_id}.raw"))
  input_list_path = graph_input_path / "input_list.txt"
  input_list_path.write_text(" ".join(lines) + "\n", encoding="utf-8")


def _push_so(adb_cmd: str, htp_arch: str, qairt_sdk: Path):
  """Push shared object (.so) and binary files required by qnn-net-run to the target device.

  Args:
      adb_cmd (str): The base adb command (e.g., "adb" or "adb -s <device_id>").
      htp_arch (str): The htp architecture of the target device (e.g., "V75").
      qairt_sdk (Path): The path to the Qairt SDK.
  """
  logging.info("Pushing .so files to the target device...")
  cmd = (
      f'{adb_cmd} shell "rm -rf {_DEVICE_WORKING_DIR} && mkdir -p'
      f' {_DEVICE_WORKING_DIR}"'
  )
  subprocess.run(cmd, check=True, shell=True)
  for file_path in [
      "bin/aarch64-android/qnn-net-run",
      "lib/aarch64-android/libQnnHtp.so",
      "lib/aarch64-android/libQnnHtpNetRunExtensions.so",
      "lib/aarch64-android/libQnnHtpPrepare.so",
      f"lib/aarch64-android/libQnnHtp{htp_arch.upper()}Stub.so",
      f"lib/hexagon-{htp_arch.lower()}/unsigned/libQnnHtp{htp_arch.upper()}Skel.so",
  ]:
    cmd = f"{adb_cmd} push {qairt_sdk / file_path} {_DEVICE_WORKING_DIR}"
    logging.debug(cmd)
    subprocess.run(cmd, check=True, shell=True)


def _get_adb_cmd(hostname: str, serial: str) -> str:
  """Get adb command with hostname and serial.

  Args:
      hostname (str): The hostname of the device.
      serial (str): The serial number of the device.

  Returns:
      str: The adb command.
  """
  cmd_parts = ["adb"]

  if hostname:
    cmd_parts += ["-H", hostname]
  if serial:
    cmd_parts += ["-s", serial]

  return " ".join(cmd_parts).strip()


def _push_target(adb_cmd: str, ctx_bin_path: str) -> None:
  """Push inputs and context binary to the device.

  Args:
      adb_cmd (str): The adb command.
      ctx_bin_path (str): The path to the context binary.
  """
  logging.info(
      "Pushing inputs, ctx binary, and .json files to the target device..."
  )
  for file_path in [
      _ASSETS_DIR / "inputs",
      _ASSETS_DIR / "htp_ext_config.json",
      _ASSETS_DIR / "config.json",
      Path(ctx_bin_path).resolve(),
  ]:
    cmd = f"{adb_cmd} push {file_path} {_DEVICE_WORKING_DIR}"
    subprocess.run(
        cmd, check=True, shell=True, cwd=Path(__file__).resolve().parent
    )
    logging.debug(cmd)
    if file_path.name == "inputs":
      shutil.rmtree(file_path)
      logging.debug("Removing inputs...")


def _run_ctx_bin(
    adb_cmd: str, ctx_bin_path: str, htp_arch: str, qairt_sdk: Path
) -> None:
  """Run qnn-net-run with the given context binary.

  Args:
      adb_cmd (str): The adb command.
      ctx_bin_path (str): The path of the context binary.
      htp_arch (str): The htp architecture of the target device (e.g., "V75").
      qairt_sdk (Path): The path to the Qairt SDK.
  """
  _push_so(adb_cmd, htp_arch, qairt_sdk)
  _push_target(adb_cmd, ctx_bin_path)
  logging.info("Exectuing qnn-net-run with the given context binary...")
  env_vars = (
      f"export LD_LIBRARY_PATH={_DEVICE_WORKING_DIR} && "
      f"export ADSP_LIBRARY_PATH={_DEVICE_WORKING_DIR} && "
      f"cd {_DEVICE_WORKING_DIR}"
  )
  run_cmd = (
      "./qnn-net-run "
      "--backend libQnnHtp.so "
      f"--retrieve_context {Path(ctx_bin_path).name} "
      "--input_list inputs/qnn_partition_0/input_list.txt "
      "--output_dir output_htp "
      "--use_native_input_files "
      "--use_native_output_files "
      "--config_file config.json "
      "--profiling_option optrace "
      "--profiling_level detailed"
  )
  full_cmd = f'{adb_cmd} shell "{env_vars} && {run_cmd}"'
  logging.debug(full_cmd)
  subprocess.run(full_cmd, check=True, shell=True)


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
  cmd = f"{adb_cmd} pull {_DEVICE_WORKING_DIR}/output_htp ./"
  subprocess.run(cmd, check=True, shell=True)
  log_path = Path("output_htp") / "qnn-profiling-data_0.log"
  schematic_path = (
      _LITERT_ROOT
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
      _ASSETS_DIR / "config_viewer.json",
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


def _setup_logging(level_name):
  level = getattr(logging, level_name.upper(), logging.INFO)
  logging.basicConfig(
      level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  )


def _generate_ctx_bin(model_path: Path, soc_model: str) -> Path:
  """Generate the ctx.bin file from a given model.

  Args:
      model_path (Path): The path to the model.
      soc_model (str): The SOC model to use.

  Returns:
      Optional[Path]: The path to the generated ctx.bin file, or None if the
      generation failed.

  Raises:
      RuntimeError: If the context binary path cannot be extracted from the
        subprocess output.
  """
  tmp_tflite_path = _ASSETS_DIR / "tmp.tflite"
  bazel_run = [
      "bazel",
      "run",
      "-c",
      "opt",
      "--cxxopt=--std=c++17",
      "--nocheck_visibility",
  ]
  apply_pluin_main_cmd = [
      *bazel_run,
      "//litert/tools:apply_plugin_main",
      "--",
      "--libs=litert/vendors/qualcomm/compiler",
      "--cmd=apply",
      f"--model={model_path}",
      f"--o={tmp_tflite_path}",
      "--soc_manufacturer=Qualcomm",
      f"--soc_model={soc_model.upper()}",
      "--qualcomm_profiling=optrace",
  ]
  logging.debug(" ".join(str(item) for item in apply_pluin_main_cmd))
  # TODO(jiunkaiy): Set check=True after seg fault fix.
  subprocess.run(
      apply_pluin_main_cmd, check=False, cwd=Path(__file__).resolve().parent
  )

  extract_bytecode_lst = [
      *bazel_run,
      "//litert/tools:extract_bytecode",
      "--",
      f"--model_path={tmp_tflite_path}",
      f"--output_dir={_ASSETS_DIR}",
  ]
  logging.debug(" ".join(str(item) for item in extract_bytecode_lst))
  result = subprocess.run(
      extract_bytecode_lst,
      check=True,
      cwd=Path(__file__).resolve().parent,
      capture_output=True,
      text=True,
  )
  os.remove(tmp_tflite_path)
  logging.info("Subprocess output:\n%s", result.stderr)
  # Parse the stderr to extract the path to the generated ctx.bin file.
  # "Wrote ... bytes to '<context binary path>"
  match = re.search(r"bytes to '([^']+)'", result.stderr)
  if match:
    output_path = Path(match.group(1))
    logging.info("Extracted path:\n%s", output_path)
  else:
    raise RuntimeError("Failed to generate context binary.")
  return output_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Given a context binary, generate QNN HTP Optrace Profiling."
  )
  parser.add_argument(
      "--model", "-m", type=str, help="path to the tflite model", required=True
  )
  parser.add_argument(
      "--output_dir",
      "-o",
      type=str,
      help="path to output folder",
      required=True,
  )
  parser.add_argument(
      "--hostname", "-H", type=str, help="hostname for adb", default="localhost"
  )
  parser.add_argument(
      "--serial", "-s", type=str, help="serial for adb", required=True
  )
  parser.add_argument(
      "--soc_model", type=str, help="SoC Model (e.g. SM8650)", required=True
  )
  parser.add_argument(
      "--htp_arch", type=str, help="HTP Arch (e.g. V75)", required=True
  )
  parser.add_argument(
      "--log_level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
      help="Set the logging level",
  )
  default_qairt_sdk = (
      Path(__file__).resolve().parents[5] / "third_party" / "qairt" / "latest"
  )
  if "LITERT_QAIRT_SDK" in os.environ:
    default_qairt_sdk = Path(os.environ["LITERT_QAIRT_SDK"]) / "latest"
  parser.add_argument(
      "--qairt_sdk",
      type=Path,
      help="Path to qairt sdk folder",
      required=False,
      default=default_qairt_sdk,
  )
  args = parser.parse_args()
  
  # LiteRT uses LITERT_QAIRT_SDK, align it with the one from args
  os.environ["LITERT_QAIRT_SDK"] = f"{str(args.qairt_sdk.parent)}/"

  _setup_logging(args.log_level)
  ctx_bin = _generate_ctx_bin(args.model, args.soc_model)
  _generate_zero_inputs(_get_ctx_bin_info(ctx_bin, args.qairt_sdk))
  ADB_CMD = _get_adb_cmd(args.hostname, args.serial)
  _run_ctx_bin(ADB_CMD, ctx_bin, args.htp_arch, args.qairt_sdk)
  _generate_profiler_output(
      ADB_CMD,
      str(ctx_bin.name).replace("_0.bin", ""),
      args.output_dir,
      args.qairt_sdk,
  )
  os.remove(ctx_bin)
  logging.info("Success! Profiling data is in %s", args.output_dir)
