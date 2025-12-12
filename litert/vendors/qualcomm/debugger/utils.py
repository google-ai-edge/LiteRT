import getpass
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Optional
import json
import subprocess
import re

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEVICE_WORKING_DIR = f"/data/local/tmp/{getpass.getuser()}/litert"
LITERT_ROOT = Path(__file__).resolve().parents[4]

def setup_logging(level_name):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
      level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  )


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

def get_ctx_bin_info(ctx_bin_path: str, qairt_sdk: Path) -> dict[str, Any]:
  """Get the context binary information.

  Args:
      ctx_bin_path (str): Path to the context binary.
      qairt_sdk (str): Path to the QNN SDK.

  Returns:
      json_data (dict): Context binary information.
  """
  json_path = ASSETS_DIR / "tmp.json"
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
    raise RuntimeError(
        f"The context binary is compiled with BUILD ID: {bin_id} "
        f"!= current qairt sdk BUILD ID: {qairt_id}"
    )
  return json_data["info"]


def get_adb_cmd(hostname: str, serial: str) -> str:
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


def generate_ctx_bin(
    model_path: Path,
    soc_model: str,
    optrace: bool = False,
    per_layer_dump: bool = False,
) -> list[Path]:
  """Generate the ctx.bin file from a given model.

  Args:
      model_path (Path): The path to the model.
      soc_model (str): The SOC model to use.
      optrace (bool): Enable optrace.
      per_layer_dump (bool): Enalbe per-layer dump.

  Returns:
      Optional[Path]: The path to the generated ctx.bin file, or None if the
      generation failed.

  Raises:
      RuntimeError: If the context binary path cannot be extracted from the
        subprocess output.
  """
  tmp_tflite_path = ASSETS_DIR / "tmp.tflite"
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
  ]
  if optrace:
    apply_pluin_main_cmd.append("--qualcomm_profiling=optrace")
  if per_layer_dump:
    apply_pluin_main_cmd.append("--qualcomm_dump_tensor_ids=-1")
  logging.debug(" ".join(str(item) for item in apply_pluin_main_cmd))
  subprocess.run(
      apply_pluin_main_cmd, check=True, cwd=Path(__file__).resolve().parent
  )

  extract_bytecode_lst = [
      *bazel_run,
      "//litert/tools:extract_bytecode",
      "--",
      f"--model_path={tmp_tflite_path}",
      f"--output_dir={ASSETS_DIR}",
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
  match = re.findall(r"bytes to '([^']+)'", result.stderr)
  if not match:
    raise RuntimeError("Failed to generate context binary.")
  return [Path(output_path) for output_path in match]


def _push_so(adb_cmd: str, htp_arch: str, qairt_sdk: Path):
  """Push shared object (.so) and binary files required by qnn-net-run to the target device.

  Args:
      adb_cmd (str): The base adb command (e.g., "adb" or "adb -s <device_id>").
      htp_arch (str): The htp architecture of the target device (e.g., "V75").
      qairt_sdk (Path): The path to the Qairt SDK.
  """
  logging.info("Pushing .so files to the target device...")
  cmd = (
      f'{adb_cmd} shell "rm -rf {DEVICE_WORKING_DIR} && mkdir -p'
      f' {DEVICE_WORKING_DIR}"'
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
    cmd = f"{adb_cmd} push {qairt_sdk / file_path} {DEVICE_WORKING_DIR}"
    logging.debug(cmd)
    subprocess.run(cmd, check=True, shell=True)


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
      ASSETS_DIR / "inputs",
      ASSETS_DIR / "htp_ext_config.json",
      ASSETS_DIR / "config.json",
      Path(ctx_bin_path).resolve(),
  ]:
    cmd = f"{adb_cmd} push {file_path} {DEVICE_WORKING_DIR}"
    subprocess.run(
        cmd, check=True, shell=True, cwd=Path(__file__).resolve().parent
    )
    logging.debug(cmd)
    if file_path.name == "inputs":
      shutil.rmtree(file_path)
      logging.debug("Removing inputs...")


def run_ctx_bin(
    adb_cmd: str,
    ctx_bin_path: Path,
    htp_arch: str,
    qairt_sdk: Path,
    graph_idx: int,
    ctx_bin_info: dict[str, Any],
    optrace: bool = False,
) -> None:
  """Run qnn-net-run with the given context binary.

  Args:
      adb_cmd (str): The adb command.
      ctx_bin_path (str): The path of the context binary.
      htp_arch (str): The htp architecture of the target device (e.g., "V75").
      qairt_sdk (Path): The path to the Qairt SDK.
      graph_idx (int): The index of the graph to run.
      ctx_bin_info (dict[str, Any]): A dictionary containing the context binary
        information, typically obtained from `get_ctx_bin_info`.
      optrace (bool): Enable optrace.
  """
  num_graphs = len(ctx_bin_info["graphs"])
  graph_name = ctx_bin_info["graphs"][graph_idx]["info"]["graphName"]
  input_list_str = ",".join(
      (
          f"inputs/{ctx_bin_path.stem}/{graph_name}/input_list.txt"
          if idx == graph_idx
          else "__"
      )
      for idx in range(num_graphs)
  )

  _push_so(adb_cmd, htp_arch, qairt_sdk)
  _push_target(adb_cmd, ctx_bin_path)
  logging.info("Exectuing qnn-net-run with the given context binary...")
  env_vars = (
      f"export LD_LIBRARY_PATH={DEVICE_WORKING_DIR} && "
      f"export ADSP_LIBRARY_PATH={DEVICE_WORKING_DIR} && "
      f"cd {DEVICE_WORKING_DIR}"
  )
  run_cmd = (
      "./qnn-net-run "
      "--backend libQnnHtp.so "
      f"--retrieve_context {Path(ctx_bin_path).name} "
      f"--input_list {input_list_str} "
      "--output_dir output_htp "
      "--use_native_input_files "
      "--use_native_output_files "
      "--config_file config.json "
  )
  if optrace:
    run_cmd += (
        "--profiling_option optrace "
        "--profiling_level detailed"
    )
  full_cmd = f'{adb_cmd} shell "{env_vars} && {run_cmd}"'
  logging.debug(full_cmd)
  subprocess.run(full_cmd, check=True, shell=True)
