"""Run the profiling script."""

import os
import subprocess
import json
import getpass
import argparse
import shutil
import numpy as np

_QNN_HOME = os.path.join(os.environ["LITERT_QAIRT_SDK"], "latest")
_TOOLS_X86 = os.path.join(_QNN_HOME, "bin", "x86_64-linux-clang")
_WRD = f"/data/local/tmp/{getpass.getuser()}/litert"
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _generate_inputs(ctx_bin_path: str) -> None:
    """
    Generate zero inputs using qnn-context-binary-utility.

    Parameters:
        ctx_bin_path (str): Path to the context binary.
    """
    json_path = os.path.join(_ASSETS_DIR, "tmp.json")
    subprocess.run(
        [
            os.path.join(_TOOLS_X86, "qnn-context-binary-utility"),
            "--context_binary",
            f"{ctx_bin_path}",
            "--json_file",
            json_path,
        ],
        check=True,
    )

    with open(json_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    assert (
        len(json_data["info"]["graphs"]) == 1
    ), "The input generator currently supports only one graph in context binary."
    graph = json_data["info"]["graphs"][0]
    dtype_map = {
        "QNN_DATATYPE_INT_8": np.int8,
        "QNN_DATATYPE_INT_16": np.int16,
        "QNN_DATATYPE_INT_32": np.int32,
        "QNN_DATATYPE_INT_64": np.int64,
        "QNN_DATATYPE_UINT_8": np.int8,
        "QNN_DATATYPE_UINT_16": np.int16,
        "QNN_DATATYPE_UINT_32": np.int32,
        "QNN_DATATYPE_UINT_64": np.int64,
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
        assert dtype is not None, f"Assertion failed: Unknown datatype {qnn_datatype}"
        input_tensor = np.zeros(inp["info"]["dimensions"]).astype(dtype)
        input_list.append(input_tensor)
    graph_name = graph["info"]["graphName"]
    graph_input_path = os.path.join(_ASSETS_DIR, "inputs", graph_name)
    os.makedirs(graph_input_path, exist_ok=True)
    for input_id, input_tensor in enumerate(input_list):
        input_tensor.tofile(os.path.join(graph_input_path, f"input_{input_id}.raw"))
    with open(
        os.path.join(graph_input_path, "input_list.txt"), "w", encoding="utf-8"
    ) as f:
        for input_id, _ in enumerate(input_list):
            inputs_path = os.path.join("inputs", graph_name, f"input_{input_id}.raw")
            f.write(inputs_path)
            if input_id != len(input_list) - 1:
                f.write("\n")
    os.remove(json_path)


def _push_so(adb_cmd: str, htp_arch: str):
    """
    Push shared object (.so) and binary files required by qnn-net-run to the target device.

    Parameters:
        adb_cmd (str): The base adb command (e.g., "adb" or "adb -s <device_id>").
        _QNN_HOME (str): The root directory of the QNN SDK.
    """
    cmd = adb_cmd + " shell " + f'"rm -rf {_WRD} && mkdir -p {_WRD}"'
    print(cmd)
    subprocess.run(cmd, check=True, shell=True)
    for file_path in [
        "bin/aarch64-android/qnn-net-run",
        "lib/aarch64-android/libQnnHtp.so",
        "lib/aarch64-android/libQnnHtpNetRunExtensions.so",
        "lib/aarch64-android/libQnnHtpPrepare.so",
        f"lib/aarch64-android/libQnnHtp{htp_arch.upper()}Stub.so",
        f"lib/hexagon-{htp_arch.lower()}/unsigned/libQnnHtp{htp_arch.upper()}Skel.so",
    ]:
        cmd = adb_cmd + " push " + f"{os.path.join(_QNN_HOME, file_path)} {_WRD}"
        subprocess.run(cmd, check=True, shell=True)


def _get_adb_cmd(hostname: str, serial: str) -> str:
    """
    Get adb command with hostname and serial.

    Parameters:
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
    """
    Push inputs and context binary to the device.

    Parameters:
        adb_cmd (str): The adb command.
        inputs_path (str): The path to the inputs.
        ctx_bin_path (str): The path to the context binary.
    """
    inputs_path = os.path.join(_ASSETS_DIR, "inputs")
    cmd = adb_cmd + " push " + f"{inputs_path} {_WRD}"
    subprocess.run(cmd, check=True, shell=True)
    cmd = adb_cmd + " push " + f"{os.path.abspath(ctx_bin_path)} {_WRD}"
    subprocess.run(cmd, check=True, shell=True)
    shutil.rmtree(inputs_path)

    assert_json_path = os.path.join(_ASSETS_DIR, "htp_ext_config.json")
    cmd = adb_cmd + f" push " f"{assert_json_path} {_WRD}"
    subprocess.run(
        cmd, check=True, shell=True, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    assert_json_path = os.path.join(_ASSETS_DIR, "config.json")
    cmd = adb_cmd + " push " + f"{assert_json_path} {_WRD}"
    subprocess.run(
        cmd, check=True, shell=True, cwd=os.path.dirname(os.path.abspath(__file__))
    )


def _run_ctx_bin(adb_cmd: str, ctx_bin_path: str) -> None:
    """
    Run qnn-net-run with the given context binary.

    Parameters:
        adb_cmd (str): The adb command.
        ctx_bin_name (str): The path of the context binary.
    """

    env_vars = (
        f"export LD_LIBRARY_PATH={_WRD} && "
        f"export ADSP_LIBRARY_PATH={_WRD} && "
        f"cd {_WRD}"
    )
    run_cmd = (
        "./qnn-net-run "
        "--backend libQnnHtp.so "
        f"--retrieve_context {os.path.basename(ctx_bin_path)} "
        "--input_list inputs/qnn_partition_0/input_list.txt "
        "--output_dir output_htp "
        "--use_native_input_files "
        "--use_native_output_files "
        "--config_file config.json "
        "--profiling_option optrace "
        "--profiling_level detailed"
    )
    full_cmd = f'{adb_cmd} shell "{env_vars} && {run_cmd}"'
    subprocess.run(full_cmd, check=True, shell=True)


def _generate_profiler_output(
    adb_cmd: str, schematic_bin_path: str, output_dir: str
) -> None:
    """
    Generate profiler output.

    Parameters:
        adb_cmd (str): The adb command.
        schematic_bin_path (str): The path to the schematic binary.
        output_dir (str): The path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = adb_cmd + f" pull {_WRD}/output_htp ./"
    subprocess.run(cmd, check=True, shell=True)
    log_path = os.path.join("output_htp", "qnn-profiling-data_0.log")
    cmd_lst = [
        os.path.join(_QNN_HOME, "bin", "x86_64-linux-clang", "qnn-profile-viewer"),
        "--input_log",
        log_path,
        "--config",
        os.path.join(_ASSETS_DIR, "config_viewer.json"),
        "--reader",
        os.path.join(
            _QNN_HOME,
            "lib",
            "x86_64-linux-clang",
            "libQnnHtpOptraceProfilingReader.so",
        ),
        "--schematic",
        schematic_bin_path,
        "--output",
        os.path.join(output_dir, "chromeTrace.json"),
    ]
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        cmd_lst,
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Given a ctx bin, generate QNN HTP Optrace Profiling."
    )
    parser.add_argument(
        "--ctx_bin", type=str, help="path to the context binary", required=True
    )
    parser.add_argument(
        "--schematic_bin", type=str, help="path to the schematic binary", required=True
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, help="path to output folder", required=True
    )
    parser.add_argument(
        "--hostname", "-H", type=str, help="hostname for adb", default="localhost"
    )
    parser.add_argument(
        "--serial", "-s", type=str, help="serial for adb", required=True
    )
    parser.add_argument(
        "--htp_arch", "-a", type=str, help="HTP Arch (e.g. V75)", required=True
    )
    args = parser.parse_args()
    _generate_inputs(args.ctx_bin)
    ADB_CMD = _get_adb_cmd(args.hostname, args.serial)
    _push_so(ADB_CMD, args.htp_arch)
    _push_target(ADB_CMD, args.ctx_bin)
    _run_ctx_bin(ADB_CMD, args.ctx_bin)
    _generate_profiler_output(ADB_CMD, args.schematic_bin, args.output_dir)
