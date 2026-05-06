<!-- Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Intel OpenVINO NPU Support for LiteRT

This directory contains the Intel OpenVINO integration for LiteRT, providing
compiler plugin and dispatch API support for Intel NPU hardware.

Validated on: Linux (PTL, Ubuntu 24.04, OpenVINO 2026.1.0, NPU driver v1.32.1)
and Windows 11 (Core Ultra 2/3, NPU driver 32.0.100.4724).

## Supported Platforms

Platform                  | NPU     | Codename           | OS
------------------------- | ------- | ------------------ | --------------
Intel Core Ultra Series 2 | NPU4000 | Lunar Lake (LNL)   | Linux, Windows
Intel Core Ultra Series 3 | NPU5010 | Panther Lake (PTL) | Linux, Windows

## Prerequisites

Requirement | Linux                              | Windows
----------- | ---------------------------------- | -------
OS          | Ubuntu 22.04 or 24.04 LTS (x86_64) | Windows 10/11 (x86_64)
Python      | 3.10 – 3.13                        | 3.11 only
NPU driver  | v1.32.1 (system `.deb`)            | 32.0.100.4724+ ([download](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html))

`openvino==2026.1.0` is installed transitively by
`ai-edge-litert-sdk-intel-nightly`. The Intel SDK nightly also fetches the NPU
compiler shared library (`libopenvino_intel_npu_compiler.so` /
`.dll`) into its `data/` dir at `pip install` time — without it, AOT fails with
`Device with "NPU" name is not registered`. To override Linux distro detection,
set `LITERT_OV_OS_ID=ubuntu22` or `ubuntu24` before `pip install`.

For building from source: Bazel 7.4.1+ via
[Bazelisk](https://github.com/bazelbuild/bazelisk) or Docker.

## Quick Start

### 1. Install NPU Drivers

See [Linux NPU Setup](#linux-npu-setup) or [Windows NPU Setup](#windows-npu-setup).
Skip if you only need AOT.

### 2. Install the pip Package

```bash
pip install ai-edge-litert-nightly ai-edge-litert-sdk-intel-nightly
```

The runtime wheel ships the compiler plugin
(`libLiteRtCompilerPlugin_IntelOpenvino.{so,dll}`) and dispatch library
(`libLiteRtDispatch_IntelOpenvino.{so,dll}`); the SDK wheel adds the NPU
compiler library and `openvino==2026.1.0`. Use the nightlies until a stable
release picks up the SDK auto-download change.

### 3. Verify Installation

```bash
python3 -c "
from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend
import ai_edge_litert_sdk_intel, openvino, os
print('Backend:', intel_openvino_backend.IntelOpenVinoBackend.id())
print('Dispatch:', intel_openvino_backend.get_dispatch_dir())
print('OpenVINO:', openvino.__version__)
print('SDK libs:', sorted(os.listdir(ai_edge_litert_sdk_intel.path_to_sdk_libs())))
"
```

`SDK libs` must list `libopenvino_intel_npu_compiler.so` (Linux) or
`openvino_intel_npu_compiler.dll` (Windows).

### 4. Run NPU Inference

```python
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator

model = CompiledModel.from_file(
    "model.tflite",
    hardware_accel=HardwareAccelerator.NPU | HardwareAccelerator.CPU,
)

sig_key = list(model.get_signature_list().keys())[0]
sig_idx = model.get_signature_index(sig_key)
input_buffers = model.create_input_buffers(sig_idx)
output_buffers = model.create_output_buffers(sig_idx)
model.run_by_index(sig_idx, input_buffers, output_buffers)
print("Fully accelerated:", model.is_fully_accelerated())
```

### 5. Benchmark

```bash
# Dispatch library and the NPU compiler are auto-discovered from the wheel.
litert-benchmark --model=model.tflite --use_npu --num_runs=50
```

Extra flags: `--require_full_delegation` (fail if not fully offloaded),
`--result_json=results.json`.

--------------------------------------------------------------------------------

## Verifying NPU Execution

NPU execution is confirmed by `Loading shared library:
.../libLiteRtDispatch_IntelOpenvino.so` in the log **and**
`model.is_fully_accelerated() == True`. Without the dispatch load line, the
model ran on XNNPACK/CPU even when `is_fully_accelerated()` is true.

--------------------------------------------------------------------------------

## AOT Compilation

AOT pre-compiles a `.tflite` for a specific Intel NPU target (PTL or LNL), so
the runtime skips the compiler plugin step. AOT does not need a physical NPU —
only the Intel SDK nightly. Output files are named
`<model>_IntelOpenVINO_<SoC>_apply_plugin.tflite`.

```python
from ai_edge_litert.aot import aot_compile

# All Intel targets, or target_models=["PTL"] for one.
aot_compile.aot_compile("model.tflite", output_dir="out", keep_going=True)
```

Run an AOT-compiled model with the same snippet as step 4 of
[Quick Start](#quick-start) — `Environment.create()` auto-discovers the
dispatch library under the installed `ai_edge_litert` package, so no explicit
`dispatch_library_path` is needed. Override it only when shipping the dispatch
library outside the wheel.

--------------------------------------------------------------------------------

## Linux NPU Setup

Skip this section if you only need AOT — a physical NPU isn't required.

NPU driver **v1.32.1** is the release paired with OpenVINO 2026.1; older
drivers fail with
`Level0 pfnCreate2 result: ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`. See the
[driver release notes](https://github.com/intel/linux-npu-driver/releases/tag/v1.32.1).

```bash
# 1. NPU driver (Ubuntu 24.04 — use -ubuntu2204 tarball for 22.04).
sudo dpkg --purge --force-remove-reinstreq \
  intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu intel-level-zero-npu-dbgsym || true
wget https://github.com/intel/linux-npu-driver/releases/download/v1.32.1/linux-npu-driver-v1.32.1.20260422-24767473183-ubuntu2404.tar.gz
tar -xf linux-npu-driver-v1.32.1.*.tar.gz
sudo apt update && sudo apt install -y libtbb12
sudo dpkg -i intel-fw-npu_*.deb intel-level-zero-npu_*.deb intel-driver-compiler-npu_*.deb

# 2. Level Zero loader v1.27.0.
wget https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260324T100000Z/pool/main/l/level-zero-loader/libze1_1.27.0-1~24.04~ppa2_amd64.deb
sudo dpkg -i libze1_*.deb

# 3. Permissions + verify.
sudo gpasswd -a ${USER} render && newgrp render
ls /dev/accel/accel0   # must exist after reboot
```

Then run the install + verify snippet from [Quick Start](#quick-start).

--------------------------------------------------------------------------------

## Windows NPU Setup

Skip this section if you only need AOT. Install the Intel NPU driver
(32.0.100.4724+) from the
[Intel Download Center](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html);
Device Manager must list "Neural processors → Intel(R) AI Boost". Then run the
install + verify snippet from [Quick Start](#quick-start) (replace `pip` with
`python -m pip`).

`import ai_edge_litert` auto-registers DLL directories via
`os.add_dll_directory()`, so Python scripts need no PATH setup. For non-Python
consumers, run `setupvars.bat` or prepend `<openvino>/libs` to `PATH`.

--------------------------------------------------------------------------------

## Building from Source

Linux (Docker, hermetic):

```bash
cd LiteRT/docker_build && ./build_wheel_with_docker.sh
```

Windows (PowerShell, Bazel in PATH):

```powershell
.\ci\build_pip_package_with_bazel_windows.ps1
```

Outputs land in `dist/`:
`ai_edge_litert-*.whl` plus one sdist per vendor (`ai_edge_litert_sdk_intel-*.tar.gz`,
qualcomm, mediatek, samsung). The Intel sdist is ~5 KB — the NPU compiler
`.so`/`.dll` is fetched at `pip install` time on the end user's machine, so
the same sdist works for both Linux and Windows users.

Set `http_proxy` / `https_proxy` / `no_proxy` before running if you are behind a
proxy; the scripts forward them into Docker and the container.

--------------------------------------------------------------------------------

## Unit Tests

```bash
bazel test \
  //litert/python/aot/vendors/intel_openvino:intel_openvino_backend_test \
  //litert/c/options:litert_intel_openvino_options_test \
  //litert/cc/options:litert_intel_openvino_options_test \
  //litert/tools/flags/vendors:intel_openvino_flags_test
```

--------------------------------------------------------------------------------

## Troubleshooting

Issue                                                    | Fix
-------------------------------------------------------- | ---
AOT fails: `Device with "NPU" name is not registered`    | NPU compiler not fetched. Check `ai_edge_litert_sdk_intel.path_to_sdk_libs()` lists `libopenvino_intel_npu_compiler.so` / `.dll`. If empty, reinstall with network access, or set `LITERT_OV_OS_ID=ubuntu22`/`ubuntu24`.
`Level0 pfnCreate2 result: ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` | Upgrade NPU driver to v1.32.1 (Linux).
`/dev/accel/accel0` not found                            | `sudo dmesg \| grep -i vpu` to debug the driver; reboot after install.
Permission denied on NPU                                 | `sudo gpasswd -a ${USER} render && newgrp render`.
Windows: NPU not in Device Manager                       | Install NPU driver 32.0.100.4724+ from [Intel Download Center](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html).
Windows: `Failed to initialize Dispatch API` / missing DLLs | Ensure `import ai_edge_litert` runs first (auto-registers DLL dirs); for non-Python callers, run `setupvars.bat` or prepend `<openvino>/libs` to `PATH`.
Windows build: `LNK2001 fixed_address_empty_string`, `C2491 dllimport`, `Python 3.12+ fails` | Protobuf ABI / Python version constraint — see `ci/build_pip_package_with_bazel_windows.ps1`; Windows builds require Python 3.11.

--------------------------------------------------------------------------------

## Limitations

Only the NPU device is supported through the OpenVINO dispatch path. For CPU
inference, use `HardwareAccelerator.CPU` alone (XNNPACK).
