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

| Requirement | Linux                    | Windows                    |
| ----------- | ------------------------ | -------------------------- |
| OS          | Ubuntu 22.04 or 24.04    | Windows 10/11 (x86_64)     |
:             : LTS (x86_64)             :                            :
| Python      | 3.10 – 3.13              | 3.11 only                  |
| NPU driver  | v1.32.1 — see [Linux NPU | 32.0.100.4724+ — see       |
:             : Setup](#linux-npu-setup) : [Windows NPU               :
:             :                          : Setup](#windows-npu-setup) :

### When is the NPU driver required?

| Task                              | Needs NPU driver installed?            |
| --------------------------------- | -------------------------------------- |
| AOT compile a `.tflite` for Intel | **No.** Cross-compilation works — e.g. |
: NPU                               : compile on Linux, run the resulting    :
:                                   : `.tflite` on Windows.                  :
| Run inference on an Intel NPU     | **Yes.** The dispatch library talks to |
: (Python or C++ API, JIT or        : the hardware via the Level Zero loader :
: AOT-compiled model)               : shipped in the NPU driver package.     :

> **In short:** the NPU driver is needed only on machines that **execute** the
> model on NPU hardware. Pure AOT-build machines can skip it.

> **Note:** `openvino==2026.1.0` is installed transitively by
> `ai-edge-litert-sdk-intel-nightly`. The Intel SDK nightly also fetches the NPU
> compiler shared library (`libopenvino_intel_npu_compiler.so` / `.dll`) into
> its `data/` dir at `pip install` time — without it, AOT fails with `Device
> with "NPU" name is not registered`. To override Linux distro detection, set
> `LITERT_OV_OS_ID=ubuntu22` or `ubuntu24` before `pip install`.

For building from source: Bazel 7.4.1+ via
[Bazelisk](https://github.com/bazelbuild/bazelisk) or Docker.

## Quick Start

### 1. Install NPU Drivers

See [Linux NPU Setup](#linux-npu-setup) or
[Windows NPU Setup](#windows-npu-setup). Skip if you only need AOT.

### 2. Install the pip Package

```bash
pip install ai-edge-litert-nightly ai-edge-litert-sdk-intel-nightly
```

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

> **Note:** `SDK libs` must list `libopenvino_intel_npu_compiler.so` (Linux) or
> `openvino_intel_npu_compiler.dll` (Windows).

### 4. AOT Compile (Optional)

-   Pre-compiles a `.tflite` for a specific Intel NPU target (PTL or LNL) so the
    runtime skips the compiler plugin step.
-   Does **not** need a physical NPU or the NPU driver — only
    `ai-edge-litert-nightly` and `ai-edge-litert-sdk-intel-nightly`.
-   Cross-compilation is supported: compile on any Linux or Windows host, ship
    the resulting `.tflite` to a target of either OS and run it there.

Output files are named `<model>_IntelOpenVINO_<SoC>_apply_plugin.tflite`.

```python
from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.intel_openvino import target as intel_target

# Compile for a single Intel NPU target (PTL or LNL).
aot_compile.aot_compile(
    "model.tflite",
    output_dir="out",
    target=intel_target.Target(soc_model=intel_target.SocModel.PTL),
)

# Or omit target= to compile for every registered backend/target.
aot_compile.aot_compile("model.tflite", output_dir="out", keep_going=True)
```

Pass the AOT-compiled `.tflite` to the inference snippet below — no extra wiring
needed. `Environment.create()` auto-discovers the dispatch library under the
installed `ai_edge_litert` package.

### 5. Run NPU Inference

LiteRT supports two inference paths on Intel NPU:

-   **JIT** — load a raw `.tflite`, the compiler plugin partitions and compiles
    supported ops for the NPU at `CompiledModel.from_file()` time. No AOT step
    needed; ~250 ms extra first-run latency for a typical model.
-   **AOT-compiled** — load a `<model>_IntelOpenVINO_<SoC>_apply_plugin.tflite`
    produced by step 4. Skips the partition/compile step at load time.

The snippet below works for both; `Environment.create()` auto-discovers the
Intel OV compiler plugin (JIT only) and dispatch library from the installed
wheel.

```python
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator

model = CompiledModel.from_file(
    "model.tflite",  # raw tflite (JIT) or ..._apply_plugin.tflite (AOT)
    hardware_accel=HardwareAccelerator.NPU | HardwareAccelerator.CPU,
)

sig_key = list(model.get_signature_list().keys())[0]
sig_idx = model.get_signature_index(sig_key)
input_buffers = model.create_input_buffers(sig_idx)
output_buffers = model.create_output_buffers(sig_idx)
model.run_by_index(sig_idx, input_buffers, output_buffers)
print("Fully accelerated:", model.is_fully_accelerated())
```

#### Mixed-vendor wheels: pinning JIT to Intel OV

The pip wheel ships compiler plugins for every registered vendor
(`intel_openvino/`, `google_tensor/`, `mediatek/`, `qualcomm/`, `samsung/`).
Auto-discovery picks the compiler plugin from the same vendor as the selected
dispatch library, and today only Intel OV ships a dispatch library, so
`CompiledModel.from_file(hardware_accel=NPU)` ends up on the Intel OV path by
default.

If you want to be explicit — or if a future wheel bundles additional dispatch
libraries and changes the auto-discovery pick — pass the Intel OV directories by
hand:

```python
from ai_edge_litert.environment import Environment
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend as ov

env = Environment.create(
    compiler_plugin_path=ov.get_compiler_plugin_dir(),   # JIT compiler
    dispatch_library_path=ov.get_dispatch_dir(),          # runtime
)
model = CompiledModel.from_file(
    "model.tflite",
    hardware_accel=HardwareAccelerator.NPU | HardwareAccelerator.CPU,
    environment=env,
)
```

The runtime loads every shared library it finds in the given directory, so
pointing at `vendors/intel_openvino/compiler/` loads only the Intel plugin; the
Google Tensor / MediaTek / Qualcomm / Samsung plugins in sibling directories are
never touched.

For the CLI, the equivalent flags are:

```bash
DISPATCH_DIR=$(python3 -c 'from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend as ov; print(ov.get_dispatch_dir())')
COMPILER_DIR=$(python3 -c 'from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend as ov; print(ov.get_compiler_plugin_dir())')

litert-benchmark --model=model.tflite --use_npu \
    --compiler_plugin_path=$COMPILER_DIR \
    --dispatch_library_path=$DISPATCH_DIR
```

#### Confirming JIT actually ran

When JIT succeeds, the log contains:

```
INFO: [compiler_plugin.cc:236] Loaded plugin at: .../libLiteRtCompilerPlugin_IntelOpenvino.so
INFO: [compiler_plugin.cc:690] Partitioned subgraph<0>, selected N ops, from a total of N ops
INFO: [compiled_model.cc:1006] JIT compilation changed model, reserializing...
```

If those lines are absent but `Fully accelerated: True` is still reported, the
model was run on XNNPACK CPU fallback, not on the NPU — see the JIT
troubleshooting row below.

### 6. Benchmark

```bash
# Dispatch library and the NPU compiler are auto-discovered from the wheel.
litert-benchmark --model=model.tflite --use_npu --num_runs=50
```

Common flags:

| Flag                        | Default | Description                          |
| --------------------------- | ------- | ------------------------------------ |
| `--model PATH`              | —       | Path to the `.tflite` model          |
:                             :         : (required).                          :
| `--signature KEY`           | first   | Signature key to run.                |
| `--use_cpu` / `--no_cpu`    | on      | Toggle the CPU accelerator / CPU     |
:                             :         : fallback.                            :
| `--use_gpu`                 | off     | Enable the GPU accelerator.          |
| `--use_npu`                 | off     | Enable the Intel NPU accelerator.    |
| `--require_full_delegation` | off     | Fail if the model is not fully       |
:                             :         : offloaded to the selected            :
:                             :         : accelerator.                         :
| `--num_runs N`              | 50      | Number of timed inference            |
:                             :         : iterations.                          :
| `--warmup_runs N`           | 5       | Untimed warm-up iterations before    |
:                             :         : measurement.                         :
| `--num_threads N`           | 1       | CPU thread count.                    |
| `--result_json PATH`        | —       | Write a JSON summary (latency stats, |
:                             :         : throughput, accelerator list).       :
| `--verbose`                 | off     | Extra runtime logging.               |

Advanced / override flags — only needed to point at custom builds:
`--dispatch_library_path`, `--compiler_plugin_path`, `--runtime_path`.

--------------------------------------------------------------------------------

## Verifying NPU Execution

To confirm the model actually ran on the NPU, check for **both** signals:

1.  The log contains `Loading shared library:
    .../libLiteRtDispatch_IntelOpenvino.so` — the Intel dispatch library was
    loaded.
2.  `model.is_fully_accelerated()` returns `True` — every op was offloaded to
    the selected accelerator.

`is_fully_accelerated()` alone is **not** sufficient: if the dispatch library
never loaded, ops were fully offloaded to XNNPACK/CPU, not the NPU.

--------------------------------------------------------------------------------

## Linux NPU Setup

> **Note:** Skip this section if you only need AOT — a physical NPU isn't
> required.

> **Info:** Use NPU driver
> **[v1.32.1](https://github.com/intel/linux-npu-driver/releases/tag/v1.32.1)**
> (paired with OpenVINO 2026.1). Older drivers fail with `Level0 pfnCreate2
> result: ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`.

<!--* pragma: { seclinter_this_is_fine: true } *-->

```bash
# 1. NPU driver (Ubuntu 24.04 use -ubuntu2204 tarball for 22.04).
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

> **Note:** Skip this section if you only need AOT — a physical NPU isn't
> required.

-   Install the Intel NPU driver (**32.0.100.4724+**) from the
    [Intel Download Center](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html).
-   Verify Device Manager lists **Neural processors → Intel(R) AI Boost**.
-   Run the install + verify snippet from [Quick Start](#quick-start), replacing
    `pip` with `python -m pip`.

> **Info:** `import ai_edge_litert` auto-registers DLL directories via
> `os.add_dll_directory()`, so Python scripts need no `PATH` setup. For
> non-Python consumers, run `setupvars.bat` or prepend `<openvino>/libs` to
> `PATH`.

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

-   `ai_edge_litert-*.whl` — the runtime wheel.
-   `ai_edge_litert_sdk_{intel,qualcomm,mediatek,samsung}-*.tar.gz` — vendor
    sdists.
-   The Intel sdist is ~5 KB; the NPU compiler `.so`/`.dll` is fetched at `pip
    install` time, so the same sdist works on Linux and Windows.
-   Behind a proxy? Export `http_proxy` / `https_proxy` / `no_proxy` — the
    scripts forward them into Docker and the container.

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

Issue                                                                                                                           | Fix
------------------------------------------------------------------------------------------------------------------------------- | ---
AOT fails: `Device with "NPU" name is not registered`                                                                           | NPU compiler not fetched. Check `ai_edge_litert_sdk_intel.path_to_sdk_libs()` lists `libopenvino_intel_npu_compiler.so` / `.dll`. If empty, reinstall with network access, or set `LITERT_OV_OS_ID=ubuntu22`/`ubuntu24`.
JIT runs on CPU instead of NPU (no `Partitioned subgraph` log, no `Loaded plugin` log, `Fully accelerated: True` still printed) | Compiler plugin was not discovered. Confirm `ov.get_compiler_plugin_dir()` returns a path under `ai_edge_litert/vendors/intel_openvino/compiler/`. If multiple vendor SDKs are installed, pass `compiler_plugin_path=ov.get_compiler_plugin_dir()` explicitly to `Environment.create()` (or `--compiler_plugin_path=...` to `litert-benchmark`).
JIT fails: `Cannot load library .../openvino/libs/libopenvino_intel_npu_compiler.so`                                            | The SDK sdist copies the NPU compiler to `openvino/libs/` on first `import ai_edge_litert_sdk_intel`. If the copy was skipped (readonly FS, missing `openvino`), reinstall `ai-edge-litert-sdk-intel` after `openvino` is installed, then `import ai_edge_litert` in a fresh process.
`Level0 pfnCreate2 result: ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`                                                                 | Upgrade NPU driver to v1.32.1 (Linux).
`/dev/accel/accel0` not found                                                                                                   | `sudo dmesg \| grep -i vpu` to debug the driver; reboot after install.
Permission denied on NPU                                                                                                        | `sudo gpasswd -a ${USER} render && newgrp render`.
Windows: NPU not in Device Manager                                                                                              | Install NPU driver 32.0.100.4724+ from [Intel Download Center](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html).
Windows: `Failed to initialize Dispatch API` / missing DLLs                                                                     | Ensure `import ai_edge_litert` runs first (auto-registers DLL dirs); for non-Python callers, run `setupvars.bat` or prepend `<openvino>/libs` to `PATH`.
Windows build: `LNK2001 fixed_address_empty_string`, `C2491 dllimport`, `Python 3.12+ fails`                                    | Protobuf ABI / Python version constraint — see `ci/build_pip_package_with_bazel_windows.ps1`; Windows builds require Python 3.11.

--------------------------------------------------------------------------------

## Limitations

Only the NPU device is supported through the OpenVINO dispatch path. For CPU
inference, use `HardwareAccelerator.CPU` alone (XNNPACK).
