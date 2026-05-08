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

## Supported Platforms

Platform                  | NPU     | Codename           | OS
------------------------- | ------- | ------------------ | --------------
Intel Core Ultra Series 2 | NPU4000 | Lunar Lake (LNL)   | Linux, Windows
Intel Core Ultra Series 3 | NPU5010 | Panther Lake (PTL) | Linux, Windows

## Prerequisites

**Linux (x86_64):**

-   Ubuntu 22.04 or 24.04 LTS
-   Python 3.10+ — install from [python.org](https://www.python.org/downloads/)
    or your distro (`sudo apt install python3 python3-venv`)
-   Intel NPU driver **v1.32.1** — see [Linux NPU Setup](#linux-npu-setup)

**Windows (x86_64):**

-   Windows 10 or 11
-   Python 3.10+ — install from
    [python.org](https://www.python.org/downloads/windows/)
-   Intel NPU driver **32.0.100.4724+** — see [Windows NPU Setup](#windows-npu-setup)

For building from source, Bazel 7.4.1+ via
[Bazelisk](https://github.com/bazelbuild/bazelisk) or the hermetic Docker
build is also required.

### When is the NPU driver required?

The NPU driver is only needed on systems that **execute** the model on NPU
hardware. Pure AOT-build systems can skip it.

| Task          | NPU driver required? | Notes |
| ------------- | -------------------- | ----- |
| AOT Compile   | No                   | Cross-compilation works — e.g. compile on Linux, run the resulting `.tflite` on Windows. |
| Inference     | Yes                  | The dispatch library talks to the hardware via the Level Zero loader shipped in the NPU driver package. |

> **Note:** `ai-edge-litert-sdk-intel-nightly` pins the matching OpenVINO
> nightly wheel by PEP 440 version (e.g. `openvino==2026.2.0.dev20260506`),
> so pip needs `--extra-index-url
> https://storage.openvinotoolkit.org/simple/wheels/nightly` to locate it —
> see the install command below. On Linux, if distro auto-detection picks
> the wrong archive, set `LITERT_OV_OS_ID=ubuntu22` or `ubuntu24` before
> `pip install`.

## Quick Start

### 1. Install NPU Drivers

See [Linux NPU Setup](#linux-npu-setup) or
[Windows NPU Setup](#windows-npu-setup). Skip if you only need AOT.

### 2. Create a Python Virtual Environment

Recommended to keep the nightly `openvino` wheel isolated from any
system-wide OpenVINO install.

```bash
python -m venv litert_env
# Linux / macOS
source litert_env/bin/activate
# Windows (PowerShell)
.\litert_env\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

### 3. Install the pip Package

```bash
pip install --pre \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly \
    ai-edge-litert-nightly ai-edge-litert-sdk-intel-nightly
```

The `--extra-index-url` lets pip resolve the pinned `openvino` nightly wheel
from OpenVINO's index alongside packages on PyPI.

### 4. Verify Installation

```bash
python -c "
from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend
import ai_edge_litert_sdk_intel, openvino, os
print('Backend:', intel_openvino_backend.IntelOpenVinoBackend.id())
print('Dispatch:', intel_openvino_backend.get_dispatch_dir())
print('OpenVINO:', openvino.__version__)
print('SDK libs:', sorted(os.listdir(ai_edge_litert_sdk_intel.path_to_sdk_libs())))
print('Available devices:', openvino.Core().available_devices)
"
```

What to check in the output:

-   `SDK libs` lists `libopenvino_intel_npu_compiler.so` (Linux) or
    `openvino_intel_npu_compiler.dll` (Windows) — required for AOT.
-   `Available devices` includes `NPU` — confirms the NPU driver is
    installed and OpenVINO can talk to the device. `NPU` will be absent on
    AOT-only systems (where the driver is not installed) and on systems
    without Intel NPU hardware.

### 5. AOT Compile (Optional)

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

Pass the AOT-compiled `.tflite` to the inference snippet below — no extra
wiring needed.

### 6. Run NPU Inference

LiteRT supports two inference paths on Intel NPU:

-   **JIT** — load a raw `.tflite`; the compiler plugin partitions and
    compiles supported ops for the NPU at `CompiledModel.from_file()` time.
    Adds some first-run latency (varies by model).
-   **AOT-compiled** — load a `<model>_IntelOpenVINO_<SoC>_apply_plugin.tflite`
    produced by step 4. Skips the partition/compile step at load time.

The snippet below works for both:

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

#### Confirming JIT actually ran

When JIT succeeds, the log contains (file extension is `.so` on Linux,
`.dll` on Windows):

```
INFO: [compiler_plugin.cc:236] Loaded plugin at: .../LiteRtCompilerPlugin_IntelOpenvino.{so,dll}
INFO: [compiler_plugin.cc:690] Partitioned subgraph<0>, selected N ops, from a total of N ops
INFO: [compiled_model.cc:1006] JIT compilation changed model, reserializing...
```

If those lines are absent but `Fully accelerated: True` is still reported,
the model was run on XNNPACK CPU fallback, not on the NPU — see the JIT
troubleshooting row below.

### 7. Benchmark

```bash
# Dispatch library and the NPU compiler are auto-discovered from the wheel.
litert-benchmark --model=model.tflite --use_npu --num_runs=50
```

Common flags:

| Flag                        | Default | Description |
| --------------------------- | ------- | ----------- |
| `--model PATH`              | —       | Path to the `.tflite` model (required). |
| `--signature KEY`           | first   | Signature key to run. |
| `--use_cpu` / `--no_cpu`    | on      | Toggle the CPU accelerator / CPU fallback. |
| `--use_gpu`                 | off     | Enable the GPU accelerator. |
| `--use_npu`                 | off     | Enable the Intel NPU accelerator. |
| `--require_full_delegation` | off     | Fail if the model is not fully offloaded to the selected accelerator. |
| `--num_runs N`              | 50      | Number of timed inference iterations. |
| `--warmup_runs N`           | 5       | Untimed warm-up iterations before measurement. |
| `--num_threads N`           | 1       | CPU thread count. |
| `--result_json PATH`        | —       | Write a JSON summary (latency stats, throughput, accelerator list). |
| `--verbose`                 | off     | Extra runtime logging. |

Advanced / override flags — only needed to point at custom builds:
`--dispatch_library_path`, `--compiler_plugin_path`, `--runtime_path`.

### Mixed-vendor wheels: pinning JIT to Intel OV

> **Note:** When `Environment.create()` is called without explicit paths,
> it auto-discovers vendors under `ai_edge_litert/vendors/` in alphabetical
> order and registers the first one it finds. In a mixed-vendor install
> this may not be Intel OV — pass the Intel OV directories explicitly to
> force the right pick.

-   The pip wheel ships compiler plugins for every registered vendor
    (`intel_openvino/`, `google_tensor/`, `mediatek/`, `qualcomm/`,
    `samsung/`).
-   To force the Intel OV path (recommended when multiple vendor SDKs are
    installed), pass the Intel OV directories by hand:

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


--------------------------------------------------------------------------------

## Verifying NPU Execution

To confirm the model actually ran on the NPU, check for **both** signals:

1.  The log contains `Loading shared library:
    .../LiteRtDispatch_IntelOpenvino.{so,dll}` — the Intel dispatch library
    was loaded (`.so` on Linux, `.dll` on Windows).
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
-   Verify Device Manager lists the NPU device under **Neural processors**
    (shown as `Intel(R) AI Boost` or `Intel(R) NPU` depending on the driver).
-   Run the install + verify snippet from [Quick Start](#quick-start), replacing
    `pip` with `python -m pip`.

> **Info:** `import ai_edge_litert` auto-registers DLL directories via
> `os.add_dll_directory()`, so Python scripts need no `PATH` setup. For
> non-Python consumers, run `setupvars.bat` or prepend `<openvino>/libs` to
> `PATH`.

--------------------------------------------------------------------------------

## Building from Source

> **Behind a proxy?** Export `http_proxy` / `https_proxy` / `no_proxy`
> before running the build scripts — they forward these into Docker and
> the container.

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
JIT fails: `Cannot load library .../openvino/libs/libopenvino_intel_npu_compiler.so` (Linux) / `openvino_intel_npu_compiler.dll` (Windows) | The SDK sdist copies the NPU compiler to `openvino/libs/` on first `import ai_edge_litert_sdk_intel`. If the copy was skipped (readonly FS, missing `openvino`), reinstall `ai-edge-litert-sdk-intel` after `openvino` is installed, then `import ai_edge_litert` in a fresh process.
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
