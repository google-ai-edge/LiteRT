<!-- Copyright (C) 2026 Intel Corporation
     SPDX-License-Identifier: Apache-2.0

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

## Table of Contents

- [Supported Platforms](#supported-platforms)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Verifying NPU Execution](#verifying-npu-execution)
- [OpenVINO Configuration Options](#openvino-configuration-options)
  - [AOT Compilation](#aot-compilation)
- [Setting Up an Intel NPU Device on Linux (PTL / LNL)](#setting-up-an-intel-npu-device-on-linux-ptl--lnl)
- [Setting Up an Intel NPU Device on Windows](#setting-up-an-intel-npu-device-on-windows)
- [Validating on a PTL Device](#validating-on-a-ptl-device)
- [Building the Python Wheel from Source](#building-the-python-wheel-from-source)
- [Running Unit Tests](#running-unit-tests)
- [Troubleshooting](#troubleshooting)
- [Component Versions](#component-versions)
- [Architecture](#architecture)

**Validated on:**
- Linux: Intel Panther Lake (PTL, NPU5010), Ubuntu 24.04, OpenVINO 2026.1.0,
  NPU driver v1.32.0.
- Windows: Intel Core Ultra Series 2/3, Windows 11, OpenVINO 2026.1.0,
  NPU driver 32.0.100.4724.

## Supported Platforms

| Platform | NPU | Codename | OS |
|----------|-----|----------|-----|
| Intel Core Ultra Series 2 | NPU4000 | Lunar Lake (LNL) | Linux, Windows |
| Intel Core Ultra Series 3 | NPU5010 | Panther Lake (PTL) | Linux, Windows |

## Prerequisites

| Requirement | Linux | Windows |
|-------------|-------|---------|
| OS | Ubuntu 24.04 LTS (x86_64) | Windows 10/11 (x86_64) |
| Python | 3.10, 3.11, 3.12, or 3.13 | 3.11 only |
| OpenVINO | `pip install openvino==2026.1.0` | `pip install openvino==2026.1.0` |
| NPU driver | v1.32.0 (system package via `dpkg`) | 32.0.100.4724+ ([download](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)) |

For building from source you also need one of:
- Bazel 7.4.1+ (install via [Bazelisk](https://github.com/bazelbuild/bazelisk))
- Docker (for hermetic builds)

## Quick Start

### 1. Install NPU Drivers

Set up your Intel NPU device first:
- **Linux:** See [Linux Device Setup](#setting-up-an-intel-npu-device-on-linux-ptl--lnl)
- **Windows:** See [Windows Device Setup](#setting-up-an-intel-npu-device-on-windows)

### 2. Install the pip Package

```bash
# Single command — installs wheel + Intel OpenVINO SDK (which pulls in openvino)
pip install ai-edge-litert[npu-intel]

# Or install separately:
pip install ai-edge-litert
pip install ai-edge-litert-sdk-intel
```

The `ai-edge-litert` wheel bundles both shared libraries:

| Library | Linux | Windows |
|---------|-------|---------|
| Compiler plugin | `libLiteRtCompilerPlugin_IntelOpenvino.so` | `LiteRtCompilerPlugin.dll` |
| Dispatch library | `libLiteRtDispatch_IntelOpenvino.so` | `LiteRtDispatch.dll` |

The `[npu-intel]` extra installs `ai-edge-litert-sdk-intel`, which
depends on `openvino==2026.1.0`. If OpenVINO is already installed, pip skips
it automatically.

> **Windows DLL discovery:** `import ai_edge_litert` automatically registers
> the required DLL directories via `os.add_dll_directory()`. Python scripts
> work out of the box without PATH setup. See [Windows Troubleshooting](#windows-1)
> if you encounter DLL errors from non-Python contexts.

### 3. Verify Installation

```bash
# Check Intel backend loads
python3 -c "
from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend
print('Backend ID:', intel_openvino_backend.IntelOpenVinoBackend.id())
print('Dispatch dir:', intel_openvino_backend.get_dispatch_dir())
"

# Check OpenVINO runtime
python3 -c "import openvino; print('OpenVINO:', openvino.__version__)"
```

Expected output:
```
Backend ID: intel_openvino
Dispatch dir: /path/to/site-packages/ai_edge_litert/vendors/intel_openvino/dispatch
OpenVINO: 2026.1.0-...
```

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

---

## Verifying NPU Execution

After running inference, check the runtime log to confirm the model is
executing on the NPU via OpenVINO rather than falling back to CPU.

### Key Log Indicators

| Log line | Meaning |
|----------|---------|
| `NPU accelerator registered.` | Dispatch library found; NPU accelerator loaded. |
| `Loading shared library: .../libLiteRtDispatch_IntelOpenvino.so` | OpenVINO dispatch loaded at model compilation time. |
| `[Openvino]Found device plugin for: CPU` | OpenVINO runtime initialized (CPU plugin always reported; NPU plugin loads internally). |
| `Fully accelerated: True` | All model ops dispatched to the accelerator. |

### NPU Execution (Success)

```
INFO: [auto_registration.cc:171] NPU accelerator registered.
INFO: [litert_dispatch.cc:145] Loading shared library: .../libLiteRtDispatch_IntelOpenvino.so
INFO: [dispatch_api.cc:115] [Openvino]Found device plugin for: CPU
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Fully accelerated: True
```

### CPU-Only Execution (No NPU)

When NPU is not available or the dispatch path is not set, only the XNNPACK
delegate appears — no dispatch library loading:

```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Fully accelerated: True
```

`Fully accelerated: True` with CPU-only means all ops ran on XNNPACK.
The key difference is the **absence** of
`Loading shared library: .../libLiteRtDispatch_IntelOpenvino.so`.

### Programmatic Check

```python
model = CompiledModel.from_file("model.tflite", ...)
print("Fully accelerated:", model.is_fully_accelerated())
```

### Known Harmless Warning

```
WARNING: NPU accelerator could not be loaded and registered: kLiteRtStatusErrorInvalidArgument.
```

This warning appears in two scenarios and **does not affect inference**:

1. **During `Environment.create()` without a dispatch path** — Auto-registration
   tries to register NPU but no dispatch directory was provided. When using the
   dispatch path is provided explicitly, NPU registers on the first attempt and
   this warning only appears on an internal environment created by the runtime.
2. **During the first `model.run()` call** — The LiteRT runtime lazily creates
   an internal environment that does not inherit the dispatch path. This is
   upstream runtime behavior; the NPU dispatch was already initialized from
   the user's environment.

If you see `Loading shared library: .../libLiteRtDispatch_IntelOpenvino.so`
and `Fully accelerated: True`, the model is running on NPU correctly.

---

## AOT Compilation

AOT (ahead-of-time) compilation pre-compiles a `.tflite` model for a specific
Intel NPU target (PTL or LNL). The resulting model loads faster at runtime
because the compiler plugin step is already done.

#### End-to-End Example

```python
from ai_edge_litert.aot import aot_compile

# Compile for all supported Intel targets (LNL + PTL)
results = aot_compile.aot_compile(
    "model.tflite",
    output_dir="compiled_output",
    keep_going=True,  # don't stop on first failure (e.g. missing MediaTek SDK)
)

# Compile for a specific target only
results = aot_compile.aot_compile(
    "model.tflite",
    output_dir="compiled_output",
    target_models=["PTL"],
    keep_going=True,
)
```

Output files are named `<model>_IntelOpenVINO_<SoC>_apply_plugin.tflite`,
e.g. `000_IntelOpenVINO_PTL_apply_plugin.tflite`.

#### Running AOT-Compiled Models

AOT-compiled models still require the dispatch library at runtime but skip the
compiler plugin step:

```python
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.environment import Environment
from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend as ov

env = Environment.create(
    dispatch_library_path=ov.get_dispatch_dir(),
)
model = CompiledModel.from_file(
    "compiled_output/000_IntelOpenVINO_PTL_apply_plugin.tflite",
    hardware_accel=HardwareAccelerator.NPU | HardwareAccelerator.CPU,
    environment=env,
)
# model.run(input_buffers) ...
```

---

## Setting Up an Intel NPU Device on Linux (PTL / LNL)

Follow these steps on your target Intel NPU machine (Panther Lake or Lunar
Lake) running Ubuntu 24.04.

### Step 1: Install NPU Drivers

```bash
# Remove old packages (if any)
sudo dpkg --purge --force-remove-reinstreq \
  intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu intel-level-zero-npu-dbgsym

# Download and extract NPU driver v1.32.0
wget https://github.com/intel/linux-npu-driver/releases/download/v1.32.0/linux-npu-driver-v1.32.0.20260402-23905121947-ubuntu2404.tar.gz
tar -xf linux-npu-driver-v1.32.0.20260402-23905121947-ubuntu2404.tar.gz

# Install dependency
sudo apt update && sudo apt install -y libtbb12

# Install all driver packages
sudo dpkg -i *.deb
```

### Step 2: Install Level Zero Loader

```bash
wget https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260324T100000Z/pool/main/l/level-zero-loader/libze1_1.27.0-1~24.04~ppa2_amd64.deb
sudo dpkg -i libze1_*.deb
```

If you encounter conflicts with existing Level Zero packages:

```bash
sudo dpkg --purge --force-remove-reinstreq level-zero level-zero-devel
sudo dpkg -i libze1_*.deb
```

### Step 3: Install OpenVINO Runtime

**Option A: pip (recommended)**

```bash
pip install openvino==2026.1.0
```

**Option B: APT repository**

```bash
wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  sudo gpg --dearmor -o /usr/share/keyrings/intel-openvino.gpg
echo "deb [signed-by=/usr/share/keyrings/intel-openvino.gpg] https://apt.repos.intel.com/openvino/2026 ubuntu24 main" | \
  sudo tee /etc/apt/sources.list.d/intel-openvino.list
sudo apt update && sudo apt install openvino
```

### Step 4: Configure User Permissions

```bash
sudo gpasswd -a ${USER} render
newgrp render
```

### Step 5: Verify NPU Device

```bash
# Check NPU device exists
ls /dev/accel/accel0

# Check driver loaded
sudo dmesg | grep -i vpu
```

After a reboot, you should see `/dev/accel/accel0` if the NPU driver is
loaded correctly.

### Step 6: Install and Test

```bash
pip install ai-edge-litert[npu-intel]

# Verify backend
python3 -c "
from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend
print('Backend ID:', intel_openvino_backend.IntelOpenVinoBackend.id())
print('Dispatch dir:', intel_openvino_backend.get_dispatch_dir())
"
```

---

## Setting Up an Intel NPU Device on Windows

Follow these steps on a Windows machine with Intel NPU hardware (Panther Lake
or Lunar Lake).

**Validated on:** Windows 11, Intel Core Ultra Series 2/3, NPU driver
32.0.100.4724, OpenVINO 2026.1.0, Python 3.11 (only supported version on
Windows — hardcoded in `.bazelrc` `build:windows` config).

### Step 1: Install Intel NPU Driver

Download and install the Intel NPU driver from:
https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html

Tested version: **32.0.100.4724**

After installation, verify the NPU device appears in Device Manager under
"Neural processors" -> "Intel(R) AI Boost".

### Step 2: Install and Verify

```powershell
python -m pip install ai-edge-litert[npu-intel]

# Verify backend
python -c "from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend; print('Backend ID:', intel_openvino_backend.IntelOpenVinoBackend.id()); print('Dispatch dir:', intel_openvino_backend.get_dispatch_dir())"

# Verify OpenVINO
python -c "import openvino; print('OpenVINO:', openvino.__version__)"
```

### Step 3: Run NPU Inference

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

---

## Validating on a PTL Device

Copy the wheel to your PTL device and run the verification steps:

```bash
# From build machine
scp dist/ai_edge_litert-*.whl intel@<PTL_HOST>:/path/to/workdir/

# On the PTL device
ssh intel@<PTL_HOST>
cd /path/to/workdir
python3 -m venv venv && source venv/bin/activate
pip install ai_edge_litert-*.whl[npu-intel]

# Verify backend
python3 -c "
from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend
print('Backend ID:', intel_openvino_backend.IntelOpenVinoBackend.id())
print('Dispatch dir:', intel_openvino_backend.get_dispatch_dir())
"
```

---

## Building the Python Wheel from Source

### Option A: Docker Hermetic Build (Recommended)

No local Bazel or build tools required. The Docker image includes all
dependencies (Bazel 7.4.1, Android SDK/NDK, Clang 18, Python).

```bash
cd LiteRT/docker_build
./build_wheel_with_docker.sh
```

#### Proxy Configuration

If you are behind a corporate proxy, export the standard proxy environment
variables **before** running any Docker build script. The scripts automatically
forward these into the Docker image build (`--build-arg`) and container
runtime (`-e`). When unset or empty, the proxy code is a complete no-op.

```bash
export http_proxy="http://proxy.example.com:912"
export https_proxy="http://proxy.example.com:912"
export no_proxy="localhost,127.0.0.1,.example.com"

cd LiteRT/docker_build
./build_wheel_with_docker.sh
```

Three layers of proxy forwarding are handled automatically:

| Layer | How | Used by |
|-------|-----|---------|
| Docker image build | `--build-arg http_proxy=...` | `apt-get`, `wget`, `pip` during image creation |
| Container runtime | `-e http_proxy=...` | `curl`, `pip`, general network access |
| Bazel JVM downloader | `--host_jvm_args=-Dhttps.proxyHost=...` | Bazel repository fetches |

Outputs in `dist/`:
- `ai_edge_litert-*.whl` (wheel with compiler plugin + dispatch library)
- `ai_edge_litert_sdk_qualcomm-*.tar.gz`
- `ai_edge_litert_sdk_mediatek-*.tar.gz`
- `ai_edge_litert_sdk_intel-*.tar.gz`

Options:
- `--use_existing_image` — skip `docker build`, reuse existing image
- `--dbg` — build in debug mode
- `--python=3.12` — set `HERMETIC_PYTHON_VERSION`

#### Building Individual Intel OpenVINO Binaries with Docker

```bash
cd LiteRT/docker_build
./build_with_docker.sh   # Build image first (only needed once)

docker run --rm \
  --security-opt seccomp=unconfined \
  --user $(id -u):$(id -g) \
  -e HOME=/litert_build \
  -e USER=$(id -un) \
  -e http_proxy="${http_proxy:-}" \
  -e https_proxy="${https_proxy:-}" \
  -e no_proxy="${no_proxy:-}" \
  -v $(cd .. && pwd):/litert_build \
  litert_build_env \
  bash -c '
source /setup_bazel_env.sh
bazel ${EXTRA_STARTUP} build //litert/vendors/intel_openvino/compiler:libLiteRtCompilerPlugin_IntelOpenvino.so
bazel ${EXTRA_STARTUP} build //litert/vendors/intel_openvino/dispatch:dispatch_api_so
bazel ${EXTRA_STARTUP} build //litert/tools:benchmark_model
'
```

### Option B: Manual Bazel Build (No Docker)

Requires Bazel 7.4.1+ (via Bazelisk), Clang, and Python 3.10+.

```bash
cd LiteRT
./configure

bazel build -c opt \
  --cxxopt=-std=gnu++17 \
  --repo_env=USE_PYWRAP_RULES=True \
  //ci/tools/python/wheel:litert_wheel

mkdir -p dist/
cp bazel-bin/ci/tools/python/wheel/dist/*.whl dist/
```

Verify Intel OpenVINO binaries are bundled:
```bash
unzip -l dist/*.whl | grep openvino
```

### Option C: CI Build Script

```bash
cd LiteRT
./configure
bash ci/build_pip_package_with_bazel.sh
```

### Building Individual Shared Libraries

```bash
bazel build //litert/vendors/intel_openvino/compiler:libLiteRtCompilerPlugin_IntelOpenvino.so
bazel build //litert/vendors/intel_openvino/dispatch:dispatch_api_so
bazel build //litert/tools:benchmark_model
```

---

## Running Unit Tests

### Bazel Tests (Source Tree)

```bash
# All Intel OpenVINO tests
bazel test \
  //litert/python/aot/vendors/intel_openvino:intel_openvino_backend_test \
  //litert/c/options:litert_intel_openvino_options_test \
  //litert/cc/options:litert_intel_openvino_options_test \
  //litert/tools/flags/vendors:intel_openvino_flags_test
```

Individual tests:

| Test | Covers |
|------|--------|
| `//litert/python/aot/vendors/intel_openvino:intel_openvino_backend_test` | AOT backend config, specialization, flags |
| `//litert/c/options:litert_intel_openvino_options_test` | C API options |
| `//litert/cc/options:litert_intel_openvino_options_test` | C++ API options |
| `//litert/tools/flags/vendors:intel_openvino_flags_test` | CLI flags parsing |

---

## Troubleshooting

### Linux

| Issue | Solution |
|-------|----------|
| `/dev/accel/accel0` not found | Check NPU driver: `sudo dmesg \| grep -i vpu`. Reboot after installing drivers. |
| Level Zero conflicts | Purge old packages: `sudo dpkg --purge --force-remove-reinstreq level-zero level-zero-devel` |
| OpenVINO import errors | Verify: `python3 -c "import openvino; print(openvino.__version__)"`. Must be 2026.1+. |
| Permission denied on NPU | `sudo gpasswd -a ${USER} render && newgrp render` |
| Compiler plugin not found in wheel | `unzip -l dist/*.whl \| grep compiler` |
| Dispatch library not found in wheel | `unzip -l dist/*.whl \| grep dispatch` |
| `get_dispatch_dir()` returns `None` | Dispatch `.so` not bundled. Check wheel contents or specify `--dispatch_library_path` manually. |
| `directory iterator cannot open directory` | Pass a directory path (not file path) to `dispatch_library_path`. |

### Windows

| Issue | Solution |
|-------|----------|
| NPU not found in Device Manager | Install NPU driver from [Intel Download Center](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html). Reboot after installation. |
| `ModuleNotFoundError: No module named 'openvino'` | `pip install openvino==2026.1.0` |
| NPU accelerator could not be loaded | Verify driver version in Device Manager -> Neural processors -> Properties -> Driver. Must be 32.0.100.4724+. |
| `Failed to initialize Dispatch API` | OpenVINO or LiteRT DLLs not found. Ensure `import ai_edge_litert` runs before dispatch loads (auto-registers DLL dirs). If running outside Python, add OpenVINO libs to PATH: `$OvDir = python -c "import openvino, os; print(os.path.dirname(openvino.__file__))"` then `$env:PATH = "$OvDir\libs;$env:PATH"`. |
| `.dll` not found errors | Same as above. For non-Python contexts, run `setupvars.bat` or set PATH manually. |
| `get_dispatch_dir()` returns `None` | Dispatch DLL not bundled. Check wheel contents or specify `--dispatch_library_path`. If running from the source tree, the fallback looks in the installed `ai_edge_litert` package. |
| `LNK2001: fixed_address_empty_string` during build | Protobuf ABI mismatch. `ci/build_pip_package_with_bazel_windows.ps1` must patch `port.h` to force `GlobalEmptyStringDynamicInit` on MSVC. Do NOT add `PROTOBUF_USE_DLLS` to `.bazelrc.user`. |
| `C2491: definition of dllimport` during build | Remove `PROTOBUF_USE_DLLS` and `LIBPROTOBUF_EXPORTS` from `.bazelrc.user`. |
| Python 3.12+ build fails on Windows | Only Python 3.11 is supported (hardcoded in `.bazelrc` `build:windows`). |

---

## Known Limitations

- **Only NPU device is supported** through the OpenVINO dispatch path. CPU
  inference is available through XNNPACK — use
  `HardwareAccelerator.CPU` without `HardwareAccelerator.NPU` for CPU-only.

---

## Component Versions

| Component | Linux | Windows |
|-----------|-------|---------|
| OpenVINO SDK | 2026.1.0 | 2026.1.0 |
| NPU Driver | v1.32.0 | 32.0.100.4724 |
| Level Zero | v1.27.0 | (bundled with driver) |
| Python | 3.10, 3.11, 3.12, 3.13 | 3.11 only |
| OS | Ubuntu 24.04 LTS | Windows 10/11 |

## Architecture

The Intel OpenVINO integration consists of two components, both bundled in
the `ai-edge-litert` pip wheel:

- **Compiler Plugin** (`compiler/`): Partitions model ops and compiles
  supported subgraphs into OpenVINO IR for NPU execution.

- **Dispatch API** (`dispatch/`): Runtime counterpart that loads compiled
  bytecode and executes it on the NPU via the Level Zero API.

### Python Packaging Layout

```
ai-edge-litert (wheel)
  ai_edge_litert/
    __init__.py                                      # version + Windows DLL auto-discovery
    vendors/intel_openvino/
      compiler/
        libLiteRtCompilerPlugin_IntelOpenvino.so     # compiler plugin
      dispatch/
        libLiteRtDispatch_IntelOpenvino.so            # dispatch library
    aot/vendors/intel_openvino/
      intel_openvino_backend.py                      # AOT backend + dispatch auto-discovery
      target.py                                      # SoC target definitions (LNL, PTL)
  extras_require:
    npu-intel -> ai-edge-litert-sdk-intel~=0.2.0 -> openvino==2026.1.0
```

User-managed prerequisites (not in wheel):
- `openvino==2026.1.0` (via `pip install`, `[npu-intel]` extra, or SDK package)
- Intel NPU driver (system package, requires `sudo` / admin)
- Level Zero loader v1.27.0 (Linux only, system package)

See [COMPILER_PLUGIN.md](../../COMPILER_PLUGIN.md) and
[DISPATCH_API.md](../../DISPATCH_API.md) for the general LiteRT vendor
integration architecture.
