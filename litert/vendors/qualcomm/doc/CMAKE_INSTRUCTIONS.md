# CMake Toolchain Instructions

This doc covers cross-compiling LiteRT with a custom CMake toolchain file.

For the host (x86 Linux) and Android builds, see
[HTP_INSTRUCTIONS.md](./HTP_INSTRUCTIONS.md) > Build with CMake — those use the
`default` and `android-arm64` presets and need no toolchain file from you.

Use the `custom-toolchain` preset to point CMake at any toolchain file:

```bash
export LITERT_TOOLCHAIN_FILE=/path/to/your.toolchain.cmake
cmake --preset custom-toolchain
cmake --build cmake_build_custom_toolchain
```

To change the build type, output directory, or other preset settings, edit the
`custom-toolchain` entry in [../../../CMakePresets.json](../../../CMakePresets.json).

## What to set in a toolchain file

### Required

| Variable | Purpose |
|---|---|
| `CMAKE_SYSTEM_NAME` | Target OS (e.g. `Linux`). Setting it enables cross-compile mode. |
| `CMAKE_SYSTEM_PROCESSOR` | Target arch (e.g. `aarch64`). |
| `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` | Target compilers. |
| `CMAKE_SYSROOT` | Target sysroot (headers + libs). |

### Required for cross-compilation

Without these, CMake's `find_program`/`find_package` can pick a target-arch
binary from the sysroot that can't run on your x86_64 host:

```cmake
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)   # never run target binaries on host
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)    # libs only from sysroot
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

### Optional

| Variable | Purpose |
|---|---|
| `LITERT_HOST_C_COMPILER` / `LITERT_HOST_CXX_COMPILER` | Host compiler used to build the host-side `flatc`. Only set if the auto-detect (`clang`→`cc`→`gcc`) picks the wrong one. |

## Example: Custom Qualcomm aarch64 (oe-linux) toolchain

A minimal toolchain file for a Qualcomm aarch64 target built from a Qualcomm
eSDK. Adjust the three paths to your eSDK install:

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   /opt/sdk/.../aarch64-oe-linux-gcc)
set(CMAKE_CXX_COMPILER /opt/sdk/.../aarch64-oe-linux-g++)
set(CMAKE_SYSROOT      /opt/sdk/.../sysroots/armv8a-oe-linux)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
A ready-to-edit copy lives at
[../toolchain/linux_aarch64_example_toolchain.cmake](../toolchain/linux_aarch64_example_toolchain.cmake).

The example above is for users who need their own toolchain file. LiteRT also
provides the `linux-aarch64-oe-gcc11.2` preset, which wraps
[../toolchain/linux_aarch64_oe_gcc11_2.toolchain.cmake](../toolchain/linux_aarch64_oe_gcc11_2.toolchain.cmake).

For Qualcomm eSDK setup and build commands for IoT devices, such as QCS8275,
including the recommended environment-setup flow and the
`LINUX_AARCH64_ESDK` alternative, see
[Build with CMake for an IoT oe-linux device](./HTP_INSTRUCTIONS.md#run-on-device-iot-device-with-oe-linux).
