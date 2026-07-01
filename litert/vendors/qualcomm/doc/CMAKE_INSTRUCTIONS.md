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

## Example: Qualcomm aarch64 (oe-linux)

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

A ready-to-edit copy lives at
[../toolchain/linux_aarch64_example_toolchain.cmake](../toolchain/linux_aarch64_example_toolchain.cmake).

For the fully-supported IQ-8275 path with eSDK setup, see
[HTP_INSTRUCTIONS.md](./HTP_INSTRUCTIONS.md) — the `linux-aarch64-iq8` preset
wraps [../toolchain/linux_aarch64_iq8275.toolchain.cmake](../toolchain/linux_aarch64_iq8275.toolchain.cmake).

