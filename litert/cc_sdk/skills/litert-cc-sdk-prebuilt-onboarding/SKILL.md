---
name: litert-cc-sdk-prebuilt-onboarding
description: >-
  Onboards, builds, and deploys open-source (OSS) C++ LiteRT CompiledModel
  applications using the prebuilt litert_cc_sdk headers and libLiteRt (.so/.dll/.dylib)
  archive via CMake dynamic linking across Linux, Android, Windows (x86_64 and
  arm64), and Qualcomm IoT on CPU, GPU (OpenCL/WebGPU), and Qualcomm NPU (AOT and
  JIT). Use when integrating LiteRT prebuilt zip archives into an external CMake
  project without building LiteRT from source, placing GPU accelerator or Qualcomm
  QNN NPU dispatch/plugin libraries, running apply_plugin_main for offline AOT
  compilation, or troubleshooting QNN version and FastRPC Skel errors.
---

# LiteRT C++ SDK & Prebuilt Library Onboarding (OSS Guide)

Use this workflow to build and deploy a standalone C++ `litert::CompiledModel`
application against the LiteRT prebuilt archive (`litert_cc_sdk/` +
`libLiteRt.so`/`libLiteRt.dll` + accelerator/dispatch/plugin libraries) using
CMake and dynamic linking, without modifying SDK headers or building LiteRT from
source.

---

## 1. Prebuilt Archive Layout & Platform Selection

Download the nightly archive from
`https://storage.googleapis.com/litert/binaries/nightly/litert_prebuilts.zip`
(or a dated snapshot under
`https://storage.googleapis.com/litert/binaries/nightly/archive/litert_prebuilts_<YYYYMMDD>.zip`):

| Subdirectory | Contents & Target Environment |
| :--- | :--- |
| `litert_cc_sdk/` | C/C++ headers (`litert/c/`, `litert/cc/`, `litert/cc/options/`), thin C++ wrapper `.cc` files (source of truth: `litert/cc_sdk/litert_cc_sdk_hdrs.txt` and `litert/cc_sdk/litert_cc_sdk_srcs.txt` in the LiteRT repo), and `CMakeLists.txt`. Builds static `litert_cc_api` and calls into `libLiteRt.so` through the single C symbol `kLiteRtRuntimeBuiltin`. |
| `linux_x86_64/` | Host `libLiteRt.so` (`glibc >= 2.26`), `libLiteRtWebGpuAccelerator.so`, `apply_plugin_main` (offline NPU AOT compiler), vendor compiler plugins (`libLiteRtCompilerPlugin_Qualcomm.so`, `libLiteRtCompilerPlugin_IntelOpenvino.so`, etc.), and `libLiteRtDispatch_IntelOpenvino.so`. |
| `android_arm64/` | Android NDK (`arm64-v8a`, API 26+) `libLiteRt.so`, `libLiteRtClGlAccelerator.so` (OpenCL/GL GPU), `libLiteRtWebGpuAccelerator.so`, and NPU dispatch/compiler plugins (`libLiteRtDispatch_Qualcomm.so`, `libLiteRtCompilerPlugin_Qualcomm.so`, etc.). |
| `linux_arm64/` | Linux `aarch64` (`glibc >= 2.26`, static C++ stdlib) `libLiteRt.so`, `libLiteRtWebGpuAccelerator.so` (Vulkan via Dawn), `libLiteRtDispatch_Qualcomm.so`, `libLiteRtCompilerPlugin_Qualcomm.so`, and `apply_plugin_main`. Recommended for Ubuntu/Linux IoT boards that need both GPU and NPU. |
| `aarch64-oe-linux-gcc11.2/` | Yocto / OpenEmbedded / Ubuntu 24.04 (`glibc >= 2.38`, dynamic `libstdc++.so.6`) `libLiteRt.so`, `libLiteRtDispatch_Qualcomm.so`, and `libLiteRtCompilerPlugin_Qualcomm.so`. Does not include a GPU accelerator `.so`. |
| `windows_arm64/` | Windows on ARM (`arm64`) `libLiteRt.dll`, `libLiteRtWebGpuAccelerator.dll` (Direct3D 12 via Dawn), Qualcomm NPU libraries (`LiteRtDispatch_Qualcomm.dll`, `LiteRtCompilerPlugin_Qualcomm.dll`), `apply_plugin_main.exe` (offline NPU AOT compiler), `analyze_model_main.exe`, `benchmark_model.exe`, and `run_model.exe`. |
| `windows_x86_64/` | Windows `x86_64` `libLiteRt.dll`, `libLiteRtWebGpuAccelerator.dll` (Direct3D 12 via Dawn), Intel OpenVINO NPU libraries (`LiteRtDispatch_IntelOpenvino.dll`, `LiteRtCompilerPlugin_IntelOpenvino.dll`), `apply_plugin_main.exe`, `analyze_model_main.exe`, `benchmark_model.exe`, and `run_model.exe`. |

---

## 2. CMake Integration & Cross-Compilation

### Staging `litert_cc_sdk/` + `libLiteRt` (Required)

`litert_cc_sdk/CMakeLists.txt` declares `litert_c_api` with `IMPORTED_LOCATION`
pointing to `${CMAKE_CURRENT_SOURCE_DIR}/libLiteRt.so` (Linux/Android),
`libLiteRt.dylib` (macOS/iOS), or `libLiteRt.dll` (Windows). Because the
archive unpacks `litert_cc_sdk/` and `<platform>/libLiteRt.*` into sibling
directories, stage both into `${CMAKE_BINARY_DIR}/litert_cc_sdk_staged` before
calling `add_subdirectory(... EXCLUDE_FROM_ALL)`:

```cmake
if(WIN32)
  set(_LITERT_LIB_NAME "libLiteRt.dll")
elseif(APPLE)
  set(_LITERT_LIB_NAME "libLiteRt.dylib")
else()
  set(_LITERT_LIB_NAME "libLiteRt.so")
endif()

set(_SDK_STAGE "${CMAKE_BINARY_DIR}/litert_cc_sdk_staged")
file(COPY "${LITERT_CC_SDK_DIR}/" DESTINATION "${_SDK_STAGE}")
file(COPY "${LITERT_PREBUILT_DIR}/${_LITERT_LIB_NAME}" DESTINATION "${_SDK_STAGE}")
add_subdirectory("${_SDK_STAGE}" "${CMAKE_BINARY_DIR}/litert_cc_sdk_build" EXCLUDE_FROM_ALL)

target_link_libraries(my_app PRIVATE litert_cc_api)
if(WIN32)
  add_custom_command(TARGET my_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${_SDK_STAGE}/libLiteRt.dll" "$<TARGET_FILE_DIR:my_app>/libLiteRt.dll")
elseif(UNIX AND NOT ANDROID)
  target_link_libraries(my_app PRIVATE "${_SDK_STAGE}/${_LITERT_LIB_NAME}" pthread dl)
  set_target_properties(my_app PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()
```

- **Windows DLL Import Library (`libLiteRt.lib`):** Prebuilt archives ship
  `libLiteRt.dll` without a `.lib` import library. When configuring on Windows
  from a Visual Studio developer shell (`x64` or `ARM64` Native Tools Command
  Prompt), `litert_cc_sdk/CMakeLists.txt` automatically generates
  `${CMAKE_BINARY_DIR}/libLiteRt.lib` from `libLiteRt.dll`'s export table using
  `dumpbin /exports` and `lib /def` (or uses an existing `libLiteRt.lib` if
  placed alongside `libLiteRt.dll`) and sets `IMPORTED_IMPLIB` on
  `litert_c_api`.
- Pass `EXCLUDE_FROM_ALL` to `add_subdirectory` so the SDK's optional sample
  targets (`run_model_simple`, `dump_model_simple`) are not built unless
  explicitly requested.
- First-time CMake configuration requires network access so
  `OverridableFetchContent.cmake` can fetch `abseil-cpp`, `flatbuffers`, and
  `opencl_headers`.
- For the complete `CMakeLists.txt` and `toolchain_aarch64_ubuntu.cmake`
  cross-compilation setup, read the
  [CMake and cross-compilation templates](references/cmake_and_toolchains.md).

### Configure & Build Commands

- **Linux `x86_64` Host:**
  ```bash
  cmake -S . -B build/linux_x86_64 -DCMAKE_BUILD_TYPE=Release \
    -DLITERT_CC_SDK_DIR=/path/to/prebuilts/litert_cc_sdk \
    -DLITERT_PREBUILT_DIR=/path/to/prebuilts/linux_x86_64
  cmake --build build/linux_x86_64 -j
  ```
- **Windows `x86_64` or `arm64` (MSVC / `clang-cl` with Ninja):**
  Run from the matching Visual Studio developer shell (`x64 Native Tools Command
  Prompt` or `ARM64 Native Tools Command Prompt`) so `cl`, `link`, `lib`, and
  `dumpbin` are on `PATH`:
  ```bat
  REM Standalone SDK build using bundled CMakePresets.json (windows-x64 or windows-arm64):
  copy <prebuilts>\windows_x86_64\libLiteRt.dll <prebuilts>\litert_cc_sdk\
  cmake -S <prebuilts>\litert_cc_sdk --preset windows-x64
  cmake --build <prebuilts>\litert_cc_sdk\cmake_build_windows_x64 -j

  REM Or configure an external application staging litert_cc_sdk + windows_x86_64 / windows_arm64:
  cmake -S . -B build\windows_x64 -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DLITERT_CC_SDK_DIR=C:\path\to\prebuilts\litert_cc_sdk ^
    -DLITERT_PREBUILT_DIR=C:\path\to\prebuilts\windows_x86_64
  cmake --build build\windows_x64 -j
  ```
- **Android `arm64-v8a` (NDK):**
  ```bash
  cmake -S . -B build/android_arm64 \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=26 -DCMAKE_BUILD_TYPE=Release \
    -DLITERT_CC_SDK_DIR=/path/to/prebuilts/litert_cc_sdk \
    -DLITERT_PREBUILT_DIR=/path/to/prebuilts/android_arm64
  cmake --build build/android_arm64 -j
  ```
- **Linux `aarch64` / Qualcomm IoT (Cross-Compile from `x86_64`):**
  ```bash
  export IOT_SYSROOT=/path/to/sysroot_noble_arm64
  export IOT_LLD=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/ld.lld
  cmake -S . -B build/iot_linux_arm64 \
    -DCMAKE_TOOLCHAIN_FILE=toolchain_aarch64_ubuntu.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DLITERT_CC_SDK_DIR=/path/to/prebuilts/litert_cc_sdk \
    -DLITERT_PREBUILT_DIR=/path/to/prebuilts/linux_arm64
  cmake --build build/iot_linux_arm64 -j
  ```

---

## 3. Writing the C++ `CompiledModel` Application

Use the headers in `litert_cc_sdk/litert/cc/` and
`litert_cc_sdk/litert/cc/options/` (including
`litert/cc/options/litert_qualcomm_options.h`):

```cpp
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"

litert::Expected<void> RunInference() {
  /* 1. Environment: configure dispatch, compiler plugin (JIT), and cache dirs */
  std::vector<litert::EnvironmentOptions::Option> env_opts;
  env_opts.emplace_back(litert::EnvironmentOptions::Tag::kDispatchLibraryDir, dispatch_dir);
  if (!plugin_dir.empty()) {
    env_opts.emplace_back(litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir, plugin_dir);
  }
  if (!cache_dir.empty()) {
    env_opts.emplace_back(litert::EnvironmentOptions::Tag::kCompilerCacheDir, cache_dir);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto env,
      litert::Environment::Create(litert::EnvironmentOptions(
          litert::Span<const litert::EnvironmentOptions::Option>(
              env_opts.data(), env_opts.size()))));

  /* 2. Options: select hardware accelerator and backend-specific options */
  LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
  LITERT_RETURN_IF_ERROR(
      options.SetHardwareAccelerators(litert::HwAccelerators::kNpu)); /* or kCpu / kGpu */

  /* CPU: XNNPACK defaults to 1 thread unless SetNumThreads is called */
  LITERT_ASSIGN_OR_RETURN(auto& cpu_opts, options.GetCpuOptions());
  LITERT_RETURN_IF_ERROR(cpu_opts.SetNumThreads(4));

  /* GPU: defaults to FP16 precision; set kFp32 for strict CPU numerical parity */
  LITERT_ASSIGN_OR_RETURN(auto& gpu_opts, options.GetGpuOptions());
  LITERT_RETURN_IF_ERROR(gpu_opts.SetPrecision(litert::GpuOptions::Precision::kFp32));

  /* Qualcomm NPU: configure HTP performance mode and optional split library dirs */
  LITERT_ASSIGN_OR_RETURN(auto& qnn,
                          options.GetOptions<litert::qualcomm::QualcommOptions>());
  qnn.SetHtpPerformanceMode(litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
  if (!qnn_lib_dir.empty()) qnn.SetQnnLibDir(qnn_lib_dir);
  if (!dsp_skel_dir.empty()) qnn.SetDspSkelDir(dsp_skel_dir);
  if (enable_jit_in_memory) qnn.SetEnableJustInTime(true);

  /* 3. Create CompiledModel and I/O TensorBuffers (always check Expected<T> before dereferencing;
   *    GPU and NPU delegates initialize lazily inside CreateInputBuffers()!) */
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          litert::CompiledModel::Create(env, model_path, options));
  LITERT_ASSIGN_OR_RETURN(auto input_buffers, compiled_model.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(auto output_buffers, compiled_model.CreateOutputBuffers());

  /* 4. Write inputs, execute, and synchronize outputs */
  LITERT_RETURN_IF_ERROR(input_buffers[0].Write<float>(input_span));
  LITERT_RETURN_IF_ERROR(compiled_model.Run(input_buffers, output_buffers));

  /* GPU Run() is async; Read<T>() or Lock(kRead) synchronizes before stopping timers. */
  LITERT_RETURN_IF_ERROR(output_buffers[0].Read<float>(output_span));
  return {};
}
```

- In functions returning `int` (such as `main()`), check every
  `litert::Expected<T>` result explicitly (`if (!res) { std::cerr <<
  res.Error().Message() << "\n"; return 1; }`) instead of dereferencing `*res`
  or `res->` unchecked.

---

## 4. Qualcomm NPU Compilation (AOT vs JIT) & QAIRT Version Alignment

### Matching the Pinned QAIRT SDK Version

`litert_cc_sdk/` does not pin or include QNN headers, but the prebuilt
`libLiteRtCompilerPlugin_Qualcomm.so` and `libLiteRtDispatch_Qualcomm.so`
compile `litert/vendors/qualcomm/qnn_manager.cc` against
`<QNN/System/QnnSystemCommon.h>`, baking `QNN_SYSTEM_API_VERSION_MINOR` and
`QNN_API_VERSION_MINOR` into the `.so` binaries. Always check
`litert/vendors/CMakeLists.txt` in the LiteRT repository as the authoritative
source of truth for the pinned QAIRT SDK version (for example, `2.51.1.261002`,
downloadable from
`https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/<version>/v<version>.zip`),
and use that matching QAIRT SDK version for both host AOT compilation and
on-device runtime libraries.

### Offline AOT Compilation with `apply_plugin_main`

Run `linux_x86_64/apply_plugin_main` on the `x86_64` host. Because
`linux_x86_64/libLiteRtCompilerPlugin_Qualcomm.so` links against `libQnnIr.so`,
`libQnnLpai.so`, `libQnnSaver.so`, `libQnnHtp.so`, `libQnnSystem.so`, and
`libc++.so.1`, include `<qairt>/lib/x86_64-linux-clang` in `LD_LIBRARY_PATH`:

```bash
QNN_HOST_LIB=/path/to/qairt/lib/x86_64-linux-clang
LD_LIBRARY_PATH="prebuilts/linux_x86_64:${QNN_HOST_LIB}" \
  ./prebuilts/linux_x86_64/apply_plugin_main \
  --cmd=apply \
  --model=model.tflite \
  --soc_manufacturer=Qualcomm \
  --soc_model=SM8650 \
  --libs=prebuilts/linux_x86_64 \
  --qualcomm_qnn_lib_dir="${QNN_HOST_LIB}" \
  --o=model_sm8650_aot.tflite
```

- Pass `--helpfull` (not `--help`) to inspect `apply_plugin_main` flags.
- Common `--soc_model` values: `SM8650` (HTP v75), `SM8750` (HTP v79), `SM8850`,
  `QCS8275` (HTP v75), `QCS8300` (HTP v75), `QCS9075` (HTP v73), `QCS6490` (HTP
  v68), `SA8255`, `SA8775`.
- Verify the output `.tflite` file size is non-zero (`apply_plugin_main` creates
  an empty output file before aborting if the plugin fails to load).

---

## 5. On-Device Library Placement Rules

### How LiteRT Resolves Runtime Libraries

| Component | Resolution Order | Placement Rule |
| :--- | :--- | :--- |
| `libLiteRt.so` / `libLiteRt.dll` | System dynamic linker (`RUNPATH=$ORIGIN` / `LD_LIBRARY_PATH` on Linux/Android; executable directory or `PATH` on Windows) | Place next to the executable (or on `LD_LIBRARY_PATH` / `PATH`). |
| GPU Accelerator (`libLiteRt*Accelerator.{so,dll}`) | `litert/runtime/accelerators/gpu_registry.cc` (source of truth in the LiteRT repo) probes bare filenames in fixed order and registers the **first library that loads** (e.g., on **Android:** `libLiteRtGpuAccelerator.so` → `libLiteRtClGlAccelerator.so` → `libLiteRtOpenClAccelerator.so` → `libLiteRtWebGpuAccelerator.so` → `libLiteRtVulkanAccelerator.so`; on **Linux:** `libLiteRtGpuAccelerator.so` → `libLiteRtWebGpuAccelerator.so` → `libLiteRtOpenClAccelerator.so` → `libLiteRtVulkanAccelerator.so`; on **Windows:** `libLiteRtWebGpuAccelerator.dll`). | `GpuOptions::SetBackend(kWebGpu)` does **not** alter the `.so` probe order. On Android, place `libLiteRtClGlAccelerator.so` (`clgl/`) and `libLiteRtWebGpuAccelerator.so` (`webgpu/`) in separate directories and include only the desired directory on `LD_LIBRARY_PATH` (or bundle only one in APK `jniLibs`). On Windows, place `libLiteRtWebGpuAccelerator.dll` plus the DirectX Shader Compiler DLLs (`dxcompiler.dll` and `dxil.dll` from Windows SDK `Redist\D3D\<arch>` or `DirectXShaderCompiler` releases) next to the `.exe`. |
| NPU Dispatch (`libLiteRtDispatch_Qualcomm.so`, `LiteRtDispatch_Qualcomm.dll`, `LiteRtDispatch_IntelOpenvino.dll`) | Searched **only** in `EnvironmentOptions::Tag::kDispatchLibraryDir` (`LD_LIBRARY_PATH` / `PATH` is not searched). | Set `EnvironmentOptions::Tag::kDispatchLibraryDir`. |
| NPU Compiler Plugin (`libLiteRtCompilerPlugin_Qualcomm.so`, `LiteRtCompilerPlugin_Qualcomm.dll`, `LiteRtCompilerPlugin_IntelOpenvino.dll`) | Searched in `EnvironmentOptions::Tag::kCompilerPluginLibraryDir` (required for on-device JIT). | Set `EnvironmentOptions::Tag::kCompilerPluginLibraryDir` when using JIT. |
| QNN Host Libraries (`libQnnSystem`, `libQnnHtp`, `libQnnHtpV*Stub`, `libQnnHtpPrepare`) | `QnnManager::LoadSharedLibHelper` checks:<br>1. `QualcommOptions::GetQnnLibDir()`<br>2. `shared_library_dir_` (`kDispatchLibraryDir` or `kCompilerPluginLibraryDir`)<br>3. Bare filename via `LD_LIBRARY_PATH` / `PATH` | • **Android:** copy from `qairt/lib/aarch64-android/`.<br>• **Linux/Ubuntu IoT:** copy from `qairt/lib/aarch64-oe-linux-gcc11.2/` (`aarch64-ubuntu-gcc9.4` in QAIRT 2.51.1 omits `libQnnHtpV75Stub.so`).<br>• **Windows ARM64:** copy from `qairt/lib/aarch64-windows-msvc/`.<br>• `libQnnHtpPrepare` is required only for JIT. |
| Hexagon DSP Skel (`libQnnHtpV*Skel.so`) | FastRPC `ADSP_LIBRARY_PATH`, which `QnnManager::Init` prepends in order:<br>1. `QualcommOptions::GetDspSkelDir()`<br>2. `QualcommOptions::GetQnnLibDir()`<br>3. `shared_library_dir_` (`kDispatchLibraryDir` / `kCompilerPluginLibraryDir`) | Copy from `qairt/lib/hexagon-v<HTP>/unsigned/libQnnHtpV<HTP>Skel.so`. |

### Choosing Between Single-Folder and Split-Folder NPU Layouts

1. **Layout A — Single Shared `npu/` Folder (Simplest):**
   Place `libLiteRtDispatch_Qualcomm.so`, `libLiteRtCompilerPlugin_Qualcomm.so`
   (JIT), `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpV<HTP>Stub.so`,
   `libQnnHtpPrepare.so` (JIT), and `libQnnHtpV<HTP>Skel.so` into one
   `<root>/npu/` directory and pass `kDispatchLibraryDir=<root>/npu` (plus
   `kCompilerPluginLibraryDir=<root>/npu` for JIT). No `QualcommOptions` path
   flags are required because `QnnManager` falls back to `shared_library_dir_`.
2. **Layout B — Split Folders via `QualcommOptions`:**
   When `dispatch/`, `plugin/`, `qnn_lib/`, and `dsp_skel/` are separate
   directories, pass:
   - `EnvironmentOptions::Tag::kDispatchLibraryDir` → `<root>/dispatch`
   - `EnvironmentOptions::Tag::kCompilerPluginLibraryDir` → `<root>/plugin`
   - `QualcommOptions::SetQnnLibDir("<root>/qnn_lib")`
   - `QualcommOptions::SetDspSkelDir("<root>/dsp_skel")`
   - Optional for JIT: `QualcommOptions::SetEnableJustInTime(true)` to pass the
     QNN context handle in-memory and skip context binary serialization.
   - **Why `SetQnnLibDir` matters when splitting folders:** Do not copy
     `libQnn*.so` into both `dispatch/` and `plugin/` without `SetQnnLibDir`.
     Loading `libQnnHtp.so` from two different file paths during JIT creates two
     DSO instances and fails on the second `QnnDevice_create` with QNN error
     `1009` (`Cannot establish more than one connection to QNN skel`).

---

## 6. Troubleshooting Matrix

| Symptom / Error | Cause | Fix |
| :--- | :--- | :--- |
| `Qnn System library version 1.14.0 is mismatched. The minimum supported version is 1.15.0` or `Context binary (2.51.1) is newer than the current SDK (2.50.0)` | QAIRT version mismatch between prebuilt `libLiteRt{CompilerPlugin,Dispatch}_Qualcomm.so` and host/device QAIRT libraries. | Use the QAIRT SDK version matching the prebuilt build (e.g., `2.51.1.261002`) for both `apply_plugin_main` and on-device `libQnn*.so`. |
| `[Qnn] failed to call device create, Error code: 14001` (`libQnnHtpV75Skel.so... file not found`) | FastRPC cannot find `libQnnHtpV*Skel.so` on `ADSP_LIBRARY_PATH`. | Call `QualcommOptions::SetDspSkelDir(skel_dir)` or place the Skel `.so` inside `qnn_lib_dir` / `dispatch_dir`. |
| `Cannot establish more than one connection to QNN skel ... Error code: 1009` during JIT | Compiler plugin and dispatch library loaded separate copies of `libQnnHtp.so` from two directories. | Point both to one QNN directory with `QualcommOptions::SetQnnLibDir(qnn_lib_dir)` or use a single shared `npu/` directory. |
| `Could not load shared library .../libLiteRtCompilerPlugin_Qualcomm.so: libQnnIr.so: cannot open shared object file` on `x86_64` | Missing QAIRT `x86_64-linux-clang` libraries or host `libc++.so.1` / `libc++abi.so.1` / `libunwind.so.1` on `LD_LIBRARY_PATH`. | Add `<qairt>/lib/x86_64-linux-clang` to `LD_LIBRARY_PATH` and ensure host `libc++1`, `libc++abi1`, and `libunwind` packages are installed. |
| `Encountered unresolved custom op: DISPATCH_OP` in `CreateInputBuffers()` | Running an AOT-compiled NPU model with CPU-only accelerator flags or missing `kDispatchLibraryDir`. | Enable `litert::HwAccelerators::kNpu` and set `EnvironmentOptions::Tag::kDispatchLibraryDir`. |
| Requesting WebGPU on Android still logs `Initializing OpenCL-based API` | `gpu_registry.cc` loads `libLiteRtClGlAccelerator.so` ahead of `libLiteRtWebGpuAccelerator.so` when both are on `LD_LIBRARY_PATH`. | Separate `clgl/` and `webgpu/` directories and place only the target GPU accelerator directory on `LD_LIBRARY_PATH`. |
| GPU `Run()` latency measures `< 1 ms` | GPU execution is asynchronous until output buffers are locked/read. | Call `output_buffer.Read<T>()` or `Lock(kRead)` / `Unlock()` before stopping the timer. |
| GPU output differs slightly from CPU (~99.2% top-1 agreement) | GPU precision defaults to `kFp16`. | Call `options->GetGpuOptions()->SetPrecision(litert::GpuOptions::Precision::kFp32)` for strict FP32 parity. |
| CPU latency is 2.5×–3× slower than expected | XNNPACK defaults to 1 thread when `SetNumThreads` is not called. | Set `options->GetCpuOptions()->SetNumThreads(4)`. |

---

## References

- [CMake and cross-compilation templates](references/cmake_and_toolchains.md)
