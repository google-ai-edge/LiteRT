# LiteRT C++ SDK

This repo is a placeholder to build LiteRT C++ SDK with a prebuilt LiteRT
runtime library (`libLiteRt.so`, `libLiteRt.dylib` or `libLiteRt.dll`).

## Build Steps (Android)

Install CMake 4.0.1 from: https://github.com/kitware/cmake/releases

1. Install the Android NDK and export the path so CMake can find it:

   ```bash
   export ANDROID_NDK_HOME=/absolute/path/to/android-ndk-r27
   ```

1. Download C++ SDK

   ```bash
   wget https://github.com/google-ai-edge/LiteRT/releases/download/<litert_version>/litert_cc_sdk.zip
   unzip litert_cc_sdk.zip -d <your_litert_cc_sdk_location>
   ```

1. Place `libLiteRt.so` under litert/cc_sdk

   ```bash
   cp <path_to_prebuilt_lib>/libLiteRt.so <your_litert_cc_sdk_location>/litert_cc_sdk/
   ```

1. Configure the LiteRT Android build using the provided preset:

   ```bash
   cd <your_litert_cc_sdk_location>
   cmake -S litert_cc_sdk -B litert_cc_sdk_build --preset android-arm64
   ```

1. Build LiteRT C++ SDK for Android:

   ```bash
   cmake --build litert_cc_sdk_build -j
   ```

Artifacts such as static libraries will be emitted under
`litert_cc_sdk_build`.

## Build Steps (Windows, ARM64 or x64)

Prerequisites: Visual Studio 2022 or later with the MSVC toolchain for the
target architecture (`cl.exe`; the optional `clang-cl` component also works),
CMake 3.23 or later and `ninja`. Run every command below from the matching
Visual Studio developer shell (for example "ARM64 Native Tools Command Prompt"),
so that `cl`, `link`, `lib` and `dumpbin` are on `PATH`.

1. Download the prebuilt package and the C++ SDK, then place the runtime DLL
   from the matching `windows_arm64` or `windows_x86_64` directory next to the
   SDK's `CMakeLists.txt`:

   ```bat
   copy <prebuilts>\windows_arm64\libLiteRt.dll <your_litert_cc_sdk_location>\litert_cc_sdk\
   ```

   The package ships no import library (`libLiteRt.lib`). CMake generates one
   from the DLL's export table during configuration using `dumpbin` and `lib`.
   If you have a `libLiteRt.lib`, place it next to the DLL and it is used
   instead.

1. Configure and build with the preset for your architecture:

   ```bat
   cd <your_litert_cc_sdk_location>
   cmake -S litert_cc_sdk --preset windows-arm64
   cmake --build litert_cc_sdk\cmake_build_windows_arm64 -j
   ```

   Use `--preset windows-x64` from an x64 developer shell for x64.

1. To run the sample executables, copy `libLiteRt.dll` next to them. The
   samples run on the CPU (XNNPACK) and, if `libLiteRtWebGpuAccelerator.dll`
   is in the same directory, on the GPU via `--accelerator=gpu,cpu`:

   * GPU: the accelerator's Direct3D 12 backend also needs the DirectX Shader
     Compiler runtime, `dxcompiler.dll` and `dxil.dll`, next to the executable.
     They are not part of LiteRT or Windows; install them from the Windows SDK
     (`Windows Kits\10\Redist\D3D\<arch>`) or the
     [DirectXShaderCompiler releases](https://github.com/microsoft/DirectXShaderCompiler/releases).
     The first run of a model compiles its shaders and can take tens of
     seconds; later runs are fast.

   * NPU (Qualcomm): use the full `run_model.exe` shipped in the prebuilt
     package's `windows_<arch>` directory rather than the SDK samples. Put the
     QNN runtime libraries from the Qualcomm AI Runtime SDK on `PATH`, then
     either compile the model ahead of time with `apply_plugin_main.exe
     --cmd=apply --soc_manufacturer=Qualcomm --soc_model=<SoC> --libs=<dir>`
     and run `run_model.exe --accelerator=npu,cpu --dispatch_library_dir=<dir>`,
     or let it compile at load time by adding
     `--compiler_plugin_library_dir=<dir>`, where `<dir>` holds
     `LiteRtDispatch_Qualcomm.dll` and `LiteRtCompilerPlugin_Qualcomm.dll`.
     An application that wants the NPU sets
     `EnvironmentOptions::Tag::kDispatchLibraryDir` (and
     `kCompilerPluginLibraryDir` for JIT) when creating its `Environment`;
     the runtime does not search the executable's directory for these
     libraries. See `litert/tools/run_model.cc` for the reference.

   * NPU (Intel OpenVINO on x64): `windows_x86_64` ships
     `LiteRtDispatch_IntelOpenvino.dll`,
     `LiteRtCompilerPlugin_IntelOpenvino.dll`, and `apply_plugin_main.exe`. Put
     the Intel OpenVINO runtime DLLs on `PATH` and pass the directory containing
     `LiteRtDispatch_IntelOpenvino.dll` (and
     `LiteRtCompilerPlugin_IntelOpenvino.dll` for JIT) via
     `EnvironmentOptions::Tag::kDispatchLibraryDir` /
     `kCompilerPluginLibraryDir`.

## Using with Coding Agents (Claude Code / Gemini CLI)

`litert_cc_sdk` bundles an agent skill under
`skills/litert-cc-sdk-prebuilt-onboarding/` that guides coding agents through
CMake integration, dynamic linking against `libLiteRt`, offline Qualcomm NPU
AOT compilation (`apply_plugin_main`), on-device JIT compilation, and GPU/NPU
runtime library placement across Linux, Android, Windows (`x86_64` and `arm64`),
and Qualcomm IoT.

- **Claude Code (auto-discovered project skill):** Symlink or copy the skill
  folder into `.claude/skills/` in your project root:

  ```bash
  mkdir -p .claude/skills
  ln -s "$PWD/litert_cc_sdk/skills/litert-cc-sdk-prebuilt-onboarding" .claude/skills/
  ```

- **Direct prompt reference:** Reference the skill directly in your prompt:

  ```text
  @litert_cc_sdk/skills/litert-cc-sdk-prebuilt-onboarding/SKILL.md
  Build a C++ CompiledModel app using this prebuilt archive for Android CPU, GPU, and Qualcomm NPU.
  ```
