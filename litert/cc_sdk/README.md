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
CMake 3.20 or later and Ninja. Run every command below from the matching
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
   runtime finds optional accelerators by name in the same directory:

   * GPU: `libLiteRtWebGpuAccelerator.dll` from the prebuilt package. Its
     Direct3D 12 backend additionally requires the DirectX Shader Compiler
     runtime, `dxcompiler.dll` and `dxil.dll`, next to the executable. These are
     not part of LiteRT or of Windows; install them from the Windows SDK
     (`Windows Kits\10\Redist\D3D\<arch>\`) or from the
     [DirectXShaderCompiler releases](https://github.com/microsoft/DirectXShaderCompiler/releases).
   * NPU (Qualcomm): `LiteRtDispatch_Qualcomm.dll` from the prebuilt package,
     plus the QNN runtime libraries from the Qualcomm AI Runtime SDK on `PATH`.