# LiteRT CMake Build Instructions

Use this guide to configure and build the LiteRT runtime with CMake on macOS
The instructions cover both Android cross-compilation targets
(built from Linux and Mac OS) machine and native macOS and Linux builds.

## Common Build Steps

Install CMake 4.0.1 from: https://github.com/kitware/cmake/releases

All build presets expect you to work from the repository root:

```bash
cd ./litert
```

The generated build trees live under `cmake_build*`. Parallel builds can be
controlled via `-j` with the desired core count.

## Android (arm64) Cross-Compilation

1.  Install the Android NDK and export the path so CMake can find it:

    ```bash
    export ANDROID_NDK_HOME=/absolute/path/to/android-ndk-r27
    ```

2.  Configure host-side flatbuffer tools

    ```bash
    cmake --preset android-arm64;
    ```

3.  Configure the LiteRT Android build using the provided preset and point to
    the generated FlatBuffers tools:

    ```bash
    cmake --preset android-arm64 \
      -DTFLITE_HOST_TOOLS_DIR="$(cd ../host_flatc_build/_deps/flatbuffers-build && pwd)"
    ```

4.  Build LiteRT for Android:

    ```bash
    cmake --build cmake_build_android_arm64 -j
    ```

Artifacts such as static libraries will be emitted under
`cmake_build_android_arm64`.

## Host Build from Mac OS and Linux

1.  Configure the default host preset:

    ```bash
    cmake --preset default
    ```

2.  Build LiteRT:

    ```bash
    cmake --build cmake_build -j
    ```

## Troubleshooting Tips

- Delete the corresponding `cmake_build*` directory if you change toolchains or
  major configuration options, then rerun the configure step.
- Inspect `CMakeCache.txt` inside each build tree for resolved dependency paths.
