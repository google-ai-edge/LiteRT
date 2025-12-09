# LiteRT C++ SDK for Android

This repo is a placeholder to build LiteRT C++ SDK with prebuilt `libLiteRt.so`.

## Build Steps

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