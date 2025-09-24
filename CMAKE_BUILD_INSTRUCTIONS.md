# LiteRT CMake Build Instructions

Install CMake 4.0.1 from: https://github.com/kitware/cmake/releases

## Build for Android (cross compilation from Mac OS or Linux)

Run the following command:

```
cd ./litert;
export ANDROID_NDK_HOME="PATH TO NDK"
cmake --preset android-arm64;
cmake --preset android-arm64 -DTFLITE_HOST_TOOLS_DIR="$(cd ../host_flatc_build/_deps/flatbuffers-build && pwd)";
cmake --build cmake_build_android_arm64 -j
```

## Build for Mac OS

Run the following command:

```
cd ./litert;
cmake --preset default;
cmake --build cmake_build -j
```