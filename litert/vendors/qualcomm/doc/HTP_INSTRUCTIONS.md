# HTP Backend

This page walks through compiling a `.tflite` model for the Qualcomm HTP backend
and executing it on target hardware. The model can be compiled ahead of time on
the host (**AOT**) or compiled just in time on the device(**JIT**). The sections
below cover the following modes:

> Throughout this page, "compile" means compiling the **model** (lowering the
> `.tflite` graph into a QNN context binary), not building the LiteRT source
> code.

| Mode | Where compiled | Recompiles each load | Section |
| ----------- | ---------- | ------------------- | ------------------------- |
| **AOT (Bazel)** | Host (x86 Linux) | No, compiled offline | [AOT Flow](#aot-flow) > [Build with Bazel](#build-with-bazel) |
| **AOT (CMake)** | Host (x86 Linux) | No, compiled offline | [AOT Flow](#aot-flow) > [Build with CMake](#build-with-cmake) |
| **Real JIT** | Device, at load time | Yes, in memory without serialization/cache | [JIT on Android](#jit-on-android) |
| **On-device AOT** | Device, on the first load | No, cached after first load | [JIT on Android](#jit-on-android) |

AOT is the most common path. Choose JIT when you cannot pre-compile on a host.
**Real JIT** recompiles in memory every load, while **On-device AOT** compiles
once and caches the result so later loads skip recompilation.

--------------------------------------------------------------------------------

## Prerequisites

Before you start, make sure the toolchain in
[PREREQUISITES.md](./PREREQUISITES.md) is installed and the QNN concepts and
libraries in [QAIRT_SDK.md](./QAIRT_SDK.md) are understood.

The commands below refer to the following `${}` variables. Configure them for
your environment before running any step.

Variable                 | Description
------------------------ | -----------
`${LITERT}`              | The path of LiteRT source code.
`${QAIRT}`               | The root path of the unzipped QAIRT SDK.
`${SOC_MODEL}`           | Target SoC. Search "Supported Snapdragon devices" in the [QNN general overview](https://docs.qualcomm.com/doc/80-63442-10/topic/QNN_general_overview.html) and use the "Snapdragon Device/Chip" value. For example, `SM8850` for "Snapdragon 8 Elite Gen 5".
`${HTP_ARCH}`            | HTP architecture of the target `${SOC_MODEL}`, matching the QNN library casing (uppercase `V` plus version number, e.g. `V81`, `V79`, `V75`). In the same "Supported Snapdragon devices" table of the [QNN general overview](https://docs.qualcomm.com/doc/80-63442-10/topic/QNN_general_overview.html), use the "Hexagon Arch" column. For example, `V81` for `SM8850`, `V79` for `SM8750`, `V75` for `SM8650`.
`${HEXAGON_ARCH}`        | The same Hexagon architecture as `${HTP_ARCH}` but lowercase, to match the QAIRT directory name `hexagon-vXX`. For example, `v81`, `v79`, `v75`.
`${SOURCE_MODEL_DIR}`    | The host directory that holds the input `.tflite` model (and where the compiled output is written).
`${SOURCE_MODEL_PATH}`   | The file name of the input `.tflite` model (e.g. `model.tflite`). On the host it lives at `${SOURCE_MODEL_DIR}/${SOURCE_MODEL_PATH}`; for JIT it is pushed into `${TEST_FOLDER}`, so on the device it lives at `${TEST_FOLDER}/${SOURCE_MODEL_PATH}`.
`${COMPILED_MODEL_PATH}` | The file name of the AOT-compiled output `.tflite` model (e.g. `model_compiled.tflite`). Written on the host to `${SOURCE_MODEL_DIR}/${COMPILED_MODEL_PATH}`, then pushed into `${TEST_FOLDER}`, so on the device it lives at `${TEST_FOLDER}/${COMPILED_MODEL_PATH}`.
`${TEST_FOLDER}`         | The path of the test folder on the device.

These variables are only needed by specific sections:

Variable                    | Used by
--------------------------- | -------
`${EXTRACTED_BYTECODE_DIR}` | [(Optional) Extract bytecode from compiled model for QNN Runtime](#optional-extract-bytecode-from-compiled-model-for-qnn-runtime)
`${ANDROID_NDK_HOME}`       | [Build with CMake](#build-with-cmake)
`${IOT_DIR}`                | [Build with CMake](#build-with-cmake) > [Run on device (IoT device with oe-linux)](#run-on-device-iot-device-with-oe-linux)
`${CACHE_DIR}`              | [On-device AOT (cached JIT)](#on-device-aot-cached-jit). A writable directory on the device for the cached context binary.

--------------------------------------------------------------------------------

## AOT Flow

The model is compiled into a QNN context binary on the host (x86 Linux) ahead of
time, then the compiled `.tflite` is pushed to the device and executed. Two
build systems are supported: Bazel and CMake.

### Build with Bazel

#### Compile on host (x86 Linux)

Build "apply_plugin_main" compilation tool and compile the model on Linux host
machine.

```bash
cd ${LITERT}

bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/tools:apply_plugin_main

export LD_LIBRARY_PATH=${QAIRT}/lib/x86_64-linux-clang
bazel-bin/litert/tools/apply_plugin_main --cmd apply --libs bazel-bin/litert/vendors/qualcomm/compiler --soc_model ${SOC_MODEL} --soc_manufacturer Qualcomm --model ${SOURCE_MODEL_DIR}/${SOURCE_MODEL_PATH} -o ${SOURCE_MODEL_DIR}/${COMPILED_MODEL_PATH}
```

#### Run on device (Android)

This section runs the model compiled by the AOT step above. To compile the model
on the device instead, see [JIT on Android](#jit-on-android) below.

Build "run_model" and required libraries on Linux host machine:

```bash
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/tools:run_model
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/vendors/qualcomm/dispatch:dispatch_api_so
```

Execute the compiled model on Android device with QNN HTP backend via adb tool.

```bash
adb push ${QAIRT}/lib/aarch64-android/libQnnSystem.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp${HTP_ARCH}Stub.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/hexagon-${HEXAGON_ARCH}/unsigned/libQnnHtp${HTP_ARCH}Skel.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/c/libLiteRt.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/tools/run_model ${TEST_FOLDER}
adb push ${SOURCE_MODEL_DIR}/${COMPILED_MODEL_PATH} ${TEST_FOLDER}

adb shell "export LD_LIBRARY_PATH=${TEST_FOLDER} && export ADSP_LIBRARY_PATH=${TEST_FOLDER} && cd ${TEST_FOLDER} && ./run_model --graph=${TEST_FOLDER}/${COMPILED_MODEL_PATH} --dispatch_library_dir=${TEST_FOLDER}"
```

You can use `--helpfull` to find all Qualcomm options and descriptions. For
example, we execute a model on Android device with burst mode via HTP Backend:

```bash
adb shell "export LD_LIBRARY_PATH=/tmp/test_folder/ && export ADSP_LIBRARY_PATH=/tmp/test_folder/ && cd /tmp/test_folder/ && ./run_model --graph=./model_compiled.tflite --dispatch_library_dir=/tmp/test_folder/ --iterations 50 --qualcomm_htp_performance_mode burst --qualcomm_log_level off"
```

### Build with CMake

There's an official CMake build tutorial for your reference:
`${LITERT}/g3doc/instructions/CMAKE_BUILD_INSTRUCTIONS.md`.

#### Compile on host (x86 Linux)

```bash
cd ${LITERT}/litert

# This takes a while to download tensorflow
cmake --preset default
cmake --build cmake_build --target apply_plugin_main qnn_compiler_plugin litert_runtime_c_api_shared_lib -j8
```

The built files are located in: -
`${LITERT}/litert/cmake_build/c/libLiteRt.so` -
`${LITERT}/litert/cmake_build/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so`
- `${LITERT}/litert/cmake_build/tools/apply_plugin_main`

Execute apply_plugin_main to compile model.

```bash
export LD_LIBRARY_PATH=${QAIRT}/lib/x86_64-linux-clang

cd cmake_build

tools/apply_plugin_main --cmd apply --libs vendors/qualcomm/compiler/ --soc_model ${SOC_MODEL} --soc_manufacturer Qualcomm --model ${SOURCE_MODEL_DIR}/${SOURCE_MODEL_PATH} -o ${SOURCE_MODEL_DIR}/${COMPILED_MODEL_PATH}
```

#### Run on device (Android)

This step needs Android NDK to build for Android. Both `android-ndk-r25c-linux`
listed above and `android-ndk-r27` from
`${LITERT}/g3doc/instructions/CMAKE_BUILD_INSTRUCTIONS.md` should work. Below is
based on `android-ndk-r25c-linux`.

```bash
cd ${LITERT}/litert
export ANDROID_NDK_HOME=${ANDROID_NDK_HOME}

# This takes a while to build flatc and download tensorflow
cmake --preset android-arm64
cmake --build cmake_build_android_arm64 --target run_model litert_runtime_c_api_shared_lib dispatch_api_qualcomm_so -j8
```

The built files are located in: -
`${LITERT}/litert/cmake_build_android_arm64/c/libLiteRt.so` -
`${LITERT}/litert/cmake_build_android_arm64/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so`
- `${LITERT}/litert/cmake_build_android_arm64/tools/run_model`

Execute the compiled model on Android device with QNN HTP backend via adb tool.

```bash
adb push ${QAIRT}/lib/aarch64-android/libQnnSystem.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp${HTP_ARCH}Stub.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/hexagon-${HEXAGON_ARCH}/unsigned/libQnnHtp${HTP_ARCH}Skel.so ${TEST_FOLDER}
adb push ${LITERT}/litert/cmake_build_android_arm64/c/libLiteRt.so ${TEST_FOLDER}
adb push ${LITERT}/litert/cmake_build_android_arm64/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so ${TEST_FOLDER}
adb push ${LITERT}/litert/cmake_build_android_arm64/tools/run_model ${TEST_FOLDER}
adb push ${SOURCE_MODEL_DIR}/${COMPILED_MODEL_PATH} ${TEST_FOLDER}

adb shell "export LD_LIBRARY_PATH=${TEST_FOLDER} && export ADSP_LIBRARY_PATH=${TEST_FOLDER} && cd ${TEST_FOLDER} && ./run_model --graph=${TEST_FOLDER}/${COMPILED_MODEL_PATH} --dispatch_library_dir=${TEST_FOLDER}"
```

You can use `--helpfull` to find all Qualcomm options and descriptions. For
example, we execute a model on Android device with burst mode via HTP Backend:

```bash
adb shell "export LD_LIBRARY_PATH=/tmp/test_folder/ && export ADSP_LIBRARY_PATH=/tmp/test_folder/ && cd /tmp/test_folder/ && ./run_model --graph=./model_compiled.tflite --dispatch_library_dir=/tmp/test_folder/ --iterations 50 --qualcomm_htp_performance_mode burst --qualcomm_log_level off"
```

#### Run on device (IoT device with oe-linux)

The cross compilation is only required when the host machine is x86 Linux and
the target machine is oe-linux (Aarch64 Linux) based.

Below is a sample with x86 linux host machine and build for IQ-8275 (oe-linux)
device. - Please follow
[Flash IoT device (oe-linux)](./IOT_DEVICE_SETUP.md#flash-iot-device-oe-linux)
to flash the device and connect to the Linux workstation first. - This step
needs to download the eSDK for your target IoT device.

Install prerequisite for Linux workstation.

```bash
sudo locale-gen en_US.UTF-8
sudo apt update
sudo apt install gawk
```

Install ESDK on the Linux workstation. - Check the META version of the IoT
device. - Find the corresponding eSDK version in
https://artifacts.codelinaro.org/ui/native/qli-ci/flashable-binaries/qimpsdk/,
select "qcs8275-iq-8275-evk-pro-sku/" - For example, Download
"x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-esdk-2.3.0.zip" for Linux x86
host machine when the META version is "6.6.119-QLI.1.8". - Install the ESDK on
Linux workstation - Input the installation path ${ESDK}, please reserve spaces
for it - Input "Y" to install

```bash
adb shell "uname -a"

cd ${IOT_DIR}
wget https://artifacts.codelinaro.org/artifactory/qli-ci/flashable-binaries/qimpsdk/qcs8275-iq-8275-evk-pro-sku/x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-esdk-2.3.0.zip
unzip x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-esdk-2.3.0.zip

cd target/
umask a+rx

sh ./qcs8275-iq-8275-evk-pro-sku/sdk/qcom-wayland-x86_64-qcom-multimedia-image-armv8-2a-qcs8275-iq-8275-evk-pro-sku-toolchain-ext-1.8-ver.1.0.sh
```

Build required targets with ESDK.

```bash
export LINUX_AARCH64_ESDK=${ESDK}
cd ${LITERT}/litert

# This takes a while to build flatc and download tensorflow
cmake --preset linux-aarch64-iq8
cmake --build cmake_build_linux_aarch64_8275/ --target run_model litert_runtime_c_api_shared_lib dispatch_api_qualcomm_so -j8
```

The built files are located in: -
`${LITERT}/litert/cmake_build_linux_aarch64_8275/c/libLiteRt.so` -
`${LITERT}/litert/cmake_build_linux_aarch64_8275/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so`
- `${LITERT}/litert/cmake_build_linux_aarch64_8275/tools/run_model`

Upload required QNN libraries, LiteRT files and compiled model from Linux
workstation to IoT device.

```bash
adb push ${QAIRT}/lib/aarch64-oe-linux-gcc11.2/libQnnSystem.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-oe-linux-gcc11.2/libQnnHtp${HTP_ARCH}Stub.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/hexagon-${HEXAGON_ARCH}/unsigned/libQnnHtp${HTP_ARCH}Skel.so ${TEST_FOLDER}
adb push ${LITERT}/litert/cmake_build_linux_aarch64_8275/c/libLiteRt.so ${TEST_FOLDER}
adb push ${LITERT}/litert/cmake_build_linux_aarch64_8275/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so ${TEST_FOLDER}
adb push ${LITERT}/litert/cmake_build_linux_aarch64_8275/tools/run_model ${TEST_FOLDER}
```

Upload the model compiled by previous step from Linux workstation to IoT
device - For example, need to use --soc_model QCS8300 for model compilation for
IQ-8275

```bash
adb push ${SOURCE_MODEL_DIR}/${COMPILED_MODEL_PATH} ${TEST_FOLDER}
```

Execute the compiled model on IoT device with QNN HTP backend via adb tool, some
error logs for IQ-8275 execution are expected.

```bash
adb shell "export LD_LIBRARY_PATH=${TEST_FOLDER} && export ADSP_LIBRARY_PATH=${TEST_FOLDER} && cd ${TEST_FOLDER} && ./run_model --graph=${TEST_FOLDER}/${COMPILED_MODEL_PATH} --dispatch_library_dir=${TEST_FOLDER}"
```

You can use `--helpfull` to find all Qualcomm options and descriptions. For
example, we execute a model on IoT device with burst mode via HTP Backend:

```bash
adb shell "export LD_LIBRARY_PATH=/tmp/test_folder/ && export ADSP_LIBRARY_PATH=/tmp/test_folder/ && cd /tmp/test_folder/ && ./run_model --graph=./model_compiled.tflite --dispatch_library_dir=/tmp/test_folder/ --iterations 50 --qualcomm_htp_performance_mode burst --qualcomm_log_level off"
```

### (Optional) Extract bytecode from compiled model for QNN Runtime

`extract_bytecode` is a tool that extracts the QNN context binary from a LiteRT
compiled `.tflite` model. The extracted context binary can then be run with the
QNN runtime (`qnn-net-run`). If you intend to run the model with the LiteRT
runtime, you can skip this section.

Build the tool and run it with either Bazel or CMake.

#### Build with Bazel

```bash
cd ${LITERT}

bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/tools:extract_bytecode

bazel-bin/litert/tools/extract_bytecode --model_path ${SOURCE_MODEL_DIR}/${COMPILED_MODEL_PATH} --output_dir ${EXTRACTED_BYTECODE_DIR}
```

#### Build with CMake

```bash
cd ${LITERT}/litert

cmake --build cmake_build --target extract_bytecode -j8
```

The built file is located at
`${LITERT}/litert/cmake_build/tools/extract_bytecode`.

```bash
cd cmake_build

tools/extract_bytecode --model_path ${SOURCE_MODEL_DIR}/${COMPILED_MODEL_PATH} --output_dir ${EXTRACTED_BYTECODE_DIR}
```

Follow the QAIRT official document to execute the extracted context binary with
the QNN Runtime (qnn-net-run):
https://docs.qualcomm.com/doc/80-63442-10/topic/general_tools.html.

--------------------------------------------------------------------------------

## JIT on Android

The [AOT flow](#aot-flow) above compiles the model into a QNN context binary on
the host ahead of time. Alternatively, the model can be compiled **on the
device** at runtime by loading the compiler plugin directly. This is referred to
as JIT (Just-In-Time) and removes the host pre-compilation step. `run_model` is
given the original `.tflite` model and compiles it on the device on its first
load.

There are two JIT sub-modes, controlled by which flag is set:

| Sub-mode | Flags | Serialization |
| ------------ | --------------------------------- | ----------------------- |
| **Real JIT** | `--compiler_plugin_library_dir` + `--qualcomm_enable_just_in_time` | Compiles in memory and **bypasses serialization**. Recompiles on every load with no cache. |
| **On-device AOT** | `--compiler_plugin_library_dir` + `--compiler_cache_dir` | Compiles on the first run and **caches** the context binary to `--compiler_cache_dir`. Later runs load from the cache instead of recompiling. |

In both cases the model is partitioned and compiled on the device, so the NPU
accelerator must be requested via `--accelerator npu`.

### Compile on host (x86 Linux)

In addition to the `run_model` and dispatch library targets used by the AOT
flow, JIT needs the Qualcomm **compiler plugin** built for Android, since model
compilation now happens on the device.

```bash
cd ${LITERT}
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/tools:run_model
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/vendors/qualcomm/dispatch:dispatch_api_so
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so
```

### Run on device (Android)

Compared with the AOT flow, JIT pushes the **original** `.tflite` model (not a
compiled one) and the compiler plugin `libLiteRtCompilerPlugin_Qualcomm.so`. The
plugin links against extra QNN libraries that the AOT runtime does not need, so
`libQnnHtpPrepare.so` (on-device graph preparation), `libQnnIr.so`, and
`libQnnSaver.so` must also be pushed.

```bash
adb push ${QAIRT}/lib/aarch64-android/libQnnSystem.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtpPrepare.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnIr.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnSaver.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp${HTP_ARCH}Stub.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/hexagon-${HEXAGON_ARCH}/unsigned/libQnnHtp${HTP_ARCH}Skel.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/c/libLiteRt.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/tools/run_model ${TEST_FOLDER}
adb push ${SOURCE_MODEL_DIR}/${SOURCE_MODEL_PATH} ${TEST_FOLDER}
```

#### Real JIT (no serialization)

Set `--qualcomm_enable_just_in_time` and do **not** set `--compiler_cache_dir`.
The model is compiled in memory on every load, bypassing serialization.

```bash
adb shell "export LD_LIBRARY_PATH=${TEST_FOLDER} && export ADSP_LIBRARY_PATH=${TEST_FOLDER} && cd ${TEST_FOLDER} && ./run_model --graph=${TEST_FOLDER}/${SOURCE_MODEL_PATH} --accelerator npu --dispatch_library_dir=${TEST_FOLDER} --compiler_plugin_library_dir=${TEST_FOLDER} --qualcomm_enable_just_in_time"
```

You can use `--helpfull` to find all Qualcomm options and descriptions. For
example, we run real JIT on Android device with burst mode via HTP Backend:

```bash
adb shell "export LD_LIBRARY_PATH=/tmp/test_folder/ && export ADSP_LIBRARY_PATH=/tmp/test_folder/ && cd /tmp/test_folder/ && ./run_model --graph=./model.tflite --accelerator npu --dispatch_library_dir=/tmp/test_folder/ --compiler_plugin_library_dir=/tmp/test_folder/ --qualcomm_enable_just_in_time --iterations 50 --qualcomm_htp_performance_mode burst --qualcomm_log_level off"
```

#### On-device AOT (cached JIT)

Set both `--compiler_plugin_library_dir` (where the plugin `.so` was pushed) and
`--compiler_cache_dir` (a writable directory for the cached context binary). The
first run compiles and caches model. Subsequent runs with the same
`--compiler_cache_dir` load the cached binary and skip recompilation.

```bash
adb shell "export LD_LIBRARY_PATH=${TEST_FOLDER} && export ADSP_LIBRARY_PATH=${TEST_FOLDER} && cd ${TEST_FOLDER} && ./run_model --graph=${TEST_FOLDER}/${SOURCE_MODEL_PATH} --accelerator npu --dispatch_library_dir=${TEST_FOLDER} --compiler_plugin_library_dir=${TEST_FOLDER} --compiler_cache_dir=${CACHE_DIR}"
```

You can use `--helpfull` to find all Qualcomm options and descriptions. For
example, we run on-device AOT on Android device with burst mode via HTP Backend:

```bash
adb shell "export LD_LIBRARY_PATH=/tmp/test_folder/ && export ADSP_LIBRARY_PATH=/tmp/test_folder/ && cd /tmp/test_folder/ && ./run_model --graph=./model.tflite --accelerator npu --dispatch_library_dir=/tmp/test_folder/ --compiler_plugin_library_dir=/tmp/test_folder/ --compiler_cache_dir=/tmp/test_folder/cache --iterations 50 --qualcomm_htp_performance_mode burst --qualcomm_log_level off"
```
