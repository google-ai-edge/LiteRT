# Environment Setup

You can follow `${LITERT}/g3doc/instructions/BUILD_INSTRUCTIONS.md` to setup environment, the instructions below are what we used in our development environment.

0. Operating System
   - Ubuntu 22.04

1. Bazel
   - Download .deb file from [Bazelisk release](https://github.com/bazelbuild/bazelisk/releases) and install.

```bash
# For x86_64 Linux
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.28.1/bazelisk-amd64.deb
sudo dpkg -i ./bazelisk-amd64.deb
```

2. Clang
   - Currently we use Clang-17.

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.2/clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04.tar.xz
tar -xvf clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04.tar.xz
```

3. Android NDK
   - Currently we use android-ndk-r25c.

```bash
wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip
unzip android-ndk-r25c-linux.zip
```

4. Android SDK
   - Currently we use android-sdk-30.

> Please refer to the Android official document to download sdkmanager first and then install the SDK: https://developer.android.com/tools/releases/platform-tools#downloads.

5. Configure bazel
   - Please set the following environment variables for "Bazel Flow (Compile & Execute)" section.

```bash
$ ${LITERT}/configure 
You have bazel 7.7.0 installed.
Please specify the location of python. [Default is /usr/bin/python3]: /path/to/python3


Found possible Python library paths:
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.10/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]

Do you wish to build TensorFlow with ROCm support? [y/N]: N
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: N
No CUDA support will be enabled for TensorFlow.

Do you want to use Clang to build TensorFlow? [Y/n]: Y
Clang will be used to compile TensorFlow.

Please specify the path to clang executable. [Default is ]: /path/to/downloaded/clang/executable


You have Clang 17.0.2 installed.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: Y
Searching for NDK and SDK installations.

Please specify the home path of the Android NDK to use. [Default is /usr/<UserName>/Android/Sdk/ndk-bundle]: /path/to/android-ndk-r25c/


Please specify the (min) Android NDK API level to use. [Available levels: [16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33]] [Default is 26]: 33


Please specify the home path of the Android SDK to use. [Default is /usr/<UserName>/Android/Sdk]: /path/to/android-sdk-30/


Please specify the Android SDK API level to use. [Available levels: ['30']] [Default is 30]: 30


Please specify an Android build tools version to use. [Available versions: ['30.0.3']] [Default is 30.0.3]: 30.0.3


Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=mkl            # Build with MKL support.
        --config=mkl_aarch64    # Build with oneDNN and Compute Library for the Arm Architecture (ACL).
        --config=monolithic     # Config for mostly static monolithic build.
        --config=numa           # Build with NUMA support.
        --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.
        --config=v1             # Build with TensorFlow 1 API instead of TF 2 API.
Preconfigured Bazel build configs to DISABLE default on features:
        --config=nogcp          # Disable GCP support.
        --config=nonccl         # Disable NVIDIA NCCL support.
```

6. CMake
   - Currently we use 4.0.1, please set up this for "Example: Compile and Execute a Model (through CMake)" section.

```bash
wget https://github.com/Kitware/CMake/releases/download/v4.0.1/cmake-4.0.1-linux-x86_64.tar.gz
tar -xzvf cmake-4.0.1-linux-x86_64.tar.gz
```

---

# QAIRT SDK

## About QAIRT SDK

QAIRT is a suite of tools that help you develop, run, and optimize AI models for Qualcomm-supported hardware.
- Official Document
  - https://docs.qualcomm.com/doc/80-63442-10/topic/general_overview.html
- Download link
  - Please download from https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community.
    - For example, users could download QAIRT-2.44.0 from https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.44.0.260225/v2.44.0.260225.zip
- Interface Header Files
  - `${QAIRT}/include/QNN/`

## Required Libraries

Here're some standard concepts of "How QNN works on the Qualcomm devices" users should know:

### Backend

Currently, LiteRT QC compiler plugin supports the following QNN backends: `Htp`, `Dsp`, `Ir`, `Saver`.

### Platform

Here're the supported operating system (OS) platforms of Qualcomm devices:

| Platform | Target |
|---|---|
| `aarch64-android` | Android devices |
| `x86_64-linux-clang` | x86 Linux devices |
| `x86_64-windows-msvc` | x86 Windows devices |
| `aarch64-oe-linux-gcc11.2` | aarch64 Linux devices (IoT devices) |
| `aarch64-windows-msvc` | aarch64 Windows devices |

### Hexagon Arch

When using HTP bakcend, users need to know which architecture is used in their devices. Please search "Supported Snapdragon devices" in https://docs.qualcomm.com/doc/80-63442-10/topic/QNN_general_overview.html and find the "Hexagon Arch" column to get the Hexagon Arch of the target device.
- For example, if you want to run on a Snapdragon 8 Elite Gen 5 (SM8850), the Hexagon Arch is V81.

### QNN libraries

After we know above information, users can locate all required libraries to execute QNN:
- `${QAIRT}/lib/{Platform}/libQnnSystem.so`
- `${QAIRT}/lib/{Platform}/libQnn{Backend}*.so`
- `${QAIRT}/lib/hexagon-v{HexagonArch}/unsigned/libQnnHtpV{HexagonArch}Skel.so`

For example, users want to execute QNN HTP backends on SM8850, they need to find the following files:
- `${QAIRT}/lib/aarch64-android/libQnnSystem.so`
- `${QAIRT}/lib/aarch64-android/libQnnHtp.so`
- `${QAIRT}/lib/aarch64-android/libQnnHtpPrepare.so`
- `${QAIRT}/lib/aarch64-android/libQnnHtpV81Stub.so`
- `${QAIRT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so`

For example, users want to compiled model for SM8850 with HTP backend in Linux work station, they need to find the following files:
- `${QAIRT}/lib/x86_64-linux-clang/libQnnHtp.so`
- `${QAIRT}/lib/x86_64-linux-clang/libQnnSystem.so`

## (Optional) Specify other QAIRT version in LiteRT

Modify `${LITERT}/third_party/qairt/workspace.bzl` and change `strip_prefix`& `url`. Please get the url of QAIRT SDK from https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community.

For example, the original workspace.bzl is using "qairt/2.42.0.251225" and users want to use 2.44, then the workspace.bzl should be changed from:
```bash
def qairt():
    configurable_repo(
        name = "qairt",
        build_file = "@//third_party/qairt:qairt.BUILD",
        local_path_env = "LITERT_QAIRT_SDK",
        strip_prefix = "qairt/2.42.0.251225",
        url = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip",
        file_extension = "zip",
    )
```

To:
```bash
def qairt():
    configurable_repo(
        name = "qairt",
        build_file = "@//third_party/qairt:qairt.BUILD",
        local_path_env = "LITERT_QAIRT_SDK",
        strip_prefix = "qairt/2.44.0.260225",
        url = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.44.0.260225/v2.44.0.260225.zip",
        file_extension = "zip",
    )
```

---

# Compile and Execute

Ensure that all `${}` variables are properly configured before proceeding:

| Variable | Description |
|---|---|
| `${LITERT}` | The path of LiteRT source code. |
| `${QAIRT}` | The root path of unzipped QAIRT SDK. |
| `${SOC_MODEL}` | Please search "Supported Snapdragon devices" in https://docs.qualcomm.com/doc/80-63442-10/topic/QNN_general_overview.html to find the correct "Snapdragon Device/Chip" column for your device. For example, use `SM8850` for "Snapdragon 8 Elite Gen 5". |
| `${SOURCE_MODEL_PATH}` | |
| `${COMPILED_MODEL_PATH}` | |
| `${TEST_FOLDER}` | The path of test folder on the device. |

Below are for specific sections:

| Variable | Description |
|---|---|
| `${EXTRACTED_BYTECODE_DIR}` | The path of extracted bytecode directory, used for "(Optional) Extract bytecode from compiled model for QNN Runtime" section. |
| `${ANDROID_NDK_HOME}` | The path of unzipped Android NDK, used for "Example: Compile and Execute a Model (through CMake)" section. |
| `${IOT_DIR}` | The path of the folder to store IoT required files, used for "Execution on target device (IoT device with oe-linux)" section. |

---

## Example: Compile and Execute a Model (through Bazel)

### AOT compilation on host machine(x86 Linux)

Build "apply_plugin_main" compilation tool and compile the model on Linux workstation (Ubuntu-22.04). 

```bash
cd ${LITERT}

bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/tools:apply_plugin_main

export LD_LIBRARY_PATH=${QAIRT}/lib/x86_64-linux-clang
bazel-bin/litert/tools/apply_plugin_main --cmd apply --libs bazel-bin/litert/vendors/qualcomm/compiler --soc_model ${SOC_MODEL} --soc_manufacturer Qualcomm --model ${SOURCE_MODEL_PATH} -o ${COMPILED_MODEL_PATH}
```

### Execution on target device (Android)

Build "run_model" and required libraries on Linux workstation:

```bash
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/tools:run_model
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/vendors/qualcomm/dispatch:dispatch_api_so
```

Execute the compiled model on Android device with QNN HTP backend via adb tool.

```bash
adb push ${QAIRT}/lib/aarch64-android/libQnnSystem.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtpV81Stub.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/c/libLiteRt.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so ${TEST_FOLDER}
adb push ${LITERT}/bazel-bin/litert/tools/run_model ${TEST_FOLDER}
adb push ${COMPILED_MODEL_PATH} ${TEST_FOLDER}

# Please use "--helpfull" to find all Qualcomm options and descriptions
adb shell "export LD_LIBRARY_PATH=${TEST_FOLDER} && export ADSP_LIBRARY_PATH=${TEST_FOLDER} && cd ${TEST_FOLDER} && ./run_model --graph=${COMPILED_MODEL_PATH} --dispatch_library_dir=${TEST_FOLDER}"
```

For example, we execute a model on Android device with burst mode via HTP Backend:

```bash
adb shell "export LD_LIBRARY_PATH=/tmp/test_folder/ && export ADSP_LIBRARY_PATH=/tmp/test_folder/ && cd /tmp/test_folder/ && ./run_model --graph=./model_compiled.tflite --dispatch_library_dir=/tmp/test_folder/ --iterations 50 --qualcomm_htp_performance_mode burst --qualcomm_log_level off"
```

---

## Example: Compile and Execute a Model (through CMake)

There's an official CMake build tutorial for your reference: `${LITERT}/g3doc/instructions/CMAKE_BUILD_INSTRUCTIONS.md`.

### AOT compilation on the host machine(x86 Linux)

```bash
cd ${LITERT}/litert

# This takes a while to download tensorflow
cmake --preset default
cmake --build cmake_build --target apply_plugin_main qnn_compiler_plugin litert_runtime_c_api_shared_lib -j8
```

The built files are located in:
- `${LITERT}/litert/cmake_build/c/libLiteRt.so`
- `${LITERT}/litert/cmake_build/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so`
- `${LITERT}/litert/cmake_build/tools/apply_plugin_main`

Execute apply_plugin_main to compile model.

```bash
export LD_LIBRARY_PATH=${QAIRT}/lib/x86_64-linux-clang

cd cmake_build

tools/apply_plugin_main --cmd apply --libs vendors/qualcomm/compiler/ --soc_model ${SOC_MODEL} --soc_manufacturer Qualcomm --model ${SOURCE_MODEL_PATH} -o ${COMPILED_MODEL_PATH}
```

#### (Optional) Extract bytecode from compiled model for QNN Runtime

Target `extract_bytecode` is a tool used to extract the context binary file from LiteRT compiled .tflite model. The extracted context binary file can be used to run the model with QNN runtime (`qnn-net-run`). 

If the users intend to run the model with LiteRT runtime, users could ignore to build this target.

```bash
cd ${LITERT}/litert

cmake --build cmake_build --target extract_bytecode -j8
```

The built files are located in:
- `${LITERT}/litert/cmake_build/tools/extract_bytecode`

Run extract_bytecode to extract the context binary file.

```bash
cd cmake_build

tools/extract_bytecode --model_path ${COMPILED_MODEL_PATH} --output_dir ${EXTRACTED_BYTECODE_DIR}
```

Follow QAIRT official document to execute the extracted context binary file with QNN Runtime (qnn-net-run) (https://docs.qualcomm.com/doc/80-63442-10/topic/general_tools.html).

### Execution on target device (Android)

This step needs Android NDK to build for Android. Both `android-ndk-r25c-linux` listed above and `android-ndk-r27` from `${LITERT}/g3doc/instructions/CMAKE_BUILD_INSTRUCTIONS.md` should work. Below is based on `android-ndk-r25c-linux`.

```bash
cd ${LITERT}/litert
export ANDROID_NDK_HOME=${ANDROID_NDK_HOME}

# This takes a while to build flatc and download tensorflow
cmake --preset android-arm64
cmake --build cmake_build_android_arm64 --target run_model litert_runtime_c_api_shared_lib dispatch_api_qualcomm_so -j8
```

The built files are located in:
- `${LITERT}/litert/cmake_build_android_arm64/c/libLiteRt.so`
- `${LITERT}/litert/cmake_build_android_arm64/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so`
- `${LITERT}/litert/cmake_build_android_arm64/tools/run_model`

Execute the compiled model on Android device with QNN HTP backend via adb tool.

```bash
adb push ${QAIRT}/lib/aarch64-android/libQnnSystem.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-android/libQnnHtpV81Stub.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so ${TEST_FOLDER}
adb push ${LITERT}/litert/cmake_build_android_arm64/c/libLiteRt.so ${TEST_FOLDER}
adb push ${LITERT}/litert/litert/cmake_build_android_arm64/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so ${TEST_FOLDER}
adb push ${LITERT}/litert/litert/cmake_build_android_arm64/tools/run_model ${TEST_FOLDER}
adb push ${COMPILED_MODEL_PATH} ${TEST_FOLDER}

# Please use "--helpfull" to find all Qualcomm options and descriptions
adb shell "export LD_LIBRARY_PATH=${TEST_FOLDER} && export ADSP_LIBRARY_PATH=${TEST_FOLDER} && cd ${TEST_FOLDER} && ./run_model --graph=${COMPILED_MODEL_PATH} --dispatch_library_dir=${TEST_FOLDER}"
```

For example, we execute a model on Android device with burst mode via HTP Backend:

```bash
adb shell "export LD_LIBRARY_PATH=/tmp/test_folder/ && export ADSP_LIBRARY_PATH=/tmp/test_folder/ && cd /tmp/test_folder/ && ./run_model --graph=./model_compiled.tflite --dispatch_library_dir=/tmp/test_folder/ --iterations 50 --qualcomm_htp_performance_mode burst --qualcomm_log_level off"
```

### Execution on target device (IoT device with oe-linux)

The cross compilation is only required when the host machine is x86 Linux and the target machine is oe-linux (Aarch64 Linux) based.

Below is a sample with x86 linux host machine and build for IQ-8275 (oe-linux) device.
- Please follow "Bringup IoT device Flash oe-linux" to flash the device and connect to the Linux workstation first.
- This step needs to download the eSDK for your target IoT device.

Install prerequisite for Linux workstation.

```bash
sudo locale-gen en_US.UTF-8
sudo apt update
sudo apt install gawk
```

Install ESDK on the Linux workstation.
- Check the META version of the IoT device.
- Find the corresponding eSDK version in https://artifacts.codelinaro.org/ui/native/qli-ci/flashable-binaries/qimpsdk/, select "qcs8275-iq-8275-evk-pro-sku/"
  - For example, Download "x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-esdk-2.3.0.zip" for Linux x86 host machine when the META version is "6.6.119-QLI.1.8".
- Install the ESDK on Linux workstation
  - Input the installation path ${ESDK}, please reserve spaces for it
  - Input "Y" to install

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

The built files are located in:
- `${LITERT}/litert/cmake_build_linux_aarch64_8275/c/libLiteRt.so`
- `${LITERT}/litert/cmake_build_linux_aarch64_8275/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so`
- `${LITERT}/litert/cmake_build_linux_aarch64_8275/tools/run_model`

Upload required QNN libraries, LiteRT files and compiled model from Linux workstation to IoT device.

```bash
adb push ${QAIRT}/lib/aarch64-oe-linux-gcc11.2/libQnnSystem.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/aarch64-oe-linux-gcc11.2/libQnnHtpV75Stub.so ${TEST_FOLDER}
adb push ${QAIRT}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so ${TEST_FOLDER}
adb push ${LITERT}/litert/cmake_build_linux_aarch64_8275/c/libLiteRt.so
adb push ${LITERT}/litert/cmake_build_linux_aarch64_8275/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so
adb push ${LITERT}/litert/cmake_build_linux_aarch64_8275/tools/run_model
```

Upload the model compiled by previous step from Linux workstation to IoT device
- For example, need to use --soc_model QCS8300 for model compilation for IQ-8275

```bash
adb push ${COMPILED_MODEL_PATH} ${TEST_FOLDER}
```

Execute the compiled model on IoT device with QNN HTP backend via adb tool, some error logs for IQ-8275 execution are expected.

```bash
# Please use "--helpfull" to find all Qualcomm options and descriptions
adb shell "export LD_LIBRARY_PATH=${TEST_FOLDER} && export ADSP_LIBRARY_PATH=${TEST_FOLDER} && cd ${TEST_FOLDER} && ./run_model --graph=${COMPILED_MODEL_PATH} --dispatch_library_dir=${TEST_FOLDER}"
```

For example, we execute a model on IoT device with burst mode via HTP Backend:

```bash
adb shell "export LD_LIBRARY_PATH=/tmp/test_folder/ && export ADSP_LIBRARY_PATH=/tmp/test_folder/ && cd /tmp/test_folder/ && ./run_model --graph=./model_compiled.tflite --dispatch_library_dir=/tmp/test_folder/ --iterations 50 --qualcomm_htp_performance_mode burst --qualcomm_log_level off"
```