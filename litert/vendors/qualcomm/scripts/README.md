# Enviroment
- Ubuntu 22.04
- Python 3.10.12
- adb 1.0.41
- clang-format 14.0.0-1ubuntu1.1


# Setup
0. Most instructions are based on https://www.tensorflow.org/install/source.

1. Install bazel: Download bazelisk from https://github.com/bazelbuild/bazelisk/releases and install bazel by it.
```sh
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-arm64.deb

sudo dkpg -i bazelisk-arm64.deb
```

2. Download Clang-17 from https://releases.llvm.org/
```sh
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.2/clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04.tar.xz

tar -xvf clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04.tar.xz
```

3. Download android-ndk-r25c from https://github.com/android/ndk/wiki/Unsupported-Downloads 
```sh
wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip

unzip android-ndk-r25c-linux.zip
```

4. Download android-sdk-30

5. Clone LiteRT from https://github.com/google-ai-edge/LiteRT.
```sh
git clone git@github.com:google-ai-edge/LiteRT.git
cd LiteRT/

# Update submodule
git submodule init && git submodule update --remote

# Configure the build
./configure
You have bazel 7.4.1 installed.
Please specify the location of python. [Default is /usr/bin/python3]: 


Found possible Python library paths:
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.10/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]

Do you wish to build TensorFlow with ROCm support? [y/N]: 
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: 
No CUDA support will be enabled for TensorFlow.

Do you want to use Clang to build TensorFlow? [Y/n]: 
Clang will be used to compile TensorFlow.

Please specify the path to clang executable. [Default is ]: <Your Clang-17 binary path>


You have Clang 17.0.2 installed.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: Y
Searching for NDK and SDK installations.

Please specify the home path of the Android NDK to use. [Default is ]: <Your android-ndk-r25c path>


Please specify the (min) Android NDK API level to use. [Available levels: [16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33]] [Default is 21]: 33


Please specify the home path of the Android SDK to use. [Default is ]: <Your android-sdk-30 path>


Please specify the Android SDK API level to use. [Available levels: ['30']] [Default is 30]: 


Please specify an Android build tools version to use. [Available versions: ['30.0.3']] [Default is 30.0.3]: 


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
Configuration finished
```

6. build buidifier. buildifier is a tool for formatting bazel BUILD and .bzl files.
```sh
git clone git@github.com:bazelbuild/buildtools.git
cd buildtools/buildifier
bazel build //buildifier:buildifier
```


# Build Qualcomm targets
1. Download Qualcomm AI Runtime SDK. For more details, please check https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-15B/qairt-install.html
```sh
# Direct download
wget https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.29.0.241129.zip

unzip v2.29.0.241129.zip

# Copy BUILD file into Qualcomm AI Runtime SDK
cp $LiteRT/litert/vendors/qualcomm/scripts/BUILD ./qairt/2.29.0.241129/

# Establish soft link to Qualcomm AI Runtime SDK
ln -s ./qairt/2.29.0.241129/ $LiteRT/litert/third_party/qairt/latest
```

2. Uncomment copybara comments and apply workaround.patch before build.
```sh
# uncomment `"//third_party/qairt/latest:qnn_lib_headers",` in BUILD files before build
$LiteRT/litert/vendors/qualcomm/scripts/copybara.sh $LiteRT/litert/vendors/qualcomm/ uncomment \"//third_party/qairt/latest:qnn_lib_headers\",

cd $LiteRT

# apply workaround.patch before build.
git apply litert/vendors/qualcomm/scripts/workaround.patch

#########################################################

# build targets you need, e.g.
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_test
# or build and run all tests by this script.
./litert/vendors/qualcomm/scripts/run_all_tests_x86.sh

#########################################################

# revert workaround.path after build.
git apply -R litert/vendors/qualcomm/scripts/workaround.patch

# comment `"//third_party/qairt/latest:qnn_lib_headers",` in BUILD files after build
$LiteRT/litert/vendors/qualcomm/scripts/copybara.sh $LiteRT/litert/vendors/qualcomm/ comment \"//third_party/qairt/latest:qnn_lib_headers\",

# format
$LiteRT/litert/vendors/qualcomm/scripts/format.sh $LiteRT/litert/vendors/qualcomm/ /usr/local/bin/buildifier
```


# Useful Scripts and Commands

## copybara
- Remove or add back copybara comments.
- example:
```sh
# Print usage
$LiteRT/litert/vendors/qualcomm/scripts/copybara.sh -h

# Remove
$LiteRT/litert/vendors/qualcomm/scripts/copybara.sh $LiteRT/litert/vendors/qualcomm/ uncomment \"//third_party/qairt/latest:qnn_lib_headers\",

# Add
$LiteRT/litert/vendors/qualcomm/scripts/copybara.sh $LiteRT/litert/vendors/qualcomm/ comment \"//third_party/qairt/latest:qnn_lib_headers\",
```

## Workaround
- If you encounter some build issue, you can apply this patch before build.
- For more details, please see `$LiteRT/litert/vendors/qualcomm/scripts/workaround.patch`.
```sh
# apply
git apply $LiteRT/litert/vendors/qualcomm/scripts/workaround.patch

# revert
git apply -R $LiteRT/litert/vendors/qualcomm/scripts/workaround.patch
```

## format
- Format source code and bazel files.
- Note: not sure about the correct format style, plase use this script and only commit the files you modified.
- example:
```sh
# Print usage
$LiteRT/litert/vendors/qualcomm/scripts/format.sh -h

# do format
$LiteRT/litert/vendors/qualcomm/scripts/format.sh $LiteRT/litert/vendors/qualcomm/ /usr/local/bin/buildifier
```

## Run all tests
- Note: Suggest to use this script before you open a PR, it can help build and run all tests.
```sh
# Print usage
$LiteRT/litert/vendors/qualcomm/scripts/run_all_tests_x86.sh -h

# Run all tests on x86 machine
$LiteRT/litert/vendors/qualcomm/scripts/run_all_tests_x86.sh $LiteRT/
```

## apply_plugin_main
```sh
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/tools:apply_plugin_main
$LiteRT/bazel-bin/litert/tools/apply_plugin_main --cmd apply --libs $LiteRT/bazel-bin/litert/vendors/qualcomm/compiler --soc_model <SoC Model> --soc_manufacturer Qualcomm --model <Input Model Path> -o <Output Model Path>
# example:
$LiteRT/bazel-bin/litert/tools/apply_plugin_main --cmd apply --libs $LiteRT/bazel-bin/litert/vendors/qualcomm/compiler --soc_model SM8650 --soc_manufacturer Qualcomm --model <Input Model Path> -o <Output Model Path>
```

## run_model
- Run a compiled model on a device.
- Please update the device info in `$LiteRT/litert/vendors/qualcomm/scripts/devices.json` and specify the device you want to use.
```sh
# Print usage
$LiteRT/litert/vendors/qualcomm/scripts/run_model.sh -h

# execute a compiled model
$LiteRT/litert/vendors/qualcomm/scripts/run_model.sh $LiteRT/ <Device Key> <Compiled Model Path>

# example:
$LiteRT/litert/vendors/qualcomm/scripts/run_model.sh $LiteRT/ "example" <Compiled Model Path>
```
