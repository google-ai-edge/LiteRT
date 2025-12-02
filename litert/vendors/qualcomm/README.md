# Setup with docker
- Please refer to [build/](../../../../build/) for more details.

# Setup without docker

## Environment
- Ubuntu 22.04
- Python 3.10.12

## Instructions
0. Most instructions are based on https://www.tensorflow.org/install/source.

1. Download and install bazelisk from https://github.com/bazelbuild/bazelisk/releases.
```sh
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-arm64.deb
sudo dkpg -i bazelisk-arm64.deb
```

2. Download Clang-17 from https://releases.llvm.org/.
```sh
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.2/clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04.tar.xz
tar -xvf clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04.tar.xz
```

3. Download android-ndk-r25c.

4. Download android-sdk-30.

5. Clone LiteRT from https://github.com/google-ai-edge/LiteRT. Update submodule and do bazel configuration.
```sh
git clone git@github.com:google-ai-edge/LiteRT.git
cd LiteRT/

# Update submodule
git submodule init && git submodule update --remote

# Do bazel configuration
./configure
You have bazel 7.4.1 installed.
Please specify the location of python. [Default is /usr/bin/python3]: <Your python path>


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


# Build Qualcomm targets with Qualcomm AI Runtime SDK
1. Download Qualcomm AI Runtime SDK. Please check [qairt-install](https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-15B/qairt-install.html) for more details.
```sh
# Direct download
wget https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.29.0.241129.zip

unzip v2.29.0.241129.zip

# Copy BUILD file into Qualcomm AI Runtime SDK
cp $LiteRT/third_party/qairt/qairt.BUILD ./qairt/2.29.0.241129/BUILD

# Establish soft link to Qualcomm AI Runtime SDK
ln -s ./qairt/2.29.0.241129/ $LiteRT/litert/third_party/qairt/latest
```

2. Build targets.
```sh
cd $LiteRT

# setup enviroment variable if you want to use downloaded qairt sdk.
export LITERT_QAIRT_SDK=$LiteRT/third_party/qairt/

# Example x86 targets:
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_test
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/tools:apply_plugin_main

# Example android arm64 targets:
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/vendors/qualcomm/dispatch:dispatch_api_so
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/tools:run_model
```