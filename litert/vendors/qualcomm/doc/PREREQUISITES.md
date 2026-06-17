# Environment Setup

You can follow
[BUILD_INSTRUCTIONS.md](../../../../g3doc/instructions/BUILD_INSTRUCTIONS.md) to
setup environment, the instructions below are what we used in our development
environment.

1.  Operating System

    -   Ubuntu 22.04

2.  Bazel

    -   Download .deb file from
        [Bazelisk release](https://github.com/bazelbuild/bazelisk/releases) and
        install.

    ```bash
    # For x86_64 Linux
    wget https://github.com/bazelbuild/bazelisk/releases/download/v1.28.1/bazelisk-amd64.deb
    sudo dpkg -i ./bazelisk-amd64.deb
    ```

3.  Clang

    -   Currently we use Clang-17.

    ```bash
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.2/clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04.tar.xz
    tar -xvf clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04.tar.xz
    ```

4.  Android NDK

    -   Currently we use android-ndk-r25c.

    ```bash
    wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip
    unzip android-ndk-r25c-linux.zip
    ```

5.  Android SDK

    -   Currently we use android-sdk-30.

    > Please refer to the Android official document to download sdkmanager first
    > and then install the SDK:
    > https://developer.android.com/tools/releases/platform-tools#downloads.

6.  Configure bazel

    -   Please set the following environment variables for the
        [Bazel Flow](./HTP_INSTRUCTIONS.md#build-with-bazel) in
        HTP_INSTRUCTIONS.md.

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

7.  CMake

    -   Currently we use 4.0.1, please set up this for the CMake flow in
        [HTP_INSTRUCTIONS.md](./HTP_INSTRUCTIONS.md).

    ```bash
    wget https://github.com/Kitware/CMake/releases/download/v4.0.1/cmake-4.0.1-linux-x86_64.tar.gz
    tar -xzvf cmake-4.0.1-linux-x86_64.tar.gz
    ```
