# LiteRT Build Instructions

This document provides detailed build instructions for LiteRT across supported
platforms: Linux, macOS, Windows, Android.

## Installing Bazelisk (Recommended)

```bash
# Linux
curl -LO https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel

# macOS (via Homebrew)
brew install bazelisk

# Windows (PowerShell)
Invoke-WebRequest -Uri https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-windows-amd64.exe -OutFile bazel.exe
Move-Item bazel.exe C:\Windows\System32\
```

## Setup local build environment

Use `./docker_build/hermetic_build.Dockerfile` as the reference to create your
own `.tf_configure.bazelrc` file

```
build --action_env PYTHON_BIN_PATH="${PYTHON_BIN_PATH}"\n\
build --action_env PYTHON_LIB_PATH="${PYTHON_LIB_PATH}"\n\
build --action_env TF_NEED_CUDA="${TF_NEED_CUDA}"\n\
build --action_env TF_NEED_ROCM="${TF_NEED_ROCM}"\n\
build --action_env TF_DOWNLOAD_CLANG="${TF_DOWNLOAD_CLANG}"\n\
build --action_env TF_SET_ANDROID_WORKSPACE="${TF_SET_ANDROID_WORKSPACE}"\n\
build --action_env ANDROID_SDK_HOME="${ANDROID_SDK_HOME}"\n\
build --action_env ANDROID_NDK_HOME="${ANDROID_NDK_HOME}"\n\
build --action_env ANDROID_BUILD_TOOLS_VERSION="${ANDROID_BUILD_TOOLS_VERSION}"\n\
build --action_env ANDROID_SDK_API_LEVEL="${ANDROID_SDK_API_LEVEL}"\n\
build --action_env ANDROID_NDK_API_LEVEL="${ANDROID_NDK_API_LEVEL}"\n\
build --action_env ANDROID_NDK_VERSION="${ANDROID_NDK_VERSION}"\n\
build --action_env TF_CONFIGURE_IOS="${TF_CONFIGURE_IOS}"\n\
build --action_env CLANG_COMPILER_PATH="${CLANG_COMPILER_PATH}"\n\
build --action_env TF_NEED_CLANG="${TF_NEED_CLANG}"\n\
build --action_env CLANG_COMPILER_PATH="${CLANG_COMPILER_PATH}"\n\
build --repo_env=CC="${CLANG_COMPILER_PATH}"\n\
build --repo_env=BAZEL_COMPILER="${CLANG_COMPILER_PATH}"\n\
```

## Platform-Specific Instructions

## Linux Build Instructions

### Step 1: Install Dependencies

```bash
# Update package manager
sudo apt-get update

# Install build essentials
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    openjdk-17-jdk \
    python3 \
    python3-pip \
    python3-dev \
    unzip \
    wget \
    zip \
    llvm-18 \
    clang-18 \
    libc++-dev \
    libc++abi-dev
```

### Step 2: Clone and Build

#### Bazel Build (Linux)

```bash
git clone https://github.com/google-ai-edge/LiteRT.git
cd LiteRT

# Configure for Linux
cat > .tf_configure.bazelrc << EOF
build --config=linux
EOF

# Build
bazel build //litert/cc:litert_api
bazel build //litert/tools:all
```

### Step 3: Verify Installation

```bash
# Test the benchmark tool
./bazel-bin/litert/tools/benchmark_model \
    --model=path/to/model.tflite \
    --num_threads=4
```

## macOS Build Instructions

### Step 1: Install Prerequisites

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install python@3.11
```

### Step 2: Build for macOS

#### Bazel Build (macOS)

```bash
git clone https://github.com/google-ai-edge/LiteRT.git
cd LiteRT

# Configure for macOS
cat > .tf_configure.bazelrc << EOF
build --config=macos
EOF

# Build for Apple Silicon
bazel build --config=macos_arm64 //litert/cc:litert_api

# Or build for Intel Mac
bazel build --config=macos_x86_64 //litert/cc:litert_api

# Build tools
bazel build //litert/tools:all
```

## Windows Build Instructions

### Step 1: Install Prerequisites

```powershell
# Install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install required packages
choco install git python3 -y

# Install Visual Studio Build Tools (if not installed)
choco install visualstudio2022buildtools -y
choco install visualstudio2022-workload-vctools -y
```

### Step 2: Configure Environment

```powershell
# Set environment variables
[System.Environment]::SetEnvironmentVariable("BAZEL_VC", "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC", "User")
[System.Environment]::SetEnvironmentVariable("BAZEL_SH", "C:\Program Files\Git\bin\bash.exe", "User")

# Restart PowerShell to apply changes
```

### Step 3: Build for Windows

#### Bazel Build (Windows)

```powershell
git clone https://github.com/google-ai-edge/LiteRT.git
cd LiteRT

# Configure for Windows
@"
build --config=windows
"@ | Out-File -Encoding ASCII .bazelrc.user

# Build
bazel build //litert/cc:litert_api
bazel build //litert/tools:all
```

## Android Build Instructions

### System Requirements

- **Host OS**: Linux, macOS, or Windows
- **Android SDK**: API level 21+ (Android 5.0+)
- **Android NDK**: r21+ recommended
- **Android Studio**: For sample apps

### Step 1: Install Android Development Tools

```bash
# Download Android SDK Command Line Tools
mkdir -p ~/android-sdk
cd ~/android-sdk
wget https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip
unzip commandlinetools-linux-9477386_latest.zip

# Set up environment
export ANDROID_HOME=~/android-sdk
export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$PATH
export PATH=$ANDROID_HOME/platform-tools:$PATH

# Install required SDK components
sdkmanager "platform-tools" "platforms;android-33" "build-tools;33.0.0" "ndk;21.4.7075529"
```

### Step 2: Build for Android with Bazel

```bash
cd LiteRT

# Configure for Android
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/21.4.7075529

# Build for ARM64
bazel build --config=android_arm64 \
  //litert/some_target
```

### Step 3: Test Accelerator Support for Android

```bash

# Push NPU libraries (Use Qualcomm as an example)
case "$PHONE" in
    's24')
        QNN_STUB_LIB="libQnnHtpV75Stub.so"
        QNN_SKEL_LIB="libQnnHtpV75Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v75"
        ;;
    's25')
        QNN_STUB_LIB="libQnnHtpV79Stub.so"
        QNN_SKEL_LIB="libQnnHtpV79Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v79"
        ;;
    ;; ...
esac

adb push "${HOST_NPU_LIB}/aarch64-android/libQnnHtp.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIB}/aarch64-android/${QNN_STUB_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIB}/aarch64-android/libQnnSystem.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIB}/aarch64-android/libQnnHtpPrepare.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIB}/${QNN_SKEL_PATH_ARCH}/unsigned/${QNN_SKEL_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"

# Push GPU libraries
adb push "${HOST_GPU_LIBRARY_DIR}/libLiteRtGpuAccelerator.so" "${DEVICE_BASE_DIR}/"

# For Qualcomm NPU (ADSP_LIBRARY_PATH needs to be specified)
FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ADSP_LIBRARY_PATH=\"${ADSP_LIBRARY_PATH}\" ${RUN_COMMAND}"

# Other cases
FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ${RUN_COMMAND}"
```

## License

Copyright 2025 Google LLC. Licensed under Apache License 2.0.
