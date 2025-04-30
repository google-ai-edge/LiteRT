# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Docker image to provide a hermetic build environment for Litert.
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
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
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Bazelisk to handle automatic Bazel version management
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64 -O bazelisk; \
    elif [ "$ARCH" = "aarch64" ]; then \
        wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-arm64 -O bazelisk; \
    else \
        echo "Unsupported architecture: $ARCH"; \
        exit 1; \
    fi && \
    chmod +x bazelisk && \
    mv bazelisk /usr/local/bin/bazel && \
    # Set USE_BAZEL_VERSION to ensure bazelisk downloads the right version
    echo "export USE_BAZEL_VERSION=7.4.1" >> /etc/bash.bashrc

# Set up Android SDK and NDK
ENV ANDROID_DEV_HOME=/android
RUN mkdir -p ${ANDROID_DEV_HOME}

# Create directory for Android config
RUN mkdir -p /root/.android

# Install Android SDK
ENV ANDROID_SDK_FILENAME=commandlinetools-linux-13114758_latest.zip
ENV ANDROID_SDK_URL=https://dl.google.com/android/repository/${ANDROID_SDK_FILENAME}
ENV ANDROID_API_LEVEL=34
ENV ANDROID_NDK_API_LEVEL=28
ENV ANDROID_SDK_API_LEVEL=34
ENV ANDROID_BUILD_TOOLS_VERSION=34.0.0
ENV ANDROID_SDK_HOME=${ANDROID_DEV_HOME}/sdk
RUN mkdir -p ${ANDROID_SDK_HOME}/cmdline-tools
ENV PATH=${PATH}:${ANDROID_SDK_HOME}/cmdline-tools/latest/bin:${ANDROID_SDK_HOME}/platform-tools
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_SDK_URL} && \
    unzip ${ANDROID_SDK_FILENAME} -d /tmp && \
    mv /tmp/cmdline-tools ${ANDROID_SDK_HOME}/cmdline-tools/latest && \
    rm ${ANDROID_SDK_FILENAME}

# Install Android NDK
ENV ANDROID_NDK_FILENAME=android-ndk-r28b-linux.zip
ENV ANDROID_NDK_URL=https://dl.google.com/android/repository/${ANDROID_NDK_FILENAME}
ENV ANDROID_NDK_HOME=${ANDROID_DEV_HOME}/ndk
ENV PATH=${PATH}:${ANDROID_NDK_HOME}
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_NDK_URL} && \
    unzip ${ANDROID_NDK_FILENAME} -d ${ANDROID_DEV_HOME} && \
    rm ${ANDROID_NDK_FILENAME} && \
    bash -c "ln -s ${ANDROID_DEV_HOME}/android-ndk-* ${ANDROID_NDK_HOME}"

# Create empty directories for SDK components (will be filled by sdkmanager in entrypoint)
RUN mkdir -p ${ANDROID_SDK_HOME}/build-tools
RUN mkdir -p ${ANDROID_SDK_HOME}/platforms
RUN mkdir -p ${ANDROID_SDK_HOME}/platform-tools

# Make android SDK and NDK executable to all users
RUN chmod -R go=u ${ANDROID_DEV_HOME}

# Install Python dependencies
RUN pip3 install numpy six absl-py wheel setuptools pybind11

# Set up environment variables for auto-configuration
ENV PYTHON_BIN_PATH=/usr/bin/python3
ENV PYTHON_LIB_PATH=/usr/lib/python3/dist-packages
ENV TF_NEED_CUDA=0
ENV TF_NEED_ROCM=0
ENV TF_DOWNLOAD_CLANG=0
ENV TF_SET_ANDROID_WORKSPACE=1
ENV TF_CONFIGURE_IOS=0
ENV USE_BAZEL_VERSION=7.4.1

# Set NDK version for configuration
ENV ANDROID_NDK_VERSION=25

RUN echo y | ${ANDROID_SDK_HOME}/cmdline-tools/latest/bin/sdkmanager --sdk_root=${ANDROID_SDK_HOME} "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" "platforms;android-${ANDROID_SDK_API_LEVEL}" "platform-tools" 
# Set up work directory
WORKDIR /litert_build

# Create a script to generate .tf_configure.bazelrc automatically and initialize git submodules
RUN echo '#!/bin/bash\n\
\n\
# Make the repository directory safe for git\n\
git config --global --add safe.directory /litert_build\n\
git config --global --add safe.directory /litert_build/third_party/tensorflow\n\
\n\
# Initialize git submodules if .gitmodules file exists\n\
if [ -f ".gitmodules" ]; then\n\
  echo "Initializing git submodules..."\n\
  git submodule init && git submodule update --remote\n\
  if [ $? -ne 0 ]; then\n\
    echo "Warning: Failed to initialize git submodules. Project may not build correctly."\n\
  else\n\
    echo "Git submodules initialized successfully."\n\
  fi\n\
else\n\
  echo "No .gitmodules file found. Skipping git submodule initialization."\n\
fi\n\
\n\
# Generate .tf_configure.bazelrc with necessary environment variables\n\
cat > /litert_build/.tf_configure.bazelrc << EOL\n\
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
EOL\n\
\n\
echo "Configuration complete. .tf_configure.bazelrc has been generated at /litert_build/.tf_configure.bazelrc"\n\
\n\
# Execute the command passed to the entrypoint\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Copy verification script
COPY verify_android_env.sh /verify_android_env.sh
RUN chmod +x /verify_android_env.sh

# Immediately execute a build.
CMD ["bash", "-c", "./build/verify_android_env.sh && USE_BAZEL_VERSION=7.4.1 bazel build //litert/runtime:metrics"]
