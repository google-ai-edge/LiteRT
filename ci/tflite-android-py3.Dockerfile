FROM us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest

ENV ANDROID_DEV_HOME /android
RUN mkdir -p ${ANDROID_DEV_HOME}

RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk llvm-18 clang-18

# Install Android SDK.
ENV ANDROID_SDK_FILENAME commandlinetools-linux-13114758_latest.zip
ENV ANDROID_SDK_URL https://dl.google.com/android/repository/${ANDROID_SDK_FILENAME}
ENV ANDROID_API_LEVEL 35
# 21 is required for LiteRT v1 packages.
ENV ANDROID_NDK_API_LEVEL 21
ENV ANDROID_SDK_API_LEVEL 35

# Build Tools Version liable to change.
ENV ANDROID_BUILD_TOOLS_VERSION 35.0.1
ENV ANDROID_SDK_HOME ${ANDROID_DEV_HOME}/sdk
RUN mkdir -p ${ANDROID_SDK_HOME}/cmdline-tools
ENV PATH ${PATH}:${ANDROID_SDK_HOME}/cmdline-tools/latest/bin:${ANDROID_SDK_HOME}/platform-tools
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_SDK_URL} && \
    unzip ${ANDROID_SDK_FILENAME} -d /tmp && \
    mv /tmp/cmdline-tools ${ANDROID_SDK_HOME}/cmdline-tools/latest && \
    rm ${ANDROID_SDK_FILENAME}

# Install Android NDK.
ENV ANDROID_NDK_FILENAME android-ndk-r25b-linux.zip
ENV ANDROID_NDK_URL https://dl.google.com/android/repository/${ANDROID_NDK_FILENAME}
ENV ANDROID_NDK_HOME ${ANDROID_DEV_HOME}/ndk
ENV PATH ${PATH}:${ANDROID_NDK_HOME}
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_NDK_URL} && \
    unzip ${ANDROID_NDK_FILENAME} -d ${ANDROID_DEV_HOME} && \
    rm ${ANDROID_NDK_FILENAME} && \
    bash -c "ln -s ${ANDROID_DEV_HOME}/android-ndk-* ${ANDROID_NDK_HOME}"

# Make android ndk executable to all users.
RUN chmod -R go=u ${ANDROID_DEV_HOME}

# Install Python development tools and build dependencies for wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      build-essential \
      libpython3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set Python environment variables
ENV PYTHON_BIN_PATH /usr/bin/python3
ENV PYTHON_LIB_PATH /usr/lib/python3/dist-packages
