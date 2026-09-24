# Copyright 2026 The AI Edge LiteRT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# LiteRT Linux IoT Build Container
#
# Purpose:
#   Provides a hermetic, reproducible cross-compilation environment for building
#   LiteRT libraries, Qualcomm NPU vendor plugins (compiler and dispatch), and
#   command-line tools (run_model, apply_plugin_main) targeting OpenEmbedded /
#   Yocto Linux ARM64 IoT and embedded devices (GCC 11.2 / glibc 2.35).
#
# Use Cases:
#   1. CI/CD Release Pipelines: Automates nightly and release artifact builds
#      of LiteRT OpenEmbedded Linux ARM64 binaries for packaging and distribution.
#   2. Local Workstation Builds: Enables developers to build and test LiteRT IoT
#      binaries using the standard OpenEmbedded / Qualcomm eSDK toolchain via
#      'build_iot_linux_with_docker.sh' without needing to manually install the
#      Yocto SDK and cross-toolchains on the host machine.
#
# Key Features:
#   - Pre-installed and pruned Qualcomm eSDK (aarch64-qcom-linux-gcc 11.2, glibc 2.35).
#   - Build tools: CMake, Ninja, chrpath, aarch64 cross-compilers, and stripping tools.
#   - Compatible with CMake preset 'linux-aarch64-oe-gcc11.2'.
# ==============================================================================

FROM us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest
# Install build essentials, cross-compilers, and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    ninja-build \
    cmake \
    gawk \
    pigz \
    zip \
    unzip \
    wget \
    curl \
    locales \
    diffstat \
    cpio \
    chrpath \
    file \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Set up toolchain root directory
ENV TOOLCHAINS_DIR=/opt/toolchains
ENV ESDK_GCC11_2_ROOT=${TOOLCHAINS_DIR}/qcom-esdk-gcc11.2

# Create non-root user 'builder' because Yocto/eSDK installers forbid installation as root
RUN groupadd -g 1000 builder && \
    useradd -u 1000 -g builder -m -s /bin/bash builder && \
    mkdir -p ${TOOLCHAINS_DIR} /tmp/esdk11_2 && \
    chown -R builder:builder ${TOOLCHAINS_DIR} /tmp/esdk11_2

USER builder
RUN cd /tmp/esdk11_2 && \
    wget -q https://artifacts.codelinaro.org/artifactory/qli-ci/flashable-binaries/qimpsdk/qcs8275-iq-8275-evk-pro-sku/x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-esdk-2.3.0.zip && \
    unzip -q x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-esdk-2.3.0.zip "target/qcs8275-iq-8275-evk-pro-sku/sdk/*.sh" && \
    rm -f x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-esdk-2.3.0.zip && \
    sh ./target/qcs8275-iq-8275-evk-pro-sku/sdk/qcom-wayland-x86_64-qcom-multimedia-image-armv8-2a-qcs8275-iq-8275-evk-pro-sku-toolchain-ext-1.8-ver.1.0.sh -y -d ${ESDK_GCC11_2_ROOT} && \
    # Prune non-compiler/sysroot overhead to minimize image footprint (< 2GB)
    rm -rf ${ESDK_GCC11_2_ROOT}/sstate-cache \
           ${ESDK_GCC11_2_ROOT}/layers \
           ${ESDK_GCC11_2_ROOT}/bitbake \
           ${ESDK_GCC11_2_ROOT}/downloads \
           ${ESDK_GCC11_2_ROOT}/buildtools \
           ${ESDK_GCC11_2_ROOT}/tmp-qcom-guestvm \
           ${ESDK_GCC11_2_ROOT}/tmp/work \
           ${ESDK_GCC11_2_ROOT}/tmp/work-shared \
           ${ESDK_GCC11_2_ROOT}/tmp/sysroots-components \
           ${ESDK_GCC11_2_ROOT}/tmp/sysroots-uninative \
           ${ESDK_GCC11_2_ROOT}/tmp/deploy \
           ${ESDK_GCC11_2_ROOT}/tmp/cache \
           ${ESDK_GCC11_2_ROOT}/tmp/pkgdata \
           ${ESDK_GCC11_2_ROOT}/tmp/log \
           ${ESDK_GCC11_2_ROOT}/tmp/sstate-control \
           ${ESDK_GCC11_2_ROOT}/tmp/stamps \
           ${ESDK_GCC11_2_ROOT}/tmp/sysroots/*/usr/share && \
    mkdir -p ${ESDK_GCC11_2_ROOT}/tmp/sysroots-uninative/x86_64-linux/lib && \
    ln -sf /lib64/ld-linux-x86-64.so.2 ${ESDK_GCC11_2_ROOT}/tmp/sysroots-uninative/x86_64-linux/lib/ld-linux-x86-64.so.2 && \
    chmod -R a+rX ${ESDK_GCC11_2_ROOT} && \
    rm -rf /tmp/esdk11_2

USER root
# Avoid Git 2.35.2+ dubious ownership errors in mounted volumes and GitHub Actions
RUN git config --system --add safe.directory '*'

# Set default environment variables for LiteRT CMake builds
ENV LINUX_AARCH64_ESDK=${ESDK_GCC11_2_ROOT}
ENV PATH=${PATH}:${ESDK_GCC11_2_ROOT}/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux
