#!/usr/bin/env bash
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
set -e
set -x

ESDK_DEFAULT_ROOT="/opt/toolchains/qcom-esdk-gcc11.2"
ESDK_URL="https://artifacts.codelinaro.org/artifactory/qli-ci/flashable-binaries/qimpsdk/qcs8275-iq-8275-evk-pro-sku/x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-esdk-2.3.0.zip"
ESDK_SHA256="4381fb56f2b47e9691cb72bf937c20e5b199bb304ec4eb3a98ec64bfd875cbe0"
ESDK_INSTALLER_REL="target/qcs8275-iq-8275-evk-pro-sku/sdk/qcom-wayland-x86_64-qcom-multimedia-image-armv8-2a-qcs8275-iq-8275-evk-pro-sku-toolchain-ext-1.8-ver.1.0.sh"

install_and_prune_esdk() {
  local esdk_root="${LINUX_AARCH64_ESDK:-${ESDK_DEFAULT_ROOT}}"
  local esdk_gcc="${esdk_root}/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-gcc"
  if [[ -x "${esdk_gcc}" ]]; then
    echo "Qualcomm OpenEmbedded eSDK already present at ${esdk_root}"
    mkdir -p "${esdk_root}/tmp/sysroots-uninative/x86_64-linux/lib"
    ln -sf /lib64/ld-linux-x86-64.so.2 "${esdk_root}/tmp/sysroots-uninative/x86_64-linux/lib/ld-linux-x86-64.so.2"
    return 0
  fi

  local tmp_esdk="/tmp/esdk11_2"
  rm -rf "${tmp_esdk}"
  mkdir -p "${esdk_root}" "${tmp_esdk}"

  echo "Downloading Qualcomm OpenEmbedded eSDK from ${ESDK_URL}..."
  pushd "${tmp_esdk}" > /dev/null
  wget -q --timeout=60 --tries=3 "${ESDK_URL}" -O esdk.zip
  echo "${ESDK_SHA256}  esdk.zip" | sha256sum -c -
  unzip -q esdk.zip "target/qcs8275-iq-8275-evk-pro-sku/sdk/*.sh"
  rm -f esdk.zip
  chmod +x "./${ESDK_INSTALLER_REL}"

  # Yocto/eSDK installers forbid execution as root (UID 0).
  if [[ "$(id -u)" -eq 0 ]]; then
    if ! id -u builder >/dev/null 2>&1; then
      groupadd builder 2>/dev/null || true
      useradd -m -g builder -s /bin/bash builder
    fi
    mkdir -p "${esdk_root}"
    chown -R builder:builder "${esdk_root}" "${tmp_esdk}"
    su -s /bin/bash builder -c 'sh "$1" -y -d "$2"' _ "${tmp_esdk}/${ESDK_INSTALLER_REL}" "${esdk_root}"
  else
    sh "./${ESDK_INSTALLER_REL}" -y -d "${esdk_root}"
  fi
  popd > /dev/null

  echo "Pruning non-compiler/sysroot overhead in ${esdk_root}..."
  rm -rf "${esdk_root}/sstate-cache" \
         "${esdk_root}/layers" \
         "${esdk_root}/bitbake" \
         "${esdk_root}/downloads" \
         "${esdk_root}/buildtools" \
         "${esdk_root}/tmp-qcom-guestvm" \
         "${esdk_root}/tmp/work" \
         "${esdk_root}/tmp/work-shared" \
         "${esdk_root}/tmp/sysroots-components" \
         "${esdk_root}/tmp/sysroots-uninative" \
         "${esdk_root}/tmp/deploy" \
         "${esdk_root}/tmp/cache" \
         "${esdk_root}/tmp/pkgdata" \
         "${esdk_root}/tmp/log" \
         "${esdk_root}/tmp/sstate-control" \
         "${esdk_root}/tmp/stamps" \
         "${esdk_root}/tmp/sysroots/"*/usr/share \
         "${esdk_root}/tmp/sysroots/"*/manifest-* \
         "${esdk_root}/tmp/sysroots/"*/sysroot-providers \
         "${esdk_root}/tmp/sysroots/"*/etc \
         "${esdk_root}/tmp/sysroots/"*/var
  find "${esdk_root}/tmp/sysroots/x86_64/usr/bin" -mindepth 1 -maxdepth 1 ! -name "aarch64-qcom-linux" -exec rm -rf {} +
  rm -rf "${esdk_root}/tmp/sysroots/x86_64/usr/include" \
         "${esdk_root}/tmp/sysroots/x86_64/usr/sbin" \
         "${esdk_root}/tmp/sysroots/x86_64/bin" \
         "${esdk_root}/tmp/sysroots/x86_64/sbin" \
         "${esdk_root}/tmp/sysroots/x86_64/lib"
  find "${esdk_root}/tmp/sysroots/x86_64/usr/lib" -mindepth 1 -maxdepth 1 -type d ! -name "aarch64-qcom-linux" -exec rm -rf {} +
  rm -f "${esdk_root}/tmp/sysroots/x86_64/usr/lib/"*.a \
        "${esdk_root}/tmp/sysroots/x86_64/usr/lib/librustc"* \
        "${esdk_root}/tmp/sysroots/x86_64/usr/lib/libLLVM"* \
        "${esdk_root}/tmp/sysroots/x86_64/usr/lib/libicudata"* \
        "${esdk_root}/tmp/sysroots/x86_64/usr/lib/libclang"* \
        "${esdk_root}/tmp/sysroots/x86_64/usr/lib/libLTO"*

  local target_sysroot="${esdk_root}/tmp/sysroots/qcs8275-iq-8275-evk-pro-sku"
  rm -rf "${target_sysroot}/usr/bin" \
         "${target_sysroot}/usr/sbin" \
         "${target_sysroot}/usr/libexec" \
         "${target_sysroot}/usr/src" \
         "${target_sysroot}/usr/games" \
         "${target_sysroot}/bin" \
         "${target_sysroot}/sbin" \
         "${target_sysroot}/usr/include/boost" \
         "${target_sysroot}/usr/include/autogen" \
         "${target_sysroot}/usr/include/opencv4" \
         "${target_sysroot}/usr/include/tensorflow" \
         "${target_sysroot}/usr/include/eigen3" \
         "${target_sysroot}/usr/include/absl" \
         "${target_sysroot}/usr/include/google" \
         "${target_sysroot}/usr/include/gstreamer"* \
         "${target_sysroot}/usr/include/chi-cdk"* \
         "${target_sysroot}/usr/include/camx"* \
         "${target_sysroot}/usr/include/vulkan"
  find "${target_sysroot}/usr/lib" -mindepth 1 -maxdepth 1 -type d ! -name "aarch64-qcom-linux" ! -name "gcc" -exec rm -rf {} +
  find "${target_sysroot}/usr/lib" -maxdepth 1 \( -type f -o -type l \) \
    ! -name "ld-*" ! -name "*crt*.o" \
    ! -name "libc.*" ! -name "libc-*" ! -name "libc_*" \
    ! -name "libm.*" ! -name "libm-*" ! -name "libmvec*" \
    ! -name "libdl*" ! -name "libpthread*" ! -name "librt*" \
    ! -name "libresolv*" ! -name "libutil*" ! -name "libanl*" ! -name "libnsl*" ! -name "libcrypt.*" ! -name "libcrypt-*" ! -name "libnss*" ! -name "libBrokenLocale*" ! -name "libthread_db*" \
    ! -name "libstdc++*" ! -name "libsupc++*" ! -name "libgcc*" ! -name "libatomic*" ! -name "libgomp*" ! -name "libitm*" ! -name "libasan*" ! -name "libtsan*" ! -name "libubsan*" ! -name "liblsan*" ! -name "libhwasan*" ! -name "libssp*" ! -name "libquadmath*" \
    ! -name "libz.*" ! -name "libz-*" \
    -delete

  mkdir -p "${esdk_root}/tmp/sysroots-uninative/x86_64-linux/lib"
  ln -sf /lib64/ld-linux-x86-64.so.2 "${esdk_root}/tmp/sysroots-uninative/x86_64-linux/lib/ld-linux-x86-64.so.2"
  chmod -R a+rX "${esdk_root}"
  rm -rf "${tmp_esdk}"
}

install_and_prune_esdk
