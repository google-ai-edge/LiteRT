#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shared Bazel environment setup for Docker builds.
# Source this file from any build script to get:
#   - EXTRA_STARTUP: Bazel startup flags for proxy and SVE workarounds
#   - Android env verification
#
# Usage:
#   source /setup_bazel_env.sh
#   bazel ${EXTRA_STARTUP} build //your:target
#
# When no proxy env vars are set, EXTRA_STARTUP stays empty and Bazel
# connects directly — all proxy code is a no-op.

./docker_build/verify_android_env.sh

# ---- Parse proxy env vars into Bazel JVM startup flags ----
# Bazel's JVM-based downloader does NOT read http_proxy/https_proxy env vars.
# We parse those variables and forward them as Java system properties
# (-Dhttps.proxyHost, -Dhttps.proxyPort, etc.) via --host_jvm_args.
EXTRA_STARTUP=""

parse_proxy() {
  # Input: a proxy URL like http://proxy.example.com:912/
  # Output: two words — HOST PORT (or empty if parsing fails)
  local url="$1"
  [ -z "$url" ] && return
  local stripped="${url#*://}"
  stripped="${stripped%%/*}"
  stripped="${stripped##*@}"
  local host="${stripped%%:*}"
  local port="${stripped##*:}"
  [ -n "$host" ] && [ -n "$port" ] && [ "$host" != "$port" ] && echo "$host $port"
}

if [ -n "${https_proxy:-}" ]; then
  read -r PH PP <<< "$(parse_proxy "$https_proxy")"
  [ -n "${PH:-}" ] && EXTRA_STARTUP="${EXTRA_STARTUP} --host_jvm_args=-Dhttps.proxyHost=${PH} --host_jvm_args=-Dhttps.proxyPort=${PP}"
fi
if [ -n "${http_proxy:-}" ]; then
  read -r PH PP <<< "$(parse_proxy "$http_proxy")"
  [ -n "${PH:-}" ] && EXTRA_STARTUP="${EXTRA_STARTUP} --host_jvm_args=-Dhttp.proxyHost=${PH} --host_jvm_args=-Dhttp.proxyPort=${PP}"
fi

# ---- SVE workaround for AArch64 (Apple Silicon under Docker) ----
arch=$(uname -m || echo unknown)
if [ "${DISABLE_SVE_FOR_BAZEL:-}" = "1" ] && { [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; }; then
  EXTRA_STARTUP="${EXTRA_STARTUP} --host_jvm_args=-XX:UseSVE=0"
  echo "[setup_bazel_env] AArch64 detected; disabling SVE for Bazel host JVM" >&2
fi

export EXTRA_STARTUP
