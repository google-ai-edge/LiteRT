#!/bin/bash
# Copyright 2026 Google LLC.
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
#
# Build the LiteRT Python wheel and vendor SDK packages using the Docker
# hermetic build environment.  Reuses the same litert_build_env image as
# build_with_docker.sh but overrides the default CMD to run the pip wheel
# build pipeline.
#
# Usage:
#   cd LiteRT/docker_build
#   ./build_wheel_with_docker.sh [--use_existing_image]
#
# Outputs are copied to ../dist/ (relative to docker_build/).

set -eo pipefail

# Change to the directory of this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Docker pre-checks ----
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# ---- Parse arguments ----
SKIP_BUILD=0
LITERT_BUILD_MODE="opt"
HERMETIC_PYTHON_VERSION=""
for arg in "$@"; do
  case "$arg" in
    --use_existing_image)
      SKIP_BUILD=1
      ;;
    --dbg)
      LITERT_BUILD_MODE="dbg"
      ;;
    --python=*)
      HERMETIC_PYTHON_VERSION="${arg#*=}"
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --use_existing_image  Skip 'docker build' and reuse existing 'litert_build_env' image"
      echo "  --dbg                 Build in debug mode (default: opt)"
      echo "  --python=VERSION      Set HERMETIC_PYTHON_VERSION (e.g., 3.12)"
      echo "  -h, --help            Show this help message"
      exit 0
      ;;
  esac
done

# ---- Build Docker image (unless --use_existing_image) ----
if [ "$SKIP_BUILD" -eq 0 ]; then
  echo "Building Docker image..."
  # Forward host proxy env vars into the Docker image build so that apt-get,
  # wget, and pip can reach the internet from behind a corporate proxy.
  # The :- default ensures unset variables expand to empty strings rather
  # than causing errors under set -eo pipefail. Empty values are harmless —
  # Docker treats them as "no proxy".
  docker build -t litert_build_env -f ./hermetic_build.Dockerfile \
    --build-arg http_proxy="${http_proxy:-}" \
    --build-arg https_proxy="${https_proxy:-}" \
    --build-arg no_proxy="${no_proxy:-}" \
    --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
    --build-arg NO_PROXY="${NO_PROXY:-}" \
    .
  if [ $? -ne 0 ]; then
    echo "Error: Docker build failed."
    exit 1
  fi
else
  echo "Using existing Docker image 'litert_build_env' (skipping build)"
fi

# ---- Container name ----
CONTAINER_NAME="litert_wheel_build_container"

# Remove any previous wheel-build container so we always get a clean run.
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing previous wheel-build container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1
fi

# ---- SVE workaround for Apple Silicon ----
HOST_OS=$(uname -s || echo unknown)
HOST_ARCH=$(uname -m || echo unknown)
DISABLE_SVE_ARG=()
if [ "$HOST_OS" = "Darwin" ] && { [ "$HOST_ARCH" = "arm64" ] || [ "$HOST_ARCH" = "aarch64" ]; }; then
  DISABLE_SVE_ARG=(-e DISABLE_SVE_FOR_BAZEL=1)
fi

# ---- Extra env vars ----
EXTRA_ENV=()
EXTRA_ENV+=(-e LITERT_BUILD_MODE="${LITERT_BUILD_MODE}")
if [ -n "${HERMETIC_PYTHON_VERSION}" ]; then
  EXTRA_ENV+=(-e HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION}")
fi

# ---- Run the wheel build ----
# Proxy env vars are forwarded into the container so that curl, pip, and
# other tools can reach the internet. When unset, empty strings are passed
# and no proxy is configured. Inside the container, these are additionally
# parsed into Bazel JVM flags (see parse_proxy_host_port below).
echo "Running wheel build in Docker container..."
docker run --name "${CONTAINER_NAME}" \
  --security-opt seccomp=unconfined \
  --user "$(id -u):$(id -g)" \
  -e HOME=/litert_build \
  -e "USER=$(id -un)" \
  -e "http_proxy=${http_proxy:-}" \
  -e "https_proxy=${https_proxy:-}" \
  -e "no_proxy=${no_proxy:-}" \
  -e "HTTP_PROXY=${HTTP_PROXY:-}" \
  -e "HTTPS_PROXY=${HTTPS_PROXY:-}" \
  -e "NO_PROXY=${NO_PROXY:-}" \
  "${DISABLE_SVE_ARG[@]}" \
  "${EXTRA_ENV[@]}" \
  -v "${REPO_ROOT}:/litert_build" \
  litert_build_env \
  bash -c '
set -eo pipefail

# Source shared Bazel environment setup (proxy-to-JVM forwarding, SVE workaround).
# Sets EXTRA_STARTUP with any needed --host_jvm_args flags.
source /setup_bazel_env.sh

# ---- Pre-fetch external archives that may be blocked by rate-limiters ----
# Google third-party-mirror.googlesource.com can return 429 / CAPTCHA for
# certain corporate networks. curl/wget honour the http_proxy env var and
# can sometimes succeed where Bazel JVM downloader fails.  We pre-populate
# the Bazel repository cache so that Bazel never needs to fetch these URLs.
prefetch_for_bazel() {
  # Usage: prefetch_for_bazel <url> <sha256|""> <cache_base>
  # Stores the downloaded archive under Bazel repository cache layout:
  #   <cache_base>/content_addressable/sha256/<hash>/file
  local url="$1"
  local sha256="$2"
  local cache_base="$3"

  if [ -z "$sha256" ]; then
    echo "[prefetch] No sha256 for $url — skipping cache pre-population"
    return 0
  fi

  local cache_dir="${cache_base}/content_addressable/sha256/${sha256}"
  if [ -f "${cache_dir}/file" ]; then
    echo "[prefetch] Already cached: ${url##*/}"
    return 0
  fi

  echo "[prefetch] Downloading: ${url##*/} ..."
  local tmpfile
  tmpfile=$(mktemp)
  if curl -fsSL --retry 3 --retry-delay 2 "$url" -o "$tmpfile" 2>/dev/null; then
    # Verify checksum if we have one
    local actual_sha256
    actual_sha256=$(sha256sum "$tmpfile" | awk "{print \$1}")
    if [ "$actual_sha256" = "$sha256" ]; then
      mkdir -p "$cache_dir"
      mv "$tmpfile" "${cache_dir}/file"
      echo "[prefetch] Cached successfully: ${url##*/}"
      return 0
    else
      echo "[prefetch] SHA256 mismatch for ${url##*/} (expected $sha256, got $actual_sha256)"
      rm -f "$tmpfile"
      return 1
    fi
  else
    echo "[prefetch] Download failed: ${url##*/} (will let Bazel retry)"
    rm -f "$tmpfile"
    return 1
  fi
}

BAZEL_REPO_CACHE="/litert_build/.cache/bazel_repo_cache"
mkdir -p "$BAZEL_REPO_CACHE"

# tqdm — only URL is a Google internal mirror that may be rate-limited.
# The workspace.bzl has no sha256 (googlesource +archive URLs are not stable),
# so we cannot use the content-addressable cache. Instead, if the fetch fails,
# we fall back to a local override built from PyPI tqdm.
echo "[prefetch] Testing access to third-party-mirror.googlesource.com ..."
TQDM_URL="https://third-party-mirror.googlesource.com/tqdm/+archive/d593e871a6b3fcc21ca5281aebda0feee0e8732e.tar.gz"
TQDM_OVERRIDE_DIR="/litert_build/.cache/tqdm_override"
USE_TQDM_OVERRIDE=""

if ! curl -fsSL --retry 1 --max-time 10 "$TQDM_URL" -o /dev/null 2>/dev/null; then
  echo "[prefetch] Google mirror blocked (429/timeout). Building tqdm from PyPI..."
  if [ ! -d "${TQDM_OVERRIDE_DIR}/tqdm" ]; then
    tmpdir=$(mktemp -d)
    # pip is available in the Docker image
    pip3 install --break-system-packages --target="$tmpdir" --no-deps tqdm 2>/dev/null \
      || pip3 install --target="$tmpdir" --no-deps tqdm 2>/dev/null \
      || { echo "[prefetch] WARNING: pip install tqdm failed"; rm -rf "$tmpdir"; }
    if [ -d "$tmpdir/tqdm" ]; then
      rm -rf "$TQDM_OVERRIDE_DIR"
      mkdir -p "$TQDM_OVERRIDE_DIR"
      cp -r "$tmpdir/tqdm" "$TQDM_OVERRIDE_DIR/tqdm"
      # Copy the BUILD file and create WORKSPACE for the override repo
      cp /litert_build/third_party/tqdm/tqdm.BUILD "$TQDM_OVERRIDE_DIR/BUILD"
      touch "$TQDM_OVERRIDE_DIR/WORKSPACE"
      echo "[prefetch] tqdm override ready from PyPI"
    fi
    rm -rf "$tmpdir"
  else
    echo "[prefetch] Using existing tqdm override"
  fi
  USE_TQDM_OVERRIDE=1
else
  echo "[prefetch] Google mirror accessible"
fi

# ---- Build mode ----
BUILD_MODE="${LITERT_BUILD_MODE:-opt}"
if [ "${BUILD_MODE}" = "opt" ]; then
  OPT_COPT="--copt=-O3"
else
  OPT_COPT=""
fi

BAZEL_FLAGS=(
  "-c" "${BUILD_MODE}"
  "--cxxopt=-std=gnu++17"
  ${OPT_COPT}
  "--repo_env=USE_PYWRAP_RULES=True"
  "--repository_cache=${BAZEL_REPO_CACHE}"
)

if [ -n "$USE_TQDM_OVERRIDE" ] && [ -d "$TQDM_OVERRIDE_DIR/tqdm" ]; then
  BAZEL_FLAGS+=("--override_repository=tqdm=${TQDM_OVERRIDE_DIR}")
fi

if [ -n "${HERMETIC_PYTHON_VERSION:-}" ]; then
  BAZEL_FLAGS+=("--action_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION}")
fi

echo "=== Building LiteRT wheel ==="
bazel ${EXTRA_STARTUP} build "${BAZEL_FLAGS[@]}" \
  //ci/tools/python/wheel:litert_wheel

echo "=== Building vendor SDK sdists ==="

# Qualcomm SDK
bazel ${EXTRA_STARTUP} build "${BAZEL_FLAGS[@]}" \
  //ci/tools/python/vendor_sdk/qualcomm:ai_edge_litert_sdk_qualcomm_sdist || \
  echo "WARNING: Qualcomm SDK build failed (non-fatal)"

# MediaTek SDK
bazel ${EXTRA_STARTUP} build "${BAZEL_FLAGS[@]}" \
  //ci/tools/python/vendor_sdk/mediatek:ai_edge_litert_sdk_mediatek_sdist || \
  echo "WARNING: MediaTek SDK build failed (non-fatal)"

# Intel OpenVINO SDK
bazel ${EXTRA_STARTUP} build "${BAZEL_FLAGS[@]}" \
  //ci/tools/python/vendor_sdk/intel:ai_edge_litert_sdk_intel_sdist || \
  echo "WARNING: Intel OpenVINO SDK build failed (non-fatal)"

# Google Tensor SDK (if present)
if [ -d "ci/tools/python/vendor_sdk/google_tensor" ]; then
  bazel ${EXTRA_STARTUP} build "${BAZEL_FLAGS[@]}" \
    //ci/tools/python/vendor_sdk/google_tensor:ai_edge_litert_sdk_google_tensor_sdist || \
    echo "WARNING: Google Tensor SDK build failed (non-fatal)"
fi

# --- Collect outputs ---
rm -rf /litert_build/dist
mkdir -p /litert_build/dist

cp bazel-bin/ci/tools/python/wheel/dist/*.whl /litert_build/dist/ 2>/dev/null || true
cp bazel-bin/ci/tools/python/vendor_sdk/qualcomm/ai_edge_litert_sdk_qualcomm*.tar.gz /litert_build/dist/ 2>/dev/null || true
cp bazel-bin/ci/tools/python/vendor_sdk/mediatek/ai_edge_litert_sdk_mediatek*.tar.gz /litert_build/dist/ 2>/dev/null || true
cp bazel-bin/ci/tools/python/vendor_sdk/intel/ai_edge_litert_sdk_intel*.tar.gz /litert_build/dist/ 2>/dev/null || true
cp bazel-bin/ci/tools/python/vendor_sdk/google_tensor/ai_edge_litert_sdk_google_tensor*.tar.gz /litert_build/dist/ 2>/dev/null || true

echo ""
echo "=== Wheel build complete ==="
echo "Output files:"
ls -lh /litert_build/dist/
'

if [ $? -ne 0 ]; then
  echo "Error: Wheel build failed inside Docker container."
  exit 1
fi

echo ""
echo "Wheel build completed successfully!"
echo ""
echo "Output files in ${REPO_ROOT}/dist/:"
ls -lh "${REPO_ROOT}/dist/" 2>/dev/null || echo "(no files found)"
echo ""
echo "To verify Intel OpenVINO plugin is bundled in the wheel:"
echo "  unzip -l dist/*.whl | grep openvino"
echo ""
echo "To install the wheel:"
echo "  pip install dist/ai_edge_litert-*.whl"
echo ""
echo "For Intel NPU support (dispatch .so is bundled in the wheel):"
echo "  pip install dist/ai_edge_litert-*.whl[npu-intel]   # also installs openvino"
echo "  # Or install openvino separately: pip install openvino==2026.1.0"
echo "  # NPU driver must be installed manually (requires sudo)"
