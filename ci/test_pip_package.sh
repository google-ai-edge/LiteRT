#!/usr/bin/env bash
# Copyright 2024 The AI Edge LiteRT Authors.
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
set -ex

# Test AI Edge LiteRT's pip package.
# Run this script under the root directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-true}"

echo "Testing on Python version ${PYTHON_VERSION}"


function create_venv {

  if [[ "${INSTALL_APT_DEPS}" == true ]]; then
    # Install libssl-dev to use pyenv.
    # https://github.com/pyenv/pyenv/wiki/Common-build-problems#0-first-check
    sudo apt-get update -y
    sudo apt install libssl-dev build-essential libbz2-dev libncurses5-dev \
      libncursesw5-dev libffi-dev libreadline-dev libsqlite3-dev liblzma-dev zlib1g-dev -y
  fi

  if ! command -v pyenv &> /dev/null; then
    PYENV_ROOT="$(pwd)/pyenv"
    if ! git clone https://github.com/pyenv/pyenv.git 2>/dev/null && [ -d "${PYENV_ROOT}" ] ; then
        echo "${PYENV_ROOT} exists"
    fi
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
  fi

  pyenv install -s "${PYTHON_VERSION}"
  pyenv global "${PYTHON_VERSION}"

  PYTHON_BIN=$(pyenv which python)
  echo "PYTHON_BIN: ${PYTHON_BIN}"
  export PYTHON_BIN

  ${PYTHON_BIN} -m pip install virtualenv
  ${PYTHON_BIN} -m virtualenv ai_edge_litert_env
  source ai_edge_litert_env/bin/activate
}

function initialize_pip_wheel_environment {
  # Build and install pip package.
  if [[ "${PYTHON_BIN}" == "" ]]; then
    echo "python is not available."
    exit 1
  fi

  ${PYTHON_BIN} -m pip install --upgrade pip
  ${PYTHON_BIN} -m pip install build wheel

  echo "------ build pip and install -----"

  # Clean up distributions.
  rm -r -f ./dist

}

function configure_litert_environment {
  configs=(
    '/usr/bin/python3'
    '/usr/lib/python3/dist-packages'
    'N'
    'N'
    'Y'
    '/usr/lib/llvm-18/bin/clang'
    '-Wno-sign-compare -Wno-c++20-designator -Wno-gnu-inline-cpp-without-extern'
    'N'
  )
  printf '%s\n' "${configs[@]}" | ./configure
}

function install_sdk {
  # SKIP_SDK_DOWNLOAD skips the (large) vendor SDKs; the LiteRT compiler plugin
  # and dispatch libraries are still fetched. Locally built SDKs carry a version
  # that may not be on GCS yet, so default to the latest nightly archive.
  local prebuilts_url="${LITERT_PREBUILTS_URL:-https://storage.googleapis.com/litert/binaries/nightly/litert_prebuilts.zip}"

  local qc_dist_pkg="$(ls ./dist/ai_edge_litert_sdk_qualcomm*.tar.gz)"
  SKIP_SDK_DOWNLOAD="true" LITERT_PREBUILTS_URL="${prebuilts_url}" \
    ${PYTHON_BIN} -m pip install ${qc_dist_pkg?} --ignore-installed

  local mtk_dist_pkg="$(ls ./dist/ai_edge_litert_sdk_mediatek*.tar.gz)"
  SKIP_SDK_DOWNLOAD="true" LITERT_PREBUILTS_URL="${prebuilts_url}" \
    ${PYTHON_BIN} -m pip install ${mtk_dist_pkg?} --ignore-installed

  local gt_dist_pkg="$(ls ./dist/ai_edge_litert_sdk_google_tensor*.tar.gz)"
  SKIP_SDK_DOWNLOAD="true" LITERT_PREBUILTS_URL="${prebuilts_url}" \
    ${PYTHON_BIN} -m pip install ${gt_dist_pkg?} --ignore-installed

  local intel_dist_pkg="$(ls ./dist/ai_edge_litert_sdk_intel*.tar.gz)"
  LITERT_PREBUILTS_URL="${prebuilts_url}" \
    ${PYTHON_BIN} -m pip install --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly ${intel_dist_pkg?} --ignore-installed

  echo
}

function install_wheel {
  local dist_pkg="$(ls ./dist/${pkg}*.whl)"
  ${PYTHON_BIN} -m pip install ${dist_pkg?} --ignore-installed

  echo
}

function uninstall_pip {
  # Uninstall pip package.
  echo "------ uninstall pip -----"

  local pip_pkg="ai-edge-litert"

  yes | ${PYTHON_BIN} -m pip uninstall ${pip_pkg}

  local qnn_pip_pkg="ai_edge_litert_sdk_qualcomm"

  yes | ${PYTHON_BIN} -m pip uninstall ${qnn_pip_pkg}

  local mtk_pip_pkg="ai_edge_litert_sdk_mediatek"

  yes | ${PYTHON_BIN} -m pip uninstall ${mtk_pip_pkg}

  local google_tensor_pip_pkg="ai_edge_litert_sdk_google_tensor"

  yes | ${PYTHON_BIN} -m pip uninstall ${google_tensor_pip_pkg}

  local intel_pip_pkg="ai_edge_litert_sdk_intel"

  yes | ${PYTHON_BIN} -m pip uninstall ${intel_pip_pkg}

  echo
}

function test_import {
  # Test whether import is successful.
  echo "------ Test import -----"
  ${PYTHON_BIN} -c "import ai_edge_litert"
  ${PYTHON_BIN} -c "import ai_edge_litert.environment"
  ${PYTHON_BIN} -c "import ai_edge_litert_sdk_qualcomm"
  ${PYTHON_BIN} -c "import ai_edge_litert_sdk_mediatek"
  ${PYTHON_BIN} -c "import ai_edge_litert_sdk_google_tensor"
  ${PYTHON_BIN} -c "import ai_edge_litert_sdk_intel"

  # Intel OpenVINO backend imports (no Intel OpenVINO support on macOS or
  # Linux aarch64).
  if [[ "$(uname -s)" != "Darwin" && "$(uname -m)" != "aarch64" ]]; then
    ${PYTHON_BIN} -c "from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend; print('Intel OpenVINO backend ID:', intel_openvino_backend.IntelOpenVinoBackend.id())"
    ${PYTHON_BIN} -c "
import os
from ai_edge_litert.aot.vendors.intel_openvino import intel_openvino_backend
d = intel_openvino_backend.get_dispatch_dir()
assert d and os.path.isdir(d), f'Dispatch dir not found: {d}'
print('Dispatch dir:', d)
"
  fi

  # Vendor compiler plugins / dispatch libraries installed by the SDK sdists
  # (no longer in the ai-edge-litert wheel), and found by runtime discovery.
  if [[ "$(uname -s)" == "Linux" ]]; then
    ${PYTHON_BIN} -c "
import importlib, os, pathlib, platform
from ai_edge_litert import environment
from ai_edge_litert.aot.core import common
expected = {
    'x86_64': [
        ('qualcomm', 'compiler/libLiteRtCompilerPlugin_Qualcomm.so'),
        ('mediatek', 'compiler/libLiteRtCompilerPlugin_MediaTek.so'),
        ('google_tensor', 'compiler/libLiteRtCompilerPlugin_google_tensor.so'),
        ('intel', 'compiler/libLiteRtCompilerPlugin_IntelOpenvino.so'),
        ('intel', 'dispatch/libLiteRtDispatch_IntelOpenvino.so'),
    ],
    'aarch64': [
        ('qualcomm', 'compiler/libLiteRtCompilerPlugin_Qualcomm.so'),
        ('qualcomm', 'dispatch/libLiteRtDispatch_Qualcomm.so'),
    ],
}[platform.machine()]
vendor_dirs = {'intel': 'intel_openvino'}
for sdk, rel in expected:
  pkg = importlib.import_module('ai_edge_litert_sdk_' + sdk)
  lib = os.path.join(os.path.dirname(pkg.__file__), 'vendors',
                     vendor_dirs.get(sdk, sdk), rel)
  assert os.path.isfile(lib), f'Missing vendor library: {lib}'
  found = common.get_resource(
      pathlib.Path('vendors', vendor_dirs.get(sdk, sdk), rel))
  assert os.path.samefile(found, lib), f'get_resource found {found}, not {lib}'
  print('OK:', lib)
dispatch = environment._autodiscover_dispatch_library_path()
assert 'ai_edge_litert_sdk_' in dispatch, f'Bad dispatch dir: {dispatch!r}'
print('Autodiscovered dispatch dir:', dispatch)
"
  fi

  # apply_plugin_main must keep its execute bit in every Linux wheel.
  if [[ "$(uname -s)" == "Linux" ]]; then
    ${PYTHON_BIN} -c "
import os
import ai_edge_litert
tool = os.path.join(os.path.dirname(ai_edge_litert.__file__), 'tools/apply_plugin_main')
assert os.access(tool, os.X_OK), f'apply_plugin_main is not executable: {tool}'
"
  fi

  if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]]; then
    ${PYTHON_BIN} -c "
from ai_edge_litert.aot.vendors.qualcomm import target
assert target.SocModel.QCS8275
"
  fi
  echo
}

function test_ai_edge_litert {
  echo "===== Test AI Edge Litert ====="

  create_venv
  configure_litert_environment
  initialize_pip_wheel_environment
  ./ci/build_pip_package_with_bazel.sh
  install_sdk
  install_wheel
  test_import
  uninstall_pip
  deactivate  # deactivate virtualenv
  echo
}

function test_ai_edge_litert_with_docker {
  echo "===== Test AI Edge Litert with Docker ====="
  export TEST_WHEEL=true
  ./ci/build_pip_package_with_docker.sh
  echo
}
