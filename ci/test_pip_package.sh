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
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
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

  PYENV_ROOT="$(pwd)/pyenv"
  if ! git clone https://github.com/pyenv/pyenv.git 2>/dev/null && [ -d "${PYENV_ROOT}" ] ; then
      echo "${PYENV_ROOT} exists"
  fi

  export PATH="$PYENV_ROOT/bin:$PATH"

  eval "$(pyenv init -)"
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
  local qc_dist_pkg="$(ls ./dist/ai_edge_litert_sdk_qualcomm*.tar.gz)"
  SKIP_SDK_DOWNLOAD="true" ${PYTHON_BIN} -m pip install ${qc_dist_pkg?} --ignore-installed

  local mtk_dist_pkg="$(ls ./dist/ai_edge_litert_sdk_mediatek*.tar.gz)"
  SKIP_SDK_DOWNLOAD="true" ${PYTHON_BIN} -m pip install ${mtk_dist_pkg?} --ignore-installed

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

  echo
}

function test_import {
  # Test whether import is successful.
  echo "------ Test import -----"
  ${PYTHON_BIN} -c "import ai_edge_litert"
  ${PYTHON_BIN} -c "import ai_edge_litert_sdk_qualcomm"
  ${PYTHON_BIN} -c "import ai_edge_litert_sdk_mediatek"
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

