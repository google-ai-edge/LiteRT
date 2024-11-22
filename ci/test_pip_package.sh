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

echo "Testing on Python version ${PYTHON_VERSION}"

function create_venv {
  PYENV_ROOT="$(pwd)/pyenv"
  if ! git clone https://github.com/pyenv/pyenv.git 2>/dev/null && [ -d "${PYENV_ROOT}" ] ; then
      echo "${PYENV_ROOT} exists"
  fi

  export PATH="$PYENV_ROOT/bin:$PATH"

  eval "$(pyenv init -)"
  pyenv install -s "${PYTHON_VERSION}"
  pyenv global "${PYTHON_VERSION}"

  PYTHON_BIN=$(which python)
  export PYTHON_BIN

  ${PYTHON_BIN} -m pip install virtualenv
  ${PYTHON_BIN} -m virtualenv ai_edge_litert_env
  source ai_edge_litert_env/bin/activate
}

function build_pip_and_install {
  # Build and install pip package.
  if [[ "${PYTHON_BIN}" == "" ]]; then
    echo "python is not available."
    exit 1
  fi

  ${PYTHON_BIN} -m pip install --upgrade pip
  ${PYTHON_BIN} -m pip install build wheel

  echo "------ build pip and install -----"

  # Clean up distributions.
  rm -r -f tflite/gen/litert_pip/python3/dist

  if [[ -n "${USE_DOCKER_BUILD}" ]]; then
    ./ci/build_pip_package_with_docker.sh
  else
    ./ci/build_pip_package_with_bazel.sh
  fi

  local dist_pkg="$(ls ./tflite/gen/litert_pip/python3/dist/${pkg}*.whl)"
  ${PYTHON_BIN} -m pip install ${dist_pkg?} --ignore-installed

  echo
}

function uninstall_pip {
  # Uninstall pip package.
  echo "------ uninstall pip -----"

  local pip_pkg="ai-edge-litert"

  yes | ${PYTHON_BIN} -m pip uninstall ${pip_pkg}
  echo
}

function test_import {
  # Test whether import is successful.
  echo "------ Test import -----"
  ${PYTHON_BIN} -c "import ai_edge_litert"
  echo
}

function test_ai_edge_litert {
  echo "===== Test AI Edge Litert ====="

  create_venv
  build_pip_and_install
  test_import
  uninstall_pip
  echo
}

test_ai_edge_litert
deactivate  # deactivate virtualenv
