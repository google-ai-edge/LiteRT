#!/bin/bash
# Copyright 2024 Google LLC
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
# Note: this script is primarily used within the Kokoro CI system.

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

function ensure_uv {
  if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "${HOME}/.local/bin/env"
  else
    echo "uv is already installed."
  fi
}

ensure_uv

PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13")

for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
  echo "----------------------------------------------------------------"
  echo "Testing on Python version ${PYTHON_VERSION}"
  echo "----------------------------------------------------------------"

  # Install Python version
  uv python install "${PYTHON_VERSION}"
  uv run --python "${PYTHON_VERSION}" pytest
done

echo "All tests passed!"
