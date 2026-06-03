#!/usr/bin/env bash
# Copyright 2026 Google LLC.
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${TMPDIR:-/tmp}/litert-nightly-simple-accel-venv}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/litert/test/testdata/mobilenet_v2_1.0_224.tflite}"
ITERATIONS="${ITERATIONS:-1}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-1}"
CPU_NUM_THREADS="${CPU_NUM_THREADS:-1}"

if [[ "${RECREATE_VENV:-false}" == "true" ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
PIP_ARGS=(--disable-pip-version-check --no-cache-dir)

"${VENV_PYTHON}" -m pip "${PIP_ARGS[@]}" install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip "${PIP_ARGS[@]}" install --upgrade --pre ai-edge-litert-nightly

"${VENV_PYTHON}" - <<'PY'
import importlib.metadata

print(
    "Installed ai-edge-litert-nightly",
    importlib.metadata.version("ai-edge-litert-nightly"),
)
PY

LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/litert-nightly-simple-accel.XXXXXX.log")"
trap 'rm -f "${LOG_FILE}"' EXIT

set +e
"${VENV_PYTHON}" "${SCRIPT_DIR}/simple_accel_smoke_test.py" \
  --model_path "${MODEL_PATH}" \
  --iterations "${ITERATIONS}" \
  --warmup_iterations "${WARMUP_ITERATIONS}" \
  --cpu_num_threads "${CPU_NUM_THREADS}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
SMOKE_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "${SMOKE_STATUS}" -ne 0 ]]; then
  exit "${SMOKE_STATUS}"
fi

grep -Fq "CPU: OK" "${LOG_FILE}"
grep -Fq "GPU: OK" "${LOG_FILE}"
grep -Fq "GPU|CPU: OK" "${LOG_FILE}"

