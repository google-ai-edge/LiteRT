#!/bin/bash
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

# Runs the minimal_compiled_model sample on a test model.
# Usage: minimal_compiled_model_test.sh <binary> <model.tflite>

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <binary> <model.tflite>"
  exit 1
fi

"$1" "$2"
