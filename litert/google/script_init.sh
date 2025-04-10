#!/bin/sh

# Copyright 2024 Google LLC.
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

function script_init {
  echo "script_init"

  set -e
  set -o pipefail

  readonly SDK_VERSION=28
  #ADB_OPTION="-d"

  readonly DBG="-c opt"
  #DBG="-c dbg --dynamic_mode=off"
  #DBG="-c dbg --dynamic_mode=off --features=asan"
  #DBG="-c dbg --dynamic_mode=off --features=hwasan"

  local -r SCRIPT_PATH=$(readlink -f "$0")
  readonly SCRIPT_BASE_DIR=$(dirname "$SCRIPT_PATH")

  readonly LRT_ROOT=$(echo "$SCRIPT_BASE_DIR" | sed -E "s/\/litert\/google$/\/litert/")
  readonly LRT_PATH_SUFFIX=$(echo "$LRT_ROOT" | sed -E "s/.*google3\///")

  local -r LRT_PACKAGE_ROOT="//$LRT_PATH_SUFFIX"
  readonly LRT_TOOLS_PACKAGE=$LRT_PACKAGE_ROOT/tools:
  readonly LRT_TEST_PACKAGE=$LRT_PACKAGE_ROOT/test:

  local -r BLAZE_ROOT=${LRT_ROOT}/../../../..
  readonly BIN_ROOT=${BLAZE_ROOT}/blaze-bin
  readonly GENFILES_ROOT=${BLAZE_ROOT}/blaze-genfiles
  readonly LRT_BIN_ROOT=$BIN_ROOT/$LRT_PATH_SUFFIX
  readonly LRT_GENFILES_ROOT=$GENFILES_ROOT/$LRT_PATH_SUFFIX

  readonly QNN_SDK_ROOT=${LRT_ROOT}/../../../../third_party/qairt/latest
  readonly MEDIATEK_SDK_ROOT=${LRT_ROOT}/../../../../third_party/neuro_pilot/neuron_sdk
  readonly GOOGLE_TENSOR_SDK_ROOT=${BIN_ROOT}/platforms/darwinn/compiler

  echo SCRIPT_BASE_DIR=$SCRIPT_BASE_DIR

  echo LRT_ROOT=$LRT_ROOT
  echo LRT_PATH_SUFFIX=$LRT_PATH_SUFFIX

  echo LRT_TOOLS_PACKAGE=$LRT_TOOLS_PACKAGE
  echo LRT_TEST_PACKAGE=$LRT_TEST_PACKAGE

  echo BIN_ROOT=$BIN_ROOT
  echo GENFILES_ROOT=$GENFILES_ROOT
  echo LRT_BIN_ROOT=$LRT_BIN_ROOT
  echo LRT_GENFILES_ROOT=$LRT_GENFILES_ROOT

  echo QNN_SDK_ROOT=$QNN_SDK_ROOT
  echo MEDIATEK_SDK_ROOT=$MEDIATEK_SDK_ROOT
  echo GOOGLE_TENSOR_SDK_ROOT=$GOOGLE_TENSOR_SDK_ROOT

  if [ ! -d $LRT_ROOT ] ; then
    echo "Invalid LRT_ROOT: $LRT_ROOT"
    exit -1
  fi

  if [ ! -d $QNN_SDK_ROOT ] ; then
    echo "Invalid QNN_SDK_ROOT: $QNN_SDK_ROOT"
    exit -1
  fi

}
