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

source $(dirname $0)/script_init.sh || exit -1

# $1: Option to apply (e.g. apply, partition, compile, info, noop).
# $2: SoC manufacturer (e.g. Qualcomm,GoogleTensor).
# $3: SOC Model (e.g. V75).
# $4: Path to input model.
# $5: Path to output model.
# $6: Compiler flags. written as a key=value string comma separated. eg: "enable_int64_to_int32_truncation=true,output_dir=/tmp/output"
# Output: $OUTPUT_MODEL

function apply_plugin {
echo "apply_plugin"

local -r LRT_TESTDATA_GENFILES=${LRT_GENFILES_ROOT}/test/testdata
local -r APPLY_PLUGIN_TARGET=${LRT_TOOLS_PACKAGE}apply_plugin_main

echo LRT_TESTDATA_GENFILES=$LRT_TESTDATA_GENFILES
echo APPLY_PLUGIN_TARGET=$APPLY_PLUGIN_TARGET


local OPTION=$1
local SOC_MAN=$2
local SOC_MODEL=$3
local INPUT_MODEL=$4
readonly OUTPUT_MODEL=$5
local COMPILER_FLAGS=$6

echo OPTION=$OPTION
echo SOC_MAN=$SOC_MAN
echo SOC_MODEL=$SOC_MODEL
echo INPUT_MODEL=$INPUT_MODEL
echo OUTPUT_MODEL=$OUTPUT_MODEL
echo COMPILER_FLAGS=$COMPILER_FLAGS

if [[ ! -d $(dirname $OUTPUT_MODEL ) ]] ; then
  mkdir -p $(dirname $OUTPUT_MODEL)
fi

if [[ $(basename $INPUT_MODEL) == $INPUT_MODEL ]] ; then
  local INPUT_MODEL_TARGET="$LRT_TEST_PACKAGE"testdata/${INPUT_MODEL%%.*}.tflite
  echo "Using testdata model"
  blaze build -c opt $INPUT_MODEL_TARGET
  INPUT_MODEL="${LRT_TESTDATA_GENFILES}/$INPUT_MODEL"
elif [[ ! -f $INPUT_MODEL ]] ; then
  echo "Input model does not exist at $INPUT_MODEL"
  exit -1
fi


blaze run -c opt $APPLY_PLUGIN_TARGET -- \
  $OPTION \
  --model=$INPUT_MODEL \
  --o=$OUTPUT_MODEL \
  --soc_man=$SOC_MAN \
  --soc_model=$SOC_MODEL \
  --compiler-flags=$COMPILER_FLAGS

if [[ ! -f $OUTPUT_MODEL ]] ; then
  echo "No output file found"
  exit -1
fi

}

function apply_plugin_main {
  local SCRIPT_NAME=$([[ -z $CMD ]] \
    && echo ${BASH_SOURCE[0]} \
    || echo "$(basename $0) $CMD")

  if (($# < 2)); then
    echo "Usage: $SCRIPT_NAME <apply|partition|compile|info|noop> <SOC manufacturer> <SOC model> <input model> <output model> <compiler flags>"
    exit
  fi
  script_init
  apply_plugin apply $@
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  apply_plugin_main $@
fi
