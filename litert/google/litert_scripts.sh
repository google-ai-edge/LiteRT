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

source $(dirname $0)/apply_plugin.sh || exit -1
source $(dirname $0)/script_init.sh || exit -1
source $(dirname $0)/run_test_on_android.sh || exit -1

function verify {
  echo "verify"
  run_test_on_android qualcomm_dispatcher_numeric_test --cpu_model=$1 --npu_model=$2
}

function invoke {
  echo "invoke"
  run_test_on_android invoke_qualcomm_test --model=$1
}

function invoke_main {
  if [[ $# < 1 ]] ; then
    echo "Usage `basename $0` $USR_CMD <input model> [cpu model]"
    exit
  fi
  script_init
  if (($# == 1 )); then
    invoke $1
  else
    verify $2 $1
  fi
}

function pipeline_main {
  if (($# < 2)); then
    echo "Usage `basename $0` $USR_CMD <input model> <output model> [skip_numerics]"
    exit
  fi
  script_init
  apply_plugin apply $@
  if (($4 == "skip_numerics")); then
    invoke $OUTPUT_MODEL
  else
    verify $3 $OUTPUT_MODEL
  fi
}

readonly USR_CMD=$1
readonly USER_SOC_MAN=$2
readonly USER_SOC_MODEL=$3
shift

case "$USR_CMD" in
  "compile" )
    script_init
    apply_plugin compile $@
    exit
    ;;
  "apply" )
    apply_plugin_main $@
    exit
    ;;
  "invoke" )
    invoke_main $@
    exit
    ;;
  "pipeline" )
    pipeline_main $@
    exit
    ;;
  *)
    echo "Usage `basename $0` <apply|invoke|pipeline|compile>"
    exit -1
    ;;
esac