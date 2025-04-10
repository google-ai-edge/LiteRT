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

function run_test_on_android {
  echo "run_test_on_android"

  local -r TEST_SRC_PATH_SUFFIX=$(echo "$PWD" | sed -E "s/.*google3//")
  local -r TEST_BIN_PATH=${BIN_ROOT}/${TEST_SRC_PATH_SUFFIX}/
  local -r DATA_PATH=${LRT_ROOT}/test/testdata

  local -r DEVICE_BIN_PATH=/data/local/tmp
  local -r DEVICE_DATA_PATH=${DEVICE_BIN_PATH}/runfiles/${LRT_PATH_SUFFIX}/test/testdata

  echo TEST_SRC_PATH_SUFFIX=$TEST_SRC_PATH_SUFFIX
  echo TEST_BIN_PATH=$TEST_BIN_PATH
  echo DATA_PATH=$DATA_PATH

  echo DEVICE_BIN_PATH=$DEVICE_BIN_PATH
  echo DEVICE_DATA_PATH=$DEVICE_DATA_PATH

  TESTS=$1
  if [[ $TESTS == "all" ]]; then
    TESTS_=$(find . -maxdepth 1 -type f -name "*_test.cc" -exec basename {} \;)
    TESTS=()
    for TEST_NAME in $TESTS_; do
      TEST_NAME=$(sed "s/.cc//g" <<< $TEST_NAME)
      TESTS+="$TEST_NAME "
    done
  else
    TESTS=($1)
  fi

  adb ${ADB_OPTION} shell "mkdir -p ${DEVICE_DATA_PATH}"
  echo $DEVICE_DATA_PATH

  for TEST_NAME in $TESTS; do
    echo Running test $TEST_NAME

    echo Building $TEST_NAME
    blaze build $DBG --config=android_arm64 \
      --android_ndk_min_sdk_version=$SDK_VERSION \
      /$TEST_SRC_PATH_SUFFIX:${TEST_NAME}

    adb ${ADB_OPTION} push --sync ${TEST_BIN_PATH}/${TEST_NAME} ${DEVICE_BIN_PATH}

    RUNFILES=${TEST_BIN_PATH}/${TEST_NAME}_on_android_arm64_api_${SDK_VERSION}.runfiles
    if [[ $TEST_NAME == *"dispatch_"* ]] || [[ $TEST_NAME == *"invoke_"* ]] || [[ $TEST_NAME == *"jit_"* ]]; then
      adb ${ADB_OPTION} push --sync ${RUNFILES}/google3/_solib_arm64-v8a/*/*.so ${DEVICE_BIN_PATH}
    fi
    if [[ -d ${RUNFILES}/google3/${LRT_PATH_SUFFIX}/test/testdata/ ]]; then
      find ${RUNFILES}/google3/${LRT_PATH_SUFFIX}/test/testdata/*tflite -exec adb ${ADB_OPTION} push --sync {} ${DEVICE_DATA_PATH} \;
    fi

    if [[ $TEST_NAME == *"_google_tensor_"* ]]; then
      echo "Enabling EdgeTPU access"
      adb ${ADB_OPTION} root
      adb ${ADB_OPTION} shell setprop vendor.edgetpu.service.allow_unlisted_app true
      echo "Push Google Tensor files to the device."
      find ${DATA_PATH}/*_google_tensor.bin -exec adb ${ADB_OPTION} push --sync {} ${DEVICE_DATA_PATH} \;
    fi
    if [[ $TEST_NAME == *"qualcomm_"* || $TEST_NAME == *"qnn_"* ]]; then
      echo "Push QNN files to the device."
      find ${DATA_PATH}/*_qualcomm.bin -exec adb ${ADB_OPTION} push --sync {} ${DEVICE_DATA_PATH} \;
      adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp*Stub.so ${DEVICE_BIN_PATH}
      adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so ${DEVICE_BIN_PATH}
      adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so ${DEVICE_BIN_PATH}
      adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so ${DEVICE_BIN_PATH}
      adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/hexagon-*/unsigned/libQnnHtp*Skel.so ${DEVICE_BIN_PATH}
      ENV_VARS="ADSP_LIBRARY_PATH=. LD_LIBRARY_PATH=."
    else
      ENV_VARS="LD_LIBRARY_PATH=."
    fi

    if [[ $TEST_NAME == *"_mediatek_"* ]]; then
      echo "Push MediaTek files to the device."
      find ${DATA_PATH}/*_mtk.bin -exec adb ${ADB_OPTION} -d push --sync {} ${DEVICE_DATA_PATH} \;
    fi

    local CMD="cd ${DEVICE_BIN_PATH}; ${ENV_VARS} ./${TEST_NAME}"

    # Pass along options. Push and normalize workstation model paths to device.
    shift
    ARGS=""
    for ARG in "$@" ; do
      MODEL_ARG=$(echo $ARG | sed "s/--(.*model)=(.*)/\1/" -E)
      MODEL=$(echo $ARG | sed "s/--(.*model)=(.*)/\2/" -E)
      if [[ -z $MODEL ]] ; then
        CMD+=" $ARG"
      elif [[ -f $MODEL ]] ; then
        adb ${ADB_OPTION} push --sync $MODEL ${DEVICE_BIN_PATH}
        CMD+=" --$MODEL_ARG=$DEVICE_BIN_PATH/$(basename $MODEL)"
      else
        echo "Invalid model: $MODEL"
        exit -1
      fi
    done

    echo "(adb)" $CMD
    adb ${ADB_OPTION} shell $CMD

  done

}

function run_test_on_android_main {
  local SCRIPT_NAME=$([[ -z $CMD ]] \
    && echo ${BASH_SOURCE[0]} \
    || echo "$(basename $0) $CMD")

  if (($# < 1)); then
    echo "Usage $SCRIPT_NAME <test name> [OPTION]..."
    exit
  fi
  script_init
  run_test_on_android "$@"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  run_test_on_android_main $@
fi
