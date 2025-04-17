#!/bin/bash
source gbash.sh || exit

adb_cmd="adb"

WD_folder=/data/local/tmp/$USER/litert
QNN_HOME=$(gbash::get_google3_dir)/third_party/qairt/latest/

INPUT_DIR=$1
MODEL_PATH=$2
# get base name of model
MODEL_NAME=$(basename $MODEL_PATH)


$adb_cmd shell "rm -rf $WD_folder && mkdir -p $WD_folder"

$adb_cmd push $QNN_HOME/bin/aarch64-android/qnn-net-run $WD_folder
$adb_cmd push $QNN_HOME/lib/aarch64-android/libQnnHtp.so $WD_folder
$adb_cmd push $QNN_HOME/lib/aarch64-android/libQnnHtpNetRunExtensions.so $WD_folder
$adb_cmd push $QNN_HOME/lib/aarch64-android/libQnnHtpPrepare.so $WD_folder
$adb_cmd push $QNN_HOME/lib/aarch64-android/libQnnHtpV68Stub.so $WD_folder
$adb_cmd push $QNN_HOME/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so $WD_folder
$adb_cmd push $QNN_HOME/lib/aarch64-android/libQnnHtpV75Stub.so $WD_folder
$adb_cmd push $QNN_HOME/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so $WD_folder
$adb_cmd push $QNN_HOME/lib/aarch64-android/libQnnHtpV79Stub.so $WD_folder
$adb_cmd push $QNN_HOME/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so $WD_folder



$adb_cmd push $MODEL_PATH $WD_folder
$adb_cmd push  $INPUT_DIR $WD_folder
$adb_cmd push "config.json" $WD_folder
$adb_cmd push "htp_backend_ext_config.json" $WD_folder

$adb_cmd shell "export LD_LIBRARY_PATH=$WD_folder && \
                export ADSP_LIBRARY_PATH="$WD_folder" &&\
                cd $WD_folder && \
                ./qnn-net-run \
                --shared_buffer \
                --backend libQnnHtp.so \
                --retrieve_context $MODEL_NAME \
                --input_list inputs/graph_0/input_list.txt \
                --output_dir outputs \
                --use_native_input_files \
                --use_native_output_files \
                --profiling_level basic \
                --config_file config.json"


OUTPUT_MODEL_PATH=/tmp/qnn_profiler_outputs
rm -rf $OUTPUT_MODEL_PATH && mkdir -p $OUTPUT_MODEL_PATH
$adb_cmd pull $WD_folder/outputs $OUTPUT_MODEL_PATH

$QNN_HOME/bin/x86_64-linux-clang/qnn-profile-viewer --input_log $OUTPUT_MODEL_PATH/outputs/qnn-profiling-data_0.log --output /tmp/gemma3_v79_qnn_profile.csv