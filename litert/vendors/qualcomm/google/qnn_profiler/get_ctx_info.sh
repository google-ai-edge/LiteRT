#!/bin/bash
source gbash.sh || exit
QNN_HOME=$(gbash::get_google3_dir)/third_party/qairt/latest/

MODEL_PATH=$1

rm -rf /tmp/qnn_context_info.json
$QNN_HOME/bin/x86_64-linux-clang/qnn-context-binary-utility \
--context_binary $MODEL_PATH \
--json_file /tmp/qnn_context_info.json

echo "Get context info from $MODEL_PATH, saved to /tmp/qnn_context_info.json"
