#!/bin/bash

MODEL_PATH=$1

# gather context info
sh get_ctx_info.sh $MODEL_PATH

rm -rf /tmp/inputs
# generate input
blaze run -c opt //litert/vendors/qualcomm/google/qnn_profiler:generate_context_binary_inputs -- --json_file /tmp/qnn_context_info.json --save_path /tmp --input_source zero

# run profiling
sh run.sh /tmp/inputs $MODEL_PATH
