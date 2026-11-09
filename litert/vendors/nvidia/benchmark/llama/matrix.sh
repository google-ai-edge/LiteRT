#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# The llama.cpp data points of the record: the counterparts of bench/matrix.sh's, each one
# llama/bench.sh run. One run directory per invocation of this script.
#
#   usage: matrix.sh OUT_DIR [MODEL...]        (default models: e2b e4b 12b 26b)
#   environment: llama/env.sh; LLAMA_GROUPS="context mtp init" selects the groups
#   LLAMA_REPETITIONS=N overrides measured latency repetitions; text adds one warm-up
#
#   context  llama-bench at 1k/8k/16k/30k (12B also 128k): prefill of the whole prompt, decode
#            at that depth (latency), and one prefill+decode pass under GPU sampling (memory)
#   mtp      llama-server completions of the real prompts of those lengths, plain and with the
#            MTP drafter: decode tok/s and draft acceptance
#   init     the server's start to ready, cold (model files evicted) and warm
set -euo pipefail
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/env.sh"
nb_resolve_paths  # $BENCH: LiteRT's benchmark/, the prompt files
out=${1:?usage: matrix.sh OUT_DIR [MODEL...]}; shift || true
models=("$@"); ((${#models[@]})) || models=(e2b e4b 12b 26b)
groups=${LLAMA_GROUPS:-context mtp init}
latency_iterations=()
text_iterations=3
if [[ -n ${LLAMA_REPETITIONS:-} ]]; then
  [[ $LLAMA_REPETITIONS =~ ^[1-9][0-9]*$ ]] || lb_die "LLAMA_REPETITIONS must be a positive integer"
  latency_iterations=(--iterations "$LLAMA_REPETITIONS")
  text_iterations=$((LLAMA_REPETITIONS + 1))
fi
mkdir -p "$out"
bench() { "$LLAMA_DIR/bench.sh" --out "$out" "$@" || echo "FAILED: $*"; }
prompt_of() { case "$1" in short) echo prompt_1024 ;; 8k) echo prompt_8k ;; 16k) echo prompt_16k ;; 30k) echo prompt_30k ;; 128k) echo prompt_128k ;; esac; }
for model in "${models[@]}"; do
  workloads=(short 8k 16k 30k); [[ $model != 12b ]] || workloads=(short 8k 16k 30k 128k)
  # The model files into the page cache first, as bench/matrix.sh does for the LiteRT files.
  cat "$(lb_model_file "$model")" > /dev/null
  for group in $groups; do
    case "$group" in
      init) bench --model "$model" --metric cold ;;
      context)
        for wl in "${workloads[@]}"; do
          bench --model "$model" --workload "$wl" --metric latency "${latency_iterations[@]}"
          bench --model "$model" --workload "$wl" --metric memory
        done ;;
      mtp)
        cat "$(lb_draft_file "$model")" > /dev/null
        for wl in "${workloads[@]}"; do
          prompt=$BENCH/prompts/$(prompt_of "$wl").txt
          # Exclude the first completion of each fresh server from measured samples.
          bench --model "$model" --workload "$wl" --metric latency --prompt-file "$prompt" --iterations "$text_iterations" --label "${model}_${wl}_plain_text"
          bench --model "$model" --workload "$wl" --metric latency --prompt-file "$prompt" --iterations "$text_iterations" --mtp --label "${model}_${wl}_mtp_text"
        done ;;
    esac
  done
done
python3 "$LLAMA_DIR/report.tool" "$out"
