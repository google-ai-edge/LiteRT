#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# The data points of the Gemma 4 record, each one bench.sh run of the LiteRT-LM CLI in
# benchmark mode. One run directory per invocation of this script; report.py writes its table.
#
#   usage: matrix.sh OUT_DIR [MODEL...]        (default models: e2b e4b 12b 26b)
#   environment: env.sh and optional env.local.sh; BENCH_GROUPS selects the groups
#                (default "context mtp init verify")
#
#   context  plain decoding on synthetic prefill tokens at 2k/8k/16k/32k (12B also 128k):
#            latency (decode steps/s, prefill tok/s) and memory (GPU peak, CPU RSS)
#   mtp      plain and speculative decoding on the real prompts of those lengths (acceptance
#            needs text): decode tok/s, acceptance, tokens per round
#   init     a cold start on an empty cache (compile phase, engine builds, init) and the
#            warm init of the latency run
#   verify   the Paris answer on the NPU and on the CPU
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
source "$here/env.sh"
nb_resolve_paths
BENCH=$here
out=${1:?usage: matrix.sh OUT_DIR [MODEL...]}; shift || true
models=("$@"); ((${#models[@]})) || models=(e2b e4b 12b 26b)
groups=${BENCH_GROUPS:-context mtp init verify}
mkdir -p "$out"
"$BENCH/build.sh"
bench() { "$BENCH/bench.sh" --out "$out" "$@"; }
prompt_of() { case "$1" in short) echo prompt_1024 ;; 8k) echo prompt_8k ;; 16k) echo prompt_16k ;; 30k) echo prompt_30k ;; 128k) echo prompt_128k ;; esac; }
for model in "${models[@]}"; do
  workloads=(short 8k 16k 30k); [[ $model != 12b ]] || workloads=(short 8k 16k 30k 128k)
  # The model file into the page cache before its group: the CLI's embedder reads it per token,
  # and a file another model's group evicted costs the 26B a third of its decode speed.
  cat "$(nb_model_file "$model")" > /dev/null
  for group in $groups; do
    case "$group" in
      verify) bench --model "$model" --metric verify ;;
      init) bench --model "$model" --metric cold ;;
      context)
        for wl in "${workloads[@]}"; do
          bench --model "$model" --workload "$wl" --metric latency
          bench --model "$model" --workload "$wl" --metric memory
        done ;;
      mtp)
        for wl in "${workloads[@]}"; do
          prompt=$BENCH/prompts/$(prompt_of "$wl").txt
          # Three iterations: the first of a process is slow, and acceptance varies with the text.
          bench --model "$model" --workload "$wl" --metric latency --prompt-file "$prompt" --iterations 3 --label "${model}_${wl}_plain_text"
          bench --model "$model" --workload "$wl" --metric latency --prompt-file "$prompt" --iterations 3 --mtp --label "${model}_${wl}_mtp_text"
        done ;;
    esac
  done
done
python3 "$BENCH/report.py" "$out"
