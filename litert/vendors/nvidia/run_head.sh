#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# NVIDIA (TensorRT-RTX) benchmarks of the Gemma 4 models with the unchanged LiteRT-LM advanced
# CLI: a thin driver over benchmark/{build.sh,bench.sh,report.py}.
#   build.sh   builds the compiler plugin, the dispatch library and the CLI
#   bench.sh   one CLI run in benchmark mode = one measurement (a run directory)
#   report.py  tables from run directories; nothing is measured outside the CLI
#
# usage:
#   run_head.sh report [--profile e2b|e4b|12b|26b | --model-file PATH] [--workload short|8k|16k|30k|32k|128k]
#                      [--mtp] [--prompt-file FILE] [--metrics latency,memory,verify,cold|all]
#                      [--iterations N] [--out DIR] [--no-build] [--cli-arg ARG]...
#   run_head.sh build
#   run_head.sh table RUN_DIR...
# Defaults: e4b, short, plain decoding, metrics latency,memory,verify, a fresh run directory
# under $LITERT_BENCH_ROOT/runs. Environment: see benchmark/env.sh.
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
source "$here/benchmark/env.sh"
nb_resolve_paths
action=${1:-help}; shift || true
case "$action" in
  build) exec "$here/benchmark/build.sh" "$@" ;;
  table) exec python3 "$here/benchmark/report.py" "$@" ;;
  report) ;;
  *) sed -n '3,17p' "$0"; exit 0 ;;
esac
model=e4b workload=short mtp= prompt= metrics=latency,memory,verify iterations= out= build=1 extra=()
while (($#)); do
  case "$1" in
    --profile|--model-file) model=$2; shift 2 ;;
    --workload) workload=$2; shift 2 ;;
    --mtp) mtp=--mtp; shift ;;
    --prompt-file) prompt=$2; shift 2 ;;
    --metrics) metrics=$2; shift 2 ;;
    --iterations) iterations=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --no-build) build=0; shift ;;
    --cli-arg) extra+=(--cli-arg "$2"); shift 2 ;;
    *) nb_die "Unknown option $1" ;;
  esac
done
metrics=${metrics/all/latency,memory,verify,cold}
out=${out:-$NB_ROOT/runs/$(date -u +%Y%m%dT%H%M%S)}
mkdir -p "$out"
((build == 0)) || "$here/benchmark/build.sh"
echo "Runs: $out"
for metric in ${metrics//,/ }; do
  "$here/benchmark/bench.sh" --model "$model" --workload "$workload" --metric "$metric" --out "$out" \
    $mtp ${prompt:+--prompt-file "$prompt"} ${iterations:+--iterations "$iterations"} "${extra[@]}"
done
python3 "$here/benchmark/report.py" "$out"
