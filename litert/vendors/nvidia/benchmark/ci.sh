#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# A bounded performance check using the same build.sh and bench.sh as matrix.sh.
# Usage: ci.sh OUT_DIR
# Host preparation is external. CI_HOST_MODE records diagnostic (default) or formal.
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
if [[ ${1:-} == --help || $# != 1 ]]; then
  sed -n '3,5p' "$0"
  [[ ${1:-} == --help ]]; exit
fi
source "$here/env.sh"
nb_resolve_paths
out=$(realpath -m "$1")
[[ ! -e $out ]] || nb_die "Use a new output directory: $out"
mkdir -p "$out" "$NB_ROOT"
exec 8>"$NB_ROOT/.ci.lock"
flock -n 8 || nb_die "Another measurement CI is running"
export CI_HOST_MODE=${CI_HOST_MODE:-diagnostic}
printf 'Host mode: %s; cache root: %s\n' "$CI_HOST_MODE" "$NB_ROOT"
step() {
  local start=$SECONDS rc=0
  printf '\nCommand:'; printf ' %q' "$@"; printf '\n'
  "$@" || rc=$?
  printf 'Step finished: exit=%s elapsed_s=%s\n' "$rc" "$((SECONDS-start))"
  return "$rc"
}
started=$SECONDS
trap 'rc=$?; printf "CI finished: exit=%s elapsed_s=%s output=%s\n" "$rc" "$((SECONDS-started))" "$out"' EXIT
step "$here/build.sh"

# Cold means fresh compiled artifacts. Model bytes are deliberately in the OS page cache.
preread() { cat "$1" > /dev/null; }
step preread "$(nb_model_file 12b)"
step "$here/bench.sh" --model 12b --workload 128k --metric cold \
  --out "$out" --label 12b_128k_cold
# Prepare this exact workload explicitly; an unrelated AOT entry is insufficient.
pool=$NB_ROOT/cache/models/12b/$(nb_build_id)/ci_30k_plain
step "$here/bench.sh" --model 12b --workload 30k --metric latency --iterations 1 \
  --pool "$pool" --out "$out" --label 12b_30k_prepare
step "$here/bench.sh" --model 12b --workload 30k --metric latency --iterations 3 \
  --pool "$pool" --out "$out" --label 12b_30k_hot
step preread "$(nb_model_file 26b)"
step "$here/bench.sh" --model 26b --workload 30k --metric cold --mtp \
  --prompt-file "$here/prompts/prompt_30k.txt" --out "$out" --label 26b_30k_mtp_cold
step python3 "$here/report.py" --check-ci "$out"
printf '\nInitialization and cache evidence (full logs under %s):\n' "$out"
rg -n 'Init Total:|Init Executor:|AOT cache (hit|miss):|Engine generation completed|compiling partition' \
  "$out"/*/cli.log
