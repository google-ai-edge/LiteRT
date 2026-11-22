#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# The teacher-forced comparison of the NVIDIA backend against LiteRT-LM's CPU over a set of
# prompts, and its pass bar (tools/teacher_forced.py gate). Fed the same tokens (the CLI's
# --score_target_text prefills the prompt, then every decode step reads the target's next token
# whatever the model picked), the two backends' logits compare at every step, where greedy texts
# part at their first near-tie.
#
# usage:
#   teacher_forced.sh reference --model M --set DIR [--tokens N] [--jobs J]
#       The CPU's side of a set, once. DIR/inputs/NAME.txt are the prompts (the user's turn);
#       DIR/turn_suffix.txt is the rest of the turn as the conversation path writes it, which the
#       scoring path does not add; DIR/tokenizer.json (or .spm) is the model's tokenizer. For
#       each prompt: the CPU's greedy answer (DIR/targets/NAME.txt, at most N tokens, 64 by
#       default) and the CPU's logits teacher-forced over it (DIR/cpu/NAME.{logits.npy,
#       steps.json}). Prompts that have them are skipped; J run side by side (default 1), each
#       with a pool of its own.
#   teacher_forced.sh run --model M --set DIR --out DIR [--env NAME=VALUE ...] [--gpu-lock CMD]
#                         [--positions P,...] [--pool DIR]
#       The NVIDIA backend teacher-forced over the set's targets (OUT/NAME.*: the step feed off,
#       each decode step's logits from the dispatch's output dumps), then `teacher_forced.py gate`
#       against the CPU: OUT/gate.txt and OUT/gate.jsonl, exit status 1 when the bar fails. CMD
#       wraps every run of the CLI (a GPU lock), --env sets the CLI's environment after the
#       pool's (env.sh), --positions the gate's position buckets (default 512,1024), --pool the
#       model cache pool (default bench.sh's, by model and build id).
# Both use the context of bench.sh's short workload (2,048 tokens, 1,024-token prefill batches).
# Environment: see env.sh
# (LITERT_BENCH_RUNTIME_LIBS chooses the build; LITERT_BENCH_PYTHON a Python 3 with numpy and,
# for the tokenizer, sentencepiece or tokenizers).
set -euo pipefail
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/env.sh"
nb_resolve_paths
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
tool=$here/../tools/teacher_forced.py

command=${1:-}; shift || true
case $command in
  reference|run) ;;
  -h|--help) sed -n '3,27p' "$0"; exit 0 ;;
  *) nb_die "usage: teacher_forced.sh reference|run ... (--help)" ;;
esac
model_arg= set= out= tokens=64 jobs=1 lock=() env_extra=() positions=512,1024 pool=
while (($#)); do
  case "$1" in
    --model) model_arg=$2; shift 2 ;;
    --set) set=$(realpath "$2"); shift 2 ;;
    --out) out=$(realpath -m "$2"); shift 2 ;;
    --tokens) tokens=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --gpu-lock) read -r -a lock <<< "$2"; shift 2 ;;
    --env) env_extra+=("$2"); shift 2 ;;
    --positions) positions=$2; shift 2 ;;
    --pool) pool=$(realpath -m "$2"); shift 2 ;;
    *) nb_die "Unknown argument: $1" ;;
  esac
done
[[ -n $model_arg && -d $set/inputs && -f $set/turn_suffix.txt ]] || nb_die "--model and --set (with inputs/ and turn_suffix.txt) are needed"
model=$(nb_model_file "$model_arg")
profile=$model_arg; [[ $profile =~ ^(e2b|e4b|12b|26b)$ ]] || profile=$(basename "${model%.litertlm}")
tokenizer=
for candidate in "$set/tokenizer.json" "$set/tokenizer.spm"; do
  [[ -n $tokenizer || ! -f $candidate ]] || tokenizer=$candidate
done
[[ -n $tokenizer ]] || nb_die "No tokenizer.json or tokenizer.spm in $set"
engine=$NB_RUNTIME_LIBS/$NB_CLI
[[ -x $engine ]] || nb_die "Run build.sh first ($NB_RUNTIME_LIBS)"
context=2048 prefill=1024
mapfile -t names < <(cd "$set/inputs" && ls -- *.txt | sed 's/\.txt$//' | sort)
((${#names[@]})) || nb_die "No prompts in $set/inputs"

# The prompt the scoring path reads: the user's turn and the rest of it.
scoring_prompt() {  # NAME -> path
  local file=$set/prompts/$1.txt
  mkdir -p "$set/prompts"
  { cat "$set/inputs/$1.txt"; cat "$set/turn_suffix.txt"; } > "$file"
  echo "$file"
}

# cli POOL_DIR LOG ARGS...: one run of the CLI from the LiteRT-LM checkout, its log kept.
cli() {
  local cache=$1 log=$2; shift 2
  (cd "$NB_LM" && "${lock[@]}" "$engine" "--model_path=$model" "--prefill_batch_sizes=$prefill" \
    "--litert_dispatch_lib_dir=$NB_RUNTIME_LIBS" --min_log_severity=0 "--max_num_tokens=$context" \
    "--cache_dir=$cache" --benchmark=false "$@" > "$log" 2>&1)
}

# The target, trailing newlines too (a command substitution drops them).
read_target() { local text; text=$(cat "$1"; printf x); printf '%s' "${text%x}"; }

reference_one() {  # NAME POOL
  local name=$1 pool=$2 target=$set/targets/$1.txt
  nb_runtime_env "$pool" "$model"
  local cache=$pool/cpu
  mkdir -p "$set/targets" "$set/cpu"
  if [[ ! -s $target ]]; then
    # The CPU's own greedy answer, as the conversation path writes the turn.
    cli "$cache" "$set/cpu/$name.answer.log" --backend=cpu "--input_prompt_file=$set/inputs/$name.txt" \
      "--max_output_tokens=$tokens" --sampler_top_k=1 --sampler_top_p=0 --sampler_temperature=1 \
      --sampler_seed=1 || { echo "$name: the CPU's answer failed; see $set/cpu/$name.answer.log" >&2; return 1; }
    # The answer as the CLI captured it: its log line and the lines up to the next log line. An
    # empty one is a run that gave up (a loaded host can time the CLI's callbacks out).
    "$NB_PYTHON" - "$set/cpu/$name.answer.log" "$target.part" <<'PY' || return 1
import re, sys
lines = open(sys.argv[1], errors='replace').read().split('\n')
for i, line in enumerate(lines):
  if 'Captured model output: ' in line:
    text = [line.split('Captured model output: ', 1)[1]]
    for more in lines[i + 1:]:
      if re.match(r'^[IWEF]\d{4} ', more) or more.startswith(('INFO: [', 'WARNING: [', 'ERROR: [', 'VERBOSE: ')):
        break
      text.append(more)
    if not ''.join(text).strip():
      sys.exit(f'{sys.argv[1]}: the captured answer is empty')
    open(sys.argv[2], 'w').write('\n'.join(text))
    sys.exit(0)
sys.exit(f'{sys.argv[1]}: no captured answer')
PY
    mv "$target.part" "$target"
  fi
  if [[ ! -s $set/cpu/$name.logits.npy ]]; then
    rm -rf "$cache/litert_lm_debugger"
    cli "$cache" "$set/cpu/$name.score.log" --backend=cpu "--input_prompt_file=$(scoring_prompt "$name")" \
      --enable_debugger=true "--score_target_text=$(read_target "$target")" || true
    "$NB_PYTHON" -I "$tool" collect --tokenizer "$tokenizer" --target "$target" \
      "$cache/litert_lm_debugger" "$set/cpu/$name" || return 1
    rm -rf "$cache/litert_lm_debugger"
    # The debugger's lines are most of the log.
    sed -i '/Captured safetensors tensor/d' "$set/cpu/$name.score.log"
  fi
}

if [[ $command == reference ]]; then
  # The prompts of each worker, every J-th; a worker's pool holds its CPU caches and the
  # debugger's captures.
  pids=()
  for ((w = 0; w < jobs; w++)); do
    (
      rc=0
      for ((i = w; i < ${#names[@]}; i += jobs)); do
        if reference_one "${names[i]}" "$NB_ROOT/cache/teacher_forced/$profile/cpu$w"; then
          echo "${names[i]}: $(wc -w < "$set/targets/${names[i]}.txt") words, CPU logits $set/cpu/${names[i]}.logits.npy"
        else
          echo "${names[i]}: FAILED (run reference again for it)"; rc=1
        fi
      done
      exit $rc
    ) &
    pids+=($!)
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  exit $rc
fi

[[ -n $out ]] || nb_die "run needs --out"
mkdir -p "$out"
pool=${pool:-$NB_ROOT/cache/models/$profile/$(nb_build_id)}
nb_runtime_env "$pool" "$model"
# The step feed runs a decode step ahead with the model's own pick, which a scored step replaces:
# off, so that every step is the plain invocation the CPU makes.
export LITERT_NVIDIA_DISPATCH_STEP_FEED=0
for assignment in "${env_extra[@]}"; do export "$assignment"; done
selected=(--selected_signatures=prefill_$prefill,decode)
pairs=()
for name in "${names[@]}"; do
  [[ -s $set/cpu/$name.logits.npy ]] || nb_die "$name has no CPU reference: run teacher_forced.sh reference first"
  rm -rf "$out/$name.dump"
  # Every output up to 2 MB: the logits, not the caches.
  LITERT_NVIDIA_DISPATCH_DUMP_OUTPUTS=$out/$name.dump:1000000:2 \
    cli "$pool/compiler" "$out/$name.log" --backend=npu "${selected[@]}" --enable_speculative_decoding=false \
    "--input_prompt_file=$(scoring_prompt "$name")" "--score_target_text=$(read_target "$set/targets/$name.txt")" \
    || nb_die "$name: the run failed; see $out/$name.log"
  "$NB_PYTHON" -I "$tool" collect-dispatch --tokenizer "$tokenizer" --target "$set/targets/$name.txt" \
    "$out/$name.dump" "$out/$name"
  rm -rf "$out/$name.dump"
  pairs+=("$set/cpu/$name" "$out/$name")
done
{
  echo "build $(cat "$NB_RUNTIME_LIBS/BUILD_ID" 2>/dev/null || echo unknown), model $model"
  echo "environment: ${env_extra[*]:-(none)}"
} > "$out/gate.txt"
rc=0
"$NB_PYTHON" -I "$tool" gate --positions "$positions" --json "$out/gate.jsonl" "${pairs[@]}" >> "$out/gate.txt" || rc=$?
cat "$out/gate.txt"
exit $rc
