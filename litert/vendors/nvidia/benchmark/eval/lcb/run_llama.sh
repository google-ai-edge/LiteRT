#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# Runs the selected LiveCodeBench problems through llama.cpp, the counterpart of run.sh: one
# llama-server per run (llama/serve.sh), one completion per problem (llama/client.tool) with the
# same prompt text in Gemma 4's user turn, the sampler LiteRT-LM uses for the model (greedy for
# E2B and E4B; top-k 64, top-p 0.95, temperature 1.0 for 12B and 26B-A4B), stop tokens honored,
# at most MAX_TOKENS tokens; with mtp, the model's MTP drafter GGUF (--spec-type draft-mtp).
#   usage: run_llama.sh MODEL plain|mtp [MAX_TOKENS] [LABEL]   ONLY="qid ..." or QUESTION=id|index selects problems
#   EVAL_RUNS: where the runs go (default $LITERT_BENCH_ROOT/eval/lcb/runs)
#   MODEL: e2b|e4b|12b|26b (llama/env.sh profile names)
# Output: runs/LABEL/<question_id>.{out,json} (LABEL defaults to MODEL_llama_MODE); .out is the
# completion, .json the server's timings in the fields grade.py reads (decode_tokens,
# decode_tokens_per_second, acceptance, ...); runs/LABEL/server/ holds the server's log.
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
source "$here/../../llama/env.sh"
if [[ -n ${QUESTION:-} ]]; then ONLY=$(python3 "$here/select.tool" --resolve "$QUESTION"); export ONLY; fi
model_arg=${1:?usage: run_llama.sh MODEL plain|mtp [MAX_TOKENS] [LABEL]} mode=$2 max_tokens=${3:-8192} label=${4:-${1}_llama_$2}
[[ $mode == plain || $mode == mtp ]] || lb_die "mode is plain or mtp"
context=$((max_tokens + 2048)); ((context % 1024 == 0)) || context=$(( (context / 1024 + 1) * 1024 ))
runs=${EVAL_RUNS:-$LITERT_BENCH_ROOT/eval/lcb/runs}/$label; mkdir -p "$runs"
handle=$runs/server
serve_args=(--ctx "$context" --port "$LLAMA_PORT" --no-gpu-sampling)
[[ $mode == plain ]] || serve_args+=(--mtp)
cleanup() { "$LLAMA_DIR/serve.sh" stop "$handle" || true; }
trap cleanup EXIT
"$LLAMA_DIR/serve.sh" start "$handle" "$model_arg" "${serve_args[@]}" > /dev/null
sampling=$(lb_sampling "$model_arg")
for prompt in "$here"/prompts/*.txt; do
  qid=$(basename "${prompt%.txt}")
  if [[ -n ${ONLY:-} ]]; then [[ " $ONLY " == *" $qid "* ]] || continue; fi
  [[ ! -s $runs/$qid.out ]] || { echo "$qid: done"; continue; }
  echo "$(date -Is) $label $qid"
  if ! python3 "$LLAMA_DIR/client.tool" --prompt-file "$prompt" --out "$runs/$qid" --n-predict "$max_tokens" \
       --port "$LLAMA_PORT" --sampling "$sampling" --seed 1 > /dev/null; then
    echo "$qid: request failed"; continue
  fi
  mv "$runs/$qid.txt" "$runs/$qid.out"
  python3 - "$runs/$qid.json" "$sampling" <<'PY'
import json, sys
path, sampling = sys.argv[1:3]
d = json.load(open(path)); t = d.get('timings', {})
meta = {'runtime': 'llama.cpp', 'sampling': sampling, 'prompt_tokens': t.get('prompt_n'),
        'prefill_tokens_per_second': t.get('prompt_per_second'), 'decode_tokens': t.get('predicted_n'),
        'decode_tokens_per_second': t.get('predicted_per_second'), 'wall_s': d.get('wall_s'),
        'stop_type': d.get('stop_type')}
if t.get('draft_n'):
    meta['acceptance'] = (t.get('draft_n_accepted') or 0) / t['draft_n']
    meta['drafted'] = t['draft_n']
json.dump(meta, open(path, 'w'))
print('  ', json.dumps({k: meta[k] for k in ('decode_tokens', 'decode_tokens_per_second', 'acceptance', 'stop_type') if k in meta}))
PY
done
