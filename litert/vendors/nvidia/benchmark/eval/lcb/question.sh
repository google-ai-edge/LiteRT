#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# One LiveCodeBench problem, end to end: generate it through LiteRT-LM on TensorRT-RTX (or
# llama.cpp), grade it, print the verdict. The predictable way to exercise one question.
#   usage: question.sh MODEL plain|mtp QUESTION [MAX_TOKENS] [LABEL]
#   QUESTION: a question id (abc397_a, 3714, ...) or its 1-based index in selection.json
#   RUNTIME=llama runs llama.cpp instead of LiteRT; EVAL_RUNS as in run.sh.
# Output: the run directory $EVAL_RUNS/LABEL (LABEL defaults to MODEL_MODE_qQUESTION) with the
# response, log, metrics and grades.json; exit 0 when the answer passed every test.
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
model=${1:?usage: question.sh MODEL plain|mtp QUESTION [MAX_TOKENS] [LABEL]} mode=${2:?} question=${3:?}
max_tokens=${4:-8192} label=${5:-${model}_${mode}_q${question}}
export QUESTION=$question
case ${RUNTIME:-litert} in
  litert) "$here/run.sh" "$model" "$mode" "$max_tokens" "$label" ;;
  llama) "$here/run_llama.sh" "$model" "$mode" "$max_tokens" "$label" ;;
  *) echo "RUNTIME is litert or llama" >&2; exit 2 ;;
esac
source "$here/../../env.sh"
nb_resolve_paths
runs=${EVAL_RUNS:-$NB_ROOT/eval/lcb/runs}/$label
python3 -I "$here/grade.tool" "$runs" | tail -2
python3 - "$runs/grades.json" <<'PY'
import json, sys
grades = json.load(open(sys.argv[1]))
rows = [r for r in grades['rows'] if r['status'] != 'missing']
for r in rows:
    print(f"{r['question_id']}: {r['status']} ({r['passed']}/{r['total']} tests), "
          f"{r.get('decode_tokens', '?')} tokens at {r.get('decode_tokens_per_second') or 0:.1f} tok/s")
sys.exit(0 if rows and all(r['status'] == 'pass' for r in rows) else 1)
PY
