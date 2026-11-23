#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# Runs the selected LiveCodeBench problems through the LiteRT-LM CLI on the TensorRT-RTX
# backend, one process per problem, and keeps each response with its benchmark report.
#   usage: run.sh MODEL plain|mtp [MAX_TOKENS] [LABEL]
#   MODEL: e2b|e4b|12b|26b (bench.sh profile names) or a litertlm file
#   ONLY="qid qid ..." selects problems by id; QUESTION=N selects the N-th of selection.json
#   (1-based) or a problem by id: one question, predictably (question.sh wraps this).
#   EVAL_RUNS: where the runs go (default $LITERT_BENCH_ROOT/eval/lcb/runs).
#   EVAL_TOP_K/EVAL_TOP_P/EVAL_TEMPERATURE/EVAL_SEED: set all four for a seeded sampler.
#   EVAL_ALLOW_CONTEXT_LENGTH_EXPANSION=1 opts into the CLI's context expansion flag.
#   EVAL_ONE_PROCESS=1 runs every problem in one process (../engine_runner.tool): the model loads
#   once and each problem gets a fresh conversation configured as the CLI's, with the same files
#   per problem. LITERT_LM_PYTHON names the directory of LiteRT-LM's Python package (litert_lm,
#   its library built from the LITERT_LM_DIR checkout the CLI comes from). EVAL_ENGINES=N (with
#   it) decodes N problems side by side, an engine each, the weights held once: faster where one
#   engine leaves the GPU idle (Gemma 4 E2B, E4B), each response the same. A model with in-place
#   recurrent state (Qwen 3.8's gated DeltaNet) is answer for answer the same in one process only
#   with EVAL_ENV adding LITERT_NVIDIA_DISPATCH_STEP_FEED=0 LITERT_NVIDIA_DISPATCH_CUDA_GRAPH=0:
#   with either on, a CUDA graph capture or a step ahead updates that state twice, and an answer
#   depends on what ran before it (the CLI's on its own capture step).
#   EVAL_BATCH=1 (with EVAL_ENGINES=N, 2 to 4) batches the engines' decode steps: the steps that
#   come together run as one launch of a batched decode engine (LITERT_NVIDIA_DISPATCH_SLOTS=4:
#   engines of 2 to 4 slots compiled beside the decode engine, one compile for any N), each
#   response the same as alone.
#   EVAL_ORDER_FROM="DIR..." (one process) runs the problems longest answer first, by their decode
#   tokens in those earlier runs (unknown ones first): the set does not end on one long answer.
#   EVAL_SIGNATURES=a,b compiles and runs only those signatures (the 12B and 26B plain runs default
#   to prefill_1024,decode): a one-process Qwen 3.8 run with prefill_1024,decode does not page.
# Output: $EVAL_RUNS/LABEL/<question_id>.{out,log,json} (LABEL defaults to MODEL_MODE); .out is
# the model's response (the CLI's stdout), .log the CLI's log with the BenchmarkInfo block (the
# prompt is real text and stop tokens end the turn: --benchmark with both token counts 0),
# .json the parsed decode speed, token count and MTP acceptance of that problem.
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
source "$here/../../env.sh"  # LiteRT's benchmark/env.sh (the staging shim resolves it)
nb_resolve_paths
BENCH=${BENCH:-$(dirname "$(realpath "$here/../../env.sh")")}
if [[ -n ${QUESTION:-} ]]; then
  ONLY=$(python3 "$here/select.tool" --resolve "$QUESTION")
  export ONLY
fi
model_arg=$1 mode=$2 max_tokens=${3:-8192} label=${4:-$1_$2}
[[ $mode == plain || $mode == mtp ]] || nb_die "mode is plain or mtp"
expansion=${EVAL_ALLOW_CONTEXT_LENGTH_EXPANSION:-0}
[[ $expansion == 0 || $expansion == 1 ]] || nb_die "EVAL_ALLOW_CONTEXT_LENGTH_EXPANSION must be 0 or 1"
context_args=()
[[ $expansion == 0 ]] || context_args=(--allow_context_length_expansion=true)
sampler_args=()
if [[ -n ${EVAL_TOP_K:-}${EVAL_TOP_P:-}${EVAL_TEMPERATURE:-}${EVAL_SEED:-} ]]; then
  [[ -n ${EVAL_TOP_K:-} && -n ${EVAL_TOP_P:-} && -n ${EVAL_TEMPERATURE:-} && -n ${EVAL_SEED:-} ]] || nb_die "Set all four EVAL_TOP_K, EVAL_TOP_P, EVAL_TEMPERATURE, EVAL_SEED"
  sampler_args=("--sampler_top_k=$EVAL_TOP_K" "--sampler_top_p=$EVAL_TOP_P"
                "--sampler_temperature=$EVAL_TEMPERATURE" "--sampler_seed=$EVAL_SEED")
fi
one_process=${EVAL_ONE_PROCESS:-0}
[[ $one_process == 0 || $one_process == 1 ]] || nb_die "EVAL_ONE_PROCESS must be 0 or 1"
engines=${EVAL_ENGINES:-1}
[[ $engines =~ ^[1-9][0-9]*$ ]] || nb_die "EVAL_ENGINES must be a positive integer"
[[ $engines == 1 || $one_process == 1 ]] || nb_die "EVAL_ENGINES needs EVAL_ONE_PROCESS=1"
batch=${EVAL_BATCH:-0}
[[ $batch == 0 || $batch == 1 ]] || nb_die "EVAL_BATCH must be 0 or 1"
[[ $batch == 0 || ( $engines -ge 2 && $engines -le 4 ) ]] || nb_die "EVAL_BATCH=1 needs EVAL_ENGINES from 2 to 4"
if [[ $one_process == 1 ]]; then
  # LiteRT-LM's C API sets no advanced executor settings: the expansion is the CLI's alone.
  [[ $expansion == 0 ]] || nb_die "EVAL_ONE_PROCESS=1 cannot expand the context (LiteRT-LM's C API has no setting for it)"
  [[ -f ${LITERT_LM_PYTHON:-}/litert_lm/__init__.py ]] || nb_die "EVAL_ONE_PROCESS=1 needs LITERT_LM_PYTHON, the directory of the litert_lm package"
fi
model=$(nb_model_file "$model_arg")
profile=$model_arg; [[ $profile =~ ^(e2b|e4b|12b|26b)$ ]] || profile=$(basename "${model%.litertlm}")
# A/B of compile-time settings: EVAL_POOL_SUFFIX names a separate cache pool (fresh engines) and
# EVAL_ENV="NAME=VALUE ..." overrides the pool's standard TensorRT-RTX environment.
pool=${EVAL_POOL:-$NB_ROOT/cache/models/$profile/$(nb_build_id)${EVAL_POOL_SUFFIX:+-$EVAL_POOL_SUFFIX}}
nb_runtime_env "$pool" "$model"
[[ $batch == 0 ]] || export LITERT_NVIDIA_DISPATCH_SLOTS=4
for kv in ${EVAL_ENV:-}; do export "$kv"; done
runs=${EVAL_RUNS:-$NB_ROOT/eval/lcb/runs}/$label; mkdir -p "$runs"
engine=$NB_RUNTIME_LIBS/litert_lm_advanced_main
selected=()  # plain runs of the larger models compile only the signatures they run
if [[ -n ${EVAL_SIGNATURES:-} ]]; then
  [[ $EVAL_SIGNATURES =~ ^[A-Za-z0-9_]+(,[A-Za-z0-9_]+)*$ ]] || nb_die "EVAL_SIGNATURES is a comma-separated list of signatures"
  selected=("--selected_signatures=$EVAL_SIGNATURES")
elif [[ $mode == plain && ( $profile == 12b || $profile == 26b ) ]]; then
  selected=(--selected_signatures=prefill_1024,decode)
fi
spec=false; [[ $mode == plain ]] || spec=true
context=$((max_tokens + 2048)); ((context % 1024 == 0)) || context=$(( (context / 1024 + 1) * 1024 ))
EVAL_HOST_CLOCK=; [[ -f $here/../../clock_check.tool ]] && { EVAL_HOST_CLOCK=$(python3 "$here/../../clock_check.tool" 3) || echo "Warning: the host's clock is not fit: $EVAL_HOST_CLOCK" >&2; }
export EVAL_HOST_CLOCK
python3 - "$runs/run.json" "$profile" "$mode" "$max_tokens" "$context" "$(nb_build_id)" "$model" "$engine" "$expansion" "$here" "$BENCH" "$one_process" "${sampler_args[@]}" <<'PY'
import hashlib, json, os, pathlib, sys
path, profile, mode, budget, context, build, model, engine, expansion, here, bench, one_process, *sampler = sys.argv[1:]
def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def model_sha(path):
    # Kept by the file's path, size and modification time: a model's tens of gigabytes take
    # minutes to read again when they are no longer in the page cache, at every run.
    stat = os.stat(path)
    key = hashlib.sha256(f'{os.path.realpath(path)}|{stat.st_size}|{stat.st_mtime_ns}'.encode()).hexdigest()
    kept = pathlib.Path(os.environ['NB_ROOT']) / 'cache' / 'digests' / key[:32]
    if kept.exists():
        return kept.read_text().strip()
    digest = sha(path)
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_text(digest + '\n')
    return digest
config = {'profile': profile, 'mode': mode, 'max_output_tokens': int(budget),
          'context': int(context), 'build_id': build, 'model': model,
          'model_sha256': model_sha(model), 'binaries': {name: sha(pathlib.Path(engine).parent / name)
          for name in ('litert_lm_advanced_main', 'libLiteRtCompilerPlugin_Nvidia.so', 'libLiteRtDispatch_Nvidia.so')},
          # The build the binaries came from (build.sh's stamp); build_id above is the checkout's heads now.
          'binaries_build_id': (pathlib.Path(engine).parent / 'BUILD_ID').read_text().strip()
          if (pathlib.Path(engine).parent / 'BUILD_ID').exists() else None,
          'environment': {k: v for k, v in sorted(os.environ.items()) if k.startswith('LITERT_NVIDIA_')},
          # The host's clocks at the run's start (clock_check.tool in this tree): the decode
          # speeds are read on the monotonic clock. None when the check is not there.
          'host_clock': json.loads(os.environ['EVAL_HOST_CLOCK']) if os.environ.get('EVAL_HOST_CLOCK') else None}
prompt_paths = sorted((pathlib.Path(here) / 'prompts').glob('*.txt'))
requested = sorted(os.environ['ONLY'].split()) if os.environ.get('ONLY') else [p.stem for p in prompt_paths]
if not requested or len(set(requested)) != len(requested) or set(requested) - {p.stem for p in prompt_paths}:
    raise SystemExit('Requested evaluation questions must exist and be nonempty and unique')
config.update(sampler_arguments=sampler, requested_questions=requested,
              prompts_sha256={p.stem: sha(p) for p in prompt_paths if p.stem in requested},
              runner_sha256={str(p): sha(p) for p in (pathlib.Path(here) / 'run.sh',
                              pathlib.Path(bench) / 'env.sh')})
if expansion == '1':
    config['allow_context_length_expansion'] = True
if one_process == '1':
    # The runner and the LiteRT-LM library it drives in place of the CLI.
    runner = pathlib.Path(here).parent / 'engine_runner.tool'
    config['one_process'] = {'runner_sha256': sha(runner), 'litert_lm_sha256': sha(
        pathlib.Path(os.environ['LITERT_LM_PYTHON']) / 'litert_lm' / 'liblitert-lm.so')}
target = pathlib.Path(path)
if target.exists() and json.loads(target.read_text()) != config:
    raise SystemExit('Existing evaluation has different settings; use a new label')
target.write_text(json.dumps(config, indent=1) + '\n')
(target.parent / 'settings.json').write_text(json.dumps(config['environment'], indent=1) + '\n')
PY
# record QID RC: the problem's metrics from its log and accounting into QID.json.
record() {
  python3 - "$runs/$1" "$2" <<'PY'
import json, re, sys
base = sys.argv[1]
log = open(base + '.log', errors='replace').read()
meta = {'exit_code': int(sys.argv[2])}
m = re.search(r'Prefill Turn 1: Processed (\d+) tokens in (\S+) duration\.\s*Prefill Speed: ([0-9.]+)', log)
if m: meta.update(prompt_tokens=int(m[1]), prefill_tokens_per_second=float(m[3]))
m = re.search(r'Decode Turn 1: Processed (\d+) tokens in (\S+) duration\.\s*Decode Speed: ([0-9.]+)', log)
if m: meta.update(decode_tokens=int(m[1]), decode_tokens_per_second=float(m[3]))
m = re.search(r'Acceptance rate:\s*([0-9.]+)%', log)
if m: meta['acceptance'] = float(m[1]) / 100
m = re.search(r'Tokens per round:\s*([0-9.]+)', log)
if m: meta['tokens_per_round'] = float(m[1])
m = re.search(r'Time to first token: ([0-9.]+) s', log)
if m: meta['time_to_first_token_s'] = float(m[1])
m = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)', open(base + '.time').read())
if m:
    parts = [float(p) for p in m[1].split(':')]
    meta['wall_s'] = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
json.dump(meta, open(base + '.json', 'w'))
print('  ', json.dumps(meta))
PY
}
status=0
pending=()
for prompt in "$here"/prompts/*.txt; do
  qid=$(basename "${prompt%.txt}")
  if [[ -n ${ONLY:-} ]]; then [[ " $ONLY " == *" $qid "* ]] || continue; fi
  if [[ -s $runs/$qid.out && -f $runs/$qid.json ]] && python3 - "$runs/$qid.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
sys.exit(0 if m.get('exit_code') == 0 and m.get('decode_tokens', 0) > 0 else 1)
PY
  then echo "$qid: done"; continue; fi
  if [[ $one_process == 1 ]]; then pending+=("$qid"); continue; fi
  echo "$(date -Is) $label $qid"
  rc=0
  (cd "$NB_LM" && /usr/bin/time -v -o "$runs/$qid.time" "$engine" "--model_path=$model" --backend=npu \
     --prefill_batch_sizes=1024 "--max_num_tokens=$context" "${context_args[@]}" "${sampler_args[@]}" "--cache_dir=$pool/compiler" \
     "--litert_dispatch_lib_dir=$NB_RUNTIME_LIBS" --min_log_severity=0 "${selected[@]}" \
     "--enable_speculative_decoding=$spec" --benchmark=true --benchmark_prefill_tokens=0 \
     --benchmark_decode_tokens=0 --num_iterations=1 "--max_output_tokens=$max_tokens" \
     "--input_prompt_file=$prompt" > "$runs/$qid.out" 2> "$runs/$qid.log") || rc=$?
  [[ $rc == 0 ]] || { status=1; echo "$qid: exit $rc"; }
  record "$qid" "$rc"
done
if ((${#pending[@]})); then
  # One process for the pending problems: the CLI's settings above, as a job list.
  echo "$(date -Is) $label: ${#pending[@]} problems in one process"
  python3 - "$runs/jobs.json" "$model" "$NB_RUNTIME_LIBS" "$pool/compiler" "$context" "$max_tokens" \
      "$spec" "$here/prompts" "$runs" "${selected[*]}" "${sampler_args[*]}" "$engines" "${pending[@]}" <<'PY'
import json, os, sys
path, model, libs, cache, context, budget, spec, prompts, runs, selected, sampler, engines, *qids = sys.argv[1:]
flags = dict(a[2:].split('=', 1) for a in sampler.split())
sampler = {'top_k': int(flags['sampler_top_k']), 'top_p': float(flags['sampler_top_p']),
           'temperature': float(flags['sampler_temperature']), 'seed': int(flags['sampler_seed'])} if flags else None
signatures = selected.split('=', 1)[1].split(',') if selected else None
# EVAL_ORDER_FROM: earlier runs' directories; the problems go longest answer first (by their decode
# tokens there, the most of any; a problem none of them ran goes first), so that the engines, each
# taking the next problem when done, do not end the set on one long answer.
known = {}
for directory in os.environ.get('EVAL_ORDER_FROM', '').split():
    for q in qids:
        try:
            tokens = json.load(open(os.path.join(directory, f'{q}.json'))).get('decode_tokens')
        except (OSError, ValueError):
            continue
        if isinstance(tokens, int):
            known[q] = max(known.get(q, 0), tokens)
if os.environ.get('EVAL_ORDER_FROM'):
    qids = sorted(qids, key=lambda q: -known.get(q, float('inf')))
jobs = [{'name': q, 'prompt_file': f'{prompts}/{q}.txt', 'max_output_tokens': int(budget), 'sampler': sampler,
         'stdout': f'{runs}/{q}.out', 'stderr': f'{runs}/{q}.log', 'time': f'{runs}/{q}.time'} for q in qids]
json.dump({'engine': {'model': model, 'dispatch_lib_dir': libs, 'cache_dir': cache, 'max_num_tokens': int(context),
                      'selected_signatures': signatures, 'speculative_decoding': spec == 'true',
                      'min_log_severity': 0},
           'jobs': jobs, 'engines': int(engines), 'summary': f'{runs}/one_process.json'}, open(path, 'w'), indent=1)
PY
  (cd "$NB_LM" && /usr/bin/time -v -o "$runs/one_process.time" python3 "$here/../engine_runner.tool" "$runs/jobs.json") || status=1
  for qid in "${pending[@]}"; do
    rc=$(sed -n 's/^\tExit status: //p' "$runs/$qid.time" 2>/dev/null || true)
    [[ ${rc:-1} == 0 ]] || { status=1; echo "$qid: exit ${rc:-none}"; }
    [[ -f $runs/$qid.log ]] && record "$qid" "${rc:-1}"
  done
fi
exit "$status"
