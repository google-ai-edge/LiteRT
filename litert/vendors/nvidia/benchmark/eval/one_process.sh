#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# Runs a job list through engine_runner.tool in the environment bench.sh gives the CLI: the
# model of a bench.sh profile (or a file), its cache pool by build id, the pool's TensorRT-RTX
# settings. The model, the dispatch libraries and the cache directory of JOBS.json's engine are
# filled in here; the caller sets the rest (context, signatures, speculative decoding, jobs).
#   usage: one_process.sh MODEL JOBS.json [--pool DIR] [--env NAME=VALUE]...
#   LITERT_LM_PYTHON: the directory of LiteRT-LM's Python package (litert_lm).
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
source "$here/../env.sh"
nb_resolve_paths
model_arg=${1:?usage: one_process.sh MODEL JOBS.json [--pool DIR] [--env NAME=VALUE]...} jobs=${2:?}
shift 2
pool= env_extra=()
while (($#)); do
  case "$1" in
    --pool) pool=$2; shift 2 ;;
    --env) env_extra+=("$2"); shift 2 ;;
    *) nb_die "Unknown option $1" ;;
  esac
done
[[ -f ${LITERT_LM_PYTHON:-}/litert_lm/__init__.py ]] || nb_die "LITERT_LM_PYTHON must name the directory of the litert_lm package"
model=$(nb_model_file "$model_arg")
profile=$model_arg; [[ $profile =~ ^(e2b|e4b|12b|26b)$ ]] || profile=$(basename "${model%.litertlm}")
pool=${pool:-$NB_ROOT/cache/models/$profile/$(nb_build_id)}
nb_runtime_env "$pool" "$model"
for assignment in "${env_extra[@]}"; do export "$assignment"; done
"$NB_PYTHON" - "$jobs" "$model" "$NB_RUNTIME_LIBS" "$pool/compiler" <<'PY'
import json, sys
path, model, libs, cache = sys.argv[1:]
spec = json.load(open(path))
spec['engine'].update(model=model, dispatch_lib_dir=libs, cache_dir=cache)
json.dump(spec, open(path, 'w'), indent=1)
PY
# The CLI runs from the LiteRT-LM checkout (bench.sh, run.sh); so does the runner.
cd "$NB_LM"
exec "$NB_PYTHON" "$here/engine_runner.tool" "$jobs"
