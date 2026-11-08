# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# The paths and conventions of the llama.cpp counterpart measurements on Linux. Source it;
# bench.sh, serve.sh and matrix.sh in this directory use the functions. A machine's values
# go in the benchmark's env.local.sh (sourced through ../env.sh): LLAMA_CPP_DIR (the checkout;
# its build in LLAMA_BIN), LLAMA_LOCAL_MODELS and LLAMA_HF_HUB (where the GGUFs are).
#
# The models are the GGUF counterparts of the LiteRT models: unsloth's QAT UD-Q4_K_XL files
# with the matching MTP drafter GGUFs (llama.cpp's --spec-type draft-mtp).
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../env.sh"
export LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/odml/llama.cpp}
export LLAMA_BIN=${LLAMA_BIN:-$LLAMA_CPP_DIR/build/bin}
export LLAMA_HF_HUB=${LLAMA_HF_HUB:-$HOME/.cache/huggingface/hub}
export LLAMA_LOCAL_MODELS=${LLAMA_LOCAL_MODELS:-${LITERT_BENCH_MODELS:-$HOME/odml/llm/models}}
export LLAMA_PORT=${LLAMA_PORT:-8089}
export LLAMA_THREADS=${LLAMA_THREADS:-20}
export LLAMA_UBATCH=${LLAMA_UBATCH:-1024}
# Shared by llama-bench and llama-server so matrix rows use one loading policy.
export LLAMA_LOAD_MODE=${LLAMA_LOAD_MODE:-mmap}
export LLAMA_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
export LITERT_BENCH_ROOT=${LITERT_BENCH_ROOT:-$HOME/.local/state/litert-nvidia-benchmark-wl}

lb_die() { echo "llama: $*" >&2; exit 1; }

# Use the benchmark root for generated CUDA kernels in both performance and evaluation
# processes. Model downloads remain in LLAMA_HF_HUB / LLAMA_LOCAL_MODELS.
lb_runtime_env() {
  LITERT_BENCH_ROOT=$(realpath -m "$LITERT_BENCH_ROOT")
  export LITERT_BENCH_ROOT CUDA_CACHE_DISABLE=0 CUDA_CACHE_PATH=$LITERT_BENCH_ROOT/cache/llama/cuda
  mkdir -p "$CUDA_CACHE_PATH"
}

# lb_hf REPO FILE -> the file in the Hugging Face cache's snapshot of REPO (user/name).
lb_hf() {
  local snapshots=$LLAMA_HF_HUB/models--${1//\//--}/snapshots f
  f=$(ls -d "$snapshots"/*/ 2>/dev/null | head -1)
  [[ -n $f && -e $f$2 ]] || lb_die "Model file not in the Hugging Face cache: $1 $2"
  realpath "$f$2"
}

# lb_model_file e2b|e4b|12b|26b|PATH -> the main GGUF; lb_draft_file PROFILE -> its MTP drafter.
lb_model_file() {
  case "$1" in
    e2b) echo "$LLAMA_LOCAL_MODELS/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf" ;;
    e4b) echo "$LLAMA_LOCAL_MODELS/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf" ;;
    12b) lb_hf unsloth/gemma-4-12B-it-qat-GGUF gemma-4-12B-it-qat-UD-Q4_K_XL.gguf ;;
    26b) lb_hf unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf ;;
    *) [[ -f $1 ]] || lb_die "Model is neither a profile name (e2b, e4b, 12b, 26b) nor a file: $1"; realpath "$1" ;;
  esac
}
lb_draft_file() {
  case "$1" in
    e2b) echo "$LLAMA_LOCAL_MODELS/gemma-4-E2B-it-GGUF/mtp-gemma-4-E2B-it.gguf" ;;
    e4b) echo "$LLAMA_LOCAL_MODELS/gemma-4-E4B-it-GGUF/mtp-gemma-4-E4B-it.gguf" ;;
    12b) lb_hf unsloth/gemma-4-12B-it-qat-GGUF MTP/mtp-gemma-4-12B-it-Q8_0.gguf ;;
    26b) lb_hf unsloth/gemma-4-26B-A4B-it-qat-GGUF MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf ;;
    *) lb_die "No MTP drafter known for $1 (give a profile name)" ;;
  esac
}

# lb_build_id -> llama.cpp's commit (a + marks local changes).
lb_build_id() {
  local sha dirty=
  sha=$(git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD 2>/dev/null) || sha=unknown
  [[ -z $(git -C "$LLAMA_CPP_DIR" status --porcelain --untracked-files=no 2>/dev/null) ]] || dirty=+
  echo "$sha$dirty"
}

# lb_model_args PROFILE -> the placement flags every llama.cpp run of that model gets: flash
# attention, every layer on the GPU (the 26B-A4B's experts too), the host threads, 2,048-token
# logical batches; LLAMA_UBATCH sets physical batches (default 1,024, like LiteRT prefill).
lb_model_args() {
  local args=(-fa on -ngl 99 -t "$LLAMA_THREADS" -b 2048 -ub "$LLAMA_UBATCH")
  [[ $1 != 26b ]] || args+=(-ncmoe 0)
  printf '%s\n' "${args[@]}"
}

# lb_sampling PROFILE -> the sampler the LiteRT-LM CLI uses for that model on this route, so
# the text runs and the evaluation sample alike: the E2B and E4B litertlm files carry no sampler
# parameters and LiteRT-LM decodes them greedily; the 12B and 26B-A4B files ask for top-k 64,
# top-p 0.95, temperature 1.0.
lb_sampling() {
  case "$1" in e2b|e4b) echo greedy ;; *) echo model ;; esac
}

# lb_evict FILE...: drop the files' pages from the page cache (a cold-start measurement).
lb_evict() {
  local f
  for f in "$@"; do dd if="$f" iflag=nocache count=0 status=none 2>/dev/null || true; done
}

lb_smi() { command -v nvidia-smi 2>/dev/null || echo /usr/lib/wsl/lib/nvidia-smi; }
