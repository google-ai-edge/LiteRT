# nvguard: the numerics and performance guard

Does a change to LiteRT's NVIDIA (TensorRT-RTX) backend regress numerics or
performance, on every Gemma 4 model and across context lengths, before the
long measurements (the full matrix, the LiveCodeBench evaluation) are spent on
it? `nvguard` answers in a few minutes per model.

```
cargo build --release                       # once; the binary is target/release/nvguard
target/release/nvguard env                  # the paths and the build id env.sh resolves
target/release/nvguard capture --name base  # the probes of the current build
# ... change the trees, build.sh ...
target/release/nvguard check --baseline base --name change --record ~/odml/measurements/bench/results.json
```

`check` exits 0 (PASS), 1 (WARN) or 2 (FAIL) and writes
`$LITERT_BENCH_ROOT/guard/<name>/against_<baseline>.md` (and `.json`); when
the verdict is not PASS it appends the localization (below), which
`localize --baseline base change` also writes on its own.

## What it measures

**Numerics.** Each probe decodes a fixed prompt greedily (`--sampler_top_k=1`,
whatever the model's own sampler) with `LITERT_NVIDIA_DISPATCH_DUMP_IO=top32:<file>`,
so the dispatch records the 32 largest logits of every step. Two builds are
compared step by step while their argmax tokens agree (the inputs of those
steps are identical): the largest |Δlogit| among the top-K, the KL divergence
of the top-K distributions, the top-K set overlap, and where the greedy texts
part. A divergence after a few sentences is a near-tie and only warns; one
within the first eight tokens, non-finite logits, a repetition loop, a mean KL
above 1e-3 or a |Δlogit| above 0.5 fail. The thresholds are in `nvguard.toml`.
The prompts: three coding problems in LiveCodeBench's format, a reasoning
question, and the benchmark's 8k document (quick tier); the 30k document
(standard); the 128k document for the 12B (long). A capture of the same build
twice gives zero deltas: the engines are deterministic.

**Where a change enters.** Each numerics probe also runs with
`LITERT_NVIDIA_DISPATCH_DUMP_OUTPUTS=<dir>:8:16`, so the dispatch writes every
output of the first eight steps of every partition whole (outputs up to
16 MB; the in-place caches are left out), with an index in production order.
`localize --baseline A B` compares two captures' dumps in that order and names
the first tensor that differs (its context, function, step and output, how
many elements moved and how far, the functions the difference has reached by
that step), the largest move, and the KL of the first logits rows over the
whole vocabulary (the top-32 of DUMP_IO is a sample of it). A partition is as
deep as it sees: TensorRT fuses the layers inside one, so a change in a
kernel shows as the first step and partition whose outputs moved, with the
greedy token where the texts part; the autopsy below names the kernel when
its time moved too.

**Texts.** Two decodes in the CLI's session mode, each run with the MTP
verifier sampling on the device and on the host
(`LITERT_NVIDIA_MTP_GPU_SAMPLING=0`): a sampled one (top-k 64, top-p 0.95,
seed 1, 200 tokens of the `code_stdin` prompt) and a constrained one (a
space-free JSON regex with repetition penalty 1.3: with this LiteRT-LM and
Gemma 4's tokenizer a regex with spaces fails on every backend). The device
and host texts must agree (a parting warns: a tie at the k-th logit, or a
sampler change), the constrained text must match its regex and both runs
must end without parser errors (a violation or a bad exit fails). The
greedy probes cannot see any of this: the samplers and the masks are what
`--sampler_top_k=1` bypasses.

**Performance.** `bench.sh` latency runs of the `short` (2k) and `8k`
workloads with few iterations, per model; `--mtp` adds the plain and MTP text
runs of those workloads (acceptance, tokens per round). Candidate/baseline
ratios below 0.97 warn, below 0.93 fail. From the measured contexts a cost
model per model (decode step time a + b·context; prefill time per token
u + v·prompt) predicts the other contexts (16k, 30k); with `--record` the
measured change ratio, interpolated in context, is applied to the record's
full curve to say what the candidate would measure there. `predict NAME
--record FILE` also reports the predictor's own error on the record (fitted
on the record's 2k and 8k, predicting its 16k and 30k).

**Kernels.** One run of the `short` workload in the dispatch's own-graph
mode with `LITERT_NVIDIA_DISPATCH_CUDA_GRAPH_AUTOPSY` and `_DOT`: every
kernel node of each function's first step graph timed alone (warm and cold)
and named through the graph's DOT (demangled with `c++filt` where it is).
Two captures' autopsies diff by function and by kernel, so a performance
change is attributed to the kernels that moved. The autopsy times kernels
one at a time: its sums are the kernels' own times, not the step's, and it
is a diagnostic, never a table cell of the measurements.

The reports say what their numbers are: the workload legend (input and
output tokens, the context) stands above the performance tables, tok/s are
tokens per second of the measured phase, and the guard's numbers belong to
captures (few iterations, the diagnostic host mode), never to the
measurements' tables.

## Layout

- `src/env.rs`: env.sh's paths, the build id, a pool's environment, through bash.
- `src/runner.rs`: one numerics, text, autopsy (bench.sh) or performance probe (bench.sh).
- `src/capture.rs`: a resumable capture under `$LITERT_BENCH_ROOT/guard/<name>`.
- `src/numerics.rs`, `src/perf.rs`: the metrics, the cost model, the verdicts.
- `src/dumps.rs`: the whole-output dumps, their diff in production order, the whole-vocabulary KL.
- `src/texts.rs`: the sampled and constrained texts, device against host.
- `src/autopsy.rs`: the per-kernel autopsy joined with the DOT's names, and its diff.
- `src/report.rs`: the comparison, the localization and their markdown.
- `nvguard.toml`, `prompts/`: the defaults and the probe prompts.

A capture is tied to a build id; probing a changed tree needs another name.
The probes use the model's standard cache pool of the build (compiling it on
first use); a numerics probe of a model runs before its performance probes.
`cargo test` covers the parsers, the metrics and the cost model.

## What it caught on its first day

- A build against the wrong SDK headers (1.5 instead of 1.7): no weight store,
  pools five to twenty times larger, timings under the record. The capture
  table now names the compiler's SDK and whether the store was used.
- The dispatch library exporting no sampler symbols after an upstream change:
  speculative decoding silently sampling on the CPU (MTP rounds 4.7 to 6.4 ms).
  The capture counts the device sampler lines.
- The 26B-A4B's logits differing from run to run (a shared-memory atomic merge
  in the decode attention): a capture of the same build twice, compared, is
  the test (`capture --name a`, `capture --name b`, `compare a b`); the other
  models are bit-identical.

## What it does not probe

- A grammar that ends inside an MTP round, and a constraint under the
  immediate-upload mode (`LITERT_NVIDIA_DISPATCH_DEFERRED_UPLOAD=0`): the
  text probes run one regex in the default mode. Both were checked by hand
  with the same A/B the text probes automate (the dispatch's device samplers
  reading a constraint mask the host had written into the logits was found
  that way).
- The samplers at exact ties: the device keeps the lowest indices at a tie
  on the k-th logit, the CPU heap does not say; a parting of the sampled
  texts warns for that reason instead of failing.
- Memory: the capture records the peak RSS of the performance runs and
  nothing of the device's.

## What the gates showed (2026-10-27, E2B, build 3c590a6e1-95cd9c0b)

- The chain against round 5's `final` capture: the ten greedy probes (five
  prompts, plain and MTP) bit-identical, every performance cell within its
  spread, acceptance equal.
- A bit-identical rebuild captured under another name and localized against
  the first: 76 tensors paired per probe, none differing, the texts
  identical, PASS. The autopsy's warm times moved by up to about 20 µs per
  kernel between the two: a kernel's change below that is noise.
- A constructed divergence in a kernel the E2B's prefill runs (the INT4
  GEMM's per-row unscale times 1.001, the branch `scratch-divergence-gemm`):
  the first differing tensor is the prefill partition's first cache output
  at step 1 (956 of 524288 int8 values moved by one quantization step,
  relative 8e-3, all thirty caches of that step reached), the whole-vocabulary
  KL of the first logits row 3e-4 with the argmax kept, the emitted tokens
  parting at token 56 of 256 (`code_grid`; 186, 134, never, 42 of 64 on the
  other prompts), the device and host texts still identical to each other
  and both changed; the check warns (a parting after the first eight tokens,
  max |Δlogit| 0.56). A first attempt scaled the decode attention's combine
  output and changed nothing: that kernel is not on the E2B's path at 2k
  (the autopsy lists 212 GEMV and 30 cache-write launches per decode step
  and 57 INT4 GEMM launches per prefill; the rest are TensorRT's fusions),
  which is the guard saying so.
- The functions in the dumps are the invocation contexts' (`c<context>
  tensorrt_partition_<n>`): which is the prefill and which the decode follows
  from the step counts and the dims, and the numbering differs between a
  numerics run and the autopsy's bench.sh run, which load different
  signatures.

## Rules of thumb

- Keep both trees clean while a capture runs: the pool key goes dirty ("+")
  with uncommitted tracked changes and later probes compile new pools.
- Never run `build.sh` or GPU unit tests beside a capture (bench.sh refuses
  another CUDA process; a rebuilt library under a running capture mixes builds).
- Host mode: timings compare within one mode (`CI_HOST_MODE=formal` after
  `zsh -lic benchmark`, else diagnostic); the manifest records it.
- A failing performance cell within the iterations' spread is noise; rerun it.
