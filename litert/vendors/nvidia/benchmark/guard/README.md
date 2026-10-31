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
within the first eight tokens, non-finite logits and a repetition loop fail.
The KL over the first four steps warns at 1e-3 and fails at 1e-2, the mean KL
warns at 1e-3 and fails at 1e-1, a |Δlogit| warns at 1.0 and fails at 8.0, the
top-K overlap warns under 0.95 and fails under 0.8 (`nvguard.toml`).
The prompts: three coding problems in LiveCodeBench's format, a reasoning
question, and the benchmark's 8k document (quick tier); the 30k document
(standard); the 128k document for the 12B (long). A capture of the same build
twice gives zero deltas: the engines are deterministic.

**Where a change enters.** Each numerics probe also runs with
`LITERT_NVIDIA_DISPATCH_DUMP_OUTPUTS=<dir>:8:16`, so the dispatch writes every
output of the first eight steps of every partition whole (outputs up to
16 MB; the in-place caches, marked `state`, at every step of a context that
emits nothing else, the prefill's chunks, and at the first step of the
others), with an index in production order that also records each step's
input tokens, which tell the prefill (a chunk's count: 128 or 1024 here)
from the decode (one) and the verify step (a few).
`localize --baseline A B` compares two captures' dumps in that order and names
the first tensor that differs (its context, function and input tokens, step
and output, how many elements moved and how far, how many of that step's
outputs differ), the largest move, and the KL of the first logits rows over
the whole vocabulary (the top-32 of DUMP_IO is a sample of it). An output
above the cap keeps its index line without a file, and the localization says
how many such outputs a probe had (the caches of a long context: the 12B's
128k caches are 33 MB each, so at that tier only the decode's outputs are
localized). A partition is as
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

**Performance.** `bench.sh` latency runs of the `short` (the record's 1k),
`8k` and `16k` workloads with few iterations, per model; `--mtp` adds the
plain and MTP text runs of those workloads (acceptance, tokens per round).
Candidate/baseline ratios below 0.97 warn, below 0.93 fail. From the measured
contexts a cost model per model (decode step time a + b·context; prefill time
c0 + c1·N + c2·N²) predicts the other contexts (30k, 128k); with `--record`
the measured change ratio, interpolated in context, is applied to the
record's full curve to say what the candidate would measure there. `predict
NAME --record FILE` also reports the predictor's own error on the record
(fitted on the record's measured workloads, predicting its other contexts).

**Conversations.** One numerics prompt (`doc_8k`, 256 tokens) decoded
greedily in two conversations of one process, the CLI's benchmark
iterations, without the dumps; every conversation's text must be the first
one's. A later conversation that parts depends on what an earlier one left
behind (a cache's stale rows, a workspace, a buffer read before it is
written), which no probe of one conversation sees: `check` fails it, or
warns when the baseline's parted too (a dependence the baseline had
already). Known failing case: the INT8 GEMM chose its outlier dims over a
prefill chunk's padding rows, whose attention averages the cache, a
previous conversation's included; on 413f8dfda the 12B's second text parted
at character 158 and the E4B's at 736, and with
`LITERT_NVIDIA_GEMM_INT8_REAL_ROWS=1` (962697400) every conversation agrees.
About 15 to 25 seconds per model; `--no-conversations` skips it.

**Memory.** bench.sh's memory metric of the short workload, plain and MTP,
per model: the most device memory the run allocated (the dispatch's memory
profile's largest device-wide use less its first, once the CUDA context
exists) and its peak RSS. These are allocations, not nvidia-smi's sampled peak
(which moves by 100 MiB between two runs of one build): a same-build pair
agrees to a few MiB. `check` fails growth of 256 MiB or 0.25 GiB and warns
from 64 MiB or 0.1 GiB (`[probes.memory]`); a run that compiled its engines
in the process holds the compiler's memory and is run again from the warm
cache. Known failing case: the step feed's tables under MTP before
LiteRT-LM 12ed7c4b, 1,254 MiB and 1.15 GiB on the E2B. About 15 to 40
seconds per model and mode; `--no-memory` skips it.

**MTP acceptance.** With `--mtp` each numerics prompt also decodes greedily
with the drafter, and the run keeps its accepted and drafted tokens. `check`
and `compare` pool them per model (accepted over drafted, across the
prompts) and judge the pool: a drop of 2 points with more prompts falling
than rising warns, a drop of 5 points with two thirds of them falling
fails (`[acceptance]` in `nvguard.toml`); fewer than three prompts paired
are reported, not judged. One prompt's acceptance moves with its text,
which any numerics change parts, so a single prompt never decides. The
performance probes' acceptance stays reported only: their texts are the
model's sampled ones. (Until 2026-10-29 no acceptance was judged at all,
and the INT8 GEMM's drop on the E4B's 16k document, 52.5% to 44.2%, passed
unnoticed.)

**Provenance.** A capture records the build its binaries came from
(runtime-libs/BUILD_ID, build.sh's stamp) and refuses to run when it is not
the trees' build id; it records clock_check.py's report on the host's
clock and skips the timed probes (performance, autopsy) when the clock is
unfit, unless `--allow-unfit-clock`, which notes them as diagnostic. The
CLI's timers read the monotonic clock, which a kernel tick off 10,000 us
scales: on cuda-wsl from 2026-10-20 every rate read 1 to 5 percent low.

**A runtime knob's A/B.** `--env NAME=VALUE` (repeatable) runs every probe
of a capture with that environment, in a model pool of its own (so a
compile-time knob compiles its own engines), and the manifest records it:
`capture --name fp16 --env LITERT_NVIDIA_GEMM_INT8=0` beside the plain
capture is the INT8 GEMM's A/B. Names the probes set themselves, and
env.sh's pool variables, are refused.

**Kernels.** One run of the `short` workload in the dispatch's own-graph
mode with `LITERT_NVIDIA_DISPATCH_CUDA_GRAPH_AUTOPSY` and `_DOT`: every
kernel node of each function's first step graph timed alone (warm and cold)
and named through the graph's DOT (demangled with `c++filt` where it is).
Two captures' autopsies diff by function and by kernel, so a performance
change is attributed to the kernels that moved. The autopsy times kernels
one at a time: its sums are the kernels' own times, not the step's, they are
noisy (between captures of one build, kernels nobody touched moved by up to
about 85 µs over a step's launches on the E2B), and it is a diagnostic,
never a table cell of the measurements: a kernel's Δ means something when it
stands well above what a same-build pair shows and agrees with the cells.

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

- A grammar that ends inside an MTP round: the text probes run one regex,
  in the default upload mode and, for the constraint, in the immediate-upload
  mode too (`LITERT_NVIDIA_DISPATCH_DEFERRED_UPLOAD=0`). The grammar case was
  checked by hand with the same A/B the text probes automate (the dispatch's
  device samplers reading a constraint mask the host had written into the
  logits was found that way).
- The samplers at exact ties: the device keeps the lowest indices at a tie
  on the k-th logit, the CPU heap does not say; a parting of the sampled
  texts warns for that reason instead of failing.
- Memory beyond the memory probe: one workload (short), one conversation;
  what a long context or a later conversation allocates is not in it, and
  the device figure is device-wide (another process's allocation during the
  run moves it).
- The step feed: the numerics probes dump every step's logits, and the
  dispatch turns the feed off under the dumps, so the plain decode path
  the feed serves (the default) is not in the numerics; the text probes run
  MTP, where the feed is inert.
- Across conversations, beyond the conversation probe: it decodes one
  prompt, greedily, in two conversations, plain decoding only, and compares
  texts, not logits; a dependence that parts the text after 256 tokens,
  needs a third conversation, or shows only in sampled or MTP decoding is
  not seen. The other probes each decode one conversation in a fresh
  process, while the benchmark measures later ones.

## What the gates showed (2026-10-27, E2B)

- The chain against round 5's `final` capture (build 3c590a6e1-95cd9c0b):
  the ten greedy probes (five prompts, plain and MTP) bit-identical, every
  performance cell within four percent (the 8k decode cells up by 2.7 and
  3.9 percent, beyond their spreads), acceptance equal.
- A bit-identical rebuild captured under another name and localized against
  the first: 76 tensors paired per probe, none differing, the texts
  identical, PASS. The autopsy's sums moved by up to 21 µs per kernel
  between the two, and by up to about 85 µs on untouched decode fusions
  between the clean capture and the scratch ones: the sums are noisy.
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
- A divergence past the first prefill chunk (the int8 cache write's scale
  times 1.001 from write position 1024 on, the branch
  `scratch-divergence-tail`, against the build 44cdec619-95cd9c0b): the
  four short prompts identical in every tensor; the 8k document named at
  `c1 tensorrt_partition_1 (1024 input tokens) step 3`, its first cache
  output differing from row 2047 on, chunks 1 and 2 identical. Step 3, not
  2: by the dumps chunk 1 writes rows 0 to 1023, chunk 2 rows 1023 to 2046
  (the first chunk's last row written again), chunk 3 rows 2047 to 3070 and
  chunk 4 rows 3071 to 4094, so chunk 2's write position is under the gate
  and chunk 3's, 2047, is the first at or past it.
- A build against the one before it (the dump's index fields only): every
  tensor identical, none differing of the 76 to 334 paired per probe (136
  for each of the three code prompts, whose prefill takes three chunks; 76
  for `reason`, one chunk; 286 for the 8k document; 184, 184, 184, 124 and
  334 for their MTP runs), the three constraint variants (device, host,
  immediate) identical and matching the regex, the sampled texts identical.
- The functions in the dumps are the invocation contexts' (`c<context>
  tensorrt_partition_<n>`), with the step's input tokens beside them: a
  prefill chunk's count (128 for the short prompts, which LiteRT-LM prefills
  through the `prefill_128` signature: three chunks for the code prompts of
  315 to 360 tokens, one for `reason`; 1024 for the documents), a decode step's 1, a verify step's few.
  The numbering differs between a numerics run and the autopsy's bench.sh
  run, which load different signatures.

## Rules of thumb

- Keep both trees clean while a capture runs: the pool key goes dirty ("+")
  with uncommitted tracked changes and later probes compile new pools. A
  dirty build id is the same string for every edit of the same heads, so it
  is refused as the default capture name: name a dirty tree's capture, and
  use a new name after each edit, or the earlier edit's probes are kept.
- Never run `build.sh` or GPU unit tests beside a capture (bench.sh refuses
  another CUDA process; a rebuilt library under a running capture mixes builds).
- Host mode: timings compare within one mode (`CI_HOST_MODE=formal` after
  `zsh -lic benchmark`, else diagnostic); the manifest records it.
- A failing performance cell within the iterations' spread is noise; rerun it.
