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
`$LITERT_BENCH_ROOT/guard/<name>/against_<baseline>.md` (and `.json`).

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

## Layout

- `src/env.rs`: env.sh's paths, the build id, a pool's environment, through bash.
- `src/runner.rs`: one numerics probe (the CLI) or one performance probe (bench.sh).
- `src/capture.rs`: a resumable capture under `$LITERT_BENCH_ROOT/guard/<name>`.
- `src/numerics.rs`, `src/perf.rs`: the metrics, the cost model, the verdicts.
- `src/report.rs`: the comparison and its markdown.
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

- Constrained decoding and the penalties: the probes decode unconstrained
  prompts. The dispatch's device samplers reading a constraint mask that the
  host had written into the logits (deferred upload) was found by hand with
  a same-seed A/B of the MTP verifier (`LITERT_NVIDIA_MTP_GPU_SAMPLING=0`
  against the default) under `--constraint_regex` in the CLI's session mode
  (`--use_session=true --async=false`), on a regex without spaces: with this
  LiteRT-LM and Gemma 4's tokenizer a regex with spaces fails on every
  backend ("token ▁\" doesn't satisfy the grammar").
- Sampled (top-k) text: the probes are greedy; the device and CPU samplers
  draw the same stream from the same uniforms but part at exact fp16 ties.

## Rules of thumb

- Keep both trees clean while a capture runs: the pool key goes dirty ("+")
  with uncommitted tracked changes and later probes compile new pools.
- Never run `build.sh` or GPU unit tests beside a capture (bench.sh refuses
  another CUDA process; a rebuilt library under a running capture mixes builds).
- Host mode: timings compare within one mode (`CI_HOST_MODE=formal` after
  `zsh -lic benchmark`, else diagnostic); the manifest records it.
- A failing performance cell within the iterations' spread is noise; rerun it.
