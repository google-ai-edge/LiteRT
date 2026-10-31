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
