# Experimental Gemma 4 LiteRT-Based Tensor API Baseline (CPU)

<!-- disableFinding(LINE_OVER_80) -->

> **Temporary experimental folder:** This directory is temporary. Its purpose is
> to establish a CPU performance and numerical baseline for the **LiteRT-based
> Tensor API** Gemma 4 implementation (`CompiledModel` + XNNPACK delegate).
> Validated optimizations from this experiment will eventually be migrated into
> the backend-agnostic implementation in `tensor/examples/gemma4/`.

## What This Experiment Explores

This runner executes a two-signature (`prefill` and `decode`) Gemma 4 E2B/E4B
`.tflite` model on CPU via LiteRT `CompiledModel` (`LitertDynamicRunner`) with
XNNPACK delegation and persistent weight caching, while performing host-side
embedding lookups from an unpacked `.litertlm` published-weight bundle.

Key differences and trade-offs compared with upstream LiteRT-LM + XNNPACK:

- **Active-prefix and sliding-window KV slicing (`kv_bank.h`)**: Instead of
  running attention over the full pre-allocated KV capacity on every step,
  `ActiveKvBank` allocates static-capacity `[capacity, head_dim]` INT8 K/V
  buffers once at startup (64-byte aligned with 64 bytes of tail padding) and
  binds only the active prefix (`[0, length)` for global layers) or active
  sliding window (for local layers), aligned to 32-token boundaries. This makes
  TTFT and decode latency largely independent of maximum KV capacity.
- **Current limitation (copy on slice/concat)**: Without custom in-place TFLite
  KV mutation ops, slicing and concatenating inside the TFLite graph still
  copies the active prefix.
- **CPU-only dynamic shapes**: Resizing active KV and (optionally) prefill query
  row dimensions at runtime relies on CPU/XNNPACK dynamic shape support and does
  not target non-CPU accelerators.
- **Host embeddings & packed dynamic INT4 prefill**: Token and per-layer
  embedding lookups run on host using compact bundle storage
  (`bundle_loader.{h,cc}`), and prefill uses packed dynamic INT4 weights (which
  changes the original static QAT activation contract relative to upstream
  LiteRT-LM).

## KV Cache Buffer Transaction Lifecycle (`ActiveKvBank`)

`ActiveKvBank` manages static-capacity, token-major INT8 KV storage with
transactional appends across Gemma 4's KV owner layers (shared consumer layers
read their owner's views without appending again):

```text
Create() / Reset()
       │
       ▼
    [ Idle ]  (length = L, pending = 0)  ◄───────────────────────┐
       │                                                         │
       │ BeginAppend(N)                                          │ Abort()
       ▼                                                         │ (discards N,
  [ Appending ]  (length = L, pending = N) ──────────────────────┘  keeps L)
       │
       │ Append(owner, keys, values, head)  [once per owner/head, writes [L, L+N)]
       ▼
  [ All owners/heads appended ]
       │
       │ Commit()  (publishes length = L + N)
       ▼
    [ Idle ]  (length = L + N, pending = 0)
```

`Reset()` transitions from any state back to `Idle (length = 0)`. Buffers never
reallocate after `Create()`, and `Abort()`/`Reset()` update logical counts
without zeroing memory—callers must request fresh spans after either call.

## Baseline Phone Measurements (Gemma 4 E2B, CPU)

Measurements compare three CPU runners on identical forced-token prompts (4
threads, affinity `1e0` on Pixel 8 and `f0` on TECNO LJ9 / Samsung S25 Edge,
max prefill chunk 128, thermal-gated, 4 measured requests across 2 launches per
configuration, 64 forced continuation tokens per request):

- **Upstream LiteRT-LM**: Baseline production path (attention over full
  allocated KV capacity).
- **Tensor API / XNNPACK**: Direct native XNNPACK Tensor API runner.
- **Tensor API / LiteRT**: This experimental implementation (`runner`).

### Representative TTFT (ms) and Decode (ms/token)

All values are medians across 4 measured requests (capacity 8,448 for prompts up
to 8,192; capacity 16,896 for prompt 16,384).

| Phone | Prompt | Cap | Upstream LiteRT-LM (TTFT / Decode) | Tensor API / XNNPACK (TTFT / Decode) | Tensor API / LiteRT (TTFT / Decode) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Pixel 8 | 1 | 8,448 | 70.31 / 69.36 | 71.11 / 46.57 | 47.02 / 50.41 |
| Pixel 8 | 128 | 8,448 | 1,117.41 / 70.39 | 401.18 / 45.15 | 386.72 / 51.34 |
| Pixel 8 | 1,024 | 8,448 | 8,821.10 / 72.52 | 3,690.74 / 49.24 | 2,495.74 / 54.58 |
| Pixel 8 | 4,096 | 8,448 | 36,872.95 / 79.04 | 16,582.67 / 55.40 | 12,779.84 / 59.43 |
| Pixel 8 | 8,192 | 8,448 | 89,164.98 / 91.99 | 45,901.78 / 93.30 | 41,081.49 / 99.75 |
| Pixel 8 | 16,384 | 16,896 | OOM | 131,114.74 / 127.68 | 125,871.03 / 136.29 |
| TECNO LJ9 | 128 | 8,448 | 1,052.95 / 65.02 | 256.28 / 35.63 | 233.13 / 37.02 |
| TECNO LJ9 | 1,024 | 8,448 | 8,014.50 / 65.19 | 2,318.73 / 40.57 | 2,250.03 / 42.53 |
| TECNO LJ9 | 8,192 | 8,448 | 67,606.09 / 69.19 | 34,579.06 / 71.87 | 32,428.79 / 75.34 |
| Samsung S25 Edge | 128 | 8,448 | 542.92 / 40.05 | 146.52 / 20.67 | 126.01 / 21.41 |
| Samsung S25 Edge | 1,024 | 8,448 | 5,931.06 / 52.18 | 1,277.01 / 24.20 | 1,146.92 / 24.07 |
| Samsung S25 Edge | 8,192 | 8,448 | 47,083.34 / 52.34 | 23,449.06 / 48.28 | 22,067.88 / 51.11 |

### Capacity Sweep Invariance (Samsung S25 Edge, Prompt = 128)

Because attention only binds the active KV range, latency stays flat as maximum
KV capacity scales from 256 to 16,896:

| Capacity | Upstream LiteRT-LM (TTFT / Decode) | Tensor API / XNNPACK (TTFT / Decode) | Tensor API / LiteRT (TTFT / Decode) |
| ---: | ---: | ---: | ---: |
| 256 | — | 132.28 / 20.89 | 126.28 / 21.49 |
| 2,048 | 299.29 / 25.20 | 136.36 / 20.54 | 126.18 / 21.40 |
| 8,448 | 542.92 / 40.05 | 146.52 / 20.67 | 126.01 / 21.41 |
| 16,896 | — | 160.07 / 21.00 | 126.42 / 21.39 |

### Memory & Numerical Parity Notes

- **Process Peak RSS**: Across prompts 8–8,192 at 8,448 capacity, this runner's
  median peak RSS is ~1,261–1,397 MiB on all three phones (vs. ~3,448–4,094 MiB
  for upstream LiteRT-LM and ~2,133–2,319 MiB for native Tensor API / XNNPACK).
  *Important caveat*: this RSS difference is **not** an isolated measurement of
  the LiteRT wrapper cost—it reflects the selected runner configurations (the
  LiteRT runner loads an already exported `.tflite` graph and file-backed packed
  cache with compact host embeddings and OS page reclamation, whereas the native
  runner keeps source weights resident and builds an in-memory packed cache).
- **Numerical Parity**: Exact 65/65 forced-token argmax agreement with the
  native Tensor API / XNNPACK runner across all prompt lengths (1 to 16,384) on
  all three phones, with 0 repeated-run divergence. Differences versus upstream
  LiteRT-LM on multi-token prompts (47–64/65 agreement) come from packed dynamic
  INT4 prefill changing the original static QAT activation contract.

## Targets

- `:kv_bank_test` — unit tests for `ActiveKvBank` transaction semantics,
  alignment, and multi-head views.
- `:runner` / `:runner_lib` — CPU `CompiledModel` runner and benchmarking binary
  (`main.cc`, `runner.h`, `kv_bank.h`, `host_buffer.h`).
- `:bundle_loader` — unpacked `.litertlm` published-bundle and KV spec loader
  (`bundle_loader.{h,cc}`, `kv_cache.h`).
- `:graph` — Gemma 4 E2B Tensor API graph layers, dynamic RoPE slicing, and
  mobile-CT static INT8 linear layers (`gemma4_graph.h`, `transformer.h`,
  `attention.h`, `feed_forward_network.h`, `mobile_fully_connected.h`,
  `bundle_matched_ops.h`, `rope.h`, `active_graph_context.h`).
- `:model_builder` — two-signature (`prefill`/`decode`) TFLite FlatBuffer graph
  authoring via `AddGemma4Signatures` (`model_builder.{h,cc}`).
