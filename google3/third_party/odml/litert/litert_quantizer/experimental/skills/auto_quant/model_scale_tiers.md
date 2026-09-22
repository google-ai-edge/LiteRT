---
name: aeq-model-scale-tiers
description: >-
  Classifies target models into scale tiers (S/M/L) by float model size and
  prescribes per-tier search budgets, validation strategies, and memory
  guardrails so the auto-quant loop remains tractable from tiny CNNs up to
  multi-billion-parameter LLMs.
---

# Model Scale Tiers: Making the Search Loop Tractable at Any Size

The canonical two-phase greedy loop in `experiment_runner.md` performs a
full-model side-by-side validation (`qt.validate()`) at every step. Each
validation holds BOTH the float and quantized graphs plus activation buffers in
RAM. This is trivially affordable for a 170 MB CNN and completely infeasible
for a multi-GB LLM. Before writing any code, classify the model and adopt the
matching tier strategy.

## 1. Tier Classification

Classify by the float `.tflite` file size on disk (the dominant cost driver
for validation memory and time):

Tier | Float Model Size | Typical Examples
:--- | :--------------- | :---------------
**S** | < 1 GB          | ISNET, U-Nets, MobileNet, small encoders, TTS vocoders
**M** | 1 – 4 GB        | Large vision transformers, Whisper-class STT, small LLMs (< 1B params)
**L** | > 4 GB          | Llama-class LLMs, multi-billion-parameter generative models

Also check the signature count: any model with multiple signatures (e.g.
`prefill` + `decode` in LLMs) inherits Tier L *procedural* rules regardless of
size, because per-signature validation multiplies the cost.

## 2. Per-Tier Search Budgets

Parameter                          | Tier S            | Tier M                | Tier L
:---------------------------------- | :---------------- | :-------------------- | :-----
Validation samples per step        | 8–32              | 4–8                   | 1–4 (short sequences for LLMs)
Full-model validations (total cap) | ~50               | ~20                   | ~8
Search direction                   | INT8 → INT4 (squash down) | INT8 → INT4 (squash down, batched) | **INT4 → INT8 (promote up, see §4)**
Greedy step size                   | 1 block per step  | Batched (see §3)      | Batched, mandatory
Intermediate tensor capture        | All tensors       | Sensitivity sweeps only | Sensitivity sweeps only, per-block
Store every `.tflite` variant      | Yes               | Yes                   | Baseline + Top 3 only (always keep all `.json` recipes + metrics)
Preferred protection mechanism     | `no_quantize` skips | 8-bit pins, then skips | Algorithm upgrades (GPTQ/Hadamard), then 8-bit pins; skips are last resort

**Smoke/calibration reserve**: treat ~10% of the validation cap (minimum 2
validations) as a reserve for environment smoke tests, backend/runtime
calibration, and re-measurement after harness fixes. Plan the greedy search
against the remaining budget. If the reserve is consumed by mid-search
methodology corrections (e.g. a stopping-metric backend fix invalidating
earlier measurements), archive the invalidated runs, document the overrun
explicitly in the exploration log, and continue — do not silently exceed the
cap.

## 3. Batched Squashing with Bisection (Tier M/L)

One-validation-per-block greedy search is unaffordable when each validation
takes many minutes. Replace it with batched commits:

1.  Rank candidate blocks by robustness exactly as in the canonical loop.
2.  Squash the top `B` candidates (start with `B = max(2, N // 4)` where `N`
    is the candidate count) in a SINGLE recipe update, then validate once.
3.  If the stopping metric holds: commit the whole batch, continue with the
    next batch.
4.  If it fails: **bisect** — revert half the batch (the least robust half),
    validate again, and recurse. This finds the fragile boundary in
    `O(log B)` validations instead of `O(B)`.
5.  Respect the total validation cap from §2. When the cap is reached, stop
    and report the best committed configuration.

## 4. Inverted Search Direction for Tier L: INT4-First

For Tier S/M models, the canonical loop starts at INT8 and greedily squashes
robust layers DOWN to INT4, because INT8 alone often already satisfies the
size bound. For Tier L models this direction is backwards and wastes the tiny
validation budget:

*   **INT8 rarely meets the deployment constraint.** A 7B-parameter model at
    INT8 weights is still ~7 GB — beyond most edge memory budgets. ~INT4 is
    the practical operating point for on-device LLMs, so treat INT4 weights as
    the DEFAULT TARGET, not the aggressive endpoint.
*   **Decode latency is memory-bandwidth-bound.** Every generated token
    streams the full weight set; halving weight bytes roughly doubles decode
    throughput. INT4 is the primary latency lever for LLMs, not just a size
    optimization.
*   **INT4 transformer weights are viable with the right algorithm** (GPTQ,
    Hadamard rotation — see `algorithms.md`), which is why the algorithm axis
    is explored first at this tier.

The Tier L loop therefore runs as follows:

1.  **Baseline**: ALL weight tensors at INT4 using **blockwise granularity**
    (`b32`/`b64`, e.g. `recipe.dynamic_wi4b32_afp32()`) — blockwise scales are
    the standard quality lever for INT4 LLM weights. Upgrade the algorithm if
    quality demands it (Hadamard rotation variants, GPTQ for linear
    projections). Pin known-sensitive tensors (e.g. per-layer embedding
    projections) to INT8 up front; note the reference LiteRT-LM Gemma recipes
    keep the main embedder at INT4 but per-layer embedding projections at
    INT8, so treat pins as candidates to be relaxed, not permanent.
    Activations remain float (dynamic) or INT8 — the INT4 target applies to
    weights only.
    *   **Per-signature recipes**: LiteRT-LM bundles split the model into
        components with separate signatures (embedder, prefill/decode). Build
        the recipe per component/signature (see the `gemma4_mixed48*` presets
        in AEQ's `recipe.py` as the reference structure) and pass
        per-signature `test_data` dicts to `qt.validate()`.
2.  **Data-free pre-screening (zero validations)**: BEFORE any validation,
    compute the per-block weight kurtosis ranking per
    `kurtosis_screening.md`. Pre-pin extreme-kurtosis blocks and the
    first/last decoder blocks (the empirically most sensitive positions) to
    higher precision in the initial recipe, and use the ranking to seed the
    fragility ordering.
3.  **Sensitivity sweep**: Run the one-shot cached sweep (§6) on this INT4
    baseline and rank decoder blocks by FRAGILITY (worst primary metric
    first), cross-checking against the kurtosis ranking. Where the two
    disagree, trust the validation-based ranking.
4.  **Greedy promotion ladder**: Batched (with bisection, §3), promote the
    most fragile blocks UP until the output stopping metric passes the bound.
    Promote along the **granularity ladder**, not straight to INT8: `int4
    b64 → int4 b32 → int8 channelwise`. Each ladder step costs only extra
    scale storage (~0.2–0.4 effective bits) instead of doubling the block's
    weight bytes, producing a much denser Pareto frontier. Also consider the
    zero-size-cost fix first: a Hadamard/GPTQ algorithm upgrade on the
    fragile block (see `algorithms.md`).
5.  **Profile mapping**: The three Pareto profiles from `experiment_runner.md`
    map to: **Compact** = pure INT4 baseline (+ pinned embedding/LM head),
    **Balanced** = minimal promotions needed to pass the bound, **Quality** =
    generous promotion of all flagged-fragile blocks to INT8. `no_quantize`
    float skips remain a last resort at this tier.
6.  **Optional INT2 demotion axis (aggressive size targets only)**: AEQ ships
    INT2 weight presets (`dynamic_wi2b32_afp32`, `dynamic_wi2b64_afp32`,
    channelwise and Hadamard-rotation variants), extending the granularity
    ladder downward: `int2 b32 → int4 b64 → int4 b32 → int8 channelwise`.
    If the INT4 baseline passes its bound with clear headroom and the user
    needs deeper compression, run a SHORT demotion pass: take the blocks
    that proved most robust during promotion (never the embedding table, LM
    head, or first/last decoder blocks), demote them to INT2 blockwise-32
    with a Hadamard/GPTQ algorithm upgrade, batched with bisection, gated on
    the same stopping metric. Expect a much higher revert rate than INT4 —
    budget at most ~2 validations for this axis and abandon it on the first
    failed bisection.
    *   **Runtime-support caveat**: XNNPACK's blockwise kernels (QB4W)
        cover INT4 only — there is no INT2 CPU kernel path, so INT2 ops
        fall back to reference/dequant execution. INT2 therefore buys
        SIZE, not CPU latency, and may regress latency versus INT4.
        Delegate support evolves, so verify what your installed runtime
        actually delegates at search time. The deployment-faithful stopping
        metric (see `experiment_runner.md`) automatically measures the
        fallback path's numerics, but you MUST benchmark latency per
        `latency_benchmarking.md` before recommending an INT2 profile, and
        state the delegation status in the report.

## 5. Memory Guardrails (All Tiers)

*   **Pre-flight estimate**: peak RSS ≈ `2 x float_model_size +
    activation_buffers`. Verify against `psutil.virtual_memory().available`
    before the first validation; abort to a smaller sample count or shorter
    sequence length if the estimate exceeds ~60% of available RAM.
*   **Strictly sequential validations** — never parallelize (see
    `experiment_runner.md`).
*   **`del` + `gc.collect()`** after every validation.
*   **Tier L extra**: prefer validating on the `decode` signature with a short
    fixed prompt over full `prefill` sequences; long sequence lengths multiply
    activation memory linearly.

## 6. Sensitivity Analysis at Tier L

Full-graph intermediate tensor capture on an LLM can produce tens of GB of
activation data. Constrain the sweep:

*   Run the sensitivity sweep ONCE on the baseline recipe, capturing
    intermediate tensors for a SINGLE short input, and cache the parsed
    per-tensor metric table to disk (`reports/sensitivity_baseline.json`).
    Re-use this table for ranking throughout the search instead of re-running
    intermediate capture.
*   Aggregate per-tensor scores to per-decoder-block scores (transformer
    blocks are the natural unit for regex targeting anyway — see
    `regex_targeting.md`). Rank and squash/promote whole blocks, not
    individual projections, unless a block sits exactly on the quality
    boundary.
*   Evaluate the embedding table and LM head as separate standalone candidates
    — they are usually the largest and most sensitive tensors.

## 7. Reporting Requirements

State the classified tier, the per-step sample count, the validation cap, and
the number of validations actually consumed in the final report so the user
can judge the thoroughness of the search.
