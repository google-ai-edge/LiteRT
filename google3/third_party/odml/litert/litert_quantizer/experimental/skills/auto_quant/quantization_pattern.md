# Quantization Exploration Patterns

Empirical patterns observed when compressing `.tflite` models via AI Edge
Quantizer. These heuristics guide the sweep order and help interpret compression
vs. quality results.

## Pattern 1: Dynamic Quantization is often the most practical starting point

Dynamic Quantization (e.g. `recipe.dynamic_wi8_afp32()`) quantizes weights only
(e.g. at 8 bits) but handles the real-time inference activations in float format
automatically at the kernel level.

*   **Pros**: Significant size reduction (usually ~4x for INT8) and memory
    usage, without requiring any calibration dataset. It is strongly recommended
    as the baseline starting point for CPU/GPU targets.
*   **Cons**: Can be slightly slower on some highly optimized NPUs compared to
    static quantization, and the on-the-fly dequantization introduces unique
    computational noise.

**Implication**: Start tuning with dynamic quantization. Only transition to
fully **Static Quantization** if your target hardware explicitly lacks integer
processing capabilities for partial graphs or if NPU latency bounds are
exceedingly strict (and you have representative calibration data available).

## Pattern 2: Validation metrics are a strong signal for evaluating quantization recipes

Tracking the difference between original signals and compressed outputs using
the appropriate mathematical metric (quantization noise prediction) predicts
fidelity drops cleanly.

*   **Layer Sensitivity and Mixed-Precision**: Not all layers respond to
    quantization equally. If aggressive quantization applied to a layer produces
    stable, favorable metrics based on modality (e.g. high SNR, low KL
    divergence), that layer is robust. If the metrics plummet, the layer is
    heavily sensitive. Instead of a single recipe, you can sweep layer-by-layer
    metrics to apply **mixed-precision**: aggressively quantizing robust layers
    and protecting degraded layers in higher precision.
*   **Comparing Algorithms**: If choosing between algorithms (like standard
    Min-Max, GPTQ, etc.), tracking layer-wise metrics allows algorithm
    evaluation offline directly on weights/activations without needing thousands
    of prompt-based evaluations.

**Implication**: Rely heavily on `quantizer.ValidationErrorMetric` outputs
defined in `error_metric_selection.md` to debug why accuracy dropped, and to
identify specifically which layers are fragile.

## Pattern 3: Navigating the Pareto Curve (Mixed-Precision)

Instead of relying on a completely different algorithm like `weight_only` when
accuracy drops, explore the **Pareto Curve** (Quality vs. Size) by mixing bit
depths dynamically.

Start with a baseline `dynamic_wi8_afp32()` and find which sensitive layers to
skip entirely (`no_quantize`). Then, for the non-skipped layers, do not treat it
as simply "8-bit or 4-bit for everything."

*   **Quality Profile**: Uses `num_bits: 8` for all non-skipped layers.
    (Maximum fidelity within the size bound)
*   **Compact Profile**: Aggressively pushes all non-skipped layers to
    `num_bits: 4`. (Smallest model within the error tolerance)
*   **Balanced Profile**: Identifies which layers are robust to 4-bit by
    comparing the Quality and Compact profiles' layer-wise SNR. Layers that
    don't degrade are pushed to `num_bits: 4`, leaving moderately sensitive
    ones at `num_bits: 8`.

**Implication**: Explore intermediate points between your least aggressive
(Quality) and most aggressive (Compact) bounds by iterating layer-wise
precision combinations (`num_bits: 8` vs `num_bits: 4`) via `op_config` on the
non-skipped layers. Build a true Pareto curve instead of just flipping to
weight-only algorithms.

## Pattern 4: Granularity is the biggest scale lever

The more independently each weight group can be represented (specifically the
scaling values mapping integer domains), the less numerical degradation occurs.

*   **TENSORWISE**: One uniform scale/zero-point for an entire tensor. (Maximum
    compression, maximum loss chance).
*   **CHANNELWISE**: A distinct scale per output channel/feature map row.
*   **BLOCKWISE (`b32`/`b64`)**: A distinct scale per block of 32 or 64
    weights within a channel. (Finest granularity; the standard quality lever
    for INT4 LLM weights, at the cost of extra scale storage.)

**Implication**: Start with `CHANNELWISE` granularity (which is the default in
8-bit configs). Only ever drop to `TENSORWISE` if you are fighting for the last
raw megabytes on memory-constrained microcontrollers. Conversely, when INT4
weights degrade quality on large transformer layers, move UP to `BLOCKWISE`
granularity (e.g. `recipe.dynamic_wi4b32_afp32()`) before retreating to 8-bit
— the reference LiteRT-LM recipes for Gemma-class models use exactly this
(mixed 4/8-bit with b32/b64 blockwise INT4 and optional Hadamard rotation).

## Pattern 5: Small bit-widths (INT4) amplify scale/zero-point choices

When pushing beyond 4x compression into 8x (e.g., `recipe.dynamic_wi4_afp32()`),
the difference between **Symmetric** (scale only, 0-point anchored) and
**Asymmetric** (scale + 0-point shift) becomes stark. With INT4, we only have 16
literal bins to map weights into. If the original weights are skewed positively,
symmetric mapping wastes half the integer domain representing theoretically
negative weights that don't exist.

**Implication**: At INT8, symmetric is often perfectly acceptable. At INT4,
ensure `symmetric=False` defaults are preserved to capture the weight
distributions faithfully during Weight-Only quantization
(`weight_only_wi4_afp32`).
However, for **Dynamic Range Quantization** (`dynamic_wi4_afp32` with runtime
integer computation), LiteRT integer compute kernels strictly require symmetric
weights (`symmetric: True`). When applying INT4 overrides to dynamic recipes,
invoke `qt.add_dynamic_config(..., num_bits=4)` directly via the python API, as
it automatically enforces supported kernel symmetry and prevents validation
failures. Do NOT try to change bit-depths by passing magic strings into
`algorithm_key`.

## Pattern 6: For transformers/LLMs, algorithms beat skip lists

For CNN-scale vision models, protecting a handful of fragile layers in float
(`no_quantize`) is cheap — the skipped layers are a few MB. For large
transformers the economics invert: a single skipped decoder projection can cost
hundreds of MB of FP32, and LLM weight matrices suffer from outlier channels
that plain min/max quantization handles poorly.

*   **Prefer algorithm upgrades over float skips**: `GPTQ` (second-order weight
    compensation) and `HADAMARD_ROTATION` (outlier smearing) recover INT4
    quality at ZERO size cost. See the selection guide in `algorithms.md`.
*   **Prefer 8-bit pins over float skips**: If a layer cannot survive INT4
    under any algorithm, pin it to INT8 before considering `no_quantize`.
*   **Embedding tables and the LM head** are typically the largest and most
    quantization-sensitive tensors in an LLM. Evaluate them separately and pin
    them to 8-bit rather than skipping them.
*   **INT4 is the default target, not the aggressive endpoint**: INT8 weights
    on a multi-billion-parameter model rarely fit edge memory budgets, and
    since decode latency is memory-bandwidth-bound, INT4 also roughly doubles
    generation throughput. Start from an all-INT4 baseline and selectively
    promote fragile blocks to INT8, rather than starting at INT8 and squashing
    down (see the inverted search direction in `model_scale_tiers.md`).

**Implication**: For Tier L models (see `model_scale_tiers.md`), search the
algorithm axis FIRST, the bit-width axis SECOND (INT4-first, promoting
upward), and use `no_quantize` only as a last resort.

## Pattern 7: Metric Mismatches (XNNPACK vs Reference)

You can call `.validate()` using `use_xnnpack=False` (mathematically pure
Reference kernels) or `use_xnnpack=True` (optimized Edge CPU kernel
simulation). Often, XNNPACK has highly optimized macro-blocks determining
mathematically fast but approximate outputs (e.g., using specialized
accumulators). If the model validates well on Reference but badly on XNNPACK,
your weights are likely inducing overflow/underflows within those accumulators.

> [!WARNING] **INT4 models REQUIRE `use_xnnpack=False` during validation.**
> Empirically verified: the XNNPACK delegate rejects INT4 weight tensors
> outright (`unsupported datatype (INT4) ... in XNNPACK delegate`, delegate
> fails to prepare), so any `validate()` call on an INT4/mixed-INT4 recipe
> with XNNPACK enabled fails before producing metrics. Always pass
> `use_xnnpack=False` when validating INT4 configurations. (Note some
> releases default `use_xnnpack=True`.)

This backend split is exactly why `qt.validate()` must NOT serve as the
search's commit/revert gate: INT8 candidates would be judged by XNNPACK and
INT4 candidates by reference kernels, making their numbers incomparable. Use
`validate()` for per-tensor sensitivity sweeps only, and gate commits on the
deployment-faithful stopping metric (exported model under the default LiteRT
runtime vs cached FP32 outputs — see `experiment_runner.md`).

## Pattern 8: Tiny output-adjacent tensors can dominate output error

Layer size and layer importance are unrelated. The final output projection
(and side/aux heads) of dense-prediction models are often just a few KB of
weights, yet their quantization noise lands DIRECTLY on the model output with
no downstream layers to average it away. Empirically (observed in a
U-Net-style dense-prediction search), skipping one few-KB final conv removed
the vast majority of the total quantized-output MSE — a larger improvement
than any multi-MB decoder-stage skip, at effectively zero size cost.

**Implication**: Always trial `no_quantize` on small output-adjacent tensors
FIRST during protection search (see the cost-aware ordering rule in
`experiment_runner.md`), and make sure statistical size filters (e.g. the
kurtosis `min_elements` threshold) never remove them from the protection
candidate list.


