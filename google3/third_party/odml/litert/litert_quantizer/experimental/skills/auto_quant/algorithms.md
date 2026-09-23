# AEQ Algorithms Registry & Selection Guide

This document serves as a registry for all the quantization algorithms offered
in AI Edge Quantizer (AEQ), corresponding to the `AlgorithmName` enum in
`algorithm_manager.py`, and guides you on WHEN to reach for each one.

## Supported Algorithms

*   **NO_QUANTIZE**: Skips quantization for the specified operations.
*   **MIN_MAX_UNIFORM_QUANT**: Standard uniform quantization using min/max
    bounds for calibration.
*   **FLOAT_CASTING**: Casts operations to float.
*   **DEQUANTIZED_WEIGHT_RECOVERY**: Reconstructs/recovers weights during
    dequantization to optimize model quality.
*   **OCTAV**: Optimizes activation ranges (Optimal Clipping and Tuning of
    Activation Variables).
*   **HADAMARD_ROTATION**: Implements Hadamard rotation using runtime custom
    ops.
*   **DECOMPOSED_HADAMARD_ROTATION**: Implements Hadamard rotation entirely
    using mathematically equivalent decomposed standard ops.
*   **MSE**: Computes quantization parameters by minimizing Mean Squared Error
    (MSE).
*   **GPTQ**: Specifically tailored post-training quantization method (Accurate
    Post-Training Quantization for Generative Pre-trained Transformers).

When configuring a quantization recipe via `quantizer.py`, specify the algorithm
using `AlgorithmName.<ALGORITHM>`.

## Algorithm Selection Guide

Algorithm choice is a **third search axis** alongside bit-width and layer
scoping (see `experiment_runner.md`), and it costs zero size budget — an
algorithm upgrade recovers quality at the SAME file size, unlike a
`no_quantize` skip or an 8-bit pin. Consult this matrix when a configuration
fails its quality bound:

Situation                                            | Recommended Algorithm | Rationale
:---------------------------------------------------- | :-------------------- | :--------
Default starting point, any model                     | `MIN_MAX_UNIFORM_QUANT` | Cheapest, no calibration search; establishes the baseline.
Weight tensors with heavy outliers (min/max stretched) | `MSE`                 | Fits quantization parameters to the bulk of the distribution instead of the extremes.
Transformer / LLM weight projections at INT4          | `GPTQ`                | Second-order weight compensation dramatically reduces INT4 error on large linear layers.
Outlier-dominated transformer activations/weights     | `HADAMARD_ROTATION` (or `DECOMPOSED_HADAMARD_ROTATION` if the runtime lacks custom ops) | Rotation smears outliers across the whole vector, making uniform quantization viable.
Aggressive activation clipping needed (static quant)  | `OCTAV`               | Optimally trades clipping error against resolution error.
Layer unrecoverable at any bit-width/algorithm        | `NO_QUANTIZE`         | Last resort — costs full FP32 storage for that layer.

**Priority rules by model scale (see `model_scale_tiers.md`):**

*   **Tier S/M (CNNs, small encoders)**: `MIN_MAX_UNIFORM_QUANT` with skip
    lists is usually sufficient. Try `MSE` on individual fragile layers before
    resorting to `no_quantize`.
*   **Tier L (LLMs / large transformers)**: The algorithm axis usually
    dominates the skip-list axis in quality-per-MB. Explore `GPTQ` and
    `HADAMARD_ROTATION` for INT4 weight quantization BEFORE adding float
    skips, because a single skipped LLM layer costs hundreds of MB.

**API usage notes (empirically verified):**

*   Configure algorithms by passing `algorithm_key` through the recipe helpers
    (e.g. `recipe.dynamic_wi4c_afp32(algorithm_key=AlgorithmName.GPTQ)`) or
    via `qt.add_dynamic_config(..., algorithm_key=...)`. Do NOT call
    `update_quantization_recipe` with a bare `algorithm_key` and default
    `op_config` — the default carries no weight config and fails with "Weight
    tensor quantization is required".
*   **GPTQ requires calibration data**: check `qt.need_calibration` and, if
    true, run `qt.calibrate(calibration_data=...)` (same per-signature dict
    format as `validate()` test data) and pass the result to `qt.quantize()`.
    Follow `calibration_data.md` for dataset rules.
*   `DECOMPOSED_HADAMARD_ROTATION` inserts extra decomposed ops; on small
    models the added graph overhead can exceed the quantization savings —
    check the exported file size, and prefer `HADAMARD_ROTATION` (custom op)
    when the target runtime supports it.

**Search discipline**: Treat each algorithm variant as a distinct Pareto point
with its own artifact names (use the `[algorithm]` component from
`file_naming.md`, e.g. `dynamic_int4_selective_gptq_ver1`). Do not silently
swap algorithms mid-loop without recording the variant.
