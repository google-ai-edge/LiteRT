# Experiment Runner Guidelines

This document describes principles for writing experiment sweeps to benchmark
different quantization recipes using AI Edge Quantizer.

Unlike PyTorch graph mode quantization that operates on `nn.Module` objects in
memory, AI Edge Quantizer operates by taking a `.tflite` model, applying a
recipe, and exporting a new quantized `.tflite` model. Because models can be
large, memory management and state isolation are important.

> [!IMPORTANT] Before designing your loop, read
> [`model_scale_tiers.md`](model_scale_tiers.md) and classify the model into a
> scale tier (S / M / L). The tier determines how many full-model validations
> you can afford, how many samples to use, and whether you must batch your
> search steps. The loop below describes the canonical Tier S algorithm;
> Tier M and Tier L modify it as described in the scale tiers guide.

## Guidelines for Experiment Sweeps

1.  **State Isolation & Memory Management:** Always re-instantiate the
    `Quantizer` object inside your sweep loop for each new configuration. This
    ensures that memory is cleaned up correctly and quantization state does not
    bleed across iterations.
    *   **Prevent Memory Overload & Workstation Crashes**: Never run multiple
        `qt.validate()` calls concurrently or launch parallel model validation
        sweeps (e.g. via multithreading, multiprocessing, or concurrent tasks).
        Running multiple validations concurrently creates massive simultaneous
        memory allocations that can exhaust system RAM and crash the
        workstation.
    *   **Sequential Execution**: Always execute validations strictly **one at a
        time** (sequentially). Wait for one model's validation to finish before
        starting the next configuration sweep.
    *   **Deployment-faithful stopping metric (CRITICAL)**: Do NOT use
        `qt.validate()` as the Phase 1/2 commit/revert gate. Its backend
        differs by recipe (XNNPACK rejects INT4 tensors, forcing
        `use_xnnpack=False` reference kernels — see `quantization_pattern.md`
        Pattern 7), so INT8 and INT4 candidates get judged by different
        kernels than they deploy on and their numbers are NOT comparable
        across phases. Instead, compute the stopping metric by running each
        **exported `.tflite` under the default LiteRT `Interpreter`** (which
        applies the same delegation a real deployment gets) and comparing its
        outputs against **cached FP32 baseline outputs** on the same fixed
        sample set. Cache the FP32 outputs once; every candidate then costs a
        single quantized-model inference pass. Reserve `qt.validate()` for
        the per-tensor sensitivity sweeps only (where per-tensor
        intermediate metrics are needed; pass `use_xnnpack=False` on
        INT4-containing recipes there).
    *   **Validation API caveats (empirically verified)**: Multi-metric
        evaluation in a single `validate()` call exists only in newer
        releases; pip releases up to at least 0.8.0 accept a single metric
        string (e.g. `error_metrics="kl_divergence"`). Check
        `inspect.signature(quantizer.Quantizer.validate)` at runtime and, on
        single-metric versions, budget one validation per metric or prioritize
        the stopping metric.
    *   **RAM Pre-flight Check**: Before the first validation, estimate the
        peak footprint (roughly 2x the float model file size plus activation
        buffers) and compare it against available system memory (e.g. via
        `psutil.virtual_memory().available`). If the estimate exceeds ~60% of
        available RAM, drop to a smaller sample count or switch to the Tier L
        strategy in `model_scale_tiers.md`.
    *   **Explicit Garbage Collection**: Call `import gc; gc.collect()` and
        clear unneeded variable references (`del qt`, `del validation_results`)
        after every validation run and before instantiating the next `Quantizer`
        object.
    *   **Sample Dataset Size**: Limit the validation dataset sample count /
        batch size during iterative exploration sweeps to maintain a low RSS
        memory footprint. Use the per-tier sample budgets from
        `model_scale_tiers.md`.
2.  **The Conservative Two-Phase Search Loop:** Selective quantization follows a
    conservative, two-phase process.

    > [!IMPORTANT] **Search direction depends on scale tier.** The loop below
    > (start at INT8, greedily squash robust layers DOWN to INT4) applies to
    > Tier S/M models. For Tier L models (LLMs), the direction INVERTS: INT4
    > weights are the default target (INT8 rarely fits edge memory budgets,
    > and decode latency is memory-bandwidth-bound, so INT4 also ~doubles
    > decode throughput). Start from an all-INT4 baseline with the best
    > algorithm and greedily PROMOTE fragile blocks up to INT8 — see the
    > "Inverted Search Direction for Tier L" section of
    > `model_scale_tiers.md` for the exact procedure. The memory-safety,
    > precedence, and artifact rules in this file apply to both directions.

    *   **Phase 0 (Free Pre-Screening)**: Before ANY validation, compute the
        data-free weight kurtosis ranking per `kurtosis_screening.md`. It
        seeds the fragile-candidate list and can pre-pin obvious offenders in
        the first recipe, saving validation budget in both phases.
    *   **Phase 1 (Finalize Skips via Output-Metric Quality Guarantee)**:
        Conduct sensitivity analysis using the model's primary metric (e.g.,
        SNR for vision models, KL divergence for LLMs) on the baseline to
        identify fragile candidate layers/blocks, cross-checked against the
        Phase 0 kurtosis ranking. Incrementally add candidate
        sensitive blocks to the `no_quantize` skip list and validate after each
        addition using the **modality-appropriate stopping metric** on the
        model's output tensor (see the "Stopping Metric per Modality" section
        of `error_metric_selection.md` — output MSE for dense vision and
        regression, output KL divergence for logit-producing models, cosine
        similarity for embedding models). Continue iterating and adding
        candidate fragile layers to the skip list until the stopping metric
        reaches an acceptable quality threshold. Once the quality criteria are
        satisfied, **freeze the skip-layers set** and proceed to Phase 2.
        *   **Cost-aware protection ordering (CRITICAL)**: Do not order
            protection trials by fragility alone — order them by expected
            **bytes spent per unit of stopping-metric error removed**. A
            fragility-only order can commit a multi-MB skip before
            discovering a near-free one, wasting size budget that a later
            ablation must claw back. In particular, ALWAYS include and trial
            **small output-adjacent tensors first** (final output
            projections, side/aux heads — often just a few KB): empirically
            they can dominate the quantized output error (in one observed
            dense-prediction search, a final conv of a few KB accounted for
            the vast majority of total output MSE) while costing
            essentially nothing to skip. Do not let a
            minimum-weight-size filter drop these tensors from the candidate
            list; that filter is for kurtosis statistics, not for protection
            candidacy.
        *   **Large-model caveat**: For Tier L models (LLMs), a `no_quantize`
            skip costs hundreds of MB of FP32 per layer. Prefer *pinning
            fragile layers to 8-bit* or *upgrading their algorithm* (e.g. GPTQ
            or Hadamard rotation, see `algorithms.md`) over float skips, and
            reserve `no_quantize` for truly unrecoverable layers.
    *   **Phase 2 (Iterative Greedy INT4 Squashing on Remaining Layers)**: With
        the `no_quantize` skip-layers set strictly finalized and frozen,
        optimize the remaining non-skipped layers using an **iterative greedy
        search** to prevent inter-layer noise accumulation. You MUST adhere to
        the skip-layers set from Phase 1 and only apply the greedy INT4 search
        to the remaining non-skip layers. The greedy search is conducted as
        follows:
        1.  Rank all remaining non-skipped layers/blocks by relative robustness
            using sensitivity analysis with the primary metric (e.g., SNR for
            image problems). Strictly exclude purely weightless operations
            (such as activation functions, normalization routines, pooling
            operations, reshaping steps, and tensor concatenations) from
            candidate lists. Exclusively target parameterized neural layers
            that contain trainable weights (such as convolutions, attention
            projections, and dense linear transformations) to ensure every
            mixed-precision iteration actively compresses physical parameter
            arrays and reduces disk footprint.
        2.  Iteratively process candidate robust blocks from most robust to
            least robust:
            -   In each step, squash the next candidate robust block to 4-bit
                (`num_bits: 4` via `add_dynamic_config`).
            -   Validate the model checking the total-model **stopping metric**
                and the primary metric.
            -   If the stopping metric remains within acceptable bounds, commit
                the 4-bit assignment and proceed to the next candidate block.
            -   If squashing a block pushes the stopping metric out of bounds,
                **revert that block to 8-bit and continue** with the next
                candidate. Do NOT halt the entire phase on the first failure —
                a single fragile block should not end the search. Only halt
                Phase 2 early after `K` consecutive reverts (default `K = 3`),
                which indicates you have descended into the fragile region of
                the robustness ranking.
        3.  **No-op commit detection**: after every commit, compare the
            exported model's byte size AND stopping metric against the
            previous committed step. If both are identical, the new rule was
            a no-op (typically fused-name regex cross-capture — see
            `regex_targeting.md` §3): log it as such, do not count the block
            as independently squashed, and treat the exported recipe JSONs
            (not your committed-block list) as ground truth.
    *   **Phase 3 (Post-Squash Audit & Ablation — MANDATORY)**: INT4 noise
        exposes fragility that Phase 1 did not observe, and empirically this
        phase is where most of the final quality is recovered
        (order-of-magnitude stopping-metric improvements have been
        observed). Budget permitting, always run it:
        1.  **Skip audit**: run one sensitivity sweep on the final mixed
            model and allow ONE round of protection additions, ordered by
            the same cost-aware rule as Phase 1 (cheapest error reduction
            first, small output-adjacent tensors always trialed). Do not
            iterate this indefinitely.
        2.  **Expensive-skip ablation**: if the audit committed a cheap,
            high-impact protection AFTER an expensive one (multi-MB skip or
            un-squash), re-evaluate the configuration WITHOUT the expensive
            protection. Greedy order often commits costly fixes that a
            later cheap fix makes redundant; the ablation frequently yields
            a strictly better size point at nearly identical quality.
    *   Export Pareto frontier profiles:
        -   **Quality Profile (INT8 + Protections)**: Non-skipped layers at
            8-bit + finalized protection list. Maximum fidelity within the
            size bound.
        -   **Compact Profile (INT4 + Protections)**: Non-skipped layers to
            4-bit + finalized protection list. Smallest model within the
            error tolerance.
        -   **Balanced Profile (Iterative Greedy Mixed-Precision)**:
            Finalized 4-bit squashed blocks + 8-bit remaining non-skipped +
            finalized protection list. Pick the frontier knee
            programmatically (the point before the largest marginal error
            jump), not by construction alone.

3.  **The Algorithm Axis (Third Search Dimension):** Bit-width and skip lists
    are not the only levers. AEQ ships advanced algorithms (GPTQ, OCTAV,
    Hadamard rotation — see the selection guide in `algorithms.md`) that can
    recover quality at a given bit-width without spending any size budget.
    *   If the INT4 Compact Profile fails its quality bound, before falling back
        to 8-bit, try re-running the failing configuration with an upgraded
        algorithm on the weight-heavy layers (e.g. `GPTQ` for transformer
        projections, `HADAMARD_ROTATION` for outlier-dominated activations).
    *   Record algorithm variants as distinct Pareto points using the
        `[algorithm]` component of the file naming schema.
    *   For transformer/LLM models, the algorithm axis usually dominates the
        skip-list axis in quality-per-MB terms. Explore it FIRST for Tier L
        models.

4.  **Recipe Overrides & Resolution Precedence:** In AEQ (`RecipeManager`),
    recipe rules are evaluated sequentially where **later rules override earlier
    matching rules**. Therefore:

    -   Default / broad rules (such as `regex=".*"` INT4/INT8 configs) MUST be
        added FIRST.
    -   Specific `algorithm_key="no_quantize"` skip rules MUST be added AFTER
        broad quantization rules so that FP32 fallbacks are preserved and not
        accidentally overwritten by `.*` or 4-bit overrides.
    -   Avoid applying 4-bit overrides to regexes matching layers in the
        `no_quantize` skip list.

5.  **Tracking Artifacts:** Explicitly save the `.tflite` models and track the
    configuration (by exporting the recipe via `qt.get_quantization_recipe()`)
    alongside the validation results for **every single recipe permutation
    tried** (not just the optimal ones) for easy comparison and Pareto curve
    plotting later. For Tier L models where storing every multi-GB `.tflite`
    variant is impractical, you MUST still save every `.json` recipe and
    validation metrics file, and keep the `.tflite` exports only for the
    baseline and the final Top 3 recommended profiles.

6.  **Task-Level Evaluation (Final Models Only):** Mathematical tensor metrics
    alone aren't enough to verify quality of the final deliverables. Run the
    modality-appropriate end-to-end evaluation on the final candidate models
    using the matching pluggable evaluation skill:
    -   Image segmentation / detection → `image_segmentation_eval.md`
        (masks + difference heatmaps saved in `results_fig/`).
    -   Generative LLMs → `llm_eval.md` (token agreement, KL, sample
        generations).
    -   Classification → `classification_eval.md` (Top-1/Top-5 agreement).
    -   Other modalities → report the primary/secondary metrics from
        `error_metric_selection.md` plus at least one qualitative sample
        (e.g. an audio clip spectrogram, a rendered depth map).

## IMPORTANT!!!! Notes on regex

Once you identify the target tensors to update quantization recipe, **NEVER**
build a regex by simply joining their full absolute tensor paths using
`re.escape()`. Doing so creates unreadable megabytes of strings, and alternating
between different precisions within the same block is highly inefficient at
runtime.

Instead, you **must read and follow the block-level extraction rules** in
[`regex_targeting.md`](regex_targeting.md) to target the entire common parent
module (like a stage, residual block, or transformer decoder block parent)
using clean, hardware-friendly regex filters.
