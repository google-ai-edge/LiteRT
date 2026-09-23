---
name: auto_quant
description: >-
  Systematically explore quantization configurations for a TFLite model using the AI Edge Quantizer API, finding the optimal recipe that balances file size and accuracy. Use this skill whenever the user wants to quantize a model, optimize a recipe, explore quantization tradeoffs, minimize size bounds, or perform selective quantization using AI Edge Quantizer (or AEQ) framework. Applies to any model scale and modality: CNNs, segmentation nets, classifiers, embedding models, audio models, and LLMs.
---

# Agent Contribution Guide: AI Edge Quantizer Recipe Exploration

Welcome, AI Agent! This skill governs how you autonomously explore and generate
the optimal selective quantization recipe for any given LiteRT FlatBuffer
(`.tflite`) model using the AI Edge Quantizer (AEQ) framework. Strict adherence
to these protocols ensures code is high-quality, memory safe, and meets
verification protocols.

Your goal is to build an iterative search loop in Python that actively
interrogates the baseline model for numerical degradation, updates the recipe on
the fly to protect highly sensitive tensors, and produces a bounding-box of
Pareto-optimal models for the user.

## Skill Index

Leverage these project-specific skill guides to navigate and execute tasks
effectively.

**Core guides (read ALL of these before proceeding):**

| File | Contents |
| --- | --- |
| [`README.md`](../../README.md) | Overview of the AEQ framework and the repo's structure |
| [`README.md`](../../../../README.md#operator-coverage) | What configurations are supported for each operator |
| [`model_scale_tiers.md`](model_scale_tiers.md) | Scale tier classification (S/M/L) with per-tier search budgets, batched squashing, and memory guardrails |
| [`algorithms.md`](algorithms.md) | Registry of available quantization algorithms and when to select each one |
| [`error_metric_selection.md`](error_metric_selection.md) | Registry of validation error metrics, task-specific selection, distribution-relative thresholding, and per-modality stopping metrics |
| [`file_naming.md`](file_naming.md) | Standardized conventions for naming exported models, recipes, evaluation artifacts, and validation reports |
| [`quantization_pattern.md`](quantization_pattern.md) | Empirical patterns: what works, what doesn't, and why |
| [`snr_best_practices.md`](snr_best_practices.md) | How to compute, parse, and threshold per-tensor metrics to locate fragile layers |
| [`kurtosis_screening.md`](kurtosis_screening.md) | Zero-validation, data-free layer sensitivity screening via weight kurtosis, plus the trimmed z-score outlier detection routine |
| [`regex_targeting.md`](regex_targeting.md) | How to write readable, unified block-level regex patterns to avoid scattered precision overlaps |
| [`size_estimation.md`](size_estimation.md) | How to compute theoretical compressed model size |
| [`experiment_runner.md`](experiment_runner.md) | Memory-safe experiment loop, batched squashing, the algorithm search axis |
| [`pareto_curve_plotting.md`](pareto_curve_plotting.md) | Guidelines on formatting the Pareto visualization, drawing frontiers, and handling outliers |
| [`output_report.md`](output_report.md) | How to format and organize the output produced |

**Pluggable task evaluation family (read ONLY the file matching the model's
modality):**

| File | Applies To |
| --- | --- |
| [`image_segmentation_eval.md`](image_segmentation_eval.md) | Segmentation / detection models (masks, mIoU/Dice, heatmaps) |
| [`llm_eval.md`](llm_eval.md) | Generative LLMs (multi-signature handling, token agreement, KL, sample generations) |
| [`classification_eval.md`](classification_eval.md) | Classifiers (Top-1/Top-5 agreement, confidence drift) |

**Conditional guides (read when applicable):**

| File | Read When |
| --- | --- |
| [`calibration_data.md`](calibration_data.md) | Using static quantization (calibration dataset rules, leakage prevention, manifests) |
| [`latency_benchmarking.md`](latency_benchmarking.md) | The user states a latency requirement or asks for on-device numbers |

## Phase 0: Understand the Framework and Classify the Model

Before beginning, explicitly use specific file reading tools to read all core
guides in the Skill Index, plus the ONE task evaluation guide matching the
model's modality, plus any applicable conditional guides.

Then, before writing any plan:

1.  **Classify the modality** (segmentation, LLM, classification, embedding,
    audio, regression) from the model's signatures, I/O tensor shapes, and the
    user's description. This selects your metrics
    (`error_metric_selection.md`) and evaluation family.
2.  **Classify the scale tier** (S/M/L per `model_scale_tiers.md`) from the
    float model file size and signature count. This selects your search
    budget, batching strategy, and protection-mechanism priority.

## 1. Implementation Plan Guidelines

Because complex selective quantization tasks can cause context loss during
execution, your first action MUST be to draft a highly explicit, self-contained
Implementation Plan. Since you will follow this plan closely, you must bake all
necessary context directly into it.

**You MUST output this plan as an artifact file** (e.g.
`implementation_plan.md`, using your file-writing tool). In interactive
sessions, **HALT after writing the plan** and explicitly ask the user for
permission to proceed before you make *any* code edits or run *any* loops. If
the user has already provided explicit bounds AND explicitly requested
unattended/autonomous execution (e.g. a CI pipeline), you may proceed without
halting, but the plan artifact is still mandatory. You must strictly follow
this plan once approved.

Your printed plan must contain:

### Step 1: Context & Role Declaration

*   **Role Judgment**: Declare your role (Contributor, Maintainer, or Reviewer).
*   **Model Classification**: State the detected modality, the scale tier, and
    the resulting metric selections and search budgets.
*   **Context Summary**: Explicitly write out a 2-3 sentence summary of the
    specific constraints you read from the skill docs. You MUST explicitly state
    that your artifact names will obey the strict rules from `file_naming.md`
    and explicitly confirm you will produce a Pareto visualization per
    `pareto_curve_plotting.md`.

### Step 2: Technical Design & File List

*   **Target Files**: Explicitly list all files you plan to create or modify.
    Please adhere strictly to the `output_report.md` requirements.
*   **Optimization Constraints**: Briefly outline how you will handle target
    file size bounds, acceptable error thresholds across MULTIPLE metrics, and
    generic regex targeting to prevent data leakage.

### Step 3: Experiment Harness Architecture

*   **Metric Selection**: Infer from the float model what metrics you should use
    for validation following the guidelines of `error_metric_selection.md`,
    including the modality-appropriate stopping metric.
*   **Experiment Blueprint**: Explicitly outline your script structure focusing
    on proper API sequential updates, memory-safe `Quantizer` instantiation per
    permutation, strictly sequential `qt.validate()` calls (one validation at a
    time), explicit garbage collection (`import gc; gc.collect()`), and
    selecting layer candidates. **Most importantly, explicitly confirm you are
    implementing the stringent Two-Phase loop mandated by `experiment_runner.md`
    (Phase 1: Freezing protection set, Phase 2: Iterative Greedy INT4 squashing
    sorted by robustness, batched per your scale tier). Do not hallucinate
    trial-and-error hacks.**
*   **Budget Declaration**: State the per-tier validation cap and sample count
    you will operate under (from `model_scale_tiers.md`).

### Step 4: Verification Plan

*   **Review Hand-off**: Document your explicit commitment to run the Two-Pass
    Review Loop (see "Agent Verification Protocol" below) on your execution
    script *before* you present the final report. If your harness supports
    launching subagents, commit to delegating the review to a reviewer
    subagent; otherwise commit to performing the structured self-review pass
    against the Reviewer checklist. ONLY start the review after you think
    you're done with the quantization exploration process.

### Step 5: Post-Execution Walkthrough

*   Acknowledge that the final response will adhere to the Post-Execution
    Walkthrough Guidelines (below).

## Phase 1: Setup & Bounding Box

1.  **Ask the User for Bounds**: If the user did not specify, explicitly ask for
    either their **Minimum Model Compression Ratio / Model Size** OR their **Max
    Tolerable Error Bound**. Our objective is to find either the best quality
    model given the minimum compression ratio or the smallest model given the
    max tolerable error bound.
2.  **Acquire Verification Preprocessing**: In order to use `qt.validate()`, you
    need numpy tensor inputs matching the model's signature(s). You may copy
    the input processing logic from existing manual human scripts in the
    directory (for example `processing.py`), or construct tokenized prompts
    for LLMs per `llm_eval.md`.
    *   **CRITICAL CONSTRAINT (NO LEAKAGE)**: You MUST NOT copy the human's
        manual selective recipe adjustments from existing sources or tutorials
        to blindly inject layer names. You must discover which layers to
        protect dynamically based on your own validation metric parsing!
        This equally applies to the empirical anecdotes and example values
        inside these skill guides: treat them as unverified priors that
        motivate WHERE to look, never as pre-validated protection lists or
        expected metric values. Every protection you commit must be
        justified by a measurement from YOUR current run.
    *   **Single-source-input warning**: If the user provides only ONE test
        input, deterministic augmentations of it are acceptable for the
        search loop, but the resulting recipe may be mildly overfit to that
        input. You MUST state this limitation explicitly in the final report
        and recommend re-validating the chosen profiles on a small held-out
        set before production use.
3.  **Effort & Sizing Logic**: The user's size bounds represent a *limit*, NOT a
    *target to match exactly*. You are authorized to return models that are
    smaller than requested! The primary goal is to **maximize accuracy** (or the
    target metric) while guaranteeing the model is small enough. If you can make
    it even smaller while retaining acceptable accuracy, do so. Never
    artificially inflate the model just to hit a bound. Try multiple iterations
    until you have found the best possible quantization recipe, within the
    validation budget of your scale tier.

## Phase 2: Building the Iteration Harness

Do not try to guess the recipe blindly or write concurrent sweeps. You must
construct a dedicated, repeatable Python evaluation script (e.g.
`explore_aeq_model.py`) that operates on ONE recipe at a time. Do not run
parallel configurations or concurrent validations.

1.  Load the float baseline `.tflite` model and run data through preprocessing.
2.  Instantiate `quantizer.Quantizer()` and load **ONE default baseline**
    starting recipe via the API.
3.  Inject targeted `qt.update_quantization_recipe(...)` rules sequentially on
    the single `qt` object.
4.  Call `qt.quantize()` and `qt.export_model()`. Output every model variation
    with a detailed, unique file name following the guidelines in
    `file_naming.md` (respecting the Tier L `.tflite` retention exception in
    `model_scale_tiers.md`). Save the `.json` recipe for EVERY variation, no
    exceptions.
5.  Measure the **stopping metric deployment-faithfully**: run each exported
    `.tflite` under the default LiteRT `Interpreter` and compare outputs
    against cached FP32 baseline outputs on a fixed sample set (see the
    "Deployment-faithful stopping metric" rule in `experiment_runner.md`).
    Do NOT gate commits on `qt.validate()` — its backend differs by recipe
    (XNNPACK rejects INT4), which makes INT8 and INT4 candidates
    incomparable. Reserve `qt.validate(...)` for per-tensor sensitivity
    sweeps, strictly sequentially (one validation at a time, never
    concurrent), with `use_xnnpack=False` on INT4-containing recipes; check
    whether your installed release supports multi-metric validation or a
    single metric string per call (see the validation API caveats in
    `experiment_runner.md`). **CRITICAL: Look at metrics in your script to
    decide which tensors to target.**

## Phase 3: The Empirical Search Loop (Execution)

1.  **Two-Phase Iterative Greedy Loop**: Read `experiment_runner.md` for the
    exact step-by-step logic.
    -   **Phase 0 (Free Pre-Screening)**: Before any validation, compute the
        data-free weight kurtosis ranking (`kurtosis_screening.md`) to seed
        the fragile-candidate list and pre-pin obvious offenders without
        spending validation budget.
    -   **Phase 1 (Finalize Protection Set)**: Identify fragile layers using
        sensitivity analysis (primary metric, e.g., SNR for vision, KL for
        LLMs) and incrementally protect them until overall model quality is
        guaranteed via the **modality-appropriate stopping metric** on the
        output tensor. Protection mechanism priority depends on scale tier:
        `no_quantize` skips for Tier S; 8-bit pins and algorithm upgrades
        before skips for Tier M/L. Freeze the protection set once the stopping
        metric is good.
    -   **Phase 2 (Iterative Greedy INT4 Squashing)**: Rank remaining
        unprotected layers by robustness and iteratively squash them
        block-by-block (or batch-by-batch with bisection, per your tier) to
        4-bit, validating the stopping metric after each step to prevent
        compounding noise. **Revert failing blocks and continue with the next
        candidate; halt only after 3 consecutive reverts or when the
        validation budget is exhausted.**
        *   **CRITICAL**: When designing the Phase 2 condition, determine
            threshold bounds dynamically relative to the optimal configuration
            found in Phase 1. Do not use hardcoded scalar assumptions because
            End-to-End metrics can output naturally large arbitrary scalars!
            This applies to EVERY script in the search, including post-hoc
            audit and refinement scripts.
    -   **Phase 3 (Post-Squash Audit & Ablation — MANDATORY, budget
        permitting)**: one cost-aware round of protection additions on the
        final mixed model, followed by an ablation of any expensive
        protection committed before a cheaper superseding one. Empirically
        this phase recovers most of the final quality — see
        `experiment_runner.md`.
    -   **Algorithm Axis**: If a configuration fails its quality bound, try an
        algorithm upgrade (GPTQ, Hadamard rotation, MSE calibration — see
        `algorithms.md`) on the failing layers before retreating to a higher
        bit-width. For Tier L transformers, explore the algorithm axis FIRST.
    -   **Tier L Direction Inversion (INT4-First)**: For Tier L models, INT4
        weights are the default target, not the aggressive endpoint. Instead
        of squashing down from INT8, start from an all-INT4 baseline (best
        algorithm, embedding/LM head pinned to INT8) and greedily PROMOTE the
        most fragile blocks up to INT8 until the stopping metric passes. See
        the "Inverted Search Direction for Tier L" section of
        `model_scale_tiers.md`.
2.  **Loop Execution**: Build out the script to execute both Phase 1
    (protection set finalization via the stopping metric) and Phase 2
    (iterative greedy INT4 squashing on unprotected layers) automatically.

## Phase 4: Exploring the Pareto Frontier

Provide options. You must output, record, and **export to disk EVERY SINGLE**
recipe permutation (`.json`) and validation metrics file generated during your
exploration (including failed baseline attempts, intermediate steps, and the
final optimized models). Export the quantized `.tflite` for every permutation
on Tier S/M; on Tier L, export `.tflite` for the baseline and Top 3 profiles
(see `model_scale_tiers.md`). Do not just keep results in memory. This output
directory is necessary so the user can plot a rich Pareto curve.

Among all explored recipes, you must explicitly highlight and **recommend the
Top 3** recipes in your final report, mapped to these key profiles. The
profiles are defined by the USER-FACING trade-off they optimize, not by their
construction (post-audit refinements often blur the constructive
definitions):

*   **Quality Profile (maximum fidelity within the size bound)**: Typically
    remaining unprotected layers strictly at 8-bit, inheriting the finalized
    protection list.
*   **Compact Profile (smallest model within the error tolerance)**:
    Typically pushes remaining unprotected layers to 4-bit (with the
    best-performing algorithm variant), inheriting the finalized protection
    list.
*   **Balanced Profile (best error-per-MB knee of the frontier)**: Result of
    differential sensitivity analysis on remaining unprotected layers
    (pushing robust layers to 4-bit and moderately sensitive layers to
    8-bit), inheriting the finalized protection list. Select the knee
    programmatically — the admissible point preceding the largest marginal
    error jump — rather than assuming the mixed recipe is automatically the
    knee.

Tag the three chosen artifacts with the profile suffix from
`file_naming.md` §G so the recommendation is readable from the filename.

For Tier L models under the inverted INT4-first search, the same three
profiles map to: **Compact** = pure INT4 baseline (+ pinned embedding/LM
head), **Balanced** = minimal INT8 promotions needed to pass the quality
bound, and **Quality** = generous promotion of all flagged-fragile blocks to
INT8 (see `model_scale_tiers.md`).

## Agent Verification Protocol (Two-Pass Loop)

To ensure code quality, verifications must follow a strict Two-Pass Recursive
Review:

1.  **Pre-flight Self-Check**: Ensure the python code parses and runs
    successfully before reviewing.
2.  **Pass 1 (Comprehensive Audit)**: If your harness supports subagents, you
    MUST literally invoke a reviewer subagent — do not just hallucinate a
    review yourself. Pass your python script implementation to the subagent
    via the prompt, and instruct it to mathematically evaluate bit depths,
    proper API usage (not raw JSON hacking), isolation of quantizers,
    multi-metric evaluations, and adherence to the scale-tier budget. Wait for
    its message back. If your harness does NOT support subagents, perform a
    structured self-review: re-read your script top-to-bottom against every
    item in the Reviewer checklist below, and record each pass/fail verdict
    explicitly in the report.
3.  **Pass 2 (Delta Verification)**: Provide the updated code for a second
    review pass and verify all fixes are applied without regressions.
4.  If an impasse is reached, yield to human.

## Role-Based Execution Guidelines

*   **Contributor**: Adhere to `experiment_runner.md` strictly. Prevent Data
    Leakage by never hardcoding explicitly named tutorial layers in your
    implementations; find and parse the actual broken scopes generically.
*   **Maintainer**: Ensure backwards compatibility when updating scripts.
*   **Reviewer**: Organize feedback into clear categories: `[Quantization
    Quality]`, `[AEQ API Health]`, `[Experiment Rigor]`, `[Reporting]`. Reject
    code that lacks multi-metric tracking or multiple Pareto options.

    *   **Rule**: You MUST reject any script that uses manual trial-and-error
        regex arrays to inflate size instead of the proper Two-Phase greedy
        loop.
    *   **Rule**: You MUST reject any script that doesn't explicitly program
        the Greedy Phase 2 squashing loop sorting layers by robustness
        (single-step or batched-with-bisection per the declared scale tier).
    *   **Rule**: You MUST reject any script that halts Phase 2 permanently on
        the FIRST failed block instead of reverting-and-continuing.
    *   **Rule**: You MUST reject any script whose stopping metric ignores the
        modality (e.g. output MSE used for an LLM instead of output KL).
    *   **Rule**: You MUST reject any script whose commit/revert gate mixes
        validation backends across phases (e.g. XNNPACK-backed numbers for
        INT8 steps vs reference-kernel numbers for INT4 steps). The gate
        must be the deployment-faithful stopping metric measured on the
        exported model under the default runtime.
    *   **Rule**: You MUST reject any script that ignores the scale-tier
        budget (e.g. unbatched per-block validation on a Tier L model).
    *   **Rule**: You MUST reject the code if the artifact paths don't strictly
        generate `_validation_metrics.json` and follow the
        `[execution]_[precision]...[ver]` schema exactly as defined in
        `file_naming.md` and `output_report.md`.

## Post-Execution Walkthrough Guidelines

After execution, format your final response strictly:

1.  **Phase 1: Executive Summary & Bound Achievements**: Technical TL;DR.
    Provide detailed mathematical reasoning (using the selected validation
    metrics) justifying your layer assignments (protected vs int8 vs int4) for
    the models on the frontier. State the scale tier and the number of
    validations consumed vs the budget.
2.  **Phase 2: Architectural Footprint**: Paths of generated scripts, quantized
    models `.tflite` files, quantization recipes `.json` files, Pareto graph,
    and `quantization_report.md`.
3.  **Phase 3: Precision & Sanity Guarantee**: Confirm validations evaluated
    distinct error paths jointly to decide configurations and verify data
    leakage checks (generic regex abstraction used over hallucinated names;
    calibration/evaluation sample disjointness if static quantization was
    used).
4.  **Phase 4: Verification Audit Trail**: Explicit review pass/fail status
    scorecard and delta fixes (subagent-based or structured self-review,
    whichever was performed).
