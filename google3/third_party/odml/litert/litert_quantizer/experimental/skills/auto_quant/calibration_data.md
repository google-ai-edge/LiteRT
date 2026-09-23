# Calibration Data Guidelines (Static Quantization)

Dynamic and weight-only recipes need no calibration data. The moment you
transition to **static quantization** (`static_wi8_ai16`, etc. — activations
quantized offline), the calibration dataset becomes a first-class input that
directly determines activation scale/zero-point quality. Bad calibration data
silently produces a model that validates well on the calibration inputs and
clips catastrophically on real inputs.

## 1. When Static Quantization Is Worth It

Per `quantization_pattern.md`, start with dynamic quantization. Move to static
only if:

*   The target hardware (NPU/DSP/microcontroller) requires fully-integer
    graphs, OR
*   Dynamic quantization's on-the-fly dequantization violates a strict latency
    bound (verify with `latency_benchmarking.md`, don't assume), AND
*   Representative calibration data is actually available or obtainable.

If the user requests static quantization but provides no calibration data,
STOP and ask for it — do not fabricate inputs from random noise. Random-noise
calibration produces activation ranges unrelated to real data and is worse
than staying dynamic.

## 2. Dataset Composition Rules

*   **Representativeness over volume**: 100–500 samples drawn from the real
    input distribution beat 10,000 synthetic ones. Cover the expected
    diversity axes (e.g. for vision: lighting, scale, subject types; for
    audio: speakers, noise conditions; for text: prompt lengths and domains).
*   **Include distribution extremes deliberately**: activation ranges are set
    by observed min/max (or clipped variants). If real usage includes
    saturated/dark images or very long prompts, the calibration set must too.
*   **Apply the EXACT inference preprocessing**: the same normalization,
    resizing, and dtype conversion the deployed model will see. A calibration
    set normalized to `[0, 1]` for a model deployed with `[-1, 1]` inputs
    invalidates every activation range.
*   **No evaluation leakage**: keep calibration samples strictly disjoint from
    the samples used in `qt.validate()` sweeps and from the final task
    evaluation sets. Otherwise your quality numbers are optimistically biased.

## 3. Reproducibility Manifest

Persist a calibration manifest alongside the recipes
(`model/quantized/recipes/calibration_manifest.json`) recording:

*   Source of the samples (paths, dataset name, or generation procedure).
*   Sample count and random seed used for selection.
*   Preprocessing function reference (file + function name).
*   Date and float model checksum.

Every static recipe in the final report must reference the manifest so results
are reproducible.

## 4. Stability Check

Activation ranges from small calibration sets can be noisy. Before trusting a
static configuration, run the calibration twice with two disjoint sample
subsets (same size, different seed) and compare the resulting validation
metrics. If the stopping metric differs materially between the two runs
(e.g. by more than 20%), the calibration set is too small or too narrow —
grow it before continuing the search.
