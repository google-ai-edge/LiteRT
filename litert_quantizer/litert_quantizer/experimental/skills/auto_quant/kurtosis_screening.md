# Data-Free Sensitivity Screening via Weight Kurtosis

Validations (`qt.validate()`) are the scarcest resource in the search loop —
especially at Tier L where the budget is ~8 total. This skill describes a
**zero-validation pre-screening pass** that ranks every layer's quantization
difficulty directly from the weight tensors, before any inference runs.

Based on the layer-sensitive quantization findings of SensiBoost/KurtBoost
(arXiv:2503.06518): weight-distribution kurtosis identifies outlier-heavy,
hard-to-quantize layers about as well as activation-based sensitivity sweeps,
at essentially zero cost.

## 1. Why Kurtosis

Layers whose weights contain heavy outliers are hard to quantize: the outliers
stretch the min/max range and squeeze the bulk of the weights into few bins.
Kurtosis (the standardized fourth moment) measures exactly this
tailedness — a high-kurtosis weight tensor is an outlier-heavy tensor. It is
computed from the weights alone: no calibration data, no float-vs-quantized
inference, no activation capture.

Empirical properties worth exploiting (from arXiv:2503.06518):

*   Sensitivity rankings are stable across calibration datasets, quantization
    methods, and bit-widths — so a ranking computed once is valid for the
    whole search. (This also independently justifies the one-shot cached
    sensitivity sweep in `model_scale_tiers.md`.)
*   Fine-tuned models inherit the sensitivity pattern of their base model — a
    ranking computed for a base model is reusable for its fine-tuned variants.
*   Sensitivity spikes concentrate at the FIRST and LAST layers of
    transformer stacks — treat the first and last decoder blocks as
    pin-to-higher-precision candidates by default.

## 2. Computing Per-Tensor Kurtosis from a TFLite Model

```python
import numpy as np
from scipy import stats
from ai_edge_litert import interpreter as tfl_interpreter

def compute_weight_kurtosis(tflite_path, min_elements=1024):
    """Returns {tensor_name: kurtosis} for weight tensors in the model."""
    interp = tfl_interpreter.Interpreter(model_path=tflite_path)
    kurtosis_by_tensor = {}
    for detail in interp.get_tensor_details():
        try:
            data = interp.tensor(detail['index'])()
        except ValueError:
            continue  # non-constant tensor without allocated data
        # Only parameterized weight tensors are interesting.
        if data.size < min_elements or not np.issubdtype(data.dtype, np.floating):
            continue
        # Fisher=False gives the Pearson definition (normal distribution == 3).
        kurtosis_by_tensor[detail['name']] = float(
            stats.kurtosis(data.flatten(), fisher=False)
        )
    return kurtosis_by_tensor
```

Discard `NaN` results (zero-variance constant tensors produce them) before
ranking. Cache the result to `reports/kurtosis_screening.json`. Aggregate per-tensor
values to per-block scores (max or mean of the block's tensors) using the
block extraction rules from `regex_targeting.md`.

## 3. Outlier Detection: Trimmed Z-Score on Adjacent Differences

Do not flag layers with ad-hoc fixed thresholds. Use this detection routine
(applicable to kurtosis AND to validation-based metric arrays — see
`error_metric_selection.md` §3):

1.  Order the per-block scores by the model's structural layer order.
2.  Build the difference series `D = {s2-s1, s3-s2, ...}` (or the ratio series
    `{s2/s1, ...}` for approximately ascending data, which suppresses points
    merely returning to the normal range).
3.  Compute the **trimmed** mean and standard deviation of `D` (discard the
    5% smallest and largest points so extremes don't inflate sigma).
4.  Flag entries with `|d - mean| / std > 3` and keep the top-m by magnitude
    (m bounded by your extra-size budget).
5.  A flagged difference at position `i` implicates layer `i+1`.

Adjacent-differencing distinguishes a genuine spike from a smooth trend across
the stack, which plain distribution-tail thresholds cannot.

## 4. How Screening Slots into the Search Loop

*   **Before any validation** (all tiers): compute the kurtosis ranking. Use
    it to (a) seed the Phase 1 fragile-candidate list, and (b) pre-pin obvious
    offenders (extreme-kurtosis blocks, first/last decoder blocks) to higher
    precision in the very first recipe, saving early loop iterations.
*   **Tier L specifically**: kurtosis ranking + the single cached sensitivity
    sweep (see `model_scale_tiers.md`) together provide the robustness
    ordering for batched promotion; spend actual validations only on
    confirming candidate promotion batches.
*   **Cross-model reuse**: when quantizing a fine-tuned variant of a
    previously-explored base model, reuse the base model's cached rankings and
    final recipe as the starting point.

## 5. Caveats

*   Kurtosis is a **weight-only** signal. It cannot see activation-driven
    sensitivity (e.g. softmax attention outliers at runtime). Use it to SEED
    rankings and save validations — never to override a contradicting
    validation result.
*   **Kurtosis is a prior, NOT a replacement for the measured sensitivity
    sweep.** Empirical counter-example (observed in a U-Net-style
    segmentation search): the kurtosis ranking and the validation-based SNR
    ranking disagreed badly — the block ranked second-highest by kurtosis
    was the single MOST robust block (first to survive INT4), while the
    second-most-fragile block by measured SNR sat mid-pack in the kurtosis
    ranking. Measured fragility followed **graph position** (the decoder
    path, whose noise reaches the output unaveraged) rather than weight
    statistics — a structural signal weight kurtosis cannot see. The cached
    one-shot sensitivity sweep costs only ~1 validation, so skipping it to
    save budget is a bad trade at every tier.
*   Kurtosis screening also silently misses **small critical tensors**
    excluded by its `min_elements` statistical filter (e.g. tiny output
    projections that dominate output error). Position-based priors — output
    adjacency, first/last blocks — must be applied independently of the
    kurtosis ranking when seeding protection candidates (see the cost-aware
    ordering rule in `experiment_runner.md`).
*   Gains from selective boosting are largest in the aggressive (~3-bit /
    low-INT4) regime and shrink at higher budgets and on larger models; do not
    expect kurtosis-guided pins to rescue a fundamentally too-aggressive
    configuration.
*   A high-kurtosis layer may respond better to an algorithm upgrade (Hadamard
    rotation smears exactly these outliers — see `algorithms.md`) than to a
    bit-width promotion. Try the zero-size-cost fix first.
