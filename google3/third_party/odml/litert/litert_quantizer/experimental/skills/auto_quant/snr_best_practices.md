<!-- disableFinding(LINE_OVER_80) -->

# Metric Parsing Best Practices (SNR Example)

While `error_metric_selection.md` guides you in deciding *which* mathematical
metric to use based on model modality, this document demonstrates how to
extract, sort, and act on metric arrays natively from the quantizer API.

We use Signal-to-Quantization-Noise Ratio (SNR) in these snippets as a common
example metric.

## 1. Understanding SNR Thresholds Relative to Bit-width

SNR is calculated in Decibels (dB). **Crucially, there is no single hardcoded
threshold that works globally, because acceptable SNR scales with the
bit-width** (INT8 naturally produces higher SNR than INT4). Instead of relying
on static numbers, use RELATIVE DISTRIBUTION and general baselines:

*   **The Distribution Outlier (Relative Drop)**: The best heuristic is tracking
    the *relative* distribution of SNR across your model. If the model's median
    SNR is high, but a few outlier layers plunge significantly below the median,
    those relative outliers are your highly degraded tensors requiring fallback
    configurations.
*   **Finding Robust Layers (Aggressive Compression)**: When looking for layers
    to aggressively compress (e.g., INT8 -> INT4), look for the layers that
    maintain the highest relative SNR in the network. If they easily tolerate
    INT8 without much degradation, they are strong candidates for INT4.
*   **Finding Fragile Layers (Protection)**: If a layer's SNR plunges
    significantly during a baseline sweep relative to the rest of the network,
    it is a prime candidate for protection (upgrading to a higher precision or
    bypassing via `no_quantize`).

## 2. The Golden Rule: Per-Tensor Metrics for Search, Output Metric for Stopping

When conducting selective quantization sweeps (like mixed precision), do not
waste time evaluating end-to-end task metrics (like mIoU, BLEU score, or
post-processed masks) at every iteration.

*   **Internal Tensor Search (Fast)**: Rely on standard internal per-tensor
    metrics (SNR in these examples) to rapidly rank and identify the most
    mathematically sensitive internal layers.
*   **Stopping Criteria (Efficient)**: Use the **modality-appropriate stopping
    metric on the model's raw final output tensor** (see the "Stopping Metric
    per Modality" table in `error_metric_selection.md`) to determine if your
    loop iterations satisfy the overall degradation bounds and to stop the
    search. For dense vision and regression outputs this is Output Tensor MSE
    (`ValidationErrorMetric.MSE`); for logit-producing models (LLMs,
    classifiers) use output `KL_DIVERGENCE`; for embedding models use output
    `COSINE_SIMILARITY`.
*   **End-to-End Metrics (For Illustration Only)**: True Task Metrics should
    only be computed at the very end of an experiment for reporting and
    illustration purposes (e.g. visualizing a segmentation mask or checking
    Top-1 accuracy). Do not use them as the mathematical stopping condition in
    your search loop.

## 3. Standard Loop for Identifying Sensitive Layers

When agents need to dynamically deduce which layers to skip (instead of
hardcoding names or guessing), they should follow this standard routine:

1.  Apply a baseline quantization recipe (e.g., `dynamic_wi8_afp32`).
2.  Run `qt.validate()` and instruct it to compute `ValidationErrorMetric.SNR`.
3.  Parse the validation result dictionary to extract SNR for each tensor.
4.  Sort the tensors by lowest SNR (worst quality).
5.  Extract the worst $N$ tensors and formulate a clean regular expression that
    targets them.
6.  Inject that regex into the quantizer to protect them on the next sweep
    iteration.

## 4. Boilerplate: Extracting SNR from Validation Results

Below is the standard snippet for parsing the nested output produced by
`qt.validate()`:

```python
import numpy as np
from ai_edge_quantizer import quantizer

# 1. Run validation requesting the SNR metric
validation_results = qt.validate(
    test_data=dataset,
    error_metrics=[quantizer.ValidationErrorMetric.SNR]
)

# 2. Extract SNR values from ComparisonResult object
# validation_results is a model_validator.ComparisonResult instance.
snr_data = []

for sig_key in validation_results.available_signature_keys():
  sig_res = validation_results.get_signature_comparison_result(sig_key)
  all_tensors = {}
  all_tensors.update(sig_res.intermediate_tensors)
  all_tensors.update(sig_res.output_tensors)

  for tensor_name, metrics in all_tensors.items():
    if quantizer.ValidationErrorMetric.SNR in metrics:
      snr_val = metrics[quantizer.ValidationErrorMetric.SNR]
      snr_db = 10.0 * np.log10(snr_val) if snr_val > 0 else 0.0
      snr_data.append((tensor_name, snr_db))

# 3. Sort by lowest SNR first (worst layers at the top)
snr_data.sort(key=lambda x: x[1])

# 4. Clean up validation structures and trigger GC to prevent memory overload
del validation_results
import gc
gc.collect()

# Print the 5 worst layers
print("Worst 5 Tensors by SNR:")
for tensor_name, snr_db in snr_data[:5]:
    print(f"{tensor_name}: {snr_db:.2f} dB")
```

## 5. Boilerplate: Abstracting the Regex (Preventing Huge Strings)

Once you identify the worst tensors, **NEVER** build a regex by simply joining
their full absolute tensor paths using `re.escape()`. Doing so creates
unreadable megabytes of strings, and alternating between different precisions
within the same block is highly inefficient at runtime.

Instead, you **must read and follow the block-level extraction rules** in
[`regex_targeting.md`](regex_targeting.md) to target the entire common parent
module using clean, hardware-friendly regex filters.

