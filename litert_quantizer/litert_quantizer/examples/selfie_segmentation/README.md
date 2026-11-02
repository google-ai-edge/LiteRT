<!-- disableFinding(HTML_ATTRIBUTE_MALFORMED) -->
# Histogram Calibration & A8W8 Static-Range Quantization: Selfie Segmentation

This example demonstrates how to perform **profiler-based histogram
calibration** and static-range uniform integer quantization (**A8W8**) using
LiteRT Quantizer (AEQ) on a multiclass semantic segmentation model
(`selfie_multiclass_256x256.tflite`).

---

## 1. Mathematical Background

Uniform integer quantization maps continuous floating-point activations
`x ∈ ℝ` to `b`-bit integers `q ∈ [q_min, q_max]` (e.g., `[-128, 127]` for signed
INT8):

<!-- disableFinding(LINE_OVER_80) -->
$$q = \text{clamp}\left(\left\lfloor \frac{x}{s} \right\rceil + z, q_{\min}, q_{\max}\right)$$

$$\hat{x} = s \cdot (q - z)$$
<!-- enableFinding(LINE_OVER_80) -->

where `s = (β - α) / (q_max - q_min)` is the scale factor,
`z = round(-α / s) + q_min` is the zero-point,
and `[α, β]` defines the quantization range.

### Quantization Noise Decomposition

The total quantization error `e(x) = x - x̂` across an activation
distribution `p(x)` decomposes into two competing components:

<!-- disableFinding(LINE_OVER_80) -->
$$E_{\text{total}}(\alpha, \beta) = \underbrace{\int_{\alpha}^{\beta} (x - \hat{x})^2 p(x) \, dx}_{\text{Granular Noise } E_{\text{granular}}} + \underbrace{\int_{-\infty}^{\alpha} (x - \alpha)^2 p(x) \, dx + \int_{\beta}^{\infty} (x - \beta)^2 p(x) \, dx}_{\text{Overload / Clipping Distortion } E_{\text{overload}}}$$
<!-- enableFinding(LINE_OVER_80) -->

1. **Granular Noise (`E_granular`):** For inputs inside `[α, β]`, the
   rounding error variance is approximately governed by the step size `Δ`:
   <!-- disableFinding(LINE_OVER_80) -->
   $$\sigma_{\text{granular}}^2 \approx \frac{\Delta^2}{12} = \frac{1}{12} \left(\frac{\beta - \alpha}{2^b - 1}\right)^2$$
   <!-- enableFinding(LINE_OVER_80) -->
   `Δ` grows linearly with the range width `β - α`, so granular noise grows
   quadratically with it.

2. **Overload Distortion (`E_overload`):** For inputs outside `[α, β]`,
   activations are clipped to the boundaries `α` and `β`, producing squared
   clipping error.

### Why Min-Max Can Be Suboptimal

`MIN_MAX_UNIFORM_QUANT` sets `α = min(X)` and `β = max(X)`, which makes
`E_overload = 0`. When an activation distribution has rare outliers, spanning
them inflates `Δ` and raises `E_granular` for the bulk of the values.

The histogram algorithms trade a small amount of overload distortion for lower
granular noise:

* **`HISTOGRAM_PERCENTILE`** clips each activation to the `[0.001, 99.999]`
  percentiles of its histogram.
* **`HISTOGRAM_MSE`** searches candidate pairs `(α, β)` drawn from histogram
  bin centers and picks the pair with the lowest estimated `E_total`. The
  default `symmetric` search expands outward from the center bin with
  coordinate refinement; `bruteforce` evaluates all bin pairs.

### Prefix-Moment MSE Evaluation

Evaluating `N` candidate ranges directly against an `M`-bin histogram costs
`O(N × M)`. AEQ instead precomputes shifted cumulative moments once in `O(M)`:

<!-- disableFinding(LINE_OVER_80) -->
$$L_0(k) = \sum_{i=0}^k f_i, \quad L_1(k) = \sum_{i=0}^k f_i (m_i - m_0), \quad L_2(k) = \sum_{i=0}^k f_i (m_i - m_0)^2$$
<!-- enableFinding(LINE_OVER_80) -->

and the corresponding suffix moments `R_0, R_1, R_2` shifted by `m_{M-1}`.
Shifting the origin keeps the moments numerically well-conditioned. For each
candidate, the clipped region boundaries are located with a binary search
(`np.searchsorted`), and the clipping errors follow in `O(1)`:

<!-- disableFinding(LINE_OVER_80) -->
$$E_{\text{clip\_left}} = L_2(l) - 2(m_l - m_0) L_1(l) + (m_l - m_0)^2 L_0(l)$$
$$E_{\text{clip\_right}} = R_2(r) - 2(m_r - m_{M-1}) R_1(r) + (m_r - m_{M-1})^2 R_0(r)$$
<!-- enableFinding(LINE_OVER_80) -->

Granular error inside the range uses the uniform approximation
`(Δ² / 12) ∑_{i=l}^r f_i`. Total cost is `O(M + N log M)`, vectorized in NumPy
across each batch of candidates.

---

## 2. Profiler-Based vs. Preserve-All-Tensors Calibration

<!-- disableFinding(LINE_OVER_80) -->
| Feature | `CALIBRATION_PRESERVE_ALL_TENSORS` | `CALIBRATION_PROFILER_BASED` |
| :--- | :--- | :--- |
| **Statistics collection** | Python: after each invoke, iterates over ops and updates min/max and histograms in NumPy from preserved tensor buffers | C++: min/max and dynamic histograms are accumulated by profiler hooks during invoke |
| **Interpreter** | XNNPACK with `preserve_all_tensors=True` (intermediate buffers are kept alive) | XNNPACK with delegate node fusion disabled and per-node profiling enabled |
| **Supported algorithms** | All | Algorithms calibrated with min/max or histograms (not e.g. GPTQ, OSCAR) |
<!-- enableFinding(LINE_OVER_80) -->

---

## 3. Results

Results below come from a single run of the example with default flags (all
four configurations) on one workstation. Wall times are machine-dependent and
cover only the `Quantizer.calibrate()` call over 32 calibration samples (the 6
bundled portraits plus 26 augmented crops/flips/jitters).

All quality metrics compare the quantized model against the **FP32 model's own
predictions** on the 6 bundled portraits, not against ground-truth labels.
These portraits are also part of the calibration set, so the numbers are
in-sample.

* **Top-1 agreement:** fraction of pixels whose argmax class matches FP32.
* **mIoU / per-class IoU:** IoU between the quantized and FP32 argmax masks.
* **KL Div.:** mean per-pixel `D_KL(P_FP32 || P_quant)`, where both class
  distributions are the softmax of the model's output logits.

### Metric Comparison

The float model is 15.61 MiB; every A8W8 model is 4.43 MiB (3.5x smaller).

<!-- disableFinding(LINE_OVER_80) -->
| Configuration | Calibration Mode | `calibrate()` Wall Time | SQNR (dB) | Output MSE | Cosine Sim. | KL Div. | Top-1 Agreement | mIoU |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MIN_MAX_UNIFORM_QUANT` | Profiler | 3.61 s | 7.43 | 4.17e-01 | 0.90799 | 0.21295 | 89.33% | 61.13% |
| `HISTOGRAM_PERCENTILE` | Profiler | 13.27 s | 5.19 | 6.98e-01 | 0.84383 | 0.37429 | 82.94% | 60.78% |
| `HISTOGRAM_MSE` | Profiler | 13.15 s | 9.36 | 2.67e-01 | 0.94095 | 0.12341 | 92.86% | 69.49% |
| `HISTOGRAM_MSE` | Preserve-all-tensors | 141.36 s | 9.54 | 2.56e-01 | 0.94343 | 0.11675 | 93.37% | 71.72% |
<!-- enableFinding(LINE_OVER_80) -->

Observations from this run:

* `HISTOGRAM_MSE` (profiler) improves on min/max across every quality metric
  in the table: +1.93 dB SQNR and +8.36 points mIoU.
* `HISTOGRAM_PERCENTILE` with the default `[0.001, 99.999]` percentiles is
  **worse** than min/max on every quality metric in the table on this model,
  although it wins on three classes (`hair`, `face_skin`, `others`; see below).
  Its fixed tail cut is not error-aware.
* For `HISTOGRAM_MSE`, profiler-based calibration took 13.15 s versus 141.36 s
  for preserve-all-tensors (about 10.7x faster), with slightly lower quality
  (9.36 vs. 9.54 dB SQNR). Both modes clipped 171 of 176 tensors tighter than
  min/max, but the per-tensor ranges differ slightly, so their results are not
  identical.

### Per-Class IoU vs. FP32 Predictions

<!-- disableFinding(LINE_OVER_80) -->
| Class | `MIN_MAX` (Profiler) | `HISTOGRAM_PERCENTILE` (Profiler) | `HISTOGRAM_MSE` (Profiler) | `HISTOGRAM_MSE` (Preserve) |
| :--- | ---: | ---: | ---: | ---: |
| `background` | 90.77% | 86.70% | 93.75% | 94.15% |
| `hair` | 49.68% | 71.01% | 54.44% | 56.64% |
| `body_skin` | 55.95% | 25.66% | 70.33% | 69.01% |
| `face_skin` | 77.45% | 91.40% | 90.26% | 91.11% |
| `clothes` | 57.69% | 49.43% | 70.35% | 73.30% |
| `others` | 35.27% | 40.47% | 37.79% | 46.09% |
<!-- enableFinding(LINE_OVER_80) -->

`HISTOGRAM_MSE` (profiler) beats min/max on all six classes in this run.

---

## 4. Running the Example

### Execution via Python / CLI

With no `--calibration_algorithm`, the script runs the four configurations
above and prints the analysis and comparison tables. To run a single
configuration and write its model to a specific path:

```bash
python3 quantize_selfie_segmentation.py \
  --model_path=/path/to/selfie_multiclass_256x256.tflite \
  --calibration_algorithm=HISTOGRAM_MSE \
  --calibration_method=CALIBRATION_PROFILER_BASED \
  --output_tflite_path=/tmp/selfie_a8w8_histogram_mse.tflite
```

### CLI Flag Reference

<!-- disableFinding(LINE_OVER_80) -->
- `--model_path` (alias of `--float_model_path`): Path to the input floating-point TFLite model.
- `--dataset_dir` (alias of `--image_dir`): Directory of `.jpg`/`.jpeg`/`.png` images (defaults to the bundled portraits in `data/`).
- `--num_calibration_steps` (alias of `--num_calib_samples`): Number of calibration samples; the images are augmented up to this count (default: 32).
- `--calibration_algorithm`: `HISTOGRAM_MSE`, `HISTOGRAM_PERCENTILE`, or `MIN_MAX_UNIFORM_QUANT`. If unset, all four configurations above are run.
- `--calibration_method`: `CALIBRATION_PROFILER_BASED` (default) or `CALIBRATION_PRESERVE_ALL_TENSORS`. Only used together with `--calibration_algorithm`.
- `--output_dir`: Directory for the quantized models and recipes (default: `/tmp/selfie_a8w8_quantization`).
- `--output_tflite_path`: Extra path to write the quantized model to. Only used together with `--calibration_algorithm`.
<!-- enableFinding(LINE_OVER_80) -->

### Running Automated Unit Tests

```bash
pytest quantize_selfie_segmentation_test.py
```
