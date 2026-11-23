<!-- disableFinding(HTML_ATTRIBUTE_MALFORMED) -->
# Histogram Calibration & A8W8 Static-Range Quantization: Selfie Segmentation

This example demonstrates how to perform **histogram calibration** and
static-range uniform integer quantization (**A8W8**) using LiteRT Quantizer
on a multiclass semantic segmentation model
(`selfie_multiclass_256x256.tflite`).

It compares the default `min_max` activation range with the histogram range
methods, selected with `activation_algorithm_params={"range_method": ...}`:

* **`histogram_percentile`** clips each activation to the `[0.001, 99.999]`
  percentiles of its histogram (configurable with `min_percentile` and
  `max_percentile`).
* **`histogram_mse`** picks the range that minimizes the expected quantization
  MSE estimated from the histogram.

See the "Histogram-Based Activation Calibration" section of the LiteRT
Quantizer README for how the histogram range methods work.

---

## 1. Profiler-Based vs. Preserve-All-Tensors Calibration

<!-- disableFinding(LINE_OVER_80) -->
| Feature | `CALIBRATION_PRESERVE_ALL_TENSORS` | `CALIBRATION_PROFILER_BASED` |
| :--- | :--- | :--- |
| **Statistics collection** | Python: after each invoke, iterates over ops and computes min/max and histograms in NumPy from preserved tensor buffers | C++: profiler hooks compute per-invoke min/max and histograms during invoke |
| **Merging across samples** | Python: per-step QSV update | Python: per-step QSV update (same) |
| **Interpreter** | XNNPACK with `preserve_all_tensors=True` (intermediate buffers are kept alive) | XNNPACK with delegate node fusion disabled and per-node profiling enabled |
| **Supported algorithms** | All | Algorithms calibrated with min/max or histograms (not e.g. GPTQ) |
<!-- enableFinding(LINE_OVER_80) -->

---

## 2. Results

Results below come from one run on one workstation (`blaze run -c opt`) using
the google3 default inputs (the `async_segmentation` sample model and image); on
GitHub, the default `portrait.jpg` image gives different numbers. `calibrate()`
time covers only the `Quantizer.calibrate()` call over 32 calibration samples
(the google3 sample image plus 31 augmented crops/flips/jitters) and is
machine-dependent.

All quality metrics compare the quantized model against the **FP32 model's own
predictions** on that google3 sample image, not against ground-truth labels. The
image is also part of the calibration set, so the numbers are in-sample.

* **Top-1 agreement:** fraction of pixels whose argmax class matches FP32.
* **mIoU:** mean IoU between the quantized and FP32 argmax masks.
* **KL Div.:** mean per-pixel `D_KL(P_FP32 || P_quant)`, where both class
  distributions are the softmax of the model's output logits.

The float model is 15.61 MiB; every A8W8 model is 4.43 MiB (3.5x smaller).

<!-- disableFinding(LINE_OVER_80) -->
| Range method | Calibration mode | `calibrate()` time | SQNR (dB) | Output MSE | Cosine Sim. | KL Div. | Top-1 Agreement | mIoU |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `min_max` | Profiler | 3.0 s | 8.24 | 3.46e-01 | 0.92269 | 0.15769 | 91.33% | 64.43% |
| `histogram_percentile` | Profiler | 10.2 s | 9.75 | 2.45e-01 | 0.94581 | 0.10389 | 94.32% | 70.81% |
| `histogram_mse` | Profiler | 10.2 s | 9.72 | 2.46e-01 | 0.94596 | 0.11198 | 93.96% | 68.44% |
| `histogram_mse` | Preserve-all-tensors | 52.6 s | 9.70 | 2.47e-01 | 0.94575 | 0.11162 | 93.86% | 67.69% |
<!-- enableFinding(LINE_OVER_80) -->

Observations from this run:

* Both histogram range methods improve on min/max across every quality metric
  in the table, by about 1.5 dB SQNR and 4 to 6 points mIoU.
* Profiler-based and preserve-all-tensors calibration give nearly the same
  `histogram_mse` quality (9.72 vs. 9.70 dB SQNR), and profiler-based
  calibration is about 5x faster (10.2 s vs. 52.6 s).

---

## 3. Running the Example

### Execution via Python / CLI

With no `--range_method`, the script runs the four configurations
above and prints the comparison table. To run a single configuration and save
`<model_name>.tflite` and `<model_name>_recipe.json` in `--output_dir`:

```bash
python3 quantize_selfie_segmentation.py \
  --float_model_path=/path/to/selfie_multiclass_256x256.tflite \
  --image_paths=/path/to/image_1.jpg,/path/to/image_2.jpg \
  --range_method=histogram_mse \
  --calibration_method=CALIBRATION_PROFILER_BASED \
  --output_dir=/tmp/selfie_a8w8
```

### CLI Flag Reference

<!-- disableFinding(LINE_OVER_80) -->
- `--float_model_path`: Path to the input floating-point TFLite model (defaults to the in-tree `async_segmentation` sample model when present, or downloads it from MediaPipe GCS into `--output_dir`).
- `--image_paths`: Comma-separated calibration and evaluation images (defaults to the in-tree `async_segmentation` sample image when present, or downloads `portrait.jpg` from MediaPipe assets into `--output_dir`).
- `--num_calib_samples`: Number of calibration samples; the images are augmented up to this count (default: 32).
- `--range_method`: Activation range method: `histogram_mse`, `histogram_percentile`, or `min_max`. If unset, all four configurations above are run.
- `--calibration_method`: `CALIBRATION_PROFILER_BASED` (default) or `CALIBRATION_PRESERVE_ALL_TENSORS`. Only used together with `--range_method`.
- `--output_dir`: Directory for downloaded sample assets, quantized models, and recipes (default: `/tmp`).
<!-- enableFinding(LINE_OVER_80) -->

### Running Automated Unit Tests

```bash
pytest quantize_selfie_segmentation_test.py
```
