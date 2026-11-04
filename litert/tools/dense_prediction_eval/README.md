# Dense prediction evaluation

These tools prepare inputs, run a LiteRT model over a dataset, and score depth,
surface normals, or semantic segmentation. The execution tool uses
[`run_model`](../run_model.cc)'s input-list mode: one compiled model and one set
of tensor buffers serve a chunk of samples. Model loading is amortized across
that chunk. Scoring runs on the host and requires NumPy; the optional dataset
adapter also requires Pillow, PyArrow, and `huggingface_hub`.

The current orchestrator accepts a single FLOAT32 `pixel_values` input and
static, batch-one FLOAT32 outputs. Quantization inside the model is independent
of this public interface. No model converter is required.

## Build and test

From the LiteRT repository root:

```bash
bazelisk build -c opt //litert/tools:run_model
bazelisk test //litert/tools/dense_prediction_eval:metrics_test \
  //litert/tools/dense_prediction_eval:evaluate_test \
  //litert/tools/dense_prediction_eval:runner_test
```

For Android, build `run_model` using the repository's Android toolchain and
`--config=android_arm64`. Supply the dispatch/runtime libraries required by the
selected accelerator separately.

## Dataset manifest

A version-1 JSON manifest records the input bytes, targets, output contract,
and sample identity. Paths are relative to the manifest. Raw tensors use
little-endian FLOAT32; targets are NumPy `.npy` files. For example:

```json
{
  "version": 1,
  "outputs": {
    "depth": {
      "shape": [1, 1, 448, 448], "dtype": "float32", "task": "depth",
      "min_depth": 0.001, "max_depth": 10.0
    }
  },
  "samples": [{
    "id": "image_0001",
    "inputs": {"pixel_values": "image_0001/input.raw"},
    "input_sha256": "<SHA-256 of input.raw>",
    "targets": {"depth": "image_0001/depth.npy"}
  }]
}
```

Output names must match model signature names. Each sample can provide targets
for only some outputs; the runner checks all outputs but retains only those
needed for that sample. Sample IDs must be unique filename-safe identifiers.

For segmentation, add `classes` and optional `ignore_label` (default 255) to
the output contract. Predictions have shape `[1, classes, height, width]`,
and labels have shape `[height, width]`. Normals have shape `[1, 3, height,
width]`; targets have shape `[3, height, width]`. Predictions and normal labels
must use the same axis directions and orientation. The scorer does not infer
coordinate transformations from predictions.

## Run and resume

```bash
python -m litert.tools.dense_prediction_eval.run \
  --manifest data/manifest.json --model model.tflite \
  --runner bazel-bin/litert/tools/run_model \
  --runtime-version '<LiteRT revision and build identity>' \
  --accelerator cpu --chunk-size 10 --output results/cpu
```

Use `--device <adb-serial>` and an Android runner for phone execution. For an
NPU-compiled model, also pass `--accelerator npu`, `--library-path` for Android
shared libraries, and repeated `--runner-arg=--flag=value` options, for example
`--runner-arg=--dispatch_library_dir=/data/local/tmp/runtime` and
`--runner-arg=--require_full_acceleration=true`. Additional vendor environment
variables use repeated `--env NAME=VALUE`. Provision matching compiler/runtime
libraries before running; full delegation alone does not establish accuracy.

Each run records model, runner, manifest and input hashes, runtime options,
and selected sample IDs. Phone runs also record device properties and thermal
and battery snapshots. Repeating the command resumes verified completed
samples. A changed recorded identity requires a new output directory. Keep
shared runtime libraries unchanged when resuming and include their build or
release identity in `--runtime-version`.

The driver locks each phone against another instance of this tool and refuses
to start while `run_model` is active. It does not stop unrelated phone apps or
benchmarks. Stop competing workloads before performance measurements.
Interrupts and timeouts stop the owned runner and preserve complete outputs.
`--limit` selects a smaller prefix for smoke tests; use a separate output
directory when changing the selected sample set.

## Score

```bash
python -m litert.tools.dense_prediction_eval.evaluate \
  --manifest data/manifest.json --results results/cpu \
  --reference results/pytorch --output results/scores.json
```

`--reference` is optional. A reference implementation writes the same named
`.raw` outputs under `<reference>/<sample-id>/`. `--sample-ids` accepts a JSON
list and applies the identical subset to predictions and references. Missing,
wrong-sized, or nonfinite output files fail evaluation.

Metric definitions:

* Segmentation: bilinearly resize logits with half-pixel coordinates before
  argmax. Accumulate one confusion matrix across the dataset; exclude classes
  with zero union from mIoU and save the confusion matrix. Ties select the lowest
  class index. Resizing proceeds one class at a time to bound memory usage.
* Depth: bilinearly resize predictions, mask targets outside the configured
  interval `(min_depth, max_depth]`, and clip predictions to the configured
  bounds. Report the mean
  of per-image AbsRel, RMSE, and delta-1 (`max(p/t, t/p) < 1.25`). Units must
  match the target, normally meters. No scale alignment or additional crop
  is applied.
* Normals: bilinearly resize and normalize predicted vectors. Normalize valid
  target vectors and report oriented angular error and threshold accuracies.
  Zero target vectors are ignored; a zero prediction incurs 90 degrees.

These definitions establish a reproducible comparison, not equivalence to a
paper's benchmark protocol. Record dataset revision, preprocessing, crops,
label mapping, and normal coordinate conventions with published results.

## Optional TIPSv2 DPT input preparation

`prepare_tipsv2.py` demonstrates adapters for ADE20K Parquet snapshots and
MTAN-format NYUv2 Parquet snapshots. Pin the dataset revision explicitly.
It samples uniformly without replacement using a fixed seed, saves the exact
prepared pixels, and records source rows. TIPSv2 preprocessing is RGB, PIL
bilinear resize to 448 by 448, NCHW, and scaling to `[0, 1]` without mean/std
normalization. Calibration and evaluation should use disjoint splits.

```bash
python -m litert.tools.dense_prediction_eval.prepare_tipsv2 \
  --dataset ade20k --repo zhoubolei/scene_parse_150 \
  --revision e660d866a1351c70bcf07925d2600b60bd6e3bc3 \
  --split validation --count 500 --seed 2026 --output data/ade20k
```

The NYUv2 adapter prepares depth targets by default. Enable normals only after
verifying the label and model coordinate conventions, using an explicit
`--normal-axis-order` signed permutation. `1,2,3` preserves the supplied axes;
`-1,2,3` flips the first axis. This mapping is recorded in the manifest and
must come from the coordinate contract, not fitting to evaluation predictions.
Labels from different NYUv2 preparations need not share axes.
Dataset access and redistribution remain subject to the source dataset terms.

## Timing

Accuracy runs save per-sample `metrics.json`: `inference_us` surrounds
`CompiledModel::Run`, while `run_and_output_us` also includes output readback
and serialization. Neither is integrated application latency. Do not treat
output-file-heavy accuracy runs as sustained accelerator benchmarks.

For repeated inference on a fixed prepared input, use the runner directly:

```bash
run_model --graph=model.tflite --input_dir=prepared_input \
  --warmup_iterations=5 --iterations=30 --minimum_duration_ms=300000 \
  --timing_output=timings.json
```

The loop meets both the iteration count and minimum duration. Warmup runs are
excluded. The JSON contains individual `inference_us` values; preprocessing,
loading, and output serialization are outside these timings. These benchmark
flags cannot be combined with `--input_list`.
