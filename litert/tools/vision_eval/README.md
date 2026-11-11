<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# Evaluate vision models with LiteRT

`litert_vision_eval` runs classification, semantic segmentation, metric depth,
and surface-normal evaluation in C++. CPU, MLDrift GPU, and vendor NPU execution
share the same runner, preprocessing, scoring, and checkpoint format. One run
can score multiple named outputs against different tasks.

## Supported evaluation tasks

The tool implements four scoring tasks. Dataset preparation is provided for
ImageNet-1K classification, TorchVision segmentation on COCO/Pascal VOC, and
TIPSv2-style ADE20K/NYUv2 evaluation:

| Task | Provided dataset workflow | What the evaluation compares | Metrics |
| --- | --- | --- | --- |
| **Image classification** | **ImageNet-1K validation** images and class labels | A predicted class for each image against its class label | Top-1 and top-5 accuracy |
| **Semantic segmentation** | **COCO val2017 with VOC labels** or **Pascal VOC validation**, with 21 classes including background; **ADE20K validation**, with 150 classes | A predicted class for every pixel against the segmentation label map | Mean intersection over union (mIoU), global confusion matrix |
| **Metric depth estimation** | **NYUv2 test**, in MTAN Parquet format | A predicted distance in meters at each pixel against the ground-truth depth map | Absolute relative error (AbsRel), RMSE, delta1 accuracy |
| **Surface-normal estimation** | **NYUv2 test**, in MTAN Parquet format, with an explicit normal-axis mapping | A predicted surface direction at each pixel against its ground-truth normal vector | Mean and exact median angular error; fractions below 11.25, 22.5, and 30 degrees |

For example, a ResNet or MobileNet classifier can be evaluated on ImageNet.
TorchVision DeepLabV3, FCN, and LRASPP exports can be evaluated on COCO with
VOC labels or on Pascal VOC, subject to the model's input shape contract.
A TIPSv2 model can be evaluated for segmentation on ADE20K, or for depth and
surface normals on NYUv2. A multi-task model can produce several named outputs
in one invocation; each sample is scored against the targets available for it.
The NYUv2 adapter prepares depth targets by default and enables normal targets
when `--normal_axis_order` is supplied.

Choose the preparation workflow for your task:

- **ImageNet classification:** follow [Start with three ImageNet images](#start-with-three-imagenet-images).
  The exporter retains encoded images and writes a TSV of image paths and labels.
- **TorchVision's published COCO segmentation protocol:** follow
  [TorchVision segmentation on COCO](#torchvision-segmentation-on-coco).
  This preserves aspect ratio and requires a compatible model for each input
  shape. A square-only subset is explicitly identified as a biased subset.
- **TorchVision semantic segmentation on Pascal VOC:** follow
  [TorchVision segmentation on Pascal VOC](#torchvision-segmentation-on-pascal-voc).
  This includes a Linux FP32 WebGPU example.
- **ADE20K segmentation or NYUv2 depth/normals with TIPSv2:** follow
  [Optional TIPSv2 dataset preparation](#optional-tipsv2-dataset-preparation).
  The adapter writes prepared raw tensors and a protobuf manifest for TIPSv2's
  input/output contract.
- **Another model or dataset using one of these four tasks:** supply matching
  inputs, targets, preprocessing, and tensor descriptions through
  [Prepared tensors and dense tasks](#prepared-tensors-and-dense-tasks).
  The scoring code is shared; a different model may need its own preparation
  adapter or manifest.

All four tasks support live model execution, comparison against a baseline
model, and [offline scoring of saved predictions](#baselines-saved-predictions-and-result-files).

## Model interfaces and file formats

The executable accepts static shapes and FP32, INT8, or UINT8 model interfaces.
Quantized interfaces must use per-tensor scales and zero points. Prepared raw
inputs can bind multiple signature inputs. Encoded images use one batch-one RGB
input in NCHW or NHWC layout. Dynamic shapes and batched dataset evaluation are
not supported. Individual tensors are limited to 512 MiB and input protobuf files
to 64 MiB.

Configuration, manifests, results, and checkpoints use
[evaluation.proto](evaluation.proto). Human-readable files end in `.txtpb`;
binary protobuf files end in `.binpb`. The evaluator does not use a JSON library.

## Build

From the repository root:

```sh
mkdir -p .cache .bazelisk-cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //litert/tools:litert_vision_eval
```

The normal executable includes LiteRT's runtime. The dynamic variant uses a
separate `libLiteRt.so`, allowing the evaluator to run against a selected runtime
build. For a separately supplied runtime, build
`//litert/tools:litert_vision_eval_dynamic_runtime` with
`--define=litert_runtime_link_mode=dynamic`. Select the library directory with
`--runtime_library_dir=DIR`; it is also the place to supply matching accelerator
libraries. See the [Android guide](../../../docs/instructions/VISION_EVAL_ANDROID.md)
for cross compilation and CPU/GPU/NPU examples.

## Start with three ImageNet images

Accept the dataset conditions on
[ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k), then
use an account with access:

```sh
uvx hf auth login
uvx hf download ILSVRC/imagenet-1k \
  data/validation-00000-of-00014.parquet --repo-type dataset \
  --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5 \
  --local-dir eval-data/hf-imagenet

uv run --no-project --python 3.13 --with absl-py --with pyarrow \
  litert/tools/vision_eval/export_imagenet_val_jpegs.py \
  --input_dir=eval-data/hf-imagenet/data \
  --out_dir=eval-data/imagenet --max_samples=3
```

The exporter retains original encoded image bytes and writes `manifest.tsv`:

```text
id	label	path
0	65	images/00000000.JPEG
1	970	images/00000001.JPEG
```

Paths are relative to the manifest directory, or to an explicit `--image_root`.
Labels are zero-based class indices. The optional TSV header has three tab-separated
columns. `--max_samples=0` extracts all rows from the downloaded shards; a small
extraction limit does not reduce the shard download size.

Download a model and run it with its checkpoint's preprocessing:

```sh
uvx hf download litert-community/resnet18 resnet18.tflite \
  --revision 7eabef429f15b4fcd4db3e225d81bd0fd7860c4f \
  --local-dir eval-data/models

bazel-bin/litert/tools/litert_vision_eval \
  --model=eval-data/models/resnet18.tflite \
  --manifest=eval-data/imagenet/manifest.tsv --output_dir=eval-data/resnet18-cpu \
  --accelerator=cpu --num_threads=4 --xnnpack_only=true \
  --resize=256 --crop_h=224 --crop_w=224 --max_samples=3
```

Image preprocessing defaults to antialiased bilinear resize of the shorter side
to 232, a 224x224 center crop, and TorchVision RGB mean/stddev normalization.
Use the resize, interpolation, crop, and normalization specified by your model.
For array settings, use a config:

```textproto
model: "eval-data/models/resnet18.tflite"
manifest: "eval-data/imagenet/manifest.tsv"
output_dir: "eval-data/resnet18-cpu"
max_samples: 3
backend { accelerator: "cpu" num_threads: 4 xnnpack_only: true }
preprocessing {
  interpolation: "bilinear"
  resize: 256
  crop_h: 224
  crop_w: 224
  antialias: true
  mean: [0.485, 0.456, 0.406]
  stddev: [0.229, 0.224, 0.225]
}
```

Save it as `run.txtpb` and pass `--config=run.txtpb`. Explicit CLI flags override
config fields regardless of argument order. CLI flags use `--name=value`.
Scalar backend and preprocessing fields also work as flags; repeated fields such
as mean and stddev belong in the protobuf config. Run `--help` for the main options.

## Prepared tensors and dense tasks

Prepare contiguous, little-endian raw files without a file header. Describe the
input and output shapes, types, layouts, and signature names in the manifest.
`name` is a filename-safe dataset alias; `binding_name` is the model signature name
and defaults to `name`. Input data must match the candidate model's interface.
INT8/UINT8 contracts can include `scale` and `zero_point`; supplied values are
checked against the model. Otherwise the model supplies them during inference.

For example, this manifest scores one depth output:

```textproto
version: 1
inputs {
  name: "image"
  binding_name: "pixel_values"
  shape: [1, 3, 448, 448]
  type: FLOAT32
  layout: NCHW
}
outputs {
  tensor { name: "depth" shape: [1, 1, 448, 448] type: FLOAT32 layout: NCHW }
  scoring { task: DEPTH min_depth: 0.001 max_depth: 10 }
}
samples {
  id: "nyuv2_test_000001"
  inputs { name: "image" file { path: "sample/input.raw" } }
  targets {
    name: "depth"
    file { path: "sample/depth.raw" }
    tensor { shape: [288, 384] type: FLOAT32 }
  }
}
```

Add more `outputs` and `targets` for a multi-task model. Outputs without a target
are not scored on that sample, but their named runtime tensors are still checked.
Use `signature_index` to select another model signature. The model is compiled
once per execution side and its buffers are reused for every sample.

Targets use these contracts:

| Task | Prediction | Target | Metrics |
| --- | --- | --- | --- |
| Classification | Class scores | Sample `label` | Top-1 and top-5 accuracy |
| Segmentation | Batch-one NCHW/NHWC logits | UINT8 or INT32 HW labels | Global confusion matrix and mean IoU |
| Depth | Batch-one, one-channel map | FLOAT32 HW, in meters | AbsRel, RMSE, delta1, averaged equally over valid images |
| Normals | Batch-one, three-channel vectors | FLOAT32 CHW | Oriented angular mean, exact global median, fractions below 11.25/22.5/30 degrees |

Dense predictions are resized to the target size with edge-clamped, half-pixel
bilinear interpolation and no antialiasing. Segmentation resizes each logit plane
before argmax; ties select the lowest channel index. Absent classes are excluded
from mean IoU. `classes` defaults to 150 and `ignore_label` to 255.

Depth accepts finite targets in `(min_depth, max_depth]`, clamps predictions to
those bounds, and uses a strict `< 1.25` threshold for delta1. Normals exclude
nonfinite or near-zero targets; zero prediction vectors have 90-degree error.
The coordinate convention must match the model; the evaluator does not infer it.
An optional `targets.mask` points to a UINT8/BOOL HW mask, where nonzero is valid.
Metrics with no valid observations omit their numeric values in protobuf.

## Optional TIPSv2 dataset preparation

[prepare_tipsv2.py](prepare_tipsv2.py) is a host adapter for ADE20K Parquet
snapshots with original labels and MTAN-format NYUv2 Parquet snapshots. It
uniformly selects samples across the complete split using a recorded seed, pins
the resolved dataset commit, and records preprocessing and source row identities.
Its output is a protobuf manifest plus raw tensors. The C++ evaluator needs no
Python or Hugging Face client.

Generate the Python protobuf module first:

```sh
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //litert/tools/vision_eval:evaluation_py_proto

PYTHONPATH="$PWD/bazel-bin" \
  uv run --no-project --python 3.13 --with absl-py --with protobuf \
  --with numpy --with pillow --with pyarrow --with huggingface-hub \
  litert/tools/vision_eval/prepare_tipsv2.py \
  --dataset=nyuv2 --repo=DATASET_REPO --revision=DATASET_COMMIT \
  --split=test --count=3 --seed=2026 --output=eval-data/nyuv2
```

Select a repository whose Parquet layout matches the adapter. NYUv2 requires RGB
in `[0,1]` with shape `[3,288,384]`, depth in meters with shape `[1,288,384]`, and
normals `[3,288,384]`. Normal targets are disabled until you pass an explicit
signed, one-based axis mapping, such as `--normal_axis_order=1,-2,3`. ADE20K maps
original labels 1..150 to 0..149 and ignores original background label 0.
The adapter uses Pillow bilinear resize to 448x448 and RGB scaling to `[0,1]`,
without ImageNet mean/stddev normalization. It requires a new output directory.

## TorchVision segmentation on COCO

The published DeepLabV3, FCN, and LRASPP weights use **COCO val2017 with VOC
labels**, rather than the Pascal VOC validation split. The adapter
[prepare_coco2017.py](prepare_coco2017.py) follows the
[TorchVision reference](https://github.com/pytorch/vision/tree/1f1a920d685d46dfbe187934b785b7ecb7a182f2/references/segmentation):

- Use all 5,000 validation images for a full evaluation. Retain images without
  objects in the selected classes; the training-only object/area filter does
  not apply to validation.
- Map COCO category IDs `[0,5,2,16,9,44,6,3,17,62,21,67,18,19,4,1,64,20,63,7,72]`
  by index to the 21 VOC classes. Background is 0. Unselected COCO categories
  remain background; they are not void pixels.
- Include crowd annotations. Pixels covered by multiple selected instances,
  including two instances of the same class, receive the ignored label 255.
- Resize the RGB image's shorter side to 520 using antialiased PIL bilinear
  interpolation and preserve aspect ratio, with TorchVision's integer output
  dimensions. Convert to FP32 NCHW, divide by 255, and apply ImageNet mean/stddev.
- Resize the semantic target to the same inference grid using nearest-neighbor
  interpolation. Score logit argmax on that grid, including background and
  excluding label 255. Sum the confusion matrices over the complete selection
  before calculating mIoU. Do not average per-image or per-shape mIoUs.

The confusion matrix also gives global pixel accuracy: its diagonal sum divided
by the total count. TorchVision averages IoU over all 21 classes. This tool
excludes zero-union classes and reports `evaluated_classes`; compare the two
mIoUs only when all 21 classes have nonzero union. Record the sample count,
selection, and per-class coverage with subset results.

**Check model shapes before evaluating.** The published fixed-520x520 exports
cannot accept most inputs produced by this protocol. The complete COCO split
produces 483 resized shapes at shorter-side 520, so the adapter emits a separate
manifest for each shape. Each manifest requires a matching model. Preparing
these manifests does not make a fixed-shape model support other shapes.

The six fixed-shape exports also embed constant final resize dimensions, and
DeepLabV3/LRASPP embed intermediate resize dimensions. Editing only their input
metadata cannot reproduce the variable-size source graphs. They need exports
with correct shape expressions, or separate exports for the required shapes.
A filename containing `dynamic_wi8_afp32` describes dynamic-range weight
quantization; it does not establish support for dynamic input shapes.

Download the original images and the pinned annotation snapshot:

```sh
mkdir -p eval-data/coco
curl -fL https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip \
  -o eval-data/coco/val2017.zip
unzip -q eval-data/coco/val2017.zip -d eval-data/coco

uvx hf download pcuenq/coco2017-instances instances_val2017.json \
  --repo-type dataset --revision 0d2b1478a7c692d6227612841b28512da3def874 \
  --local-dir eval-data/coco/annotations
```

The original image ZIP has SHA256
`4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`.
The annotation SHA256 below also matches the file from COCO's original
`annotations_trainval2017.zip`. After building `evaluation_py_proto` as above:

```sh
PYTHONPATH="$PWD/bazel-bin" \
  uv run --no-project --python 3.13 --with absl-py --with protobuf \
  --with numpy --with pillow --with pycocotools \
  litert/tools/vision_eval/prepare_coco2017.py \
  --annotations=eval-data/coco/annotations/instances_val2017.json \
  --annotation_sha256=e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f \
  --images=eval-data/coco/val2017 --output=eval-data/coco-prepared --count=0
```

This writes `HEIGHTxWIDTH/manifest.txtpb` and its prepared inputs/targets, a
global `image_ids.txt`, source image hashes, and the recorded protocol. Source
annotations, tensors, selection, reference revision, and preprocessing are
identified in each manifest. Use `--count=N --seed=2026` for a reproducible uniform
subset of the complete split, grouped by shape after selection. All-prepared
FP32 inputs need tens of GB; choose a data volume with enough free space.

For an existing 520x520 model, add `--input_shape=520,520` to prepare only the
176 compatible validation images. This includes a few nearly square images
whose reference resize rounds to 520x520. The selection is shape-biased and
has no potted-plant ground-truth pixels. Report it as a
**COCO shape-compatible subset**, with its coverage, rather than a full or
representative COCO benchmark.

The default signature names are `x` and `upsample_bilinear2d_1` for DeepLabV3
and LRASPP. For FCN, pass `--output_name=upsample_bilinear2d`. Use the Linux
WebGPU command below with the manifest whose shape matches the model.

## TorchVision segmentation on Pascal VOC

[prepare_voc2012.py](prepare_voc2012.py) prepares validation samples for the FP32
TorchVision DeepLabV3, FCN, and LRASPP exports. The published exports use NCHW
inputs `[1,3,520,520]` and logits `[1,21,520,520]`. This workflow preserves
background class 0, foreground classes 1..20, and void label 255.

Generate `evaluation_py_proto` with the command in the TIPSv2 preparation section
above. Download the validation shard and prepare three samples:

```sh
uvx hf download shijli/voc2012 \
  segmentation/validation-00000-of-00001.parquet --repo-type dataset \
  --revision fba53aa390b729cb2bb848de59e1439deea3e88f \
  --local-dir eval-data/voc2012

PYTHONPATH="$PWD/bazel-bin" \
  uv run --no-project --python 3.13 --with absl-py --with protobuf \
  --with numpy --with pillow --with pyarrow \
  litert/tools/vision_eval/prepare_voc2012.py \
  --input_dir=eval-data/voc2012/segmentation --output=eval-data/voc3 \
  --max_samples=3 --revision=fba53aa390b729cb2bb848de59e1439deea3e88f
```

The helper resizes RGB images directly to the fixed 520x520 export shape with
Pillow bilinear interpolation, scales them to `[0,1]`, and applies ImageNet
mean/stddev normalization. Ground-truth masks retain their original resolution.
The evaluator resizes logits to each mask before argmax. The prepared manifest
records this fixed-square protocol. Three-sample scores from this protocol are
smoke evaluations; they do not reproduce the published full COCO/VOC benchmark.

The default signature names are `x` and `upsample_bilinear2d_1`, as used by the
DeepLabV3 and LRASPP exports. For FCN, pass `--output_name=upsample_bilinear2d`
when preparing its manifest. Tensor names and shapes must match your export.

### Linux FP32 WebGPU execution

Build `litert_vision_eval_dynamic_runtime` with the dynamic build option shown
in [Build](#build). Keep the built runtime and WebGPU accelerator together:

```sh
mkdir -p eval-data/runtime
cp bazel-bin/litert/c/libLiteRt.so eval-data/runtime/
cp .bazel-output/external/litert_prebuilts/linux_x86_64/libLiteRtWebGpuAccelerator.so \
  eval-data/runtime/

uvx hf download litert-community/deeplabv3_resnet50 deeplabv3_resnet50.tflite \
  --revision d0c4b73db7c77a68e7cc33cb1a3615f11b616c43 \
  --local-dir eval-data/deeplabv3_resnet50

LD_LIBRARY_PATH="$PWD/eval-data/runtime" \
  bazel-bin/litert/tools/litert_vision_eval_dynamic_runtime \
  --runtime_library_dir="$PWD/eval-data/runtime" \
  --model=eval-data/deeplabv3_resnet50/deeplabv3_resnet50.tflite \
  --manifest=eval-data/voc3/manifest.txtpb --output_dir=eval-data/deeplab-webgpu \
  --accelerator=gpu --gpu_backend=webgpu --gpu_precision=fp32 \
  --gpu_use_ir_model=true --require_full_acceleration=true --max_samples=3
```

The accelerator library path above assumes the `.bazel-output` output base used
in this guide. The selected GPU adapter is reported in stderr. Setting
`require_full_acceleration` rejects partial delegation.

For a CPU XNNPACK comparison, run the same model and manifest with
`--accelerator=cpu --num_threads=8 --xnnpack_only=true --dump_predictions=true`
into a separate output directory, then give its `samples` directory to the
WebGPU run with `--reference_dir=CPU_RESULTS/samples`. This records CPU and GPU
mIoU plus numerical differences in one summary.

The following TorchVision models use this segmentation workflow:

- `deeplabv3_resnet50`
- `deeplabv3_resnet101`
- `deeplabv3_mobilenet_v3_large`
- `fcn_resnet50`
- `fcn_resnet101`
- `lraspp_mobilenet_v3_large`

Choose numerical parity tolerances separately from dataset accuracy metrics;
small FP32 logit differences may leave the predicted class unchanged.

## Baselines, saved predictions, and result files

Add `--baseline_model=MODEL --baseline_accelerator=cpu` for a separate reference
CPU model. It uses reference kernels, one thread by default, and no vendor
accelerator options. `--baseline_accelerator=inherit` uses the same backend options
for both models. A compatible FP32 baseline can be compared with an INT8/UINT8
candidate; each receives the same real inputs with its own IO quantization.
Raw inputs must have compatible shapes and binding names.

`--require_output_parity=true` turns `atol`/`rtol` comparison into a pass/fail gate.
For INT8 outputs, `--compare_int8_codes=true` compares code-space differences using
`max_code_error` and `mean_code_error` instead. Classification can additionally
require `--require_prediction_agreement=true`.

A run directory contains:

- `summary.txtpb` by default, or `summary.binpb` with
  `--summary_format=binaryproto`: status, provenance, task metrics, execution
  setup/inference times, and optional baseline comparisons.
- `identity.binpb`: the configuration, models, executable, runtime/plugin library
  identities, device/OS metadata, signature contracts, delegation metrics, and selected sample IDs.
- `resolved_manifest.txtpb`: the bound tensor contracts and resolved data paths.
- `samples/ID/result.binpb`: completion state, source file paths, typed
  tensor descriptions, per-sample metrics, and timings.

`--dump_predictions=true` writes each named output as dequantized FLOAT32 bytes
under `samples/ID/NAME.raw`; baseline dumps are in `samples/ID/baseline/`.
`--dump_inputs=true` writes the actual model input bytes under `inputs/input_INDEX.raw`.
The sample protobuf records the stored type, shape, quantization, and optional dump path.

With `protoc` installed, inspect a binary sample checkpoint as text:

```sh
protoc -I . --decode=litert.vision.SampleResult \
  litert/tools/vision_eval/evaluation.proto < RESULTS/samples/ID/result.binpb
```

To score saved predictions without loading a runtime:

```sh
bazel-bin/litert/tools/litert_vision_eval \
  --score_only=true \
  --manifest=eval-data/resnet18-cpu/resolved_manifest.txtpb \
  --predictions_dir=eval-data/resnet18-cpu/samples \
  --output_dir=eval-data/offline-score
```

Offline manifests must declare output contracts. Prediction files are always
FLOAT32, in the declared output layout. `--reference_dir=DIR` reads the same
`ID/NAME.raw` structure for an offline comparison; it can also compare a live
model against saved outputs.

## Resume and validation

Repeat the identical command to reuse completed samples. The evaluator verifies
run provenance and recorded file paths, and resumes only samples with complete,
valid protobuf checkpoints. An interrupted sample is evaluated again. Changes
to models, configuration, selected samples, manifest, executable, or recorded
libraries require a new output directory. Keep source files and normals error
spools available until evaluation and resumption are finished.

A directory lock prevents concurrent writers. Each file is published by atomic
replacement; a sample's completion record is written after its data and metrics.
Runtime failures produce a failed summary and a nonzero exit status. Configuration
parse errors are reported to stderr. A provenance mismatch preserves the existing
run's summary.

The exact normals median uses sorted runs with a 64 MiB sorting buffer and merges
at most 64 files at a time. Other accumulators retain only one aggregate per output;
predictions are released after each sample. This keeps memory independent of the
number of evaluated images, apart from small file-identity lists for the median.

Run the targeted C++ and NumPy parity checks with:

```sh
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" test -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //litert/tools/vision_eval:all
```

The optional preparation helpers have standalone tests with their host dependencies:

```sh
PYTHONPATH="$PWD/bazel-bin" \
  uv run --no-project --python 3.13 --with absl-py --with protobuf \
  --with numpy --with pillow litert/tools/vision_eval/prepare_tipsv2_test.py
PYTHONPATH="$PWD/bazel-bin" \
  uv run --no-project --python 3.13 --with absl-py --with protobuf \
  --with numpy --with pillow --with pyarrow \
  litert/tools/vision_eval/prepare_voc2012_test.py
PYTHONPATH="$PWD/bazel-bin" \
  uv run --no-project --python 3.13 --with absl-py --with protobuf \
  --with numpy --with pillow --with pycocotools \
  litert/tools/vision_eval/prepare_coco2017_test.py
PYTHONPATH="$PWD/bazel-bin" \
  uv run --no-project --python 3.13 --with absl-py --with pyarrow \
  litert/tools/vision_eval/export_imagenet_val_jpegs_test.py
```
