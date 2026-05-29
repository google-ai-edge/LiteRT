# LiteRT to MediaTek Neuron Compiler Support

This document covers **compile-time** behavior of the MediaTek Neuron compiler
plugin: supported types/ops, compile-time options, partitioning, and how to
extend the plugin. For **runtime** behavior (bytecode loading, library
resolution, lifecycle), see
[`MediaTek_Neuro_Dispatcher.md`](../dispatch/MediaTek_Neuro_Dispatcher.md).

Source of truth: `litert/vendors/mediatek/compiler/`.

## Building the Compiler Plugin

The compiler plugin is built as a shared library
(`libLiteRtCompilerPlugin_MediaTek.so`) that LiteRT loads during AOT
compilation.

```bash
bazel build //litert/vendors/mediatek/compiler:compiler_plugin_so
```

The plugin links against `libneuron_adapter.so` from the NeuroPilot SDK. The
exact dependency wiring lives in
[`litert/vendors/mediatek/mediatek_build_defs.bzl`](../mediatek_build_defs.bzl)
and is selected per-SDK version (v7 / v8 / v9). On host builds, the SDK is
resolved via the `@neuro_pilot` external repository.

It is also packaged into the LiteRT Python wheel (see
`ci/tools/python/wheel/BUILD`) and consumed by the Python AOT backend in
`litert/python/aot/vendors/mediatek/mediatek_backend.py`.

## Compile-time Options

Options are configured through the C API in
[`litert/c/options/litert_mediatek_options.h`](../../../c/options/litert_mediatek_options.h)
or via `apply_plugin` / benchmark CLI flags declared in
[`litert/tools/flags/vendors/mediatek_flags.h`](../../../tools/flags/vendors/mediatek_flags.h).
They can also be supplied as a TOML payload through
`LrtCreateMediatekOptionsFromToml` (the same payload the dispatcher consumes
via the `"mediatek"` opaque-options key).

| Option | CLI flag | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `NeronSDKVersionType` | `--mediatek_sdk_version_type` | v8 | Selects NeuroPilot SDK version (v7 / v8 / v9). Determines which `libneuron_adapter` symbols are loaded and which `third_party/neuro_pilot/vN_latest` host path is searched. |
| `GemmaCompilerOptimizations` | `--mediatek_enable_gemma_compiler_optimizations` | false | Enables Gemma-specific compiler passes. |
| `PerformanceMode` | `--mediatek_performance_mode_type` | `SustainedSpeed` | Sets `NeuronAdapterPreferenceCode` at compile time: `LowPower`, `FastSingleAnswer`, `SustainedSpeed`, `TurboBoost`. |
| `L1CacheOptimizations` | `--mediatek_enable_l1_cache_optimizations` | false | Enables L1 cache optimization pass. |
| `OptimizationHint` | `--mediatek_optimization_hint` | `Normal` | Bitmask hint for the compiler: `Normal`, `LowLatency`, `DeepFusion`, `BatchProcessing`. |
| `DisableDlaDirRemoval` | `--mediatek_disable_dla_dir_removal` | false | Keeps the temporary DLA scratch directory after compilation (debug). |
| `MediatekDlaDir` | `--mediatek_dla_dir` | (auto) | Override the DLA scratch directory path. |
| `AotCompilationOptions` | `--mediatek_aot_compilation_options` | SDK default | Free-form passthrough string forwarded to the Neuron compiler. |
| `UseGetSupportedOperations` | `--mediatek_use_get_supported_operations` | true | When true, partitioning queries the Neuron compiler via `GetSupportedOperations` for a per-op verdict on top of the static `IsOpSupported` list. When false, only the static list plus per-op constraint checks are used. |

## Supported Data Types

The compiler maps `litert::ElementType` to a corresponding `NeuronTensorType`.
The final Neuron type depends on the element type and its quantization scheme.

| LiteRT Element Type | Mapped Neuron Tensor Type | Note |
| :--- | :--- | :--- |
| `Float32` | `NEURON_TENSOR_FLOAT32` | |
| `Float16` | `NEURON_TENSOR_FLOAT16` | |
| `Int32`   | `NEURON_TENSOR_INT32` | |
| `Int16`   | `NEURON_TENSOR_QUANT16_SYMM` | Only per-tensor quantization is supported. |
| `UInt8`   | `NEURON_TENSOR_QUANT8_ASYMM` (per-tensor) or `NEURON_EXT_TENSOR_QUANT8_ASYMM_PER_CHANNEL` (per-channel) | |
| `Int8`    | `NEURON_TENSOR_QUANT8_SYMM` / `NEURON_TENSOR_QUANT8_ASYMM_SIGNED` (per-tensor) or `NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL` / `NEURON_EXT_TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL` (per-channel) | The final type depends on the quantization scheme. |
| `Int4`    | `NEURON_EXT_TENSOR_QUANT4_SYMM` / `NEURON_EXT_TENSOR_QUANT4_ASYMM_SIGNED` (per-tensor) or `NEURON_EXT_TENSOR_QUANT4_SYMM_PER_CHANNEL` / `NEURON_TENSOR_QUANT4_ASYMM_SIGNED_PER_CHANNEL` (per-channel) | The final type depends on the quantization scheme. |
| `Int2`    | `NEURON_EXT_TENSOR_QUANT2_SYMM` / `NEURON_EXT_TENSOR_QUANT2_ASYMM_SIGNED` (per-tensor) or `NEURON_EXT_TENSOR_QUANT2_SYMM_PER_CHANNEL` / `NEURON_EXT_TENSOR_QUANT2_ASYMM_SIGNED_PER_CHANNEL` (per-channel) | The final type depends on the quantization scheme. |
| `Bool`    | `NEURON_TENSOR_BOOL8` | |
| `Int64`   | `NEURON_TENSOR_INT32` (per-tensor / unquantized) or `NEURON_EXT_TENSOR_INT32_SYMM_PER_CHANNEL` (per-channel) | Only supported for constant tensors (weights). Values are cast from `Int64` to `Int32`. |

## Supported Quantization Types

| LiteRT Quantization Type | Supported | Note |
| :--- | :--- | :--- |
| `kLiteRtQuantizationPerTensor` | **Yes** | |
| `kLiteRtQuantizationPerChannel` | **Yes** | |
| `kLiteRtQuantizationBlockWise` | **No** | Explicitly marked as unsupported. |
| `kLiteRtQuantizationNone` | **Yes** | |

<!-- LINT.IfChange(supported_ops) -->
## Supported Operations

The following `LiteRtOpCode` operations are supported. For an operation to be
delegated, it must also pass a set of constraints checked by
the `VerifyCommonOp` function.

| LiteRT Op Code | Note |
| :--- | :--- |
| `kLiteRtOpCodeTflAbs` | Legalized to `NEURON_ABS`. |
| `kLiteRtOpCodeTflAdd` | Legalized to `NEURON_ADD`. Supports fused activation. |
| `kLiteRtOpCodeTflAveragePool2d` | Legalized to `NEURON_AVERAGE_POOL_2D`. Supports padding, stride, filter size, and fused activation. |
| `kLiteRtOpCodeTflBatchMatmul` | Legalized to `NEURON_BATCH_MATMUL`. Supports `adj_x` and `adj_y` attributes. |
| `kLiteRtOpCodeTflCast` | Legalized to `NEURON_CAST`. |
| `kLiteRtOpCodeTflConcatenation` | Legalized to `NEURON_CONCATENATION`. Supports `axis` attribute. |
| `kLiteRtOpCodeTflConv2d` | Legalized to `NEURON_CONV_2D`. Supports padding, stride, fused activation, data format, and dilation. |
| `kLiteRtOpCodeTflDequantize` | Legalized to `NEURON_DEQUANTIZE`. |
| `kLiteRtOpCodeTflDepthwiseConv2d` | Legalized to `NEURON_DEPTHWISE_CONV_2D`. Supports padding, stride, depth multiplier, fused activation, data format, and dilation. |
| `kLiteRtOpCodeTflDiv` | Legalized to `NEURON_DIV`. Supports fused activation. |
| `kLiteRtOpCodeTflFullyConnected` | Legalized to `NEURON_FULLY_CONNECTED`. Supports fused activation. Adds a zero bias if one is not provided. |
| `kLiteRtOpCodeTflGelu` | Legalized to `NEURON_GELU_V2` with `approximate` set to true. |
| `kLiteRtOpCodeTflGreater` | Legalized to `NEURON_GREATER`. |
| `kLiteRtOpCodeTflHardSwish` | Legalized to `NEURON_HARD_SWISH`. |
| `kLiteRtOpCodeTflLogistic` | Legalized to `NEURON_LOGISTIC`. |
| `kLiteRtOpCodeTflMaximum` | Legalized to `NEURON_MAXIMUM`. |
| `kLiteRtOpCodeTflMaxPool2d` | Legalized to `NEURON_MAX_POOL_2D`. Supports padding, stride, filter size, and fused activation. |
| `kLiteRtOpCodeTflMean` | Legalized to `NEURON_MEAN`. Supports `keep_dims` attribute. |
| `kLiteRtOpCodeTflMinimum` | Legalized to `NEURON_MINIMUM`. |
| `kLiteRtOpCodeTflMul` | Legalized to `NEURON_MUL`. Supports fused activation. |
| `kLiteRtOpCodeTflPad` | Legalized to `NEURON_PAD`. |
| `kLiteRtOpCodeTflPadv2` | Legalized to `NEURON_PAD_V2`. |
| `kLiteRtOpCodeTflPrelu` | Legalized to `NEURON_PRELU`. |
| `kLiteRtOpCodeTflLeakyRelu` | Legalized to `NEURON_PRELU` via treating alpha in LeakyRelu as the input tensor of Prelu. |
| `kLiteRtOpCodeTflQuantize` | Legalized to `NEURON_QUANTIZE`. |
| `kLiteRtOpCodeTflReduceMax` | Legalized to `NEURON_REDUCE_MAX`. Supports `keep_dims` attribute. |
| `kLiteRtOpCodeTflRelu` | Legalized to `NEURON_RELU`. |
| `kLiteRtOpCodeTflReshape` | Legalized to `NEURON_RESHAPE`. |
| `kLiteRtOpCodeTflResizeBilinear` | Legalized to `NEURON_RESIZE_BILINEAR`. Supports `align_corners` and `half_pixel_centers`. |
| `kLiteRtOpCodeTflResizeNearestNeighbor` | Legalized to `NEURON_RESIZE_NEAREST_NEIGHBOR`. Supports `align_corners` and `half_pixel_centers`. |
| `kLiteRtOpCodeTflRsqrt` | Legalized to `NEURON_RSQRT`. |
| `kLiteRtOpCodeTflSlice` | Legalized to `NEURON_SLICE`. |
| `kLiteRtOpCodeTflSoftmax` | Legalized to `NEURON_SOFTMAX`. Supports `beta` attribute. |
| `kLiteRtOpCodeTflSplit` | Legalized to `NEURON_SPLIT`. Supports `num_splits` attribute. |
| `kLiteRtOpCodeTflSqrt` | Legalized to `NEURON_SQRT`. |
| `kLiteRtOpCodeTflSquaredDifference` | Legalized as an OEM extension: `MTKEXT_SQUARED_DIFFERENCE`. |
| `kLiteRtOpCodeTflStridedSlice` | Legalized to `NEURON_STRIDED_SLICE`. Supports `begin_mask`, `end_mask`, and `shrink_axis_mask`. |
| `kLiteRtOpCodeTflSub` | Legalized to `NEURON_SUB`. Supports fused activation. |
| `kLiteRtOpCodeTflSum` | Legalized to `NEURON_REDUCE_SUM`. Supports `keep_dims` attribute. |
| `kLiteRtOpCodeTflTanh` | Legalized to `NEURON_TANH`. |
| `kLiteRtOpCodeTflTile` | Legalized to `NEURON_TILE`. |
| `kLiteRtOpCodeTflTranspose` | Legalized to `NEURON_TRANSPOSE`. |
| `kLiteRtOpCodeTflTransposeConv` | Legalized to `NEURON_TRANSPOSE_CONV`. Supports padding and stride. Adds a zero bias if one is not provided. |
| `kLiteRtOpCodeTflUnpack` | Legalized as an OEM extension: `unpackmtk`. Supports `axis` attribute. |
| `kLiteRtOpCodeShloComposite` | Supports `odml.rms_norm` (as OEM extension `MTKEXT_RMS_NORMALIZATION`) and `odml.l2_norm` (as `NEURON_L2_NORMALIZATION`). |
| `kLiteRtOpCodeTflL2Normalization` | Legalized to `NEURON_L2_NORMALIZATION`. |

<!-- LINT.ThenChange(./compiler_plugin.cc:supported_ops) -->

## Supported SoCs

Recommended and compatible NeuroPilot versions per SoC are tracked in
[`supported_soc.csv`](../supported_soc.csv) and consumed by
`compile_model.cc` when resolving a `soc_model` string. Refer to the CSV
for the canonical list.
