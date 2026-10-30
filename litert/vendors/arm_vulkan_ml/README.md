# LiteRT Arm ML Extensions for Vulkan integration

This worktree implements JIT compilation and synchronous execution through the
public Vulkan tensor and data graph extensions. LiteRT lowers supported graph
regions directly using the `tosa-for-spirv-codegen` (TFSC) core library. The
production compiler/dispatch path does not use the TensorFlow, TCFT, MLIR or
StableHLO converter frameworks, Arm Model Converter, or VGF serialization.
Ordinary LiteRT runtime dependencies remain, including its StableHLO-named
builtin kernels and TensorFlow/XLA shim headers. The Bazel C++ toolchain also
ships LLVM/MLIR files; their presence is not a model-converter dependency.

The [architecture](DIRECT_COMPILATION_DESIGN.md) explains the boundaries.
The [TOSA capability database](tools/tosa_capabilities/README.md) is a development
reference extracted from the local SPIR-V Registry and TOSA specification; it is
not loaded at runtime. Actual operation selection uses the same C++ legality
analysis as compilation. Device initialization checks Vulkan features and TOSA
instruction-set profiles.

## Implemented scope

The current path accepts fixed positive shapes and dense tensors. Quantized
activations are signed int8 with per-tensor affine quantization; convolution and
fully connected weights are symmetric int8 with per-channel scales and int32
biases. Analysis proves accumulator and rescale ranges before selecting an op.

| LiteRT operations | Constraints |
| --- | --- |
| CONV_2D | NHWC activation, constant OHWI weights/bias, validated padding/stride/dilation; grouped convolution lowers to slices, convolutions, concat |
| DEPTHWISE_CONV_2D | Depth multiplier 1; constant weights/bias |
| ADD, MUL | Validated broadcasting, int8 inputs/output, NONE/RELU/RELU6 fused activation |
| LOGISTIC | Int8 output scale 1/256, zero point -128; reference-compatible lookup table |
| RELU, RELU6, HARD_SWISH | Int8 lookup table preserves quantized semantics |
| TRANSPOSE, RESHAPE, SLICE | Constant permutation/shape/slice parameters; preserved quantization |
| PAD, PADV2 | Constant padding; PADV2 fill has matching quantization |
| CONCATENATION | Compatible shapes and identical quantization; no fused activation |
| MAX_POOL_2D | VALID window, preserved quantization |
| AVERAGE_POOL_2D | Global or disjoint tiled VALID windows, preserved quantization, bounded integer sum |
| SUM | Constant axes and keep_dims=true; bounded integer sum |
| FULLY_CONNECTED | Constant dense int8 weights/int32 bias; default weight format; flattened output |
| DEQUANTIZE, QUANTIZE | Int8 ↔ FP32 or int8 → int8 requantization, identical fixed shape |

A limited FP32 path supports TRANSPOSE, RESHAPE, PAD, CONV_2D, depthwise convolution,
HARD_SWISH, LOGISTIC, broadcast ADD/MUL and global VALID AVERAGE_POOL_2D. This
covers the mixed-precision prefixes of MobileNetV3 Large and EfficientNet-B0,
and SqueezeNet1.1's FP32 output pooling.
It requires PRO-FP in addition to PRO-INT. This is not general floating-point
model support. EfficientNet SiLU is represented as LOGISTIC followed by MUL.

Requantization composes base TOSA integer operations to reproduce LiteRT's
rounding where range proofs permit it, with per-channel decisions. It does not
assume EXT-DOUBLEROUND support. Invalid ranges are rejected rather than wrapped.

## Build and validate

From the worktree root:

```bash
mkdir -p .bazelisk-cache .cache .bazel-output
PATH=/usr/local/bin:/usr/bin:/bin \
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //litert/vendors/arm_vulkan_ml/compiler:compiler_plugin_so \
  //litert/vendors/arm_vulkan_ml/dispatch:dispatch_api_so \
  //litert/vendors/arm_vulkan_ml/tools/qualification:qualify
```

Enable LiteRT's JIT compiler option and select the NPU accelerator. Compiled
partitions own SPIR-V, constant bytes and typed binding metadata. Each dispatch
invocation owns Vulkan resources and retains the compiled artifact. Host input
and output transfers use explicit synchronization; invocation is synchronous.

See [model preparation](tools/models/README.md) and
[bounded qualification](tools/qualification/README.md) for reproducible checks.
The target set contains five ResNets, three ResNeXts, two Wide ResNets,
MobileNetV2, MobileNetV3 Large, EfficientNet-B0, DenseNet121 and SqueezeNet1.1.
MobileNetV3 Small is
explicitly excluded.

Host emulation is development evidence, not native Arm GPU qualification.
`LITERT_ARM_VULKAN_ML_ALLOW_EMULATION=1` is an explicit development override for
the emulation layer's nonconformant profile advertisement. Native device
performance, Android execution and dynamic shapes are not qualified here.
