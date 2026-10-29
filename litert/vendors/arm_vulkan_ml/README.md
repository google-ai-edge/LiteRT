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

Dispatch loads `libvulkan.so` on Android and `libvulkan.so.1` on Linux.
Android's linker resolves the system loader for the process ABI; no absolute
`/system/lib64` path is required. Load failures include the dynamic linker's
error message.

## Vulkan API revision

The backend targets **VK_ARM_tensors Revision 2**. Compilation requires headers
with `VK_ARM_TENSORS_SPEC_VERSION >= 2`, and device selection requires the
advertised tensor extension's `specVersion >= 2`. The data-graph extension and
TOSA instruction-set extension remain at Revision 1; the compiled artifact ABI
and TOSA instruction-set selection are unchanged.

The [tensor extension revision history](https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_tensors.html)
identifies Revision 2's interactions with unified image layouts, bfloat16 and
float8. This revision requirement does not enable those optional formats in
our lowering; the supported types remain those listed below.

Bazel explicitly uses the pinned Vulkan-Headers 1.4.349, independently of the
Android NDK's bundled Vulkan headers. Android NDK r30 (`30.0.16248370`) ships
Vulkan-Headers 1.4.335: its `vulkan_core.h` still declares tensors Revision 1
and lacks the TOSA instruction-set extension. Use the pinned headers even when
building with that NDK. The NDK is installed side by side at
`/data/bt/os/android_sdk/ndk/30.0.16248370` on the development host; older NDKs
remain available.

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
| RELU, RELU6 | Direct CLAMP when quantization is unchanged; lookup table otherwise |
| HARD_SWISH | Int8 lookup table preserves quantized semantics |
| TRANSPOSE, RESHAPE, SLICE | Constant permutation/shape/slice parameters; preserved quantization |
| PAD, PADV2 | Constant padding; PADV2 fill has matching quantization |
| CONCATENATION | Compatible shapes and identical quantization; no fused activation |
| MAX_POOL_2D | VALID window, preserved quantization |
| AVERAGE_POOL_2D | VALID windows, including overlap and cropped unused trailing input; preserved quantization and bounded integer sum |
| SUM | INT8: constant axes, keep_dims=true, bounded integer sum; FP32: constant axes, either keep_dims mode, non-scalar output |
| REDUCE_MAX, REDUCE_MIN | INT8/FP32, constant axes (negative/duplicate axes normalized), keep_dims true or false, non-scalar output; INT8 quantization preserved |
| GATHER | INT8/FP32 values, constant in-range INT32 indices, batch_dims=0, any valid axis; INT8 quantization preserved |
| RESIZE_NEAREST_NEIGHBOR | INT8/FP32 NHWC, constant size, align_corners or half_pixel_centers, TOSA scale/offset/border limits; INT8 quantization preserved |
| RESIZE_BILINEAR | INT8/FP32 NHWC and constant size; INT8 uses Q10 coordinates, signed rounding and preserved quantization |
| TRANSPOSE_CONV | INT8/FP32 NHWC, constant OHWI weights/output shape and optional bias, canonical SAME/VALID geometry, stride ≤ kernel, NONE/RELU/RELU6 activation; INT8 requires symmetric weights, matching INT32 bias scales and proven rescaling |
| FULLY_CONNECTED | Constant dense INT8 weights/INT32 bias or FP32 weights/bias; default weight format; flattened 2D output |
| DEQUANTIZE, QUANTIZE | Int8 ↔ FP32 or int8 → int8 requantization, identical fixed shape; identity INT8 QUANTIZE is eliminated |

A limited FP32 path supports TRANSPOSE, RESHAPE, PAD, CONV_2D, depthwise convolution,
HARD_SWISH, LOGISTIC, broadcast ADD/MUL and global VALID AVERAGE_POOL_2D,
plus the FP32 variants in the table above. This
covers the mixed-precision prefixes of MobileNetV3 Large and EfficientNet-B0,
and SqueezeNet1.1's FP32 output pooling.
It requires PRO-FP in addition to PRO-INT. This is not general floating-point
model support. EfficientNet SiLU is represented as LOGISTIC followed by MUL.

Accumulator requantization emits one native TOSA `RESCALE` with `SINGLE_ROUND`,
including per-channel scales, tiny scales and large legal bias values. Ordinary
convolutions retain `CONV2D` (INT8 to INT32) → `RESCALE` (INT32 to INT8) → `CLAMP`;
depthwise convolution uses the equivalent `DEPTHWISE_CONV2D` pattern. This lets
the driver fuse requantization without extra comparisons, shifts, selections or
channel splitting to emulate LiteRT's double rounding. FULLY_CONNECTED,
TRANSPOSE_CONV and quantized arithmetic share this rescaling policy.
Invalid accumulator/rescale ranges remain rejected. No unadvertised
`EXT-DOUBLEROUND` support is assumed. A local rescale can differ by one code
from the reference kernel; differences can accumulate through a model.

INT8 average pooling emits TOSA `AVG_POOL2D` directly. Overlapping VALID
windows have the same fixed divisor as disjoint windows; their centered sum is
bounded before selection. Explicit PAD operations remain part of the graph.
SAME padding and mismatched input/output quantization remain unsupported for
INT8 pooling. Native pooling rounding can also differ from LiteRT.

### Runtime capability checks

Before selecting a device, dispatch queries `VkTensorFormatPropertiesARM`
through `vkGetPhysicalDeviceFormatProperties2` for every format used by inputs,
outputs and constants. Each must advertise linear-tiling
`VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM`. It also queries
`VkPhysicalDeviceTensorPropertiesARM` and checks rank, dimensions, element and
byte counts, and input/output descriptor counts before allocating resources.
These checks supplement the existing engine-specific TOSA profile and extension
queries. TOSA operator/type legality is checked during lowering; Vulkan exposes
profiles and extensions rather than an individual operator/type support table.
Format support alone does not establish operator support. Graph pipeline
creation remains the final driver acceptance check for the complete graph,
including implementation and TOSA-level constraints.

The Arm 0.11 emulator advertises no TOSA extensions. A diagnostic using direct
`RESCALE` with `DOUBLE_ROUND` executed successfully on ResNet18, MobileNetV2 and
DenseNet121, but required omitting its `EXT-DOUBLEROUND` requirement. That bypass
is not part of the backend: unadvertised extensions remain rejected, including
in emulation mode.

The new gather, min/max reduction, resize and transpose-convolution paths passed
34 isolated emulator fixtures with four synthetic inputs each. Ten unsupported
cases were rejected, and every supported fixture passed SPIR-V validation.
These establish operator-level coverage, not qualification of new complete
models. Quantized bilinear resize and transpose convolution now have separately
qualified subsets described below. Runtime gather indices,
nonzero gather batch_dims, scalar reduction outputs and aligned singleton
resize dimensions remain unsupported. The [capability audit](tools/tosa_capabilities/README.md)
separates these implemented subsets from the full specification inventory.

The FP32 `mobilenet_v2.tflite` artifact (SHA-256
`34853c702f8da7298c6ae6c100b8a23df98b11d4fb41aa200e6fa45e96d52ea6`)
now compiles all 70 operations into one Vulkan partition. Its previously missing
FP32 SUM lowers to REDUCE_SUM, and FULLY_CONNECTED lowers to MATMUL plus bias
and optional activation. Three ImageNet images matched the CPU reference at
`atol=1e-4, rtol=1e-3`, with maximum absolute logit error `9.54e-7` and identical
top-1/top-5 predictions. Both new paths passed four isolated synthetic inputs,
and the complete graph passed SPIR-V validation. This is emulator smoke evidence,
not native-device performance or full ImageNet accuracy qualification.

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
  //litert/vendors/arm_vulkan_ml/tools:litert_arm_vulkan_ml_tool
```

Enable LiteRT's JIT compiler option and select the NPU accelerator. Compiled
partitions own SPIR-V, constant bytes and typed binding metadata. Each dispatch
invocation owns Vulkan resources and retains the compiled artifact. Host input
and output transfers use explicit synchronization; invocation is synchronous.
Per-tensor memory barriers are supplied through
`VkDependencyInfo::pNext → VkTensorDependencyInfoARM::pTensorMemoryBarriers`,
with `VK_QUEUE_FAMILY_IGNORED` for both queue-family indices.

See the [developer tool](tools/devtool/README.md) for the pinned public example
model suite and [regression testing](tools/testing/README.md) for numerical and
lifecycle checks.
The target set contains five ResNets, three ResNeXts, two Wide ResNets,
MobileNetV2, MobileNetV3 Large, EfficientNet-B0, DenseNet121 and SqueezeNet1.1.
MobileNetV3 Small is
explicitly excluded.

Host emulation is development evidence, not native Arm GPU qualification.
`LITERT_ARM_VULKAN_ML_ALLOW_EMULATION=1` is an explicit development override for
the emulation layer's nonconformant profile advertisement. Native device
performance, Android execution and dynamic shapes are not qualified here.

## C++ developer diagnostics

Use [`litert_arm_vulkan_ml_tool`](tools/devtool/README.md) to inspect coverage,
export validated SPIR-V, query device capabilities, validate ImageNet models,
and trace intermediate differences against CPU reference. It builds for Linux
and Android and defaults to small image samples.

The [portable FP32 matrix](tools/tosa_capabilities/PORTABLE_FP32_COVERAGE.md)
records the expanded arithmetic, tensor manipulation, pooling, reduction,
softmax and batch-matmul subsets. Use the developer tool's `--tensor_inputs`
path to qualify arbitrary graphs and diagnose individual operator differences.

## TOSA breadth expansion

Production lowering now references **60/66** mapped TOSA operators. The reviewed
`tools/tosa_capabilities/implementation_coverage.json` inventory and the developer
tool's offline `coverage` command separate emission from qualification. These are
restricted subsets, not support for all 350 specification type combinations.

New FP32 paths cover POW (static broadcasting), SIN, COS, exact GELU via ERF,
REVERSE_V2 with constant axes, and static NDHWC CONV_3D with constant DHWIO
weights/bias, SAME/VALID padding, stride/dilation and NONE/RELU/RELU6 activation.
Approximate GELU remains rejected. FP32 SCATTER_ND requires constant unique
in-range indices and output shape; duplicate-index accumulation is rejected.

BOOL interfaces use byte INT8 storage with explicit TOSA CAST operations at
graph boundaries. Logical AND/OR/NOT, Boolean NOT_EQUAL via LOGICAL_XOR,
comparisons to BOOL, and constant-axis REDUCE_ALL/ANY are enabled. ARG_MAX
supports FP32 input and INT32 output with a constant axis and non-scalar result.
INT32 BITWISE_XOR supports broadcasting and recognizes an all-ones constant as
BITWISE_NOT. INT32 FLOOR_DIV uses INTDIV with signed floor correction and
requires positive constant divisors, excluding division-by-zero and signed
quotient overflow. UINT32 RIGHT_SHIFT uses logical shifts with constant counts
0..31; its bits cross the Vulkan interface in the mapped 32-bit integer storage
format without arithmetic signedness conversion. Other UINT32 arithmetic is
not enabled.

Five operators have no selected TFLite frontend route: FFT2D, CLZ, integer
BITWISE_AND/OR and LOGICAL_LEFT_SHIFT. StableHLO-named builtins and custom ops
remain outside this expansion. RFFT2D remains pending: supporting its complex
intermediates requires partition pattern handling; the local emulator also
advertises no EXT-FFT. No native-device qualification is claimed.

## Quantized coverage expansion

The following paths now support static dense per-tensor affine INT8 activations:

| LiteRT operations | Lowering and constraints |
| --- | --- |
| TANH | Compile-time 256-entry TABLE; standard output scale 1/128 and zero point 0 |
| ABS | Compile-time TABLE, including range-proven requantization when scales differ |
| SUB | INT32 scaling/subtraction/requantization; static broadcasting; NONE/RELU/RELU6 |
| MAXIMUM, MINIMUM | INT32 casts and comparison; all inputs/output must share quantization |
| SQUEEZE, EXPAND_DIMS, PACK, UNPACK, SPLIT, SPLIT_V, TILE, STRIDED_SLICE, REVERSE_V2 | Preserve INT8 codes and quantization; constant controls; existing rank, shape and slice-mask limits |
| RESIZE_BILINEAR | Native INT8-to-INT32 RESIZE with LiteRT Q10 coordinates, followed by signed rounding and INT8 cast; matching quantization and static NHWC sizes; TOSA coordinate/border limits |
| TRANSPOSE_CONV | Native INT8-to-INT32 transpose convolution, range-proven requantization and activation clamp; constant symmetric weights with per-tensor/per-output-channel scales; optional matching INT32 bias; canonical SAME/VALID geometry and stride no larger than kernel |

The quantized fixture group requires **exact** CPU reference agreement in
validation and both trace modes. TANH/ABS cover all 256 codes; the SUB broadcast
fixture covers all 65,536 code pairs with differing quantization. Integer resize
uses the signed rounding convention selected by `TFLITE_SINGLE_ROUNDING`.
TANH/ABS use TABLE rather than claiming direct INT8 TANH/ABS instruction support.
These additions do not enable quantized DIV, MEAN, REDUCE_PROD or general affine
NEG. LiteRT's INT8 NEG kernel is raw integer negation, not an affine quantized
negation contract. Existing INT8 average-pooling differences are unchanged.

Run the fixture group using the developer tool (configure the emulation-layer
environment as described in the [test guide](tools/testing/README.md)):

```bash
python3 litert/vendors/arm_vulkan_ml/tools/testing/qualify_portable_ops.py \
  --quantized_only --output_dir /tmp/litert-quantized-fixtures \
  --tool bazel-bin/litert/vendors/arm_vulkan_ml/tools/litert_arm_vulkan_ml_tool \
  --allow_emulation --vulkan_validation sync
```

Generated models, inputs and reports stay outside source control. Qualification
is emulator evidence; native Arm GPU qualification remains outstanding.

## Shared IO buffers

Dispatch prefers Android BLOB AHardwareBuffer, then DMA-BUF, then HostMemory
on Android; Linux prefers DMA-BUF followed by HostMemory. Shared types are
advertised only when the selected Vulkan device exposes the relevant external
memory extensions and the exact linear tensor description is importable.
The runtime allocates the negotiated buffer size; attachment additionally checks
actual allocation properties, tensor memory-type bits, offset alignment and
required padding. Incompatible supplied shared allocations fail explicitly.

Imported allocations bind directly to IO tensors. They require no Vulkan CPU
mapping, host-visible memory type or per-inference staging copies. HostMemory
slots retain the existing copy-in/copy-out path. Native IO resources are created
lazily and reused until detach/replacement. Each invocation still owns its Vulkan
device, and the same registration can be imported independently on another
invocation's device. Imported allocations count against that device's allocation
limit, with the budget restored when an import is freed.

The Android AHardwareBuffer adapter uses API 31+ allocation identities (the
validated Android build targets API 35). AHardwareBuffers must be unprotected
GPU_DATA_BUFFER BLOB allocations with
height/layers 1 and zero offset. DMA-BUF views may have aligned nonzero offsets
when dedicated allocation rules permit them. Writable overlapping views are
rejected. Invocations with writable shared IO use one external handle kind;
mixed DMA-BUF/AHardwareBuffer writable IO is rejected because portable cross-kind
allocation identity is unavailable. Host/shared mixtures are supported.

Buffer registration owns a duplicated FD or acquired AHardwareBuffer reference;
Vulkan receives an additional import FD whose ownership transfers on successful
allocation. Attached buffers cannot be unregistered. The caller must keep the
LiteRT buffer object valid until unregister and must not access the shared
allocation concurrently with inference. Producer events are waited before host
locks or graph access. External/foreign tensor ownership is acquired before
execution and released afterwards; completion remains synchronous. No async
output fences or opaque camera-image layouts are added.

`litert_arm_vulkan_ml_tool capabilities --model MODEL` lists per-IO handle
support and rejection reasons. Direct validation/trace operator records expose
`buffer_io` counters for staging-copy payload bytes and imported slots. The
ordinary image-evaluator runtime-plugin path does not expose these counters.
Native import paths are covered by host resolver/adapter tests, including
fully imported execution without host mapping/copying, import reuse, dedicated
allocation, alias/alignment rejection, failure rollback and allocation-budget
reclamation. Real DMA-BUF/AHardwareBuffer correctness and performance still
require a compatible driver; Android builds are compile-checked only.

## VGF graph dumps

The C++ developer tool can package each lowered partition for inspection or
standalone benchmarking:

```sh
bazel-bin/litert/vendors/arm_vulkan_ml/tools/litert_arm_vulkan_ml_tool compile \
  --model model.tflite --output_dir compiled --dump_vgf \
  --report_json compiled/compile.json
```

The output contains one `.vgf` per accelerated partition, alongside the existing
SPIR-V, constants and lowering metadata. The VGF embeds the lowered graph and
its constants/IO descriptions; the JSON retains profile/extension requirements
and original model mappings. Partial coverage remains explicitly reported.
See the [dump guide](tools/devtool/README.md#dump-lowered-partitions-as-vgf).
VGF encoding is confined to the developer tool; production dependencies are
unchanged.

## Command-buffer reuse

Each invocation records its barriers, pipeline/descriptor bindings and graph
dispatch once, then resubmits that executable command buffer while IO bindings
remain unchanged. Host input data and caller pointer changes reuse the existing
staging allocations and recording. Producer-event waits, fence reset/submission,
synchronous completion and any host staging copies still occur on every run.

Successful tensor/descriptor replacement or detachment invalidates the cached
recording. The next run resets and records it again after all IO resources are
ready. No-op attachments, failed transactional replacements and capability
queries preserve the recording. Partial recording failures are reset before a
retry; failed submissions/completion waits continue to poison the invocation.
The command buffer uses neither one-time-submit nor simultaneous-use flags;
per-invocation synchronization prevents overlapping submissions.
