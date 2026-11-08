# LiteRT Arm ML Extensions for Vulkan Backend

This directory implements JIT compilation (`libLiteRtCompilerPlugin_ArmVulkanML.so`) and synchronous execution (`libLiteRtDispatch_ArmVulkanML.so`) for LiteRT through the public `VK_ARM_tensors`, `VK_ARM_data_graph`, and `VK_ARM_data_graph_instruction_set_tosa` Vulkan extensions.

LiteRT lowers supported subgraphs directly to `TOSA.001000.1` SPIR-V using Arm's `tosa-for-spirv-codegen` (TFSC) core library and passes an owned in-memory artifact directly to dispatch—without TensorFlow, LLVM/MLIR, StableHLO converter pipelines, TOSA Converter for TFLite (TCFT), Arm Model Converter, or runtime VGF serialization.

- **Architecture & internal design**: [DIRECT_COMPILATION_DESIGN.md](DIRECT_COMPILATION_DESIGN.md)
- **Diagnostic CLI (`litert_arm_vulkan_ml_tool`)**: [tools/devtool/README.md](tools/devtool/README.md)
- **TOSA capability database & coverage audits**: [tools/tosa_capabilities/README.md](tools/tosa_capabilities/README.md) and [PORTABLE_FP32_COVERAGE.md](tools/tosa_capabilities/PORTABLE_FP32_COVERAGE.md)
- **Qualification & emulator test harnesses**: [tools/testing/README.md](tools/testing/README.md)
- **Dataset accuracy & throughput evaluation**: [`litert_vision_eval`](../../tools/vision_eval/README.md)

---

## Vulkan API and Toolchain Requirements

- **Loader resolution**: Dispatch loads `libvulkan.so` on Android and `libvulkan.so.1` on Linux via the platform dynamic linker (no hardcoded `/system/lib64` path).
- **Extension revisions**: Requires [`VK_ARM_tensors`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_tensors.html) **Revision 2** (`specVersion >= 2`), `VK_ARM_data_graph` Revision 1, and `VK_ARM_data_graph_instruction_set_tosa` Revision 1.
- **Pinned Vulkan headers**: Both Bazel and CMake (`litert/vendors/CMakeLists.txt`) build against pinned `Vulkan-Headers` 1.4.349+ independently of the Android NDK's bundled headers (Android NDK r30 `30.0.16248370` ships Vulkan-Headers 1.4.335, which still declares `VK_ARM_tensors` Revision 1 and lacks the TOSA instruction-set extension).
- **Runtime capability & format verification**: Before selecting a device or allocating resources, dispatch verifies:
  1. Queue-family data-graph engine support for `TOSA.001000.1` and every profile (`PRO-INT`, `PRO-FP`) and extension (such as `EXT-INT16` or `EXT-FFT`) required by the compiled partition, plus `VkDataGraphTOSALevelARM` level bounds. Unadvertised extensions are always rejected.
  2. `VkPhysicalDeviceTensorPropertiesARM` limits (rank, per-dimension bounds, element/byte counts, and descriptor counts).
  3. Linear-tiling `VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM` via `VkTensorFormatPropertiesARM` for every input, output, and constant format.

---

## Supported Numeric Regimes and Operator Scope

Delegated partitions require static positive tensor shapes and dense layouts (rank-0 LiteRT scalars are stored as 1-element physical tensors while retaining rank-0 buffer validation). Partition selection runs the exact same C++ legality and overflow proofs used during compilation, leaving unsupported ops on CPU or other accelerators.

### Quantization and Data Types

- **`INT8` (`PRO-INT`)**:
  - **Activations**: Signed `INT8` with per-tensor affine quantization (`scale`, `zero_point`).
  - **Weights & biases (per-channel and per-tensor)**: `CONV_2D` (axis 0), `DEPTHWISE_CONV_2D` (axis 3), `TRANSPOSE_CONV` (axis 0), `CONV_3D` (DHWIO axis 4), and `FULLY_CONNECTED` (axis 0) support symmetric **per-output-channel** (`QuantizationTypeId::PerChannel`, `zero_point == 0`) or per-tensor `INT8` weights and matching `INT32` biases. Lowering proves `INT32` accumulator bounds per channel and emits channelwise `SINGLE_ROUND` TOSA `RESCALE` (`per_channel = true`) without requiring `EXT-DOUBLEROUND`.
- **`A16W8` / Symmetric `INT16` (`PRO-INT` + `EXT-INT16`)**:
  - **Activations**: Signed `INT16` with symmetric per-tensor quantization (`zero_point == 0`).
  - **Weights & biases (per-channel and per-tensor)**: `FULLY_CONNECTED` supports symmetric **per-channel** (axis 0) or per-tensor `INT8` weights with matching `INT64` biases, `INT48` accumulation (`shaderInt64`), and channelwise 16-bit multiplier `RESCALE`.
  - **LUTs, MatMul & Normalizations**: `BATCH_MATMUL` uses `INT48` accumulation with 3D `RESCALE`; `GELU`, `TANH`, `LOGISTIC`, `RSQRT`, and two-LUT `CLZ`-normalized `SOFTMAX` use 513-entry `INT16` `TABLE` lookups; `LayerNorm`, `RMSNorm`, and `L2Norm` (`STABLEHLO_COMPOSITE`) accumulate in `FP32` when `PRO-FP` is available.
- **`FP32` (`PRO-FP`) and Auxiliary Types**:
  - Static `FP32` convolutions, matmul, pooling, reductions, elementwise math, indexing, resize, and spectral (`RFFT2D` with `EXT-FFT` and interleaved `COMPLEX64` storage).
  - `BOOL` tensors cross the Vulkan interface as 8-bit integers with boundary `CAST`s; `INT32` / `UINT32` subsets cover indexing, comparisons, bitwise ops, signed floor division/modulo, and logical right shifts (`0..31`).

### Operator Summary

Production lowering emits **64 of 66** `TOSA.001000.1` opcodes (`FFT2D` and `BITWISE_OR` have no TFLite builtin frontend route). Run `litert_arm_vulkan_ml_tool coverage --op_set=all --json` or inspect [`implementation_coverage.json`](tools/tosa_capabilities/implementation_coverage.json) for exact per-type constraints:

| Category | Supported LiteRT Operations | Notes & Key Constraints |
| --- | --- | --- |
| Convolutions & Linear | `CONV_2D`, `DEPTHWISE_CONV_2D`, `TRANSPOSE_CONV`, `CONV_3D`, `FULLY_CONNECTED`, `BATCH_MATMUL` | Static NHWC/NDHWC, constant weights/biases, per-channel or per-tensor symmetric `INT8` weights, grouped `CONV_2D` decomposition, `FP32` / `INT8` / `INT16` (`FULLY_CONNECTED`, `BATCH_MATMUL`) |
| Activations & Softmax | `RELU`, `RELU6`, `RELU_N1_TO_1`, `RELU_0_TO_1`, `LEAKY_RELU`, `PRELU`, `LOGISTIC`, `TANH`, `HARD_SWISH`, `GELU`, `ELU`, `SOFTMAX`, `LOG_SOFTMAX` | Direct `CLAMP` or compile-time 256-entry (`INT8`) / 513-entry (`INT16`) `TABLE`s; `CLZ`-normalized two-LUT `INT16` `SOFTMAX` |
| Elementwise & Logical | `ADD`, `SUB`, `MUL`, `DIV`, `ADD_N`, `ABS`, `NEG`, `EXP`, `LOG`, `SQRT`, `RSQRT`, `SQUARE`, `SQUARED_DIFFERENCE`, `POW`, `SIN`, `COS`, `ATAN2`, `FLOOR`, `CEIL`, `ROUND`, `SIGN`, `MINIMUM`, `MAXIMUM`, `FLOOR_DIV`, `FLOOR_MOD`, `BITWISE_XOR`, `RIGHT_SHIFT`, `EQUAL`, `NOT_EQUAL`, `GREATER`, `GREATER_EQUAL`, `LESS`, `LESS_EQUAL`, `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `SELECT`, `SELECT_V2` | Static TOSA broadcasting; proven integer rescale ranges; `NONE`/`RELU`/`RELU6` fused activations where applicable |
| Pooling, Reductions & Norm | `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `SUM`, `MEAN`, `REDUCE_MAX`, `REDUCE_MIN`, `REDUCE_PROD`, `REDUCE_ALL`, `REDUCE_ANY`, `ARG_MAX`, `ARG_MIN`, `L2_NORMALIZATION`, `STABLEHLO_COMPOSITE` (`odml.group_norm`, `odml.rms_norm`, `odml.l2_norm`) | Static windows and constant reduction axes; native TOSA `AVG_POOL2D` rounding may differ by 1–2 codes at `INT8` quantization boundaries |
| Data Movement & Indexing | `RESHAPE`, `TRANSPOSE`, `SQUEEZE`, `EXPAND_DIMS`, `PAD`, `PADV2`, `MIRROR_PAD`, `SLICE`, `STRIDED_SLICE`, `CONCATENATION`, `PACK`, `UNPACK`, `SPLIT`, `SPLIT_V`, `TILE`, `BROADCAST_TO`, `REVERSE_V2`, `SPACE_TO_DEPTH`, `DEPTH_TO_SPACE`, `SPACE_TO_BATCH_ND`, `BATCH_TO_SPACE_ND`, `GATHER`, `GATHER_ND`, `SCATTER_ND`, `ONE_HOT`, `SPARSE_TO_DENSE`, `BUCKETIZE`, `FILL`, `ZEROS_LIKE`, `SHAPE`, `RANK`, `RESIZE_BILINEAR`, `RESIZE_NEAREST_NEIGHBOR` | Constant shape/slice/index controls (`ONE_HOT` and `BUCKETIZE` accept runtime inputs); preserved quantization on data-movement ops |
| Quantization, Cast & FFT | `QUANTIZE`, `DEQUANTIZE`, `FAKE_QUANT`, `CAST`, `RFFT2D`, `REAL`, `IMAG` | `FP32` ↔ `INT8`/`INT16` and `INT8` ↔ `INT16` conversions; identity `QUANTIZE` elimination; `RFFT2D` requires `EXT-FFT` |

---

## Shared IO Buffers and Command-Buffer Reuse

### Zero-Copy External Buffers

- **Negotiation order**: Android prefers `AHardwareBuffer` (`AHARDWAREBUFFER_FORMAT_BLOB` with `GPU_DATA_BUFFER`), then `DMA-BUF`, then `HostMemory`; Linux prefers `DMA-BUF`, then `HostMemory`.
- **Direct tensor binding**: External handle types are advertised only when the selected Vulkan device exposes the corresponding external-memory extensions and `vkGetPhysicalDeviceExternalTensorPropertiesARM` reports linear tensor importability. Attached external buffers bind directly to `VkTensorARM` with no CPU mapping or staging copies.
- **Safety & synchronization**: Attachment verifies allocation size, memory-type bits, offset alignment, and non-overlapping writable views (writable shared IO within an invocation uses a single external handle kind; mixing shared and host slots is supported). Producer events are waited prior to graph execution, and external/foreign queue-family tensor ownership is acquired before `vkCmdDispatchDataGraphARM` and released afterward.

### Command-Buffer Caching

Each `VulkanInvocation` records tensor memory barriers (`VkDependencyInfo` → `VkTensorDependencyInfoARM`), pipeline/descriptor bindings, and `vkCmdDispatchDataGraphARM` once into a reusable command buffer. Subsequent runs resubmit the cached command buffer whenever IO buffer attachments are unchanged, re-recording automatically only after an IO tensor replacement or detachment.

---

## Build, Diagnostics, and VGF Export

### Build Host Targets

From the repository root:

```bash
bazelisk build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //litert/vendors/arm_vulkan_ml/compiler:compiler_plugin_so \
  //litert/vendors/arm_vulkan_ml/dispatch:dispatch_api_so \
  //litert/vendors/arm_vulkan_ml/tools:litert_arm_vulkan_ml_tool
```

See [tools/devtool/README.md](tools/devtool/README.md) for `--config=android_arm64` cross-compilation flags and on-device deployment.

### Inspect, Validate, and Export VGF

Use [`litert_arm_vulkan_ml_tool`](tools/devtool/README.md) on Linux or Android to inspect partition coverage, query device capabilities, compare/trace outputs against CPU references, or export lowered partitions as `.vgf` containers for standalone inspection:

```bash
bazel-bin/litert/vendors/arm_vulkan_ml/tools/litert_arm_vulkan_ml_tool compile \
  --model model.tflite \
  --output_dir /tmp/compiled_partitions \
  --dump_vgf \
  --report_json /tmp/compiled_partitions/compile.json
```

VGF encoding (`ml_sdk_vgf_library`) is confined to `litert_arm_vulkan_ml_tool`; the production compiler and dispatch plugins have no VGF dependency.

### Host Emulation vs. Hardware Qualification

Host correctness tests can run the production compiler and dispatch plugins over Arm's ML Emulation Layer for Vulkan on Mesa Lavapipe (see [tools/testing/README.md](tools/testing/README.md)). Setting `LITERT_ARM_VULKAN_ML_ALLOW_EMULATION=1` opts into the emulation layer's profile advertisement for host development only; unadvertised extensions remain rejected, and software emulation does not substitute for native Vulkan driver conformance or on-device performance qualification.

### Observed Retail Silicon Constraints (`Arm Mali-G2-Ultra-NX`)

Capability checks on commercially available MediaTek Dimensity 9600 retail phones purchased through standard retail channels (`Arm Mali-G2-Ultra-NX`) highlight several practical differences between abstract `TOSA.001000.1` / host emulation and shipping mobile hardware:

- **Hardware acceleration vs. conformant fallback (`VkDataGraphTosaQualityFlagBitsARM`)**: The queue-family data-graph engine advertises `TOSA.001000.1` at `VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM` (`shaderInt16` and `shaderInt64` enabled). `PRO-INT` and `EXT-INT16` report `ACCELERATED | CONFORMANT`, executing `INT8` and `A16W8` (`INT16`) partitions on the dedicated neural accelerator, whereas `PRO-FP` reports `CONFORMANT` without `ACCELERATED`, executing `FP32` partitions conformantly via shader fallback.
- **Unadvertised TOSA extensions**: `EXT-DOUBLEROUND` is not advertised (nor are `EXT-INT4`, `EXT-BF16`, `EXT-FP8*`, `EXT-FFT`, `EXT-VARIABLE`, `EXT-DYNAMIC`, or `EXT-CONTROLFLOW`), confirming why integer requantization lowers exclusively through base-profile `SINGLE_ROUND` TOSA `RESCALE` and why unadvertised extensions are rejected before pipeline creation.
- **4D physical tensor rank ceiling**: Although `VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM` permits up to rank 6 in abstract TOSA, `VkPhysicalDeviceTensorPropertiesARM::maxTensorDimensionCount` on shipping `Mali-G2-Ultra-NX` silicon is `4`. Dispatch validates physical tensor limits independently of TOSA level bounds, keeping higher-rank tensors on CPU.
- **Linear formats & zero-copy external memory**: Linear-tiling `R8_SINT`, `R16_SINT`, `R32_SINT`, `R64_SINT`, and `R32_SFLOAT` report `VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM`, and both `AHardwareBuffer` (`VK_ANDROID_external_memory_android_hardware_buffer`) and `DMA-BUF` (`VK_EXT_external_memory_dma_buf`) report `importable = true` (`dedicated_only = false`) for linear graph IO tensors.
