# LiteRT Arm&reg; Integration

This directory contains the LiteRT vendor integration for Arm targets. It is
currently a work in progress: the compiler plugin only accepts the JIT flow,
and the dispatch implementation does not yet execute models.

- `compiler/` builds `libLiteRtCompilerPlugin_Arm.so`.
- `dispatch/` builds `libLiteRtDispatch_Arm.so`.
- `common/` contains types shared by the Arm integration.

## Compiler plugin support

The compiler plugin currently targets an Arm hardware using the generic SoC model.
It participates in partitioning only when the Arm `enable_just_in_time` option
is set to `true`.

### Supported data types

An operation is selected only when all of its input and output tensors use one
of the following element types:

| Category | LiteRT element types |
| --- | --- |
| Boolean | `Bool` |
| Integer | `Int8`, `UInt8`, `Int16`, `UInt16`, `Int32`, `UInt32` |
| Floating point | `Float16`, `Float32` |

These types correspond to the TOSA PRO-INT and PRO-FLOAT profiles. The
partitioner does not currently apply additional quantization-scheme checks.

### Supported operations

The following LiteRT operations are currently accepted by the TOSA
legalization flow:

| LiteRT Op Code |
| --- |
| `kLiteRtOpCodeTflAbs` |
| `kLiteRtOpCodeTflCeil` |
| `kLiteRtOpCodeTflFloor` |
| `kLiteRtOpCodeTflExp` |
| `kLiteRtOpCodeTflLog` |
| `kLiteRtOpCodeTflRsqrt` |
| `kLiteRtOpCodeTflLogicalNot` |
| `kLiteRtOpCodeTflCast` |
| `kLiteRtOpCodeTflLogicalAnd` |
| `kLiteRtOpCodeTflLogicalOr` |
| `kLiteRtOpCodeTflBitwiseXor` |
| `kLiteRtOpCodeTflPow` |
| `kLiteRtOpCodeTflGelu` |
| `kLiteRtOpCodeTflRelu` |
| `kLiteRtOpCodeTflReluN1To1` |
| `kLiteRtOpCodeTflRelu0To1` |
| `kLiteRtOpCodeTflRelu6` |
| `kLiteRtOpCodeTflEqual` |
| `kLiteRtOpCodeTflNotEqual` |
| `kLiteRtOpCodeTflGreater` |
| `kLiteRtOpCodeTflGreaterEqual` |
| `kLiteRtOpCodeTflAdd` |
| `kLiteRtOpCodeTflSub` |
| `kLiteRtOpCodeTflMul` |
| `kLiteRtOpCodeTflSquare` |
| `kLiteRtOpCodeTflSquaredDifference` |
| `kLiteRtOpCodeTflSign` |
| `kLiteRtOpCodeTflRound` |
| `kLiteRtOpCodeTflDiv` |
| `kLiteRtOpCodeTflMaximum` |
| `kLiteRtOpCodeTflMinimum` |
| `kLiteRtOpCodeTflFloorMod` |
| `kLiteRtOpCodeTflFloorDiv` |
| `kLiteRtOpCodeTflAddN` |
| `kLiteRtOpCodeTflAveragePool2d` |
| `kLiteRtOpCodeTflMaxPool2d` |
| `kLiteRtOpCodeTflConcatenation` |
| `kLiteRtOpCodeTflReshape` |
| `kLiteRtOpCodeTflRank` |
| `kLiteRtOpCodeTflShape` |
| `kLiteRtOpCodeTflExpandDims` |
| `kLiteRtOpCodeTflSqueeze` |
| `kLiteRtOpCodeTflFill` |
| `kLiteRtOpCodeTflElu` |
| `kLiteRtOpCodeTflSoftmax` |
| `kLiteRtOpCodeTflLogSoftmax` |
| `kLiteRtOpCodeTflSqrt` |
| `kLiteRtOpCodeTflL2Normalization` |
| `kLiteRtOpCodeTflReduceAll` |
| `kLiteRtOpCodeTflReduceAny` |
| `kLiteRtOpCodeTflReduceMax` |
| `kLiteRtOpCodeTflReduceMin` |
| `kLiteRtOpCodeTflMean` |
| `kLiteRtOpCodeTflReduceProd` |
| `kLiteRtOpCodeTflSum` |
| `kLiteRtOpCodeTflConv2d` |
| `kLiteRtOpCodeTflConv3d` |
| `kLiteRtOpCodeTflTransposeConv` |
| `kLiteRtOpCodeTflDepthwiseConv2d` |
| `kLiteRtOpCodeTflFullyConnected` |
| `kLiteRtOpCodeTflBatchMatmul` |
| `kLiteRtOpCodeTflSplit` |
| `kLiteRtOpCodeTflSplitV` |
| `kLiteRtOpCodeTflPack` |
| `kLiteRtOpCodeTflUnpack` |
| `kLiteRtOpCodeTflTranspose` |
| `kLiteRtOpCodeTflTile` |
| `kLiteRtOpCodeTflSlice` |
| `kLiteRtOpCodeTflStridedSlice` |
| `kLiteRtOpCodeTflHardSwish` |
| `kLiteRtOpCodeTflZerosLike` |
| `kLiteRtOpCodeTflLess` |
| `kLiteRtOpCodeTflLessEqual` |
| `kLiteRtOpCodeTflPad` |
| `kLiteRtOpCodeTflMirrorPad` |
| `kLiteRtOpCodeTflPadv2` |
| `kLiteRtOpCodeTflResizeBilinear` |
| `kLiteRtOpCodeTflResizeNearestNeighbor` |
| `kLiteRtOpCodeTflSelect` |
| `kLiteRtOpCodeTflSelectV2` |
| `kLiteRtOpCodeTflSpaceToBatchNd` |
| `kLiteRtOpCodeTflBatchToSpaceNd` |
| `kLiteRtOpCodeTflSpaceToDepth` |
| `kLiteRtOpCodeTflDepthToSpace` |
| `kLiteRtOpCodeTflBucketize` |
| `kLiteRtOpCodeTflSin` |
| `kLiteRtOpCodeTflCos` |
| `kLiteRtOpCodeTflAtan2` |
| `kLiteRtOpCodeTflLogistic` |
| `kLiteRtOpCodeTflTanh` |
| `kLiteRtOpCodeTflPrelu` |
| `kLiteRtOpCodeTflLeakyRelu` |
| `kLiteRtOpCodeTflNeg` |
| `kLiteRtOpCodeTflReverseV2` |
| `kLiteRtOpCodeTflQuantize` |
| `kLiteRtOpCodeTflDequantize` |
| `kLiteRtOpCodeTflGather` |
| `kLiteRtOpCodeTflGatherNd` |
| `kLiteRtOpCodeTflScatterNd` |
| `kLiteRtOpCodeTflSparseToDense` |
| `kLiteRtOpCodeTflOneHot` |
| `kLiteRtOpCodeTflArgMax` |
| `kLiteRtOpCodeTflArgMin` |
| `kLiteRtOpCodeTflFakeQuant` |
| `kLiteRtOpCodeTflWhile` |
| `kLiteRtOpCodeTflReal` |
| `kLiteRtOpCodeTflImag` |
| `kLiteRtOpCodeTflRfft2d` |
| `kLiteRtOpCodeTflBroadcastTo` |

The source of truth for these lists is
[`capabilities.cc`](capabilities.cc).

## Arm SDK dependencies

Bazel downloads the integration's pinned source dependencies on demand. These
archives are upstream releases; it is their use by the LiteRT Arm integration
that is still under development.

- [AI/ML SDK VGF Library v0.9.0](https://github.com/arm/ai-ml-sdk-vgf-library/tree/v0.9.0)
  provides VGF decoding and parsing.
- [TOSA for SPIR-V&trade; Codegen v2.0.0](https://github.com/arm/tosa-for-spirv-codegen/tree/v2.0.0)
  provides TOSA to SPIR-V&trade; code generation.
- [Khronos&reg; Vulkan&reg; Headers v1.4.349](https://github.com/KhronosGroup/Vulkan-Headers/tree/v1.4.349)
  provides the required Vulkan&reg; definitions.
- [Khronos&reg; SPIR-V&trade; Headers from Vulkan&reg; SDK 1.4.328.0](https://github.com/KhronosGroup/SPIRV-Headers/tree/vulkan-sdk-1.4.328.0)
  provides the required SPIR-V&trade; definitions.

The dependency declarations and checksums are in
[`third_party/arm/workspace.bzl`](../../../third_party/arm/workspace.bzl).

To verify that the shared libraries build:

```sh
bazel test //litert/vendors/arm:build_so_test
```

Arm is a registered trademark of Arm Limited (or its subsidiaries or affiliates).
