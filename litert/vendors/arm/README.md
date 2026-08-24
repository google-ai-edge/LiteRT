# LiteRT Arm&reg; Integration

This directory contains the LiteRT vendor integration for Arm targets. It is
currently a work in progress: the compiler plugin only accepts the JIT flow,
and the dispatch implementation does not yet execute models.

- `compiler/` builds `libLiteRtCompilerPlugin_Arm.so`.
- `dispatch/` builds `libLiteRtDispatch_Arm.so`.
- `common/` contains types shared by the Arm integration.

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
