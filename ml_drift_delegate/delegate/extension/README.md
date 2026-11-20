# Extension kernels

This directory is for GPU kernels that this delegate adds on top of the kernels
ML Drift provides, for TFLite builtin operators and for custom operators. It
keeps them apart from the StableHLO composite operators in
[`../composite/`](../composite/).

To add a kernel, follow
[Adding a Builtin GPU Operator](../../ADDING_BUILTIN_GPU_OP.md).

Two things to know before you start:

-   Each kernel needs a library target in this directory's `BUILD` file.
-   The op selectors that call the kernels are in `../composite/`. A kernel here
    is not used until a selector has a branch for it and depends on its target.

## Kernels

| Operator  | Kernel                 | Supported form                           |
| :-------- | :--------------------- | :--------------------------------------- |
| ScatterND | `scatter_nd_kernel.cc` | Index depth 3: indices `[N, 3]` or `[B, H, W, 3]` address (batch, height, width) of a 4D output `[B, H, W, C]`; updates have the indices shape with the last dim replaced by C; float32 or bool updates; duplicate indices are summed for float updates and OR-ed for bool updates; index vectors that are out of range are ignored, whereas the CPU kernel fails with "scatter_nd index out of bounds" |

Known limitation of the ScatterND kernel: each output element scans all N index
vectors, because duplicate indices have to be combined. The cost is
O(N x B x H x W x S), which is fine for small N. The support check puts no limit
on N, so a model with a large N may run slower on the GPU than on the CPU fallback.

## ScatterND as an example

ScatterND was added by following the guide above, so its code shows every step
in one place. Paths are relative to the delegate's top directory.

| Step                      | Files                                    |
| :------------------------ | :--------------------------------------- |
| 1. GraphFloat32 pipeline  | `ScatterNdOperationParser` and the int32 input list in `tflite/model_builder.cc` |
| 2a. IrModel support check | `tflite/support/support_scatter_nd.{h,cc}` |
| 2b. IrModel converter     | `tflite/convert/convert_scatter_nd.{h,cc}` |
| 3. Op selector            | The `scatter_nd` branches in `delegate/composite/litert_op_selector.cc` and `delegate/composite/ir/litert_op_selector.cc` |
| 4. GPU kernel             | `delegate/extension/scatter_nd_kernel.{h,cc}` |
| Tests                     | `tflite/support/support_scatter_nd_test.cc`, `tflite/convert/convert_scatter_nd_test.cc`, `delegate/extension/scatter_nd_gpu_test.cc` |
