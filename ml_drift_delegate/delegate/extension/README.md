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
| ScatterND | `scatter_nd_kernel.cc` | Indices of shape `[N, 1]` that select along the first axis of a 4D output |

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
