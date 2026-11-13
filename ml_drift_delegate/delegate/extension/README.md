# Extension kernels

This directory is for GPU kernels that this delegate adds on top of the kernels
ML Drift provides, for TFLite builtin operators and for custom operators. It
keeps them apart from the StableHLO composite operators in
[`../composite/`](../composite/).

There are no kernels here yet. To add one, follow
[Adding a Builtin GPU Operator](../../ADDING_BUILTIN_GPU_OP.md).

Two things to know before you start:

-   Each kernel needs a library target in this directory's `BUILD` file.
-   The op selectors that call the kernels are in `../composite/`. A kernel here
    is not used until a selector has a branch for it and depends on its target.
