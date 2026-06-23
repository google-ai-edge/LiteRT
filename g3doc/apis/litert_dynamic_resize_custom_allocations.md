# LiteRT Dynamic Resize with Custom Tensor Buffers: Known Issue

This note describes a current LiteRT runtime issue that shows up when a model
with dynamic input shapes is executed repeatedly with different shapes while
using LiteRT-managed custom input/output `TensorBuffer`s.

## Problem Summary

The first invocation usually succeeds:

1. Create `CompiledModel`
2. Create input/output buffers
3. Resize the dynamic input tensor to shape `A`
4. Run inference

The failure appears when the same `CompiledModel` instance is reused for a new
shape:

1. Reuse the same `CompiledModel`
2. Resize the dynamic input tensor to shape `B`
3. Recreate input/output buffers
4. Run inference or create output buffers

At that point LiteRT can fail with an error similar to:

```text
Custom allocation is too small for tensor idx: ...
```

## Affected Pattern

This affects the combination of:

- dynamic-shape models
- `ResizeInputTensor(...)`
- custom I/O tensor buffer registration
- shape changes across multiple invocations on the same `CompiledModel`

Repeated runs at the same shape are fine. The issue is triggered by changing the
shape after a prior run has already attached custom allocations to the
interpreter state.

## Observed Behavior

LiteRT allows the initial dynamic resize and execution, but a later resize to a
different shape can fail even if the caller destroys old Python/C++ buffer
objects and recreates fresh `TensorBuffer`s for the new shape.

This indicates the failure is not just stale wrapper state in the caller. The
runtime is still carrying forward an older custom allocation binding into the
next allocation cycle.

## Likely Root Cause

TFLite `Subgraph::SetCustomAllocationForTensor(...)` stores custom allocations
in `custom_allocations_` and marks the tensor as `kTfLiteCustom`. Those stored
allocations are revalidated during later `AllocateTensors()` calls.

When LiteRT resizes a signature runner input tensor, it marks the signature as
needing allocation, but it does not clear or invalidate previously registered
custom allocations for the affected signature. If the new tensor shape is
larger, the old custom allocation can fail validation on the next
`AllocateTensors()`.

Relevant code paths:

- `litert/runtime/compiled_model.cc`
- `tflite/core/signature_runner.cc`
- `tflite/core/subgraph.cc`

## Current Workaround

The current safe workaround at the application layer is:

- keep using the same `CompiledModel` while the input shape stays unchanged
- if the input shape changes after a prior run, create a fresh `CompiledModel`
  instance before resizing and recreating buffers

This is a workaround, not the intended long-term behavior.

## Desired Runtime Fix

The runtime should clear or invalidate stale custom allocations when
`ResizeInputTensor(...)` changes the shape for a signature that has previously
registered custom input/output buffers.

The expected behavior is:

1. resize input tensor
2. recreate input/output buffers for the new shape
3. run again on the same `CompiledModel`

without requiring the caller to reopen the model.

## Scope

This note is about runtime behavior only. It is not a model conversion issue,
and it is not specific to a single model architecture. Any dynamic-shape model
that relies on repeated shape changes plus custom tensor buffers can hit the
same failure mode.
