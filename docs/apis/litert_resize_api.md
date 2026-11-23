# LiteRT Dynamic Resize API User Guide

This document outlines how to resize LiteRT input tensors at runtime so models
with dynamic shapes can run against varying batch sizes or image dimensions on
CPU.

## Introduction

Many models accept inputs whose shapes change between invocations—for example,
object detectors that support different input resolutions. LiteRT mirrors
TensorFlow Lite’s dynamic tensor support through a dedicated `ResizeInputTensor`
API and automatic shape adaptation during buffer registration.

Note that `ResizeInputTensor` only works when model has dynamic shapes.

## Explicit Resize Workflow

When you know the new shape ahead of time, resize the tensor explicitly and
reattach buffers before invoking the model.

### Code Snippet: C++ Explicit Resize

```cpp
// Target shape: [1, new_height, new_width, 3]
std::array<int, 4> new_shape = {1, 720, 1280, 3};

LITERT_ASSERT_OK(compiled_model.ResizeInputTensor(
    /*signature_index=*/0, /*input_index=*/0, absl::MakeSpan(new_shape)));

// After resizing, refresh buffer requirements and recreate buffers if needed.
LITERT_ASSERT_OK_AND_ASSIGN(
    std::vector<litert::TensorBuffer> input_buffers,
    compiled_model.CreateInputBuffers(/*signature_index=*/0));
```

## Automatic Resize During Buffer Registration

If you skip the explicit resize call and register an input buffer whose layout
advertises a different shape, LiteRT:

1.  Verifies that the tensor has dynamic dimensions and that the rank matches.
2.  Ensures each static dimension remains unchanged.
3.  Calls `ResizeInputTensor` internally and clears cached buffer requirements
    so new allocations can be queried.

This is especially useful when your input buffers are produced by a
preprocessing pipeline that naturally encodes the desired shape.
