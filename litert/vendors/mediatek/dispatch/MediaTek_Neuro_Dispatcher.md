# LiteRT MediaTek Neuron Dispatcher

This document covers **runtime** behavior of the MediaTek Neuron dispatcher:
how compiled bytecode is loaded onto the device, how the Neuron Adapter
shared library is resolved, which runtime options apply, and how to build
and test the dispatcher. For **compile-time** behavior (op legalization,
partitioning, compile options), see
[`MediaTek_Neuro_Compiler.md`](../compiler/MediaTek_Neuro_Compiler.md).

Source of truth: `litert/vendors/mediatek/dispatch/`.

## Overview

The dispatcher is the runtime counterpart of the compiler plugin. LiteRT
loads it as a vendor `dispatch_api` shared library, hands it the bytecode
produced by AOT compilation, and the dispatcher executes the model on the
MediaTek NPU through the NeuronAdapter API.


## Building

```bash
# Dispatcher shared library loaded by the LiteRT runtime.
bazel build //litert/vendors/mediatek/dispatch:dispatch_api_so

# For on-device runs, build for Android arm64.
bazel build --config=android_arm64 -c opt \
    //litert/vendors/mediatek/dispatch:dispatch_api_so
```

The dispatcher dynamically links against `libneuron_adapter.so` (or one of
its USDK / mgvi variants — see *Shared Library Resolution* below). It does
**not** statically embed the NeuroPilot SDK.

