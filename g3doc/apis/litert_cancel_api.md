# LiteRT Cancellation API User Guide

This guide describes how to abort long-running LiteRT inference calls by using
the cancellation hooks exposed on `LiteRtCompiledModel`.

## Introduction

Model execution can occasionally stall. LiteRT exposes a cooperative
cancellation mechanism so that applications can stop an in-flight invocation
and reclaim control promptly.

Note that LiteRT APIs are not inherently thread safe, please handle data
synchronization properly when using LiteRT in a multithreading environment.

## Core Concepts

LiteRT’s cancellation support centers on two concepts:

*   **Cancellation callbacks**: User-provided functions that the runtime
    invokes between operator executions. The callback should return `true` when
    cancellation is requested.
*   **Cancellation status**: When a callback reports `true`, LiteRT terminates
    the invocation and surfaces `kLiteRtStatusCancelled`. Callers should check
    for this status and perform cleanup as needed.

## Installing a Cancellation Callback

Register the callback right after creating the compiled model. The callback is
free to capture state (via `absl::AnyInvocable`) in C++ or receive an opaque
pointer via the C API.

### Code Snippet: C++ Cancellation Setup

```cpp
std::atomic<bool> cancel_requested{false};

compiled_model.SetCancellationFunction([&cancel_requested]() -> bool {
  return cancel_requested.load(std::memory_order_relaxed);
});

// Keep the compiled model running on a worker thread.
std::thread worker([&]() {
  auto status = compiled_model.Run(signature_index, input_buffers,
                                   output_buffers);
  if (!status) {
    if (status.Error().Status() == kLiteRtStatusCancelled) {
      std::cout << "Inference cancelled by user request\n";
    }
  }
});

// Request cancellation from another thread.
cancel_requested.store(true, std::memory_order_relaxed);
worker.join();
```

## Note

*   **Use atomic flags or lock-free state**: The callback may run on the
    inference thread while cancellation requests occur on external control
    threads. Prefer atomics to avoid deadlocks.
