# LiteRT Constant Output Support Guide

This guide covers how LiteRT handles constant output tensors so applications
can consume read-only model results without performing extra synchronization.

## Introduction

Some TensorFlow Lite models expose outputs backed by constant buffer data like
lookup tables.

LiteRT detects these read-only tensors and ensures their contents are copied to
application-accessible buffers after each invocation.

## Core Concepts

*   **Constant outputs**: Tensors whose allocation type is `kTfLiteMmapRo` and
    whose memory originates from the model flatbuffer rather than the runtime.
*   **Registration tracking**: During buffer registration LiteRT flags constant
    outputs and captures the locked host addresses returned by
    `LiteRtLockTensorBuffer`.
*   **Post-invoke synchronization**: After `SignatureRunner::Invoke` completes,
    LiteRT copies the constant tensor data into the locked output buffers so
    downstream code can read stable values.

## Using Constant Outputs

Constant outputs appear alongside regular outputs in the buffer vector returned
by `CreateOutputBuffers`.
### Constant vs. Non-Constant Outputs

| Aspect | Constant Output Tensor | Regular (Non-Constant) Output Tensor |
| --- | --- | --- |
| Backing memory | Read-only flatbuffer payload (`kTfLiteMmapRo`) embedded in the model | Allocated by LiteRT / delegates for each invocation |
| Data updates across runs | Static contents copied after every invoke; values never change | Populated with fresh results for each invocation |
| Buffer registration | Runtime tracks locked buffer addresses and copies constants post-invoke | Runtime register buffers directly and relies on normal writes |

### Code Snippet: Reading Constant Outputs (C++)

Note that the following implementation is compatible with both constant output
and non-constant output.

```cpp
LITERT_ASSERT_OK_AND_ASSIGN(
    std::vector<litert::TensorBuffer> output_buffers,
    compiled_model.CreateOutputBuffers());

// Run inference as usual.
LITERT_ASSERT_OK(
    compiled_model.Run(input_buffers, output_buffers));

// Lock each buffer and inspect its contents.
for (auto& buffer : output_buffers) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto byte_size, buffer.Size());
  size_t element_count = byte_size / sizeof(float);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_ptr,
      litert::TensorBufferScopedLock::Create<const float>(
          buffer, litert::TensorBuffer::LockMode::kRead));
  absl::Span<const float> values(lock_and_ptr.second, element_count);
  // Constant outputs retain the same contents across invocations.
  ProcessOutput(values);
}
```

The runtime copies constant tensors into the locked buffer immediately after
invocation, so no additional synchronization is required on the application
side.
