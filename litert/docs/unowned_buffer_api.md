# LiteRT Unowned Buffer API

## Overview

The LiteRT Unowned Buffer API allows you to create a model from an external buffer without copying the data. This is useful for scenarios where:

- You have the model data in memory and want to avoid the overhead of copying
- You're memory-constrained and need zero-copy model loading
- You're loading models from memory-mapped files or shared memory

## API Reference

### C++ API

```cpp
#include "litert/cc/litert_model.h"

// Create a model from an unowned buffer
litert::BufferRef<uint8_t> buffer_ref(data_ptr, data_size);
auto result = litert::Model::CreateFromUnownedBuffer(buffer_ref);
if (result.ok()) {
  litert::Model model = std::move(result.value());
  // Use the model...
}
```

### C API

```c
#include "litert/c/litert_model.h"

// Create a model from an unowned buffer
LiteRtModel model = NULL;
LiteRtStatus status = LiteRtCreateModelFromUnownedBuffer(
    buffer_data, buffer_size, &model);
if (status == kLiteRtStatusOk) {
  // Use the model...
  LiteRtDestroyModel(model);
}
```

## Important Considerations

### Buffer Lifetime

**Critical**: The buffer MUST remain valid for the entire lifetime of the model. The model does not copy the buffer data, so deallocating or modifying the buffer while the model is in use will result in undefined behavior.

### Thread Safety

The buffer should not be modified by any thread while the model is in use. Read-only concurrent access is safe.

### Example: Correct Usage

```cpp
// Good: Buffer outlives the model
std::vector<uint8_t> model_data = LoadModelData();
{
  litert::BufferRef<uint8_t> buffer_ref(model_data.data(), model_data.size());
  auto model_result = litert::Model::CreateFromUnownedBuffer(buffer_ref);
  if (model_result.ok()) {
    auto model = std::move(model_result.value());
    // Use model...
  }
  // Model destroyed here
}
// Buffer can be safely deallocated after model is destroyed
```

### Example: Incorrect Usage (Undefined Behavior)

```cpp
// Bad: Buffer destroyed before model
litert::Model model;
{
  std::vector<uint8_t> temp_data = LoadModelData();
  litert::BufferRef<uint8_t> buffer_ref(temp_data.data(), temp_data.size());
  auto result = litert::Model::CreateFromUnownedBuffer(buffer_ref);
  model = std::move(result.value());
  // temp_data destroyed here!
}
// Using model here is undefined behavior - buffer is gone!
```

## Use Cases

### Memory-Mapped Files

```cpp
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int fd = open("model.tflite", O_RDONLY);
struct stat sb;
fstat(fd, &sb);

void* mapped = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
if (mapped != MAP_FAILED) {
  litert::BufferRef<uint8_t> buffer_ref(
      static_cast<uint8_t*>(mapped), sb.st_size);
  auto model = litert::Model::CreateFromUnownedBuffer(buffer_ref);
  // Use model...
  
  // Clean up when done
  munmap(mapped, sb.st_size);
}
close(fd);
```

### Embedded Systems

For embedded systems with model data in ROM:

```c
// Model data in read-only memory
extern const uint8_t model_data[];
extern const size_t model_size;

LiteRtModel model = NULL;
LiteRtStatus status = LiteRtCreateModelFromUnownedBuffer(
    model_data, model_size, &model);
// Model can be used without any heap allocation for the buffer
```

## Performance Comparison

| Method | Memory Usage | Load Time | Notes |
|--------|-------------|-----------|--------|
| `CreateFromFile` | Model size (copied) | File I/O + copy | Standard method |
| `CreateFromBuffer` | Model size (copied) | Copy time | Copies the buffer |
| `CreateFromUnownedBuffer` | 0 (no copy) | ~0 | Zero-copy, fastest |

## Migration Guide

If you're currently using `CreateFromBuffer` and want to switch to the unowned version:

```cpp
// Before:
auto model = litert::Model::CreateFromBuffer(buffer);

// After:
auto model = litert::Model::CreateFromUnownedBuffer(buffer);
// Ensure buffer remains valid!
```

Remember: The key difference is buffer lifetime management. With `CreateFromBuffer`, you can discard the buffer after model creation. With `CreateFromUnownedBuffer`, you must keep the buffer alive.