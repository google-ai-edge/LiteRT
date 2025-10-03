# LiteRT Custom Op Dispatcher API

## Overview

The `CustomOpDispatcher` is the API replacement for defining custom CPU
operations and custom op resolvers in LiteRT. It provides a cleaner interface to
integrating custom operations into LiteRT models.

## Why Use CustomOpDispatcher?

### Traditional Approach (Deprecated)

Previously, developers had to:

- Manually create `TfLiteRegistration` structures
- Implement TFLite-specific callback functions (init, prepare, invoke, free)
- Work directly with low-level TFLite structures (`TfLiteContext`, `TfLiteNode`,
  `TfLiteTensor`)
- Use `MutableOpResolver::AddCustom()` directly

### New CustomOpDispatcher Approach

The `CustomOpDispatcher` provides:

- Clean abstraction layer over TFLite internals
- Integration with LiteRT's compiled model

## Flow

```
┌─────────────────┐
│  User Custom Op │ (Your implementation)
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│   LiteRtCustomOpKernel  │ (C API interface)
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   CustomOpDispatcher    │ (Bridge layer)
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  TfLiteRegistration     │ (TFLite runtime)
└─────────────────────────┘
```

## Key Components and Files

### Core Implementation

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

- **[runtime/custom_op_dispatcher.h](../litert/runtime/custom_op_dispatcher.h)**:
  Main dispatcher class header

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

- **[runtime/custom_op_dispatcher.cc](../litert/runtime/custom_op_dispatcher.cc)**:
  Implementation bridging LiteRT to TFLite

### API Headers

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

- **[c/litert_custom_op_kernel.h](../litert/c/litert_custom_op_kernel.h)**: C
  API for custom op kernel interface

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

- **[cc/litert_custom_op_kernel.h](../litert/cc/litert_custom_op_kernel.h)**:
  C++ wrapper providing object-oriented interface

### Options System

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

- **[core/options.h](../litert/core/options.h)**: Core options structure with
  `CustomOpOption`

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

- **[c/litert_options.h](../litert/c/options.h)**: C API for managing
  compilation options

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

- **[cc/litert_options.h](../litert/cc/options.h)**: C++ wrapper for options
  management

### Test Examples

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

- **[cc/litert_custom_op_test.cc](../litert/cc/litert_custom_op_test.cc)**: C++
  API usage example

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

- **[c/litert_custom_op_test.cc](../litert/c/litert_custom_op_test.cc)**: C API
  usage example

## API Reference

### Core Kernel Interface (C API)

```c
typedef struct {
  LiteRtStatus (*Init)(void* user_data, const void* init_data,
                       size_t init_data_size);
  LiteRtStatus (*GetOutputLayouts)(void* user_data, size_t num_inputs,
                                   const LiteRtLayout* input_layouts,
                                   size_t num_outputs,
                                   LiteRtLayout* output_layouts);
  LiteRtStatus (*Run)(void* user_data, size_t num_inputs,
                      const LiteRtTensorBuffer* inputs, size_t num_outputs,
                      LiteRtTensorBuffer* outputs);
  LiteRtStatus (*Destroy)(void* user_data);
} LiteRtCustomOpKernel;
```

### C++ Abstract Base Class

```cpp
class CustomOpKernel {
public:
  virtual const std::string& OpName() const = 0;
  virtual int OpVersion() const = 0;
  virtual Expected<void> Init(const void* init_data, size_t init_data_size) = 0;
  virtual Expected<void> GetOutputLayouts(
      const std::vector<Layout>& input_layouts,
      std::vector<Layout>& output_layouts) = 0;
  virtual Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                             std::vector<TensorBuffer>& outputs) = 0;
  virtual Expected<void> Destroy() = 0;
};
```

## Implementation Guide

### Step 1: Define Your Custom Operation

#### C++ Implementation

```cpp
#include "litert/cc/litert_custom_op_kernel.h"

class MyCustomOpKernel : public litert::CustomOpKernel {
public:
  const std::string& OpName() const override {
    return op_name_;
  }

  int OpVersion() const override {
    return 1;
  }

  Expected<void> Init(const void* init_data, size_t init_data_size) override {
    // Initialize any persistent state
    return {};
  }

  Expected<void> GetOutputLayouts(
      const std::vector<Layout>& input_layouts,
      std::vector<Layout>& output_layouts) override {
    // Define output tensor shapes based on inputs
    output_layouts[0] = input_layouts[0];
    return {};
  }

  Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                     std::vector<TensorBuffer>& outputs) override {
    // Lock input buffers for reading
    LITERT_ASSIGN_OR_RETURN(auto input_lock,
        TensorBufferScopedLock::Create<float>(
            inputs[0], TensorBuffer::LockMode::kRead));

    // Lock output buffer for writing
    LITERT_ASSIGN_OR_RETURN(auto output_lock,
        TensorBufferScopedLock::Create<float>(
            outputs[0], TensorBuffer::LockMode::kWrite));

    const float* input_data = input_lock.second;
    float* output_data = output_lock.second;

    // Perform computation
    // ... your custom operation logic ...

    return {};
  }

  Expected<void> Destroy() override {
    // Clean up resources
    return {};
  }

private:
  const std::string op_name_ = "MyCustomOp";
};
```

#### C Implementation

```c
#include "litert/c/litert_custom_op_kernel.h"

LiteRtStatus MyOp_Init(void* user_data, const void* init_data,
                       size_t init_data_size) {
  // Initialize state
  return kLiteRtStatusOk;
}

LiteRtStatus MyOp_GetOutputLayouts(void* user_data, size_t num_inputs,
                                   const LiteRtLayout* input_layouts,
                                   size_t num_outputs,
                                   LiteRtLayout* output_layouts) {
  // Set output shape to match first input
  output_layouts[0] = input_layouts[0];
  return kLiteRtStatusOk;
}

LiteRtStatus MyOp_Run(void* user_data, size_t num_inputs,
                     const LiteRtTensorBuffer* inputs, size_t num_outputs,
                     LiteRtTensorBuffer* outputs) {
  // Lock buffers
  void* input_addr;
  LITERT_RETURN_IF_ERROR(LiteRtLockTensorBuffer(
      inputs[0], &input_addr, kLiteRtTensorBufferLockModeRead));

  void* output_addr;
  LITERT_RETURN_IF_ERROR(LiteRtLockTensorBuffer(
      outputs[0], &output_addr, kLiteRtTensorBufferLockModeWrite));

  // Perform computation
  float* in = (float*)input_addr;
  float* out = (float*)output_addr;
  // ... your custom operation logic ...

  // Unlock buffers
  LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(inputs[0]));
  LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(outputs[0]));

  return kLiteRtStatusOk;
}

LiteRtStatus MyOp_Destroy(void* user_data) {
  // Clean up
  return kLiteRtStatusOk;
}
```

### Step 2: Register the Custom Operation

#### C++ Registration

```cpp
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_options.h"

// Create environment
LITERT_ASSERT_OK_AND_ASSIGN(Environment env, Environment::Create({}));

// Load model
Model model = /* load your model */;

// Create options and register custom op
LITERT_ASSERT_OK_AND_ASSIGN(Options options, Options::Create());
options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);

// Register custom op kernel
MyCustomOpKernel my_custom_op;
ASSERT_TRUE(options.AddCustomOpKernel(my_custom_op));

// Create compiled model with custom op
LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                            CompiledModel::Create(env, model, options));
```

#### C Registration

```c
#include "litert/c/litert_environment.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_options.h"

// Create options
LiteRtOptions options;
LiteRtCreateOptions(&options);
LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu);

// Define kernel
LiteRtCustomOpKernel kernel = {
    .Init = MyOp_Init,
    .GetOutputLayouts = MyOp_GetOutputLayouts,
    .Run = MyOp_Run,
    .Destroy = MyOp_Destroy,
};

// Register custom op
LiteRtAddCustomOpKernelOption(options, "MyCustomOp", 1, &kernel, NULL);

// Create environment
LiteRtEnvironment env;
LiteRtCreateEnvironment(0, NULL, &env);

// Create compiled model
LiteRtCompiledModel compiled_model;
LiteRtCreateCompiledModel(env, model, options, &compiled_model);
```

### Step 3: Execute the Model

#### C++ Execution

```cpp
// Create buffers
LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                            compiled_model.CreateInputBuffers());
LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                            compiled_model.CreateOutputBuffers());

// Fill input data
input_buffers[0].Write<float>(your_input_data);

// Run inference
compiled_model.Run(input_buffers, output_buffers);

// Read output
LITERT_ASSERT_OK_AND_ASSIGN(auto lock,
    TensorBufferScopedLock::Create<const float>(
        output_buffers[0], TensorBuffer::LockMode::kRead));
const float* results = lock.second;
```

#### C Execution

```c
// Create buffers (see test files for complete buffer creation)
LiteRtTensorBuffer input_buffers[num_inputs];
LiteRtTensorBuffer output_buffers[num_outputs];
// ... buffer creation code ...

// Write input data
void* input_addr;
LiteRtLockTensorBuffer(input_buffers[0], &input_addr,
                      kLiteRtTensorBufferLockModeWrite);
memcpy(input_addr, your_data, data_size);
LiteRtUnlockTensorBuffer(input_buffers[0]);

// Run inference
LiteRtRunCompiledModel(compiled_model, 0, num_inputs, input_buffers,
                       num_outputs, output_buffers);

// Read output
void* output_addr;
LiteRtLockTensorBuffer(output_buffers[0], &output_addr,
                      kLiteRtTensorBufferLockModeRead);
// Process output_addr
LiteRtUnlockTensorBuffer(output_buffers[0]);
```
