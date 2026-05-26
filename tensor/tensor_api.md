# LiteRT Tensor API Documentation

This document provides an overview of the LiteRT Tensor API, including core
tensor manipulations, arithmetic operations, buffer management, and execution
runners. This documentation is specific to the **LiteRT backend** and **LiteRT
runners**.

## Table of Contents
1. [Overview](#overview)
2. [Tensor API](#tensor-api)
    * [TensorHandle](#tensorhandle)
    * [Tensor](#tensor)
    * [TensorInit](#tensorinit)
    * [Creation Functions](#creation-functions)
3. [Buffer API](#buffer-api)
    * [Core Interfaces](#core-interfaces)
    * [RAII Lock Management](#raii-lock-management)
    * [CPU Buffer Implementations](#cpu-buffer-implementations)
4. [LiteRT Buffer](#litert-buffer)
5. [Core Op APIs](#core-op-apis)
    * [Elementwise Operations](#elementwise-operations)
    * [Activations](#activations)
    * [Shape Operations](#shape-operations)
    * [Reduction Operations](#reduction-operations)
    * [Matrix Operations](#matrix-operations)
    * [Neural Network Operations](#neural-network-operations)
    * [Lookup and Indexing Operations](#lookup-and-indexing-operations)
    * [Quantization Operations](#quantization-operations)
    * [Miscellaneous Operations](#miscellaneous-operations)
6. [LiteRT Runners](#litert-runners)
    * [CompiledModelRunner](#compiledmodelrunner)
    * [LitertDynamicRunner](#litertdynamicrunner)
7. [Usage Example](#usage-example)

---

## Overview

LiteRT Tensor is a lightweight, tensor-centric C++ library designed
for high-performance tensor manipulation on mobile devices. It simplifies
complex pre- and post-processing of tensor data and integrates seamlessly
with the LiteRT framework.

---

## Tensor API

Defined in `third_party/odml/litert/tensor/tensor.h`.

### TensorHandle

`TensorHandle` is the core class for managing tensors. It acts as a handle
to the underlying graph tensor representation.

#### Key Methods:

*   **Constructors**:
    *   `TensorHandle()`: Creates an invalid handle.
    *   `TensorHandle(TensorInit init)`: Creates a handle with initialization parameters.
    *   `static TensorHandle Invalid()`: Returns an invalid handle.
*   **Status**:
    *   `absl::Status GetStatus() const`: Checks the validity of the tensor.
*   **Properties**:
    *   `absl::string_view GetName() const`: Gets the tensor name.
    *   `Type GetType() const`: Gets the element type (e.g., `kFP32`, `kI32`).
    *   `const Shape& GetShape() const`: Gets the shape of the tensor.
*   **Data Access**:
    *   `absl::StatusOr<Buffer&> GetBuffer() const`: Gets the underlying buffer.
*   **Modification** (Fluent API):
    *   `SetName(std::string name)`
    *   `SetType(Type t)`
    *   `SetShape(Shape shape)`
    *   `SetBuffer(std::shared_ptr<Buffer> buffer)`

### Tensor

`Tensor<Mixins...>` is a template class that extends `TensorHandle` and allows
mixing in additional functionality based on tags (e.g., backend-specific
extensions).

### TensorInit

A struct used to initialize a `Tensor`.

```cpp
struct TensorInit {
  std::string name;
  Type type = Type::kUnknown;
  Shape shape;
  std::variant<std::shared_ptr<Buffer>, std::vector<float>,
               std::vector<int32_t>, std::vector<int8_t>>
      buffer;
  std::shared_ptr<Quantization> quantization;
};
```

### Creation Functions

Helper functions to create `TensorHandle` instances.

*   `TensorHandle Create(const char* name, Type type, Shape shape)`: For placeholders without a buffer.
*   `template <typename Buffer> TensorHandle Create(const char* name, Type type, Shape shape, Buffer&& buffer)`: For tensors with data.

---

## Buffer API

Defined in `third_party/odml/litert/tensor/buffer.h`.

The Buffer API provides a common interface for accessing tensor data, whether
it is stored in CPU memory or accessible via hardware acceleration.

### Core Interfaces

*   **Buffer**: The base interface for read-only buffers.
    *   `virtual LockedBufferSpan<const std::byte> Lock() = 0`: Locks the
        buffer for CPU access.
*   **MutableBuffer**: Extends `Buffer` for read-write access.
    *   `virtual LockedBufferSpan<std::byte> LockMutable() = 0`: Locks the buffer for mutable CPU access.

### RAII Lock Management

*   **LockedBufferSpan\<T\>**: An RAII object returned by `Lock()` and `LockMutable()`. It provides access to the data and unlocks the buffer when destroyed.
    *   `T* data() const`: Returns the pointer to the data.
    *   `size_t size() const`: Returns the size in elements of type `T`.

### CPU Buffer Implementations

*   **SpanCpuBuffer**: A non-owning view over existing constant CPU data.
*   **MutableSpanCpuBuffer**: A non-owning view over existing mutable CPU data.
*   **OwningCpuBuffer**: Manages its own aligned memory allocation on the CPU.
    *   `static std::shared_ptr<OwningCpuBuffer> Allocate<type>(size_t count)`: Allocates a buffer.
    *   `static std::shared_ptr<OwningCpuBuffer> Copy<type>(Sequence&& seq)`: Creates a buffer by copying data.

---

## LiteRT Buffer

Defined in `third_party/odml/litert/tensor/runners/litert/litert_buffer.h`.

### LitertBuffer

`LitertBuffer` is a custom implementation of `MutableBuffer` that wraps a LiteRT
`TensorBuffer`. This allows passing accelerated buffers (e.g., GPU) between
runners using the high-level Tensor abstraction.

#### Key Features:

*   Wraps `litert::TensorBuffer`.
*   Implements `Lock()` and `LockMutable()` by duplicating the underlying
 `TensorBuffer` and locking it with appropriate modes (`kRead` or `kReadWrite`).

---

## Core Op APIs

Defined in `third_party/odml/litert/tensor/arithmetic.h`.

These APIs allow building computation graphs using a familiar operator syntax or
function calls. They return new `Tensor` instances representing the operation
result.

### Elementwise Operations

Support standard arithmetic and logical operations. Many support broadcasting.

*   `Add(Tensor a, Tensor b)`: Elementwise addition.
*   `Mul(Tensor a, Tensor b)`: Elementwise multiplication.
*   `Sub(Tensor a, Tensor b)`: Elementwise subtraction.
*   `Div(Tensor a, Tensor b)`: Elementwise division.
*   `Minimum(Tensor a, Tensor b)`: Elementwise minimum.
*   `Maximum(Tensor a, Tensor b)`: Elementwise maximum.
*   `Equal(Tensor a, Tensor b)`: Returns a boolean tensor.
*   `NotEqual(Tensor a, Tensor b)`: Returns a boolean tensor.
*   `Less(Tensor a, Tensor b)`: Elementwise less than.
*   `Greater(Tensor a, Tensor b)`: Elementwise greater than.
*   `Abs(Tensor a)`: Elementwise absolute value.
*   `Neg(Tensor a)`: Elementwise negation.
*   `Sqrt(Tensor a)`: Elementwise square root.
*   `Rsqrt(Tensor a)`: Elementwise reciprocal square root.
*   `Pow(Tensor a, Tensor b)`: Elementwise power.
*   `Exp(Tensor a)`: Elementwise exponential.
*   `Log(Tensor a)`: Elementwise natural logarithm.
*   `Cos(Tensor a)`: Elementwise cosine.
*   `Sin(Tensor a)`: Elementwise sine.

### Activations

*   `Relu(Tensor a)`: Rectified Linear Unit.
*   `Relu6(Tensor a)`: Relu capped at 6.
*   `LeakyRelu(Tensor a, float alpha = 0.2f)`: Leaky Relu.
*   `Elu(Tensor a)`: Exponential Linear Unit.
*   `HardSwish(Tensor a)`: Hard Swish activation.
*   `Logistic(Tensor a)`: Sigmoid activation.
*   `Tanh(Tensor a)`: Hyperbolic tangent.
*   `Softmax(Tensor a, float beta = 1)`: Softmax activation.
*   `LogSoftmax(Tensor a)`: Log-Softmax activation.
*   `Gelu(Tensor input, bool approximate = false)`: Gaussian Error Linear Unit.

### Shape Operations

*   `Reshape(Tensor input, std::vector<int> new_shape)`: Reshapes the tensor.
*   `ExpandDims(Tensor input, int axis)`: Inserts a dimension of size 1 at the specified axis.
*   `Squeeze(Tensor input, std::vector<int> squeeze_dims = {})`: Removes dimensions of size 1.
*   `Pad(Tensor a, Tensor b)`: Pads a tensor according to paddings in `b`.
*   `Pack(absl::Span<Tensor> inputs, int axis)`: Packs a list of tensors along a new axis.
*   `Unpack(Tensor input, int num, int axis)`: Unpacks a tensor along an axis.
*   `Split(Tensor input, int axis, int num_splits)`: Splits a tensor into multiple sub-tensors.
*   `Transpose(Tensor input, const std::vector<int>& perm)`: Permutes the dimensions of a tensor.
*   `Tile(Tensor input, const std::vector<int>& multiples)`: Replicates a tensor.
*   `SpaceToDepth(Tensor input, int block_size)`: Rearranges blocks of spatial data into depth.
*   `DepthToSpace(Tensor input, int block_size)`: Rearranges data from depth into blocks of spatial data.
*   `Slice(Tensor input, const std::vector<int>& begin, const std::vector<int>& size)`: Extracts a slice from a tensor.
*   `DynamicUpdateSlice(Tensor operand, Tensor update, const std::vector<int>& start_indices)`: Updates a slice of a tensor.

### Reduction Operations

*   `Sum(Tensor a, std::vector<int> axes, bool keep_dims)`: Sum reduction.
*   `Mean(Tensor a, std::vector<int> axes, bool keep_dims)`: Mean reduction.
*   `ReduceMax(Tensor a, std::vector<int> axes, bool keep_dims)`: Max reduction.
*   `ArgMax(Tensor a, int axis, Type output_type = Type::kI64)`: Returns the indices of the maximum values along an axis.

### Matrix Operations

*   `BatchMatMul(Tensor x, Tensor y, bool adj_x = false, bool adj_y = false)`: Multiplies slices of two tensors in batches.
*   `FullyConnected(Tensor input, Tensor weights,
    std::optional<Tensor> bias = std::nullopt,
    FusedActivation activation = kActNone, bool keep_num_dims = true)`:
    Computes a matrix multiplication with optional bias.

### Neural Network Operations

*   `Conv2D(Tensor input, Tensor filter, Tensor bias, int stride_h,
    int stride_w, Padding padding, int dilation_h_factor = 1,
    int dilation_w_factor = 1, FusedActivation activation = kActNone)`: 2D
    Convolution.
*   `DepthwiseConv2D(Tensor input, Tensor filter, Tensor bias, int stride_h,
    int stride_w, Padding padding, int dilation_h_factor = 1,
    int dilation_w_factor = 1, int depth_multiplier = 1,
    FusedActivation activation = kActNone)`: Depthwise 2D Convolution.
*   `AveragePool2D(Tensor input, int filter_height, int filter_width,
    int stride_h, int stride_w, Padding padding,
    FusedActivation activation = kActNone)`: Average Pooling.
*   `MaxPool2D(Tensor input, int filter_height, int filter_width, int stride_h,
    int stride_w, Padding padding, FusedActivation activation = kActNone)`: Max
    Pooling.
*   `TransposeConv(Tensor filter, Tensor input, Tensor bias,
    const std::vector<int>& output_shape, Padding padding, int stride_h,
    int stride_w, FusedActivation activation = kActNone)`: Transpose
    Convolution.
*   `Lstm(Tensor intermediate, Tensor prev_state)`: Long Short-Term Memory cell.

### Lookup and Indexing Operations

*   `EmbeddingLookup(Tensor ids, Tensor value, Type output_type = Type::kFP32)`: Looks up embeddings.
*   `Gather(Tensor input, Tensor indices, int axis, int batch_dims = 0)`: Gathers slices from `input` according to `indices`.
*   `GatherNd(Tensor input, Tensor indices)`: Gathers slices from `input` into a new tensor with shape specified by `indices`.
*   `OneHot(Tensor indices, Tensor depth, Tensor on_value, Tensor off_value, int axis = -1)`: Generates a one-hot tensor.
*   `TopK(Tensor input, int k)`: Finds values and indices of the `k` largest entries.

### Quantization Operations

*   `Quantize(Tensor a, Type type, std::vector<float> scale, std::vector<int64_t> zero_point)`: Quantizes a float tensor to an integer type.
*   `Dequantize(Tensor a)`: Dequantizes a tensor to float.

### Miscellaneous Operations

*   `Custom(absl::Span<Tensor> inputs, std::string custom_code,
    std::vector<uint8_t> custom_options, const
    std::vector<std::vector<int>>& output_shapes, const std::vector<Type>&
    output_types)`: Creates a custom operation.
*   `Cumsum(Tensor input, int axis, bool exclusive = false, bool reverse = false)`: Computes the cumulative sum.
*   `Reverse(Tensor input, Tensor axes)`: Reverses specific dimensions of a tensor.
*   `ResizeBilinear(Tensor input, const std::vector<int>& size, bool align_corners = false, bool half_pixel_centers = false)`: Resizes images using bilinear interpolation.
*   `ResizeNearestNeighbor(Tensor input, const std::vector<int>& size, bool align_corners = false, bool half_pixel_centers = false)`: Resizes images using nearest neighbor interpolation.
*   `NonMaxSuppressionV5(Tensor boxes, Tensor scores, int max_output_size, Tensor iou_threshold, Tensor score_threshold, Tensor soft_nms_sigma)`: Performs non-max suppression on bounding boxes.
*   `Probe(Tensor a)`: Inserts a probe operation for debugging.

---

## LiteRT Runners

These classes handle the execution of models built with the Tensor API using th
LiteRT infrastructure.

### CompiledModelRunner

Defined in `third_party/odml/litert/tensor/runners/litert/compiled_model_runner.h`.

A template class that takes a model functor and runs it using LiteRT's
`CompiledModel`.

#### Key Methods:

*   **Constructor**: `CompiledModelRunner(Environment& env, Options& options, ModelFunctor model_func, bool build_model_now = true)`
*   `absl::Status BuildModel(...)`: Serializes the graph and creates the LiteRT `CompiledModel`.
*   `absl::Status SetInput(const std::string& name, const std::vector<T>& data)`: Sets input data.
*   `absl::Status SetInput(const std::string& name, const TensorHandle& tensor)`: Sets input from another tensor.
*   `absl::Status Run()`: Executes the model.
*   `absl::StatusOr<std::vector<T>> GetOutput(const std::string& name)`: Retrieves output data.
*   `absl::StatusOr<TensorHandle> GetOutput(const std::string& name)`: Retrieves output as a tensor handle.

### LitertDynamicRunner

Defined in `third_party/odml/litert/tensor/runners/litert/litert_dynamic_runner.h`.

A runner that supports models with multiple signatures and dynamic execution.

#### Key Methods:

*   **Creation**:
    *   `static absl::StatusOr<LitertDynamicRunner> Create(Environment& env, const std::string& model_path, Options& options)`
    *   `static absl::StatusOr<LitertDynamicRunner> Create(Environment& env, absl::Span<const uint8_t> model_buffer, Options& options)`
*   **Input/Output** (Support specifying signature name):
    *   `absl::Status SetInput(const std::string& signature_name, const std::string& name, const TensorHandle& tensor)`
    *   `absl::Status SetInput(const std::string& signature_name, const std::string& name, absl::Span<const uint8_t> data)`
    *   `absl::StatusOr<TensorHandle> GetOutput(const std::string& signature_name, const std::string& name)`
*   **Execution**:
    *   `absl::Status Run(const std::string& signature_name)`: Runs a specific signature.

---

## Usage Example

Based on `third_party/odml/litert/tensor/examples/segmentation/segmentation_example_webgpu.cc`.

This example demonstrates how to use the `LitertDynamicRunner` to load a model,
set inputs, run inference, and access the output data.

```cpp
#include <iostream>
#include <vector>
#include "third_party/odml/litert/litert/cc/litert_environment.h"
#include "third_party/odml/litert/litert/cc/litert_options.h"
#include "third_party/odml/litert/tensor/tensor.h"
#include "third_party/odml/litert/tensor/runners/litert/litert_dynamic_runner.h"

using namespace litert::tensor;

void RunSegmentation(const std::string& model_path, const std::vector<float>& input_image_data) {
  // 1. Initialize LiteRT Environment and Options
  auto env_or = litert::Environment::Create({});
  auto env = std::move(*env_or);
  
  auto options_or = litert::Options::Create();
  auto options = std::move(*options_or);
  
  // Request GPU acceleration if available
  options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);

  // 2. Create the LitertDynamicRunner
  auto runner_or = LitertDynamicRunner::Create(env, model_path, options);
  if (!runner_or.ok()) {
    std::cerr << "Failed to create runner: " << runner_or.status() << std::endl;
    return;
  }
  auto runner = std::move(*runner_or);

  // 3. Prepare Input Tensor
  // We assume the model expects a 512x512 RGB image (float32)
  Shape input_shape = {1, 512, 512, 3};
  TensorHandle input_tensor = Create("input", Type::kFP32, input_shape, input_image_data);

  // 4. Set Input for the default signature
  absl::Status status = runner.SetInput("default", "input", input_tensor);
  if (!status.ok()) {
    std::cerr << "Failed to set input: " << status << std::endl;
    return;
  }

  // 5. Run Inference
  status = runner.Run("default");
  if (!status.ok()) {
    std::cerr << "Failed to run model: " << status << std::endl;
    return;
  }

  // 6. Get Output Tensor
  auto output_or = runner.GetOutput("default", "output");
  if (!output_or.ok()) {
    std::cerr << "Failed to get output: " << output_or.status() << std::endl;
    return;
  }
  TensorHandle output_tensor = std::move(*output_or);

  // 7. Access Output Data
  auto buffer_or = output_tensor.GetBuffer();
  if (!buffer_or.ok()) {
    std::cerr << "Failed to get output buffer: " << buffer_or.status() << std::endl;
    return;
  }
  
  // Lock the buffer to access CPU memory
  auto lock_span = buffer_or->Lock();
  const float* output_data = reinterpret_cast<const float*>(lock_span.data());

  // Process the output data (e.g., find argmax for segmentation)
  // ...
}
```
