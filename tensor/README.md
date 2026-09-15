# LiteRT Tensor

**A lightweight, Tensor-centric C++ library for high-performance tensor
manipulation on mobile devices.**

LiteRT Tensor provides an expressive and intuitive API for developers working
with the LiteRT framework, simplifying complex pre- and post-processing of
tensor data, model authoring, and kernel verification.

## Key Features

*   **Expressive C++ API:** Offers a familiar, tensor-centric syntax with fluent
    interfaces and operator overloading for building computation graphs.
*   **Backend Agnostic with Mixins:** Easily switch between different execution
    backends (e.g., TfLite) using template mixins.
*   **Zero-Copy Execution:** Optimized for mobile with a strong focus on
    zero-copy operations. Shared buffers (`Buffer`, `TensorBuffer`) avoid
    unnecessary data duplication between pipeline stages.
*   **Seamless LiteRT Integration:** Designed to work directly with LiteRT's
    runtime, allowing easy compilation of custom graphs into LiteRT models.
*   **Dynamic & Multi-Signature Support:** Build and run models with multiple
    entry points and dynamic shapes.

---

## Core Use Cases & Examples

### 1. LLM Model Authoring

LiteRT Tensor is powerful enough to author complex neural network layers
directly in C++ and compile them to TFLite flatbuffers.

Here is a simplified example of defining a **Multi-Head Attention (MHA)**
layer:

```cpp
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "third_party/odml/litert/tensor/tensor.h"

using namespace litert::tensor;

template <class... Mixins>
Tensor<Mixins...> MultiHeadAttention(Tensor<Mixins...> input, int d_model, int num_heads) {
  using Tensor = Tensor<Mixins...>;
  const int d_k = d_model / num_heads;

  // Define weights and biases (placeholders)
  Tensor w_q({.name = "w_q", .type = Type::kFP32, .shape = {d_model, d_model}});
  Tensor b_q({.name = "b_q", .type = Type::kFP32, .shape = {d_model}});
  // ... define w_k, b_k, w_v, b_v ...

  // Project Q, K, V using FullyConnected
  Tensor q = FullyConnected(input, w_q, b_q, kActNone, true);
  Tensor k = FullyConnected(input, w_k, b_k, kActNone, true);
  Tensor v = FullyConnected(input, w_v, b_v, kActNone, true);

  // Reshape and Transpose for Multi-Head (assuming batch_size=1, seq_len=128)
  q = Reshape(q, {1, 128, num_heads, d_k});
  q = Transpose(q, Tensor({.type = Type::kI32, .shape = {4},
                           .buffer = OwningCpuBuffer::Copy<Type::kI32>({0, 2, 1, 3})}));

  // ... similarly reshape and transpose k and v ...

  // Scaled Dot-Product Attention
  Tensor scores = Mul(q, k);
  Tensor scaled_scores = Div(scores, Sqrt(Tensor({
      .type = Type::kFP32, .shape = {1},
      .buffer = OwningCpuBuffer::Copy<Type::kFP32>({(float)d_k})})));
  Tensor attention_weights = Softmax(scaled_scores);
  Tensor attention_output = Mul(attention_weights, v);

  // Concatenate and Final Projection
  attention_output = Transpose(attention_output, Tensor({
      .type = Type::kI32, .shape = {4},
      .buffer = OwningCpuBuffer::Copy<Type::kI32>({0, 2, 1, 3})}));
  attention_output = Reshape(attention_output, {1, 128, d_model});

  return attention_output;
}

int main() {
  // Define input tensor tagged with TfLiteMixinTag
  Tensor<TfLiteMixinTag> input({.name = "input", .type = Type::kFP32, .shape = {1, 128, 256}});

  // Build MHA graph
  Tensor output = MultiHeadAttention(input, 256, 4);
  output.SetName("output");

  // Serialize and save to TFLite model
  ModelFactory model_builder;
  model_builder.AddSignature({output}, "serving_default");
  model_builder.Save("/tmp/mha.tflite");
  
  return 0;
}
```

### 2. End-to-End Pre/Post Processing with Zero-Copy

You can chain pre-processing, core inference, and post-processing models
together.
By using `LambdaRunner` and sharing buffers, you can achieve **zero-copy**
execution between GPU/CPU pipeline stages.

```cpp
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/runners/litert/lambda_model_runner.h"
#include "third_party/odml/litert/tensor/runners/litert/litert_dynamic_runner.h"

using namespace litert::tensor;

void RunPipeline(litert::Environment& env, litert::Options& options) {
  // 1. Define Pre-processing (e.g., Resize and Normalize)
  auto pre_runner = CreateLambdaRunner(env, options,
      {{"raw_image", Tensor<TfLiteMixinTag>({.name = "raw_image", .type = Type::kFP32, .shape = {1, 512, 512, 3}})}},
      [](const auto& inputs) {
        Tensor resized = ResizeBilinear(inputs.at("raw_image"), {256, 256});
        Tensor scaled = Mul(resized, 2.0f);
        Tensor normalized = Add(scaled, -1.0f);
        return absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>{{"normalized_image", normalized}};
      });

  // 2. Load Core Model
  auto core_runner = LitertDynamicRunner::Create(env, "core_model.tflite", options).value();

  // 3. Connect Pre-processing Output to Core Input (Zero-Copy!)
  auto core_input = core_runner.GetInput(0).value();
  pre_runner.SetOutput("normalized_image", core_input);

  // 4. Set raw input data and run
  std::vector<float> raw_image_data = ...; // Load image data
  auto raw_input_tensor = Create("raw_image", Type::kFP32, {1, 512, 512, 3}, std::move(raw_image_data));
  pre_runner.SetInput("raw_image", raw_input_tensor);

  pre_runner.Run();   // Pre-processing runs, outputs go directly to core_input
  core_runner.Run();  // Core model runs
}
```

### 3. Kernel & Op Verification Test Suite

LiteRT Tensor supports `GraphProbe`, enabling you to insert "probes" into the
computation graph. This is extremely useful for building test suites that
verify backend correctness by comparing intermediate outputs against a
reference CPU implementation.

```cpp
#include "third_party/odml/litert/tensor/internal/graph_probe.h"
#include "third_party/odml/litert/tensor/runners/litert/compiled_model_runner.h"

// Define probes to intercept intermediate tensors
absl::flat_hash_map<GraphProbe::StableTensorId, std::string, GraphProbe::StableTensorIdHash> probes;
probes[{fc_op_id, 0}] = "fc_output_probe";

// Register probes with both GPU (optimized) and CPU (reference) runners
optimized_runner.AddTensorsAsOutputs(probes);
reference_runner.AddTensorsAsOutputs(probes);

// Run inference
optimized_runner.Run();
reference_runner.Run();

// Retrieve and compare intermediate results
std::vector<float> opt_intermediate = optimized_runner.GetFloatOutput("fc_output_probe").value();
std::vector<float> ref_intermediate = reference_runner.GetFloatOutput("fc_output_probe").value();

// Assert results are within tolerance
EXPECT_THAT(opt_intermediate, Pointwise(testing::FloatNear(kTolerance), ref_intermediate));
```

---

## Getting Started

For a complete API reference, see [Tensor API Documentation](tensor_api.md).

Here is a simple, self-contained example of building and running an element-wise
addition on CPU using the LiteRT Tensor API:

```cpp
#include <iostream>
#include <vector>
#include "third_party/odml/litert/litert/cc/litert_environment.h"
#include "third_party/odml/litert/litert/cc/litert_options.h"
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/runners/litert/lambda_model_runner.h"
#include "third_party/odml/litert/tensor/tensor.h"

using namespace litert::tensor;

int main() {
  // 1. Initialize LiteRT Environment and Options
  auto env = std::move(*litert::Environment::Create({}));
  auto options = std::move(*litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);

  // 2. Define the computation graph using LambdaRunner
  auto runner = CreateLambdaRunner(env, options,
      {
          {"a", Tensor<TfLiteMixinTag>({.name = "a", .type = Type::kFP32, .shape = {3}})},
          {"b", Tensor<TfLiteMixinTag>({.name = "b", .type = Type::kFP32, .shape = {3}})}
      },
      [](const auto& inputs) {
        Tensor c = Add(inputs.at("a"), inputs.at("b"));
        return absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>{{"c", c}};
      });

  // 3. Set input data
  auto a_tensor = Create("a", Type::kFP32, {3}, std::vector<float>{1.0f, 2.0f, 3.0f});
  auto b_tensor = Create("b", Type::kFP32, {3}, std::vector<float>{4.0f, 5.0f, 6.0f});
  runner.SetInput("a", a_tensor);
  runner.SetInput("b", b_tensor);

  // 4. Execute the graph
  runner.Run();

  // 5. Access the output
  auto c_tensor = std::move(*runner.GetOutput("c"));
  auto buffer = c_tensor.GetBuffer().value();
  auto locked_span = buffer->Lock();
  const float* c_data = reinterpret_cast<const float*>(locked_span.data());

  std::cout << "Result: " << c_data[0] << " " << c_data[1] << " " << c_data[2] << std::endl;
  // Output: Result: 5 7 9

  return 0;
}
```
