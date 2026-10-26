---
name: custom-ops-gpu
description: >-
  Guides authoring, testing, and benchmarking LiteRT and MLDrift GPU custom operations
  across Apple Metal, OpenCL, and WebGPU backends. Covers unified kernel dispatch patterns
  incorporating reviewer feedback (unifying legacy AST and modern IR selectors),
  in-memory TFLite flatbuffer test subgraph construction, numerical verification, and
  cross-platform micro-benchmarking using the LiteRT CompiledModel C++ API.
  Use when implementing new GPU custom ops, refactoring delegate op selectors, writing
  unit tests for LiteRT GPU delegate kernels, or measuring kernel latency across Metal,
  OpenCL, or WebGPU. Don't use for CPU-only XNNPACK/Eigen kernels or standard TFLite builtin ops.
---

# LiteRT GPU Custom Op Authoring, Testing & Benchmarking

This skill encodes end-to-end best practices, architectural patterns, test utilities, and cross-platform benchmarking workflows for LiteRT and MLDrift GPU custom operations across **Apple Metal**, **OpenCL**, and **WebGPU**.

---

## Triggers and Anti-Triggers

*   **Use when**:
    *   Implementing a new GPU custom op or composite op in MLDrift / LiteRT.
    *   Refactoring or unifying custom op kernel selectors between the legacy AST delegate (`composite/litert_op_selector.cc`) and modern IR delegate (`composite/ir/litert_op_selector.cc`).
    *   Constructing programmatic in-memory TFLite test models for GPU delegate unit tests using FlatBuffers.
    *   Writing numerical correctness tests comparing GPU delegate kernels against CPU reference implementations.
    *   Benchmarking custom op latency across Apple Silicon (Metal), Android/Linux (OpenCL), or browser/native (WebGPU).
    *   Integrating custom GPU ops into `litert_lm` runtime engine and verifying 100% graph delegation.

*   **Don't use for**:
    *   Implementing standard CPU-only TFLite kernels (e.g. Eigen/XNNPACK-only).
    *   Modifying standard built-in TFLite operators that do not require custom op registration.

---

## 1. Authoring Patterns & Reviewer Feedback

### Architectural Overview

When LiteRT compiles a model for GPU execution, custom operations travel through either the **legacy AST path** or the **modern IR path**:

```
[TFLite Model Buffer (FlatBuffer)]
           │
           ├─── (Modern IR Path) ───► IrParser ──► IrOp ───────► composite/ir/litert_op_selector.cc ──┐
           │                                                                                           ▼
           └─── (Legacy AST Path) ──────────────► tflite::Node ─► composite/litert_op_selector.cc ────► Kernel Builder ──► GPU Graph (GPUOperation)
```

### Reviewer Feedback: The Unified Kernel Dispatch Pattern

**The Problem**:
A common anti-pattern is for `composite/litert_op_selector.cc` and `composite/ir/litert_op_selector.cc` to separately parse attributes, allocate input/output ID vectors, and call `model_builder->AddGpuOperation(...)`. This duplicates 20–40 lines of error-prone boilerplate in multiple places.

**The Solution (Canonical Pattern)**:
Follow the pattern used by canonical composite operations (`swiglu`, `short_conv_step`, `moe_experts`):
1. The kernel header exposes only two high-level entry points accepting `GpuModelBuilder*`.
2. The kernel implementation file (`.cc`) encapsulates attribute extraction, ID collection, GPU operation instantiation, and graph registration into a single shared function.
3. The selectors become clean, single-line delegations.

#### A. Kernel Header Interface (`composite/<op_name>_kernel.h`)

Expose only these two functions returning `absl::Status`:

```cpp
#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_MY_OP_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_MY_OP_KERNEL_H_

#include "absl/status/status.h"
#include "third_party/ml_drift/common/gpu_model_builder.h"
#include "third_party/ml_drift/ir/ir.h"
#include "third_party/tensorflow/lite/core/c/common.h"

namespace litert {
namespace ml_drift {

// Legacy AST selector entry point.
absl::Status CreateMyOpFromNode(const tflite::Node* node,
                                ::ml_drift::GpuModelBuilder* model_builder);

// Modern IR selector entry point.
absl::Status CreateMyOpFromIrOp(const ::ml_drift::ir::IrOp* ir_op,
                                ::ml_drift::GpuModelBuilder* model_builder);

}  // namespace ml_drift
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_MY_OP_KERNEL_H_
```

#### B. Kernel Implementation & Generic ID Extractor (`composite/<op_name>_kernel.cc`)

Use a template helper `GetTensorIds<T>` to unify tensor ID extraction:

```cpp
#include "ml_drift/delegate/composite/my_op_kernel.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "third_party/absl/status/status_macros.h"
#include "third_party/ml_drift/common/gpu_model_builder.h"
#include "third_party/ml_drift/common/gpu_operation.h"
#include "third_party/ml_drift/common/operations/my_op.h"
#include "third_party/ml_drift/ir/ir.h"
#include "third_party/tensorflow/lite/core/c/common.h"

namespace litert {
namespace ml_drift {
namespace {

// Generic helper that extracts uint32_t IDs across both Value* and IrTensor*.
template <typename T>
std::vector<uint32_t> GetTensorIds(const std::vector<T>& tensors) {
  std::vector<uint32_t> ids;
  ids.reserve(tensors.size());
  for (const auto& t : tensors) {
    ids.push_back(t->id);
  }
  return ids;
}

template <typename TensorType>
absl::Status BuildMyOpGpuGraph(
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs,
    const ::ml_drift::MyOpAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  std::unique_ptr<::ml_drift::GPUOperation> gpu_op;
  RETURN_IF_ERROR(::ml_drift::CreateMyOp(
      model_builder->GetGpuInfo(), attr, &gpu_op));

  std::vector<uint32_t> input_ids = GetTensorIds(inputs);
  std::vector<uint32_t> output_ids = GetTensorIds(outputs);

  return model_builder->AddGpuOperation(input_ids, output_ids,
                                        std::move(gpu_op));
}

}  // namespace

absl::Status CreateMyOpFromNode(const tflite::Node* node,
                                ::ml_drift::GpuModelBuilder* model_builder) {
  ::ml_drift::MyOpAttributes attr;
  // Parse attributes from node->custom_initial_data if needed.
  return BuildMyOpGpuGraph(node->inputs, node->outputs, attr, model_builder);
}

absl::Status CreateMyOpFromIrOp(const ::ml_drift::ir::IrOp* ir_op,
                                ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* custom_attrs =
      ir_op->attributes.As<::ml_drift::ir::MyOpAttributes>();
  if (!custom_attrs) {
    return absl::InvalidArgumentError("Missing MyOpAttributes in IrOp");
  }
  ::ml_drift::MyOpAttributes attr = custom_attrs->attr;
  return BuildMyOpGpuGraph(ir_op->inputs, ir_op->outputs, attr, model_builder);
}

}  // namespace ml_drift
}  // namespace litert
```

#### C. Delegate Op Selectors (`litert_op_selector.cc` & `ir/litert_op_selector.cc`)

With the unified pattern, op selectors are clean and concise:

```cpp
// In composite/litert_op_selector.cc:
if (custom_op_name == "MyCustomOp") {
  RETURN_IF_ERROR(CreateMyOpFromNode(node, model_builder));
  return absl::OkStatus();
}

// In composite/ir/litert_op_selector.cc:
if (ir_op->op_type == ::ml_drift::ir::OpType::kMyCustomOp) {
  RETURN_IF_ERROR(CreateMyOpFromIrOp(ir_op, model_builder));
  return absl::OkStatus();
}
```

#### D. BUILD Dependency Requirements

In `composite/BUILD`, ensure `:my_op_kernel` includes:
```starlark
cc_library(
    name = "my_op_kernel",
    srcs = ["my_op_kernel.cc"],
    hdrs = ["my_op_kernel.h"],
    deps = [
        "//third_party/absl/status",
        "//third_party/absl/status:status_macros",
        "//third_party/ml_drift/common:gpu_model_builder",
        "//third_party/ml_drift/common:gpu_operation",
        "//third_party/ml_drift/common/operations:my_op",
        "//third_party/ml_drift/ir",
        "//third_party/tensorflow/lite/core/c:common",
    ],
)
```

---

## 2. Testing GPU Custom Ops

Thorough validation requires three levels of testing:
1. **IR Parser Tests**: Verifies schema parsing and type/shape inference.
2. **In-Memory Subgraph Construction**: Builds minimal TFLite FlatBuffers on the fly without checking in binary model files.
3. **GPU Delegate Numerical Tests**: Verifies execution correctness against a CPU baseline.

### Level 1: IR Parser Test (`composite/ir/<op_name>_parser_test.cc`)

Test that custom options are parsed into the expected `IrOp` attributes:

```cpp
TEST(MyOpParserTest, ParsesAttributesCorrectly) {
  // Construct a minimal buffer with custom options.
  std::vector<uint8_t> custom_options = BuildMyOpCustomOptions(/*dim=*/64);
  auto ir_op = ParseMyOpCustomOptions(custom_options);
  ASSERT_OK(ir_op);
  EXPECT_EQ(ir_op->attributes.As<MyOpAttributes>()->dim, 64);
}
```

### Level 2: In-Memory TFLite Subgraph Builder (`<op_name>_test_util.h`)

> [!IMPORTANT]
> **Never check in raw `.tflite` binaries for unit tests.**
> Instead, programmatically generate a self-contained FlatBuffer model using `flatbuffers::FlatBufferBuilder` and `tflite::schema`. This ensures test models are maintainable, diffable, and easily parameterized.

Reference implementation pattern (as seen in `litert_compiled_model_gpu_test.cc`):

```cpp
#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_MY_OP_TEST_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_MY_OP_TEST_UTIL_H_

#include <cstdint>
#include <vector>

#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"
#include "third_party/tensorflow/lite/schema/schema_generated.h"

namespace litert {
namespace testing {

inline std::vector<uint8_t> CreateMyOpModelBuffer(
    const std::vector<int32_t>& input_shape,
    const std::vector<int32_t>& output_shape,
    tflite::TensorType tensor_type = tflite::TensorType_FLOAT32) {
  flatbuffers::FlatBufferBuilder fbb(1024);

  // 1. Buffers (Buffer 0 is empty dummy buffer).
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(fbb, fbb.CreateVector(std::vector<uint8_t>())),
  };

  // 2. Tensors.
  auto input_shape_vec = fbb.CreateVector(input_shape);
  auto output_shape_vec = fbb.CreateVector(output_shape);
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      tflite::CreateTensor(fbb, input_shape_vec, tensor_type, /*buffer=*/0,
                           fbb.CreateString("input")),
      tflite::CreateTensor(fbb, output_shape_vec, tensor_type, /*buffer=*/0,
                           fbb.CreateString("output")),
  };

  // 3. Custom Operator Code.
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes = {
      tflite::CreateOperatorCodeDirect(
          fbb, tflite::BuiltinOperator_CUSTOM, "MyCustomOp"),
  };

  // 4. Operator Instance.
  std::vector<int32_t> inputs = {0};
  std::vector<int32_t> outputs = {1};
  std::vector<flatbuffers::Offset<tflite::Operator>> operators = {
      tflite::CreateOperatorDirect(fbb, /*opcode_index=*/0, &inputs, &outputs),
  };

  // 5. SubGraph & Model.
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
      tflite::CreateSubGraphDirect(fbb, &tensors, &inputs, &outputs, &operators,
                                   "main"),
  };

  auto model = tflite::CreateModelDirect(
      fbb, TFLITE_SCHEMA_VERSION, &opcodes, &subgraphs,
      fbb.CreateString("my_op_model"), &buffers);
  fbb.Finish(model);

  const uint8_t* buf = fbb.GetBufferPointer();
  return std::vector<uint8_t>(buf, buf + fbb.GetSize());
}

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_MY_OP_TEST_UTIL_H_
```

### Level 3: Numerical Correctness Test against CPU Baseline

Execute the model with LiteRT and assert that the GPU delegate matches CPU output within numerical tolerance:

```cpp
TEST(MyOpGpuTest, MatchesCpuReference) {
  auto model_buffer = CreateMyOpModelBuffer({1, 128}, {1, 128});

  // Run on CPU.
  std::vector<float> cpu_output = RunCpuInterpreter(model_buffer, test_input);

  // Run on GPU delegate.
  std::vector<float> gpu_output = RunGpuDelegate(model_buffer, test_input);

  ASSERT_EQ(cpu_output.size(), gpu_output.size());
  for (size_t i = 0; i < cpu_output.size(); ++i) {
    EXPECT_NEAR(gpu_output[i], cpu_output[i], 1e-3f);
  }
}
```

#### Test Tags for BUILD

Mark GPU-dependent tests with appropriate tags:
```starlark
cc_test(
    name = "my_op_gpu_test",
    srcs = ["my_op_gpu_test.cc"],
    tags = [
        "requires-gpu-nvidia",  # For OpenCL on Linux CI/Forge
    ],
    deps = [
        ":my_op_test_util",
        "//testing/base/public:gunit_main",
    ],
)
```

---

## 3. Cross-Platform Benchmarking (Metal, OpenCL, WebGPU)

To measure kernel performance without framework overhead, implement a standalone micro-benchmark binary using the **LiteRT C++ CompiledModel API**.

### Standalone Benchmark Binary Pattern (`<op_name>_gpu_benchmark.cc`)

```cpp
#include <iostream>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "ml_drift/delegate/my_op_test_util.h"

int main(int argc, char** argv) {
  auto env = litert::Environment::Create();
  auto model_data = litert::testing::CreateMyOpModelBuffer({1, 512}, {1, 512});

  litert::Options options;
  litert::GpuOptions gpu_options;
  options.Set(gpu_options);

  auto compiled_model = litert::CompiledModel::Create(
      *env, litert::BufferRef<uint8_t>(model_data.data(), model_data.size()),
      options);
  if (!compiled_model) {
    std::cerr << "Failed to compile model on GPU: "
              << compiled_model.Error().Message() << std::endl;
    return 1;
  }

  // Allocate host buffers.
  auto input_buf = litert::TensorBuffer::CreateHostBuffer(*env, ...);
  auto output_buf = litert::TensorBuffer::CreateHostBuffer(*env, ...);

  // Warmup (critical for shader compilation & pipeline caching).
  for (int i = 0; i < 10; ++i) {
    compiled_model->Run("main", {*input_buf}, {*output_buf});
  }

  // Benchmark loop.
  const int kIterations = 100;
  auto start = absl::Now();
  for (int i = 0; i < kIterations; ++i) {
    compiled_model->Run("main", {*input_buf}, {*output_buf});
  }
  auto elapsed = absl::Now() - start;

  std::cout << "Avg Latency: "
            << absl::ToDoubleMilliseconds(elapsed) / kIterations << " ms\n";
  return 0;
}
```

### Platform 1: Apple Metal (macOS / iOS / Darwin)

#### BUILD Rule Configuration
Use platform selection to switch between Metal and OpenCL accelerators:

```starlark
cc_test(
    name = "my_op_gpu_benchmark",
    srcs = ["my_op_gpu_benchmark.cc"],
    deps = [
        ":my_op_test_util",
        "//third_party/absl/time",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc/options:litert_gpu_options",
    ] + select({
        "//third_party/bazel_platforms/os:macos": [
            "//litert/runtime/accelerators/gpu:ml_drift_metal_accelerator",
        ],
        "//conditions:default": [
            "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",
        ],
    }),
)
```

#### Running on Apple Silicon (M1/M2/M3/M4)
Run with Darwin ARM64 configuration and framework linkopts:

```bash
blaze run -c opt --config=darwin_arm64 \
  --features=-module_maps \
  --linkopt="-framework" --linkopt="IOKit" \
  --linkopt="-framework" --linkopt="IOSurface" \
  //ml_drift/delegate:my_op_gpu_benchmark
```

#### Verification Checklist (Metal)
- Look for log confirmation:
  ```text
  Registering MLDrift Metal Accelerator
  LITERT_METAL: Created Metal delegate successfully.
  ```
- Verify 100% delegation (0 nodes running on CPU fallback).

---

### Platform 2: OpenCL (Android / Linux GPU / Nvidia Forge)

#### Running on Android Device (Adreno / Mali)
Build with `--config=android_arm64` and execute via ADB or an Android test runner:

```bash
blaze run --config=android_arm64 -c opt \
  //ml_drift/delegate:my_op_gpu_benchmark
```

#### Running on Linux Workstation / Nvidia Forge
On Linux machines with physical GPU access:

```bash
blaze test -c opt --test_tag_filters=requires-gpu-nvidia --test_output=streamed \
  //ml_drift/delegate:my_op_gpu_benchmark
```

#### Verification Checklist (OpenCL)
- Verify OpenCL environment initialization:
  ```text
  MLDrift OpenCL Delegate initialized on device: [Adreno / Mali / NVIDIA]
  OpenCL version: OpenCL 2.0 / 3.0
  FP16 support enabled.
  ```

---

### Platform 3: WebGPU (Dawn / WGSL / Browser)

#### Architecture
For WebGPU, MLDrift generates WGSL compute shaders executed via Google Dawn (C++) or the browser (Wasm / WebGPU API).

#### Validating WGSL Shader Generation
Verify that the operation generates valid WGSL source code conforming to WebGPU constraints:
- **Workgroup Size**: Keep within WebGPU's minimum guaranteed limits (`workgroup_size_x * workgroup_size_y * workgroup_size_z <= 256`).
- **Workgroup Memory**: Shared workgroup memory must not exceed 16 KB (WebGPU baseline limit; some devices support 32 KB).
- **Uniform Buffer Alignment**: Ensure uniform structs adhere to standard 16-byte alignment rules.

#### Benchmarking with Dawn C++ Harness
Link against `//third_party/gpu/dawn:dawn_native` to instantiate a native `wgpu::Device` and benchmark shader execution without launching a browser.

---

## 4. End-to-End Model Integration & Benchmarking

Once micro-benchmarks pass, integrate the custom op into full model inference:

1. **Register in Runtime Engine**:
   Add custom op registration in `//third_party/odml/litert_lm/runtime/engine/executor.cc`.
2. **Verify 100% Delegation**:
   Ensure downstream model passes (such as RoPE, KV cache updates, RMSNorm) don't create graph partitions that force CPU fallbacks.
3. **Measure Tokens/Second**:
   Run `litert_lm_advanced_main`:
   ```bash
   blaze run -c opt --config=darwin_arm64 \
     --features=-module_maps \
     --linkopt="-framework" --linkopt="IOKit" \
     --linkopt="-framework" --linkopt="IOSurface" \
     //third_party/odml/litert_lm/runtime/engine:litert_lm_advanced_main -- \
     --model_path=/tmp/model.litertlm \
     --backend=gpu \
     --prompt="Tell me a joke" \
     --num_tokens=128
   ```

---

## 5. Troubleshooting & Gotchas

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `GetTensorIds` compiler error | Trying to pass `std::vector<TensorPtr>` directly to a function expecting `std::vector<Value*>` | Use templated `GetTensorIds<T>` with `for (const auto& t : tensors) ids.push_back(t->id);`. |
| Graph partitioning / fallback to CPU | Custom op inputs or outputs have mismatched quantization or layout expectations | Ensure tensors use standard row-major float16/float32 layout or insert layout conversion nodes. |
| Metal `IOSurface` linker errors on macOS | Missing macOS framework linkopts in blaze invocation | Pass `--linkopt="-framework" --linkopt="IOKit" --linkopt="-framework" --linkopt="IOSurface"`. |
| Flaky / high initial latency | GPU pipeline compilation & shader compilation on first run | Always run 5–10 warmup iterations before starting performance timers. |
| Presubmit BUILD format error | BUILD file reordering or formatting violations | Run `hg fix --format=build <path/to/BUILD>` to auto-format. |
