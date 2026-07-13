# Unified Backend Operator Numerical Test Suite

This directory contains the unified, type-parameterized numerical test suite
for LiteRT backend operators. It enables developers to execute a standardized
set of functional and numerical operator tests across multiple backends (e.g.,
ML Drift, WebGPU, TFLite) without duplicating test logic.

---

## Architecture

The testing framework is split into two core layers:

1.  **`numerical_test_bridge.h`**: Defines the abstract `TestBackendBridge`
    interface. This interface decouples GTest assertion logic from
    backend-specific initialization, compilation, execution, and memory
    read/write details.
2.  **`numerical_test_suite.h`**: Implements Google Test type-parameterized
    tests (`NumericalTestSuite`) using the bridge interface. It covers 50+
    operations including Unary, Binary, Activations, Reductions, Shape
    operations, Neural Network layers, and Quantized weights.

---

## How to Hook Up a New Backend

Integrating a new backend to run the unified operator tests requires four
steps:

### Step 1: Implement `TestBackendBridge`

Create a bridge class that overrides `TestBackendBridge`. This class manages
the backend's runtime state, models/graphs, and interpreter contexts.

```cpp
#include "litert/tensor/backends/testing/numerical_test_bridge.h"

class MyBackendTestBridge : public litert::tensor::TestBackendBridge {
 public:
  ~MyBackendTestBridge() override = default;

  // Initialize backend environment. Return non-OK status if required
  // hardware/drivers (e.g., Vulkan, GPU) are not available.
  absl::Status Initialize() override {
    // ... initialize device/runtime ...
    return absl::OkStatus();
  }

  // Compile inputs and outputs into an executable graph/model.
  absl::Status BuildGraph(absl::Span<const TensorHandle> inputs,
                          absl::Span<const TensorHandle> outputs) override {
    // ... lower graph to backend representation and compile ...
    return absl::OkStatus();
  }

  // Write input tensor data to the backend's memory buffer.
  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data) override {
    // ... lookup input buffer and copy data ...
    return absl::OkStatus();
  }

  // Execute the graph.
  absl::Status Execute() override {
    // ... run compiled model ...
    return absl::OkStatus();
  }

  // Read output tensor data back from the backend's memory buffer.
  absl::Status GetOutput(const TensorHandle& tensor,
                         absl::Span<std::byte> data) override {
    // ... lookup output buffer and copy data out ...
    return absl::OkStatus();
  }
};
```

> [!IMPORTANT]
> Ensure any flatbuffers or memory arrays used by the backend wrapper are
> stored as class members of the bridge to guarantee their lifetimes span
> across `BuildGraph`, `SetInput`, `Execute`, and `GetOutput` calls.

### Step 2: Define Backend Traits

Define a traits struct exposing the backend's Mixin tag type and a static
bridge instantiation method:

```cpp
struct MyBackendTraits {
  using Tag = MyBackendMixinTag;
  static std::unique_ptr<TestBackendBridge> CreateBridge() {
    return std::make_unique<MyBackendTestBridge>();
  }
};
```

### Step 3: Instantiate the Suite

Instantiate the type-parameterized test suite in your test source file:

```cpp
#include "litert/tensor/backends/testing/numerical_test_suite.h"

namespace litert::tensor {
namespace {

INSTANTIATE_TYPED_TEST_SUITE_P(MyBackend, NumericalTestSuite, MyBackendTraits);

}  // namespace
}  // namespace litert::tensor
```

### Step 4: Update the BUILD target

Add a dependency on the unified test suite target to your backend's numerical
test target in the `BUILD` file:

```python
cc_test(
    name = "my_backend_numerical_test",
    srcs = ["my_backend_numerical_test.cc"],
    deps = [
        "//litert/tensor/backends/testing:numerical_test_suite",
        # ... other dependencies ...
    ],
)
```

---

## Handling Environment Constraints

If a backend environment cannot be initialized on a specific test runner (for
instance, a WebGPU test executing on a standard Forge builder without Vulkan
driver libraries), the bridge should return `absl::StatusCode::kUnavailable`
(or any non-OK status) from `Initialize()`.

The `SetUp()` method of `NumericalTestSuite` checks the initialization
status. If initialization fails, it calls `GTEST_SKIP()`:

```cpp
void SetUp() override {
  bridge_ = BackendTraits::CreateBridge();
  auto status = bridge_->Initialize();
  if (!status.ok()) {
    GTEST_SKIP() << "Skipping test suite because backend initialization failed: "
                 << status.ToString();
  }
}
```

This ensures tests compile but skip execution gracefully when runner
constraints are not met, avoiding build breaks.

---

## Handling Unsupported Operators

If a backend does not implement or support a specific operator, its
`BuildGraph` implementation should return `absl::StatusCode::kUnimplemented`.

Each operator test in `NumericalTestSuite` checks the result of `BuildGraph`
and skips gracefully using `GTEST_SKIP()` if it is unimplemented:

```cpp
auto status = this->bridge_->BuildGraph({input}, {output});
if (status.code() == absl::StatusCode::kUnimplemented) {
  GTEST_SKIP() << "Op is unimplemented on this backend.";
}
ASSERT_OK(status);
```

