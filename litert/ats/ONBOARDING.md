# Onboarding New Ops into LiteRT Accelerator Test Suite (ATS)

The Accelerator Test Suite (ATS) is LiteRT's conformance and micro-benchmarking
framework. It allows kernel developers to verify op correctness against CPU
golden references and measure acceleration performance across backends (CPU,
Metal, WebGPU, and NPUs) without building full models.

This guide walks you through onboarding a new op—either a canonical TFLite op
or a custom composite op (such as `odml.sdpa_transposed` or `odml.swiglu`)—into
ATS using the 3-step workflow.

---

## The 3-Step Onboarding Workflow

Every ATS op consists of three components:

```
1. Generator Class
┌────────────────────────────────────────────────────────┐
│ test/generators/<op>.h                                 │
│ • struct Params                                        │
│ • Create(Params / Rng)                                 │
│ • CPU reference (or ReferenceEvaluator for composites) │
└───────────────────────────┬────────────────────────────┘
                            │
2. Registration Hook        ▼
┌────────────────────────────────────────────────────────┐
│ ats/register_<op>.h/.cc                                │
│ • RegisterCombinations<Fixture, Op, ...>               │
│ • Suite prefix ("CoreSingleOp" or "CompositeOp")       │
│ • Compile & Inference Fixture hooks                    │
└───────────────────────────┬────────────────────────────┘
                            │
3. Op Bundle Integration    ▼
┌────────────────────────────────────────────────────────┐
│ ats/register_{core,single,composite}_ops.cc & BUILD    │
│ • Add to op bundle (:register_core_ops, etc.)          │
│ • Propagates automatically to all ATS targets          │
└────────────────────────────────────────────────────────┘
```

---

## Step 1: Author the Generator Class (`test/generators/<op>.h`)

Create a header file under
`litert/test/generators/<op>.h`.
Your generator must inherit from `TestGraph`.

### Responsibilities of the Generator

1. **Define `struct Params`**: Strongly typed parameters controlling tensor
   shapes, precisions, attributes, and flags. Optionally provide named presets
   (e.g. `Params::Default()`).
2. **Choose a Parameter Strategy (`Create`)**: ATS invokes `Create(rng)`
   repeatedly across test iterations (`iters`). Choose the generation approach
   that best fits your op:
   - **Stratified Grid**: Deterministically cycle through an array of curated
     configurations (e.g. standard model dimensions, sequence lengths, edge
     cases). Recommended for complex kernels or composites where specific
     workloads must be verified predictably (see `sdpa_transposed.h`).
   - **Randomized Fuzzing**: Randomly sample dimensions, ranks, and attributes
     using `rng`. Recommended for broad conformance testing and fuzzing
     arbitrary shape combinations (see `softmax.h`).
   - **Custom / Hybrid**: Dynamically combine fixed presets with randomized
     attributes as appropriate for your op domain.
3. **Build the Graph (`BuildGraph`)**: Construct input tensors and operations
   using `litert::tensor` (`tensor/`), returning the
   serialized model via `litert::testing::SaveTensorGraph`.
4. **Allocate Runtime Inputs (`MakeInputs`)**: Populate input `SimpleBuffer`s
   matching the graph inputs using `data_builder` (e.g.
   `input.template WriteRandom<T>(data_builder, device)`) or explicit data.
5. **Provide Reference Implementation (`Reference`)**: For primitive ops,
   invoke the corresponding CPU reference kernel defined in `core/model/ops/`.
   For composite ops, use
   `ReferenceEvaluator::EvaluateCompositeReference(Graph(), inputs, outputs)`.
6. **Define Conformance Policy (`GetConformanceSpec`)**: Configure output
   comparison policy (`ConformanceComparatorKind`) and tolerances
   (`absolute_tolerance`, `relative_tolerance`, `accumulation_depth`).

### Graph Authoring and Reference Kernels

*   **Graph Construction (`litert::tensor`)**: ATS constructs computational
    graphs using LiteRT's C++ tensor authoring library (`litert/tensor/`).
    Using `Tensor<backends::Tflite>`, nodes are instantiated fluently with
    `litert::tensor::Create` and wired into ops (`litert::tensor::<Op>` or
    `StableHLOComposite`). Returning
    `litert::testing::SaveTensorGraph({output})` serializes the graph into an
    in-memory `LiteRtModelT`.
*   **Primitive Reference Kernels (`core/model/ops/`)**: LiteRT defines CPU
    reference kernels for primitive ops under `core/model/ops/` (e.g.
    `matmul.h`, `simple_binary.h`, `conv_2d.h`). When onboarding a new
    primitive op, inspect the corresponding header under `core/model/ops/` to
    locate its `Reference<Op>` kernel (or add it if missing), and call it
    directly from your generator's `Reference(...)` method.

### Example Generator Skeleton

```cpp
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MY_OP_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MY_OP_H_

#include <array>
#include <memory>
#include <type_traits>
#include <vector>
#include "third_party/absl/strings/string_view.h"
#include "third_party/flatbuffers/include/flatbuffers/flexbuffers.h"
#include "litert/cc/litert_expected.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/generators/reference_evaluator.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename T, typename WithFeature = std::false_type>
class MyOp : public TestGraph {
 private:
  static constexpr bool kWithFeature = WithFeature::value;

 public:
  static constexpr absl::string_view Name() { return "MyOp"; }

  struct Params {
    size_t batch = 1;
    size_t in_dim = 2048;
    size_t out_dim = 4096;
    float alpha = 1.0f;

    static Params Default() {
      return {.batch = 1, .in_dim = 2048, .out_dim = 4096, .alpha = 1.0f};
    }
  };

  using Ptr = std::unique_ptr<MyOp>;

  MyOp(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  // 1. Factory using explicit parameters
  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<MyOp>(std::move(params), std::move(model));
  }

  // 2. Harness factory: choose stratified grid, randomized fuzzing, or custom

  // Approach A: Stratified Grid (deterministic cycling through configurations)
  // static constexpr Params kGrid[] = {
  //     {.batch = 1, .in_dim = 1024, .out_dim = 2048, .alpha = 1.0f},
  //     {.batch = 1, .in_dim = 2048, .out_dim = 4096, .alpha = 1.0f},
  //     {.batch = 2, .in_dim = 4096, .out_dim = 8192, .alpha = 0.5f},
  // };
  // template <typename Rng>
  // static Expected<Ptr> Create(Rng& /*rng*/) {
  //   static size_t sample_counter = 0;
  //   static constexpr size_t kGridSize = sizeof(kGrid) / sizeof(kGrid[0]);
  //   return Create(kGrid[(sample_counter++) % kGridSize]);
  // }

  // Approach B: Randomized Fuzzing (random shape & parameter generation)
  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params params;
    params.batch = rng.Uniform<size_t>(1, 4);
    params.in_dim = rng.Choice<size_t>({1024, 2048, 4096});
    params.out_dim = rng.Choice<size_t>({2048, 4096, 8192});
    return Create(params);
  }

  bool HasReference() const override { return true; }

  // 3. Output comparison policy and numerical tolerances
  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kFloatAccumulationAware;
    spec.accumulation_depth = params_.in_dim;
    if constexpr (std::is_same_v<T, float>) {
      spec.relative_tolerance = 1e-3;
      spec.absolute_tolerance = 1e-3;
    } else {
      spec.relative_tolerance = 5e-2;
      spec.absolute_tolerance = 5e-2;
    }
    return spec;
  }

  // 4. Generate input runtime buffers for execution
  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(1);

    std::array<Layout::Dim, 2> shape = {
        static_cast<Layout::Dim>(params_.batch),
        static_cast<Layout::Dim>(params_.in_dim)};

    LITERT_ASSIGN_OR_RETURN(auto input, SimpleBuffer::Create<T>(shape));
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T>(data_builder, device)));
    inputs.push_back(std::move(input));

    return inputs;
  }

  // 5. Golden reference execution
  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    // For composite ops: use ReferenceEvaluator to execute decomposition.
    // For primitive ops: call the reference kernel defined in
    // core/model/ops/ (e.g. litert::internal::Reference<Op>).
    return ReferenceEvaluator::EvaluateCompositeReference(
        Graph(), inputs, outputs);
  }

 private:
  // Graph authoring using litert::tensor (tensor/)
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::backends::Tflite>;

    // 1. Create Inputs with litert::tensor::Create
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value,
        {params.batch, params.in_dim});

    // 2. Build Flexbuffer Attributes (for StableHLO composite ops)
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Float("alpha", params.alpha);
    });
    fbb.Finish();
    auto composite_attributes = fbb.GetBuffer();

    // 3. Define Golden Reference Math
    auto decompose = [&params](auto in_tensor) {
      // CPU ground truth implemented via litert::tensor operators
      return litert::tensor::Mul(in_tensor, params.alpha);
    };

    // 4. Build Target Op / Composite via litert::tensor
    // (For primitive ops, e.g.: output = litert::tensor::Relu(input);)
    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.my_op",
        .composite_attributes = composite_attributes,
    };

    TensorTf output = litert::tensor::StableHLOComposite(
        composite_options, decompose, input);

    // 5. Finalize Graph and return LiteRtModelT via SaveTensorGraph
    output.SetName("output");
    return litert::testing::SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MY_OP_H_
```

### Build Target (`test/generators/BUILD`)

Expose your generator header as a `cc_library`:

```starlark
cc_library(
    name = "my_op",
    testonly = True,
    hdrs = ["my_op.h"],
    deps = [
        ":common",
        ":graph_helpers",
        ":reference_evaluator",
        "//third_party/absl/strings:string_view",
        "//third_party/flatbuffers:runtime_cc",
        "//litert/cc:litert_expected",
        "//litert/test:simple_buffer",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
    ],
)
```

> [!IMPORTANT]
> Avoid delegate-specific abstraction leaks: test generators should construct
> backend-agnostic op representations. Never require delegate-specific internal
> state flags (such as GPU buffer packing) in the generator. Delegates should
> infer layout requirements from graph topology or fallback cleanly.

### Numerical Conformance (`ConformanceSpec`)

ATS validates accelerator outputs against the reference implementation using
the `ConformanceSpec` returned by `GetConformanceSpec()`. Overriding this
method allows your generator to define the appropriate comparison strategy:

*   `ConformanceComparatorKind::kFloatAccumulationAware`:
    Recommended for ops with reductions or matrix products (`BatchMatmul`,
    `Conv2d`, `SdpaTransposed`, `Softmax`). ATS enforces a dynamic tolerance
    floor of `10 * eps * sqrt(K)` where `K` is `accumulation_depth`, taking
    `max(relative_tolerance, 10 * eps * sqrt(K))`. Set `accumulation_depth`
    to the inner reduction dimension (e.g. `in_dim` or `head_dim`).
*   `ConformanceComparatorKind::kFloatElementwise`:
    Standard pointwise floating-point comparison checking
    `|actual - expected| <= atol + rtol * |expected|`. Best for pointwise ops
    (`Add`, `Mul`, `Gelu`).
*   `ConformanceComparatorKind::kExact`:
    Exact bitwise/byte-level match. Used for ops that rearrange or slice
    tensors without arithmetic rounding (`Transpose`, `Reshape`, `Slice`).
*   `ConformanceComparatorKind::kQuantizedBucket`:
    Verifies that integer values differ by at most `bucket_tolerance`
    (typically 1). Ideal for quantized int8/int16 ops where off-by-one
    rounding occurs across platforms.
*   `ConformanceComparatorKind::kMse`:
    Mean Squared Error comparison against `absolute_tolerance`. Default
    fallback when no reference is available.

For floating-point ops supporting both FP32 and FP16, use `if constexpr` to
provide tighter tolerances for `float` (e.g. `1e-3`) and looser tolerances for
`tflite::half` (e.g. `5e-2`) to account for lower precision.

---

## Step 2: Author the Registration Hook (`ats/register_<op>.h` and `.cc`)

Registration bridges your `TestGraph` generator into the ATS test execution
harness for inference validation (`AtsInferenceTest`) and compilation
(`AtsCompileTest`).

### 1. Header: `litert/ats/register_<op>.h`

```cpp
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_MY_OP_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_MY_OP_H_

#include <cstddef>
#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"

namespace litert::testing {

void RegisterMyOp(const AtsConf& options, size_t& test_id, size_t iters,
                  AtsInferenceTest::Capture& cap);

void RegisterMyOp(const AtsConf& options, size_t& test_id, size_t iters,
                  AtsCompileTest::Capture& cap);

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_MY_OP_H_
```

### 2. Implementation: `litert/ats/register_<op>.cc`

Use `RegisterCombinations` to generate the compile-time Cartesian product of
types and template options. Pass the appropriate suite prefix as the last
argument:

- `"CoreSingleOp"`: For P0 core ops registered in `:register_core_ops`.
- `"CompositeOp"`: For composite ops registered in `:register_composite_ops`.
- `"SingleOp"`: For other single ops in `:register_single_ops` (or omit for
  default).

```cpp
#include "litert/ats/register_my_op.h"

#include <cstddef>
#include <type_traits>
#include "third_party/absl/strings/string_view.h"
#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/my_op.h"
#include "third_party/tensorflow/lite/types/half.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterMyOpImpl(const AtsConf& options, size_t& test_id,
                      size_t iters, typename Fixture::Capture& cap) {
  // Generates Cartesian product: (float, half) x (false, true).
  // (For a single type parameter, pass just TypeList<float, tflite::half>).
  RegisterCombinations<
      Fixture,
      MyOp,
      TypeList<float, tflite::half>,              // Dtypes
      TypeList<std::false_type, std::true_type>>  // WithFeature
    (iters, test_id, options, cap, "CoreSingleOp");
}

}  // namespace

void RegisterMyOp(const AtsConf& options, size_t& test_id,
                  size_t iters, AtsInferenceTest::Capture& cap) {
  RegisterMyOpImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterMyOp(const AtsConf& options, size_t& test_id,
                  size_t iters, AtsCompileTest::Capture& cap) {
  RegisterMyOpImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
```

---

## Step 3: Integrate into an Op Bundle (`ats/BUILD` & Bundle Libraries)

Instead of modifying `ats.cc` or individual platform test targets directly,
ATS organizes ops into modular op bundles. You only need to define your
target and add it to the corresponding bundle.

### 1. Expose the Build Target (`ats/BUILD`)

Add the `cc_library` target in `litert/ats/BUILD`:

```starlark
cc_library(
    name = "register_my_op",
    testonly = True,
    srcs = ["register_my_op.cc"],
    hdrs = ["register_my_op.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//third_party/absl/strings:string_view",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:my_op",
        "//third_party/tensorflow/lite/types:half",
    ],
)
```

### 2. Add to the Appropriate Bundle Library

Choose the bundle matching your op:

*   **Core Single Ops (`:register_core_ops`)**:
    *   Scope: Canonical P0 ops supported across all accelerators (Add,
        Conv2d, BatchMatmul, Softmax, etc.).
    *   Suite Prefix: Use `"CoreSingleOp"`.
    *   Action: Add `":register_my_op"` to the `deps` of `:register_core_ops` in
        `ats/BUILD`. In `ats/register_core_ops.cc`, `#include` your header and
        call `RegisterMyOp(options, test_id, /*iters=*/10, cap);`.

*   **Other Single Ops (`:register_single_ops`)**:
    *   Scope: Remaining or auxiliary single ops (Pooling, Reshape, OneHot,
        Pad, etc.). Note that `:register_single_ops` automatically includes
        `:register_core_ops`.
    *   Suite Prefix: Use `"SingleOp"` (or omit for default).
    *   Action: Add `":register_my_op"` to the `deps` of `:register_single_ops`
        in `ats/BUILD`. In `ats/register_single_ops.cc`, `#include` your header
        and call `RegisterMyOp(options, test_id, /*iters=*/10, cap);`.

*   **Composite Ops (`:register_composite_ops`)**:
    *   Scope: Multi-op fused subgraphs and custom composite ops
        (`SdpaTransposed`, SwiGLU, RMSNorm, etc.).
    *   Suite Prefix: Use `"CompositeOp"`.
    *   Action: Add `":register_my_op"` to the `deps` of
        `:register_composite_ops` in `ats/BUILD`. In
        `ats/register_composite_ops.cc`, `#include` your header and call
        `RegisterMyOp(options, test_id, /*iters=*/10, cap);`.

*(Note: Multi-layer model blocks like `TransformerLayer` remain in standalone
targets).*

Adding your op to a bundle automatically propagates it to all ATS platform
targets (`:cpu_ats`, `:metal_macos_ats`, `:webgpu_macos_ats`, etc.) without
modifying backend targets or `ats.cc`. Suite prefixes are checked by
`:register_ops_contract_test`.

---

## Running Your Op

Refer to the litert/ats/README.md for full instructions on running ATS targets
across backends.

To run only your op on host CPU:

```bash
blaze run //litert/ats:ats -- \
  --do_register=".*my_op.*"
```

To run on macOS GPU (Metal) or WebGPU:

```bash
blaze test //litert/ats:metal_macos_ats \
  --config=darwin_arm64 \
  --test_filter="*my_op*" \
  --test_output=streamed
```
