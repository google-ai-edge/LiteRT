# Primitive Op Generator Patterns (`CoreSingleOp` / `SingleOp`)

Use these patterns when authoring a generator for a standard TFLite primitive
operator in `litert/test/generators/<op>.h`.

---

## 0. Authoring a Missing Reference Kernel (`litert/core/model/ops/<op>.h`)

If the primitive op you are onboarding does **not** yet have a
`litert::internal::Reference<Op>` implementation in
`litert/core/model/ops/` (for example, Tier 2 ops like
`Gather`, `Pack`, `Unpack`, `Split`, `TopKV2`, `BroadcastTo`, `Range`, or a
brand-new op), **always add `Reference<Op>` to `litert/core/model/ops/<op>.h`
first** rather than inlining the math inside `test/generators/<op>.h`.

### Why Reference Kernels Belong in `litert/core/model/ops/<op>.h`

1. **Composite Op Reusability (`ReferenceEvaluator`)**:
   [`ReferenceEvaluator`](litert/test/generators/reference_evaluator.cc)
   evaluates decomposed StableHLO composite ops by calling
   `litert::internal::Reference<Op>` from `core/model/ops/*.h`. Placing the
   reference kernel in `core/model/ops/<op>.h` allows any future composite op to
   use this primitive op in its decomposition.
2. **Standalone Unit Testing**: You can unit-test `Infer<Op>` and
   `Reference<Op>` directly in
   `litert/core/model/ops/<op>_test.cc`
   (`//litert/core/model:ops_<op>_test`) before building
   flatbuffer models.

### LiteRT's Reference Kernel Methodology (Do NOT Wrap `tflite::reference_ops`)

LiteRT authors its own standalone, header-only reference kernels in
`litert/core/model/ops/` rather than wrapping
`tflite::reference_ops` (only `reductions.h` is a legacy exception). This avoids
coupling `litert/core/model` to
`//third_party/tensorflow/lite/kernels/internal:reference_base` or
`tflite::RuntimeShape`, and implements a **Float-First** architecture:

#### Pattern 1 (Default for All Compute / Math Ops): Float-First Kernel (`inline void Reference<Op>(const float* ..., float* ...)`)

All arithmetic, activation, reduction, normalization, convolution, pooling, and
linear algebra ops in LiteRT implement a **single `float`-only (FP32) reference
kernel** taking `const float*` inputs and `float*` outputs.

- **How Non-Float Types (`tflite::half`, quantized, etc.) Work**: Instead of
  templating the reference kernel or writing separate type-specific loops, the
  generator's `Reference()` method unpacks inputs/weights/biases to
  `std::vector<float>` via `UnpackToFloat(...)`, runs
  `litert::internal::Reference<Op>` in full `float` precision, and packs the
  result back into the output buffer via
  `PackFromFloat(absl::MakeConstSpan(out_f32), out.data)` (see **Pattern C** in
  Section 2 below).
- **Exemplars**:
  - `ReferenceBatchMatmul` & `ReferenceFullyConnected` in
    [matmul.h](litert/core/model/ops/matmul.h)
  - `ReferenceConv2D` & `ReferenceDepthwiseConv2D` in
    [convolution.h](litert/core/model/ops/convolution.h)
  - `ReferencePool2D` in
    [pooling.h](litert/core/model/ops/pooling.h)
  - `ReferenceSoftmax`, `ReferenceLogSoftmax`, `ReferenceTanh`, `ReferenceRsqrt`
    in
    [simple_unary.h](litert/core/model/ops/simple_unary.h)
  - `ReferenceAdd`, `ReferenceSub`, `ReferenceMul`, `ReferenceDiv` in
    [simple_binary.h](litert/core/model/ops/simple_binary.h)
- **Reusable Helpers**:
  - **N-D Broadcasting**: Include
    `"litert/core/model/ops/simple_binary.h"` and call
    `ComputeBroadcastStrides(input_dims, input_rank, output_rank, strides)`
    (sets stride to `0` along broadcast size-1 dimensions; see `simple_binary.h`
    and `matmul.h`).
  - **Fused Activations**: Call
    `ApplyActivation(output_data, output_dims, rank, faf)` from
    `simple_binary.h` (see `convolution.h` and `matmul.h`).

#### Pattern 2 (Pure Data-Movement / Structural / Comparison Ops Only): Type-Generic Kernel (`template <typename T> inline void Reference<Op>(...)`)

Use `template <typename T>` **only** for pure data-movement, indexing, slicing,
permutation, concatenation, selection, and comparison ops where elements are
copied, rearranged, or compared without any floating-point arithmetic rounding
across `float`, `tflite::half`, `int32_t`, `int64_t`, `int8_t`, `uint8_t`, and
`bool`.

- **Distinct Input & Output Types
  (`template <typename InT, typename OutT = bool>`)**: When an op's output
  element type differs from its input element type (such as `Comparison` ops
  taking `InT` inputs and producing `bool` outputs), template the reference
  kernel on both `InT` and `OutT = bool` (`const InT* lhs, ..., OutT* output`).
  This allows the generator's `Reference()` method to pass `bool*`
  (`out.data.data()`) directly while
  [`ReferenceEvaluator`](litert/test/generators/reference_evaluator.cc)
  (which stores boolean and integer tensors in `TensorData::i32_data`) can pass
  `int32_t*` (`out.i32_data.data()`).
- **Exemplars**:
  - `ReferenceTile<T>` in
    [tile.h](litert/core/model/ops/tile.h)
  - `ReferenceSlice<T>` in
    [slice.h](litert/core/model/ops/slice.h)
  - `ReferenceSelect<T>` in
    [select.h](litert/core/model/ops/select.h)
  - `ReferenceTranspose<T>` in
    [transpose.h](litert/core/model/ops/transpose.h)
  - `ReferenceConcatenation<T>` in
    [concatenation.h](litert/core/model/ops/concatenation.h)
- **Reusable Helpers**:
  - **N-D Broadcasting**: Include
    `"litert/core/model/ops/simple_binary.h"` and call
    `ComputeBroadcastStrides(input_dims, input_rank, output_rank, strides)`
    (sets stride to `0` along broadcast size-1 dimensions; see `select.h`).
  - **Fused Activations**: Call
    `ApplyActivation(output_data, output_dims, rank, faf)` from
    `simple_binary.h` (see `concatenation.h` and `matmul.h`).
  - **Sub-byte Types (`int4_t`, `uint4_t`, `int2_t`)**: If the op supports
    sub-byte packed types, use `HasSubByte4<T>` / `HasSubByte2<T>` traits as
    shown in `transpose.h`.

```cpp
// In litert/core/model/ops/gather.h:
namespace litert::internal {

template <typename T, typename IndexT = int32_t>
inline void ReferenceGather(const T* params_data,
                            absl::Span<const int32_t> params_dims,
                            const IndexT* indices_data, size_t num_indices,
                            int32_t axis, int32_t batch_dims, T* output_data) {
  if (axis < 0) axis += params_dims.size();
  int64_t outer_size = 1;
  for (int i = 0; i < batch_dims; ++i) outer_size *= params_dims[i];
  int64_t pre_axis_size = 1;
  for (int i = batch_dims; i < axis; ++i) pre_axis_size *= params_dims[i];
  int64_t axis_dim = params_dims[axis];
  int64_t inner_size = 1;
  for (size_t i = axis + 1; i < params_dims.size(); ++i) {
    inner_size *= params_dims[i];
  }
  int64_t indices_per_batch = num_indices / outer_size;

  int64_t out_idx = 0;
  for (int64_t b = 0; b < outer_size; ++b) {
    for (int64_t o = 0; o < pre_axis_size; ++o) {
      for (int64_t i = 0; i < indices_per_batch; ++i) {
        IndexT gather_idx = indices_data[b * indices_per_batch + i];
        int64_t in_offset =
            (((b * pre_axis_size + o) * axis_dim) + gather_idx) * inner_size;
        for (int64_t k = 0; k < inner_size; ++k) {
          output_data[out_idx++] = params_data[in_offset + k];
        }
      }
    }
  }
}

}  // namespace litert::internal
```

### Checklist When Adding or Extending `litert/core/model/ops/<op>.h`

1. Implement `Reference<Op>` (and `Infer<Op>` if creating a new header) in
   `litert/core/model/ops/<op>.h`.
2. Ensure `cc_library(name = "ops_<op>", hdrs = ["ops/<op>.h"], ...)` is defined
   in
   [core/model/BUILD](litert/core/model/BUILD)
   and add `"//litert/core/model:ops_<op>"` to your
   generator's `deps` in `test/generators/BUILD`.
3. Add a `TEST(<Op>OpTest, Reference<Op>...)` in
   `litert/core/model/ops/<op>_test.cc` and verify it
   passes:
   ```bash
   /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test //litert/core/model:ops_<op>_test
   ```
4. *(Recommended)* Register a handler for `kLiteRtOpCodeTfl<Op>` in
   [`ReferenceEvaluator::RegisterStandardOps()`](litert/test/generators/reference_evaluator.cc)
   so composite op generators can also evaluate this op.

---

## 1. Canonical Generator Skeleton (`test/generators/<op>.h`)

Modern generators inherit from `TestGraph`, build the flatbuffer model using
`litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>` +
`SaveTensorGraph({output})`, and evaluate reference outputs via
`litert::internal::Reference<Op>` in
`litert/core/model/ops/<op>.h`.

Exemplars in the codebase:

- [slice.h](litert/test/generators/slice.h)
  (data movement, static `begin`/`size` constants, `kExact` conformance)
- [select_v2.h](litert/test/generators/select_v2.h)
  (3 dynamic runtime inputs with broadcasting, `kExact` conformance)
- [unary.h](litert/test/generators/unary.h),
  [binary_broadcast.h](litert/test/generators/binary_broadcast.h),
  and
  [reduction.h](litert/test/generators/reduction.h)
  (multi-opcode family generators templated on `typename OpCode` and expanded
  via `OpCodeListC`)
- [mean.h](litert/test/generators/mean.h)
  (reduction, static `axis` tensor, `UnpackToFloat`/`PackFromFloat`,
  `kFloatAccumulationAware`)
- [softmax.h](litert/test/generators/softmax.h)
  (shape inference `InferSoftmax`, `UnpackToFloat`/`PackFromFloat`,
  `kFloatAccumulationAware`)
- [batch_matmul.h](litert/test/generators/batch_matmul.h)
  (2 inputs, boolean `AdjX`/`AdjY` traits, `InferBatchMatmul`)
- [fully_connected.h](litert/test/generators/fully_connected.h)
  (mixed dynamic/static weights & biases, affine quantization, fused
  activations)

### Template & Class Structure

```cpp
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MY_OP_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MY_OP_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "third_party/absl/strings/string_view.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/my_op.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename Rank, typename T,
          typename OpCode = OpCodeC<kLiteRtOpCodeTflMyOp>>
class MyOp : public TestGraph {
 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr LiteRtOpCode kOpCode = OpCode::value;

  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

 public:
  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    std::array<Layout::Dim, kRank> output_shape;
    // Add any op-specific parameters or static constant tensor values here.
  };

  // IMPORTANT: First TypeList only contains DYNAMIC runtime subgraph inputs!
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<MyOp>;

  static constexpr absl::string_view Name() { return "MyOp"; }

  template <typename Rng>
  static Expected<MyOp::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 8);
    for (size_t i = 0; i < kRank; ++i) {
      params.input_shape[i] = dim_dist(rng);
      params.output_shape[i] = params.input_shape[i];
    }
    return Create(std::move(params));
  }

  static Expected<MyOp::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<MyOp>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  // Data-movement example (templated on T + kExact); for float-first compute
  // ops using UnpackToFloat/PackFromFloat, see Pattern C & Pattern D below.
  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kExact;
    return spec;
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    RandomTensorDataBuilder builder = data_builder;
    if (!builder.IsFloatDummy()) {
      builder.SetFloatRange(-5.0f, 5.0f);
    }

    LITERT_ASSIGN_OR_RETURN(auto input,
                            SimpleBuffer::Create<T>(params_.input_shape));
    LITERT_RETURN_IF_ERROR((input.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(input));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [in] = ref_inputs;
    auto [out] = ref_outputs;

    // Data-movement example; for compute ops see Pattern C (UnpackToFloat).
    litert::internal::ReferenceMyOp<T>(
        in.data.data(), params_.input_shape.data(), kRank, out.data.data());
    return {};
  }

  MyOp(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());

    TensorTf input = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims);

    TensorTf output = litert::tensor::MyOp(input);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MY_OP_H_
```

---

## 2. Special Patterns & Idioms

### Pattern A: Resolving Output Shapes via `litert::internal::Infer<Op>`

There are two shape inference signatures in
`litert/core/model/ops/`:

1. **Output-shape vector signature** (used by `Softmax`, `BatchMatmul`,
   `Concatenation`, `Transpose`, `Tile`, `Gather`):
   ```cpp
   // Infer<Op>(const LiteRtOpT& op, absl::Span<const Dims> input_shapes,
   //           std::vector<Dims>& output_shapes)
   LiteRtOpT op;
   op.SetOpCode(kLiteRtOpCodeTflSoftmax);
   auto options = std::make_unique<tflite::SoftmaxOptionsT>();
   options->beta = 1.0f;
   TflOptions tfl_opts;
   tfl_opts.type = tflite::BuiltinOptions_SoftmaxOptions;
   tfl_opts.value = options.release();
   litert::internal::SetTflOptions(op, std::move(tfl_opts));

   std::vector<litert::internal::Dims> input_shapes = {
       {params.input_shape.begin(), params.input_shape.end()}};
   std::vector<litert::internal::Dims> output_shapes(1);
   LITERT_RETURN_IF_ERROR(litert::internal::InferSoftmax(
       op, absl::MakeSpan(input_shapes), output_shapes));
   ```

2. **`Infer<Op>(const ShapeInferenceContext& ctx, InferenceResult& result)`**
   (used by `Squeeze`, `ExpandDims`, `Reshape`, `Split`, `Pack`, `Unpack`):
   - Either use `DummyShapeInferenceContext ctx(dummy_op);` from
     [graph_helpers.h](litert/test/generators/graph_helpers.h),
     or compute the output shape directly in `Create(Rng&)`.

### Pattern B: Static Constant Tensor Inputs (Axes, Indices, Weights)

When an op takes a constant parameter tensor (like `axis` in
`Mean`/`ExpandDims`, `begin`/`size` in `Slice`, or `multiples` in `Tile`):

1. **Do NOT include the constant tensor in `TestLogicTraits` `InputTypes`**
   (`TestLogicTraits<TypeList<T>, TypeList<T>, Params>`).
2. Store its values in `Params`.
3. In `BuildGraph()`, create the constant tensor with data or pass the vector to
   `litert::tensor::<Op>` (many builders in `tensor/arithmetic.h` like
   `Mean(input, axis, keep_dims)`, `Slice(input, begin, size)`,
   `Tile(input, multiples)`, `ExpandDims(input, axis)` automatically create the
   constant tensor for you!).

### Pattern C: `UnpackToFloat` / `PackFromFloat` for Float-First Reference Kernels

Because all compute/math `litert::internal::Reference<Op>` kernels in LiteRT
operate on `float` (`const float*`, `float*`), generators support `float`,
`tflite::half` (FP16), and quantized/packed types uniformly by unpacking
inputs/weights/biases via `UnpackToFloat` and packing outputs back via
`PackFromFloat` (defined in `common.h`):

```cpp
int64_t batch = 1;
for (size_t i = 0; i < kRank - 1; ++i) {
  batch *= params_.input_shape[i];
}
int64_t depth = params_.input_shape[kRank - 1];

std::vector<float> in_f32 = UnpackToFloat(in.data);
std::vector<float> out_f32(out.data.size());

litert::internal::ReferenceSoftmax(
    in_f32.data(), out_f32.data(), batch, depth, /*beta=*/1.0f);

PackFromFloat(absl::MakeConstSpan(out_f32), out.data);
```

### Pattern D: Accumulation-Aware Conformance (`kFloatAccumulationAware`)

For ops that accumulate across `K` elements (reductions, softmax, matmul),
configure `GetConformanceSpec()` with the reduction count so FP16/FP32
tolerances scale accurately with accumulation depth:

```cpp
ConformanceSpec GetConformanceSpec() const override {
  ConformanceSpec spec;
  spec.comparator_kind = ConformanceComparatorKind::kFloatAccumulationAware;
  int64_t in_elements = 1;
  for (auto d : params_.input_shape) in_elements *= d;
  int64_t out_elements = 1;
  for (auto d : params_.output_shape) out_elements *= d;
  spec.accumulation_depth =
      out_elements > 0 ? (in_elements / out_elements) : 1;
  return spec;
}
```

### Pattern E: Multi-Op Family Generators (`OpCodeListC`) & Boundary Value Seeding

When multiple TFLite opcodes share the exact same rank, shape, and type
signatures (such as `Unary`, `BinaryBroadcast`, `BinaryNoBcast`, `Reduction`, or
comparison op families), author a **single family generator** templated on
`typename OpCode` (see `unary.h`, `binary_broadcast.h`, and `reduction.h`):

1. Dispatch on `if constexpr (kOpCode == kLiteRtOpCodeTfl...)` in `BuildGraph()`
   and `Reference()`.
2. Expand all opcodes in a single `RegisterCombinations` call in
   `register_<op>.cc` using `OpCodeListC<kLiteRtOpCodeTfl..., ...>`.
3. **Seed boundary / tie cases in `MakeInputs()`** when purely uniform random
   data would rarely exercise an important branch (for example, when authoring a
   comparison op generator, copy `rhs_span[i] = lhs_span[i]` at every 3rd
   element so `Equal`, `GreaterEqual`, and `LessEqual` test exact equality as
   well as strict inequality).

### Pattern F: Avoiding Legacy `SingleOpModel` and Generator Test Anti-Patterns

When authoring generators with `litert::tensor` + `SaveTensorGraph`:

- **Never create standalone `<op>_test.cc` files or `cc_test` targets in
  `test/generators/`**. Existing `*_test.cc` files in `test/generators/`
  (`sdpa_test.cc`, `unary_test.cc`, `binary_no_bcast_test.cc`) are legacy.
  Reference math is unit-tested in `core/model/ops/<op>_test.cc` (or
  `reference_evaluator_test.cc`), and the generator itself is verified
  end-to-end by `:register_ops_contract_test`, `:builtin_ats`, `:ats`, and
  `:ynnpack_ats`.
- **Never add `FbOpTraits` / `FbOpTraitsNoOptions` entries to `FbOpTypes` in
  `test/generators/common.h`**. `FbOpTypes` is only used by legacy
  `SingleOpModel` (`OpDetails<OpCode>` in `graph_helpers.h`).
- **Never write a custom `GetTensorType<T>()` helper**. Always pass
  `litert::tensor::ApiType<T>::value` directly to `litert::tensor::Create(...)`
  (it already supports `float`, `int32_t`, `bool`, and `tflite::half` via
  `tensor/backends/tflite/arithmetic_tflite.h`).
- **Never include `litert_c_types_printing.h`** in generator headers.

---

## 3. ATS Registration Template (`ats/register_<op>.h` & `ats/register_<op>.cc`)

### `litert/ats/register_my_op.h`

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

### `litert/ats/register_my_op.cc`

```cpp
#include "litert/ats/register_my_op.h"

#include <cstddef>
#include <cstdint>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/my_op.h"
#include "third_party/tensorflow/lite/types/half.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterMyOpImpl(const AtsConf& options, size_t& test_id, size_t iters,
                      typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture,
      MyOp,
      SizeListC<1, 2, 3, 4>,
      TypeList<float, tflite::half, int32_t>>
    (iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterMyOp(const AtsConf& options, size_t& test_id, size_t iters,
                  AtsInferenceTest::Capture& cap) {
  RegisterMyOpImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterMyOp(const AtsConf& options, size_t& test_id, size_t iters,
                  AtsCompileTest::Capture& cap) {
  RegisterMyOpImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
```

Then include `register_my_op.h` and call
`RegisterMyOp(options, test_id, /*iters=*/10, cap);` inside
`RegisterSingleOpsImpl` in
[register_single_ops.cc](litert/ats/register_single_ops.cc)
(and add `":register_my_op"` to `deps` of `:register_single_ops` in
[ats/BUILD](litert/ats/BUILD)).

> **Automatic `"CoreSingleOp"` vs. `"SingleOp"` Classification**:
> All primitive ops are registered in `register_single_ops.cc`. During
> registration, `TestNames::Create` in
> [ats/common.h](litert/ats/common.h) inspects
> the op's `LiteRtOpCode` using `IsCoreSingleOp(LiteRtOpCode)` and automatically
> assigns `"CoreSingleOp"` to curated Core Single Ops and `"SingleOp"` to all
> other primitive ops.
