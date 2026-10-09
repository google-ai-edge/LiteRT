---
name: litert-ats-op-generator
description: Authors, registers, and tests new operator test generators for the LiteRT Accelerator Test Suite (ATS) in litert/test/generators/ and litert/ats/. Use when onboarding a new TFLite primitive op or StableHLO composite op into LiteRT ATS, expanding ATS op coverage, creating or debugging a TestGraph generator, wiring op registration into register_single_ops or register_composite_ops, or picking a candidate op to onboard to ATS.
---

# LiteRT ATS Op Generator Onboarding Skill

This skill guides you through the complete end-to-end workflow for onboarding a
new operator into the **LiteRT Accelerator Test Suite (ATS)**
(`//litert/ats/`).

ATS validates hardware accelerator backends (CPU/XNNPACK, GPU, Qualcomm QNN,
Metal, WebGPU, MediaTek, Google Tensor, Intel OpenVINO) by generating
self-contained single-op or composite-op `LiteRtModelT` graphs with randomized
shapes/parameters, executing them on the target accelerator, and comparing
outputs against a known-good CPU reference implementation.

---

## Reference Guides (Progressive Disclosure)

Read the relevant reference file before writing code:

- **[references/primitive_op_patterns.md](references/primitive_op_patterns.md)**:
  Copy-pasteable C++ templates and idioms for **Primitive TFLite Ops**
  (`SingleOp` / `CoreSingleOp`), covering authoring missing `Reference<Op>`
  kernels, simple ops, shape-inferred ops, static constant tensor inputs (axes,
  weights, biases), and affine quantization.
- **[references/composite_op_patterns.md](references/composite_op_patterns.md)**:
  Copy-pasteable C++ templates and idioms for **StableHLO Composite Ops**
  (`CompositeOp`), covering `kStratifiedGrid`, Flexbuffer attributes, single-
  and multi-output `StableHLOComposite` decompositions, and extending
  `ReferenceEvaluator`.

---

## End-to-End Onboarding Workflow (The 4 Touchpoints)

Every ATS op onboarding touches **ONLY 4 places** in
`litert/test/generators/` and
`litert/ats/` (aside from any missing upstream
`litert::tensor` or `core/model/ops/<op>.h` prerequisites in Step 0):

1. **Generator Header**: `test/generators/<op>.h` + header-only `cc_library` in
   `test/generators/BUILD` (**never** create a `test/generators/<op>_test.cc`
   file or `cc_test` target)
2. **Umbrella Export**: `#include` in `test/generators/generators.h` +
   dependency in `test/generators/BUILD` (`:generators`)
3. **ATS Registration**: `ats/register_<op>.h` + `ats/register_<op>.cc` +
   `cc_library` in `ats/BUILD`
4. **Bundle & Backend Wiring**: `ats/register_{single,composite}_ops.cc` +
   `ats/BUILD` (including `--dont_register` exclusions if a backend lacks
   support)

---

### Step 0: Classify the Op & Verify Prerequisites

Before creating files, determine whether the target op is a **Primitive TFLite
Op** or a **StableHLO Composite Op**, and check its upstream building blocks in
the live workspace. If the user instead asks *"what op should I onboard next?"*,
dynamically compare what is already registered in
[register_single_ops.cc](litert/ats/register_single_ops.cc),
[register_composite_ops.cc](litert/ats/register_composite_ops.cc),
and
[test/generators/BUILD](litert/test/generators/BUILD)
against the ops available in
[core/model/BUILD](litert/core/model/BUILD) and
[tensor/arithmetic.h](tensor/arithmetic.h):

#### For Primitive TFLite Ops:

1. **Graph Builder (`litert::tensor`)**: Check
   [tensor/arithmetic.h](tensor/arithmetic.h)
   and
   [arithmetic_tflite.h](tensor/backends/tflite/arithmetic_tflite.h)
   for `litert::tensor::<Op>(...)`.
   - Always prefer the declarative
     `litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>` +
     `SaveTensorGraph({output})` API over legacy `SingleOpModel`.
   - **If the op is missing from `litert::tensor`**, wire it through all **5
     places** in `tensor/` so `SaveTensorGraph` can
     both serialize and parse the TFLite flatbuffer:
     1. [arithmetic_graph.h](tensor/arithmetic_graph.h):
        Define `struct <Op>Operation`.
     2. [arithmetic.h](tensor/arithmetic.h):
        Add the `litert::tensor::<Op>(...)` builder (if an elementwise op's
        output dtype differs from its inputs—such as `Type::kBOOL` for
        comparison ops—override `GetInfo(output.GetRaw())->type` inside
        `ElementwiseOp`).
     3. [arithmetic_tflite.h](tensor/backends/tflite/arithmetic_tflite.h)
        &
        [arithmetic_tflite.cc](tensor/backends/tflite/arithmetic_tflite.cc):
        Declare and implement `Build<Op>`.
     4. [tflite_flatbuffer_conversion.cc](tensor/backends/tflite/tflite_flatbuffer_conversion.cc):
        Register `graph::<Op>Operation` in `OpConverter`.
     5. [tflite_flatbuffer_parser.cc](tensor/parsers/tflite/tflite_flatbuffer_parser.cc):
        Add the `tflite::BuiltinOperator_<OP>` case so `SaveTensorGraph` can
        parse the serialized flatbuffer back into `LiteRtModelT`.
2. **Reference Kernel & Shape Inference (`litert/core/model/ops/`)**: Check
   [core/model/BUILD](litert/core/model/BUILD)
   (`ops_*` targets) for `litert::internal::Reference<Op>` and
   `litert::internal::Infer<Op>`.
   - **LiteRT Reference Kernel Methodology**: LiteRT authors its own
     **standalone, self-contained C++ reference kernels** in
     `litert/core/model/ops/<op>.h`—do **not** wrap
     legacy `tflite::reference_ops`
     (`//third_party/tensorflow/lite/kernels/internal:reference_base`).
   - **Float-First Design**: Compute/math reference kernels operate purely on
     `float` buffers (`const float*`, `float*`), while generators convert
     arbitrary input/output types (`float`, `tflite::half`, etc.) via
     `UnpackToFloat` and `PackFromFloat`. Pure data-movement/indexing/comparison
     ops (`Slice`, `Select`, `Transpose`, `Tile`, `Concatenation`) use
     `template <typename T>` (or `template <typename InT, typename OutT>` when
     input and output element types differ).
   - **If `Reference<Op>` is missing**: Add `Reference<Op>` (and `Infer<Op>` if
     creating a new header) to
     `litert/core/model/ops/<op>.h`, add unit tests in
     `litert/core/model/ops/<op>_test.cc`, and register
     it in
     [`ReferenceEvaluator::RegisterStandardOps()`](litert/test/generators/reference_evaluator.cc).
     See **Section 0 of
     [references/primitive_op_patterns.md](references/primitive_op_patterns.md)**
     for full details.

#### For StableHLO Composite Ops:

1. **Decomposed Primitive Ops in `litert::tensor`**: Verify that every primitive
   op used inside your `decompose` lambda exists in
   [tensor/arithmetic.h](tensor/arithmetic.h).
2. **Decomposed Primitive Ops in `ReferenceEvaluator`**: Composite ops compute
   reference outputs by running the decomposed subgraph through
   [`ReferenceEvaluator::EvaluateCompositeReference`](litert/test/generators/reference_evaluator.h).
   - Inspect
     [`ReferenceEvaluator::RegisterStandardOps()`](litert/test/generators/reference_evaluator.cc)
     to confirm every `kLiteRtOpCodeTfl*` emitted by your `decompose` lambda is
     registered.
   - If any primitive op is missing from `reference_evaluator.cc`, register its
     handler there and add a test case in
     [reference_evaluator_test.cc](litert/test/generators/reference_evaluator_test.cc).

---

### Step 1: Author the Generator (`test/generators/<op>.h` & `test/generators/BUILD`)

Read [references/primitive_op_patterns.md](references/primitive_op_patterns.md)
(for primitive ops) or
[references/composite_op_patterns.md](references/composite_op_patterns.md) (for
composite ops) and create
`litert/test/generators/<op>.h`.

> [!WARNING]
> **Avoid Legacy `SingleOpModel` and Generator Test Anti-Patterns**:
> - **Do NOT create standalone `<op>_test.cc` files or `cc_test` targets in
>   `test/generators/`**: Existing `*_test.cc` files in `test/generators/`
>   (`sdpa_test.cc`, `unary_test.cc`, `binary_no_bcast_test.cc`) are legacy.
>   Standalone generator unit tests are redundant because reference math is
>   tested in `core/model/ops/<op>_test.cc` (or `reference_evaluator_test.cc`)
>   and the generator itself is verified end-to-end by
>   `:register_ops_contract_test`, `:builtin_ats`, `:ats`, and `:ynnpack_ats`.
> - **Do NOT edit `FbOpTypes` in `test/generators/common.h`** (`FbOpTraits` /
>   `FbOpTraitsNoOptions`) when building graphs with `litert::tensor` +
>   `SaveTensorGraph`. `FbOpTypes` is only used by legacy `SingleOpModel`.
> - **Do NOT write custom `GetTensorType<T>()` helpers** in your generator. Use
>   `litert::tensor::ApiType<T>::value` directly (including for `tflite::half`,
>   which is specialized in `tensor/backends/tflite/arithmetic_tflite.h`).
> - **Do NOT include `litert_c_types_printing.h`** in generator headers.

#### The 7 Non-Negotiable Generator Invariants

1. **All Template Parameters Must Be Types**:
   - [`RegisterCombinations`](litert/ats/register.h)
     and `ExpandProduct` only expand C++ **types**, never non-type template
     parameters (`size_t`, `bool`, enums).
   - Wrap compile-time constants using the wrappers in
     [test/generators/common.h](litert/test/generators/common.h):
     - Ranks / integers / axes: `SizeC<N>` (and `SizeListC<1, 2, 3, 4>`)
     - Opcodes: `OpCodeC<kLiteRtOpCodeTfl...>` (and `OpCodeListC<...>`)
     - Fused activations: `FaC<tflite::ActivationFunctionType_...>` (and
       `FaListC<...>`)
     - Booleans: `std::true_type` / `std::false_type`
       (`std::bool_constant<bool>`)
2. **`TestLogicTraits` InputTypes Must Only List Dynamic Runtime Inputs**:
   - `using Traits = TestLogicTraits<TypeList<InTypes...>, TypeList<OutTypes...>, Params>;`
   - Only include tensors passed at runtime in `MakeInputs()` in `InputTypes`.
     Static constant tensors baked into the graph via
     `litert::tensor::Create(..., data)` or `OwningCpuBuffer::CopyAs(...)`
     (such as reduction `axes`, `Slice` `begin`/`size` constants,
     `FullyConnected` static weights/biases, or static normalization scale
     weights) are **not** subgraph runtime inputs—omit them from `InputTypes`
     and `MakeInputs()`. (`ReferenceEvaluator::EvaluateCompositeReference`
     automatically binds constant composite inputs from the graph's buffers.)
3. **Provide Both `Create(Rng& rng)` and `Create(Params params)`**:
   - Implement `static Expected<Ptr> Create(Rng& rng)` to sample randomized
     shapes/parameters and delegate to
     `static Expected<Ptr> Create(Params params)`.
   - Exposing `Create(Params)` separates parameter sampling from graph
     construction and allows deterministic instantiation with explicit shapes.
4. **`Name()` Must Return a Static PascalCase Family Name**:
   - [`TestNames::Create`](litert/ats/common.h)
     uses `Logic::Name()` as the `<Family>` component of the GTest suite name
     (`<Prefix>_<Fixture>_<Family>`, e.g., `CoreSingleOp_inference_Slice` or
     `CompositeOp_inference_Sdpa`).
   - Return a static string literal for both primitive and composite ops (e.g.,
     `return "Slice";`, `return "SelectV2";`, `return "Sdpa";`)—never embed
     random runtime dimensions in `Name()`.
   - The individual GTest case name is automatically formatted from the
     generated `LiteRtModelT` graph via `NormalizeGraphName(FormatGraph(graph))`
     and verified by
     [`register_ops_contract_test.cc`](litert/ats/register_ops_contract_test.cc).
5. **Preserve Deterministic Dummy Builders in `MakeInputs()`**:
   - When constraining float ranges in `MakeInputs()` (to prevent `NaN`/`Inf`
     overflow in FP16 or transcendental ops), **always** guard with
     `if (!data_builder.IsFloatDummy())`:
     ```cpp
     RandomTensorDataBuilder builder = data_builder;
     if (!builder.IsFloatDummy()) {
       builder.SetFloatRange(-2.0f, 2.0f);
     }
     ```
   - Without `if (!builder.IsFloatDummy())`, callers that set
     `data_builder.SetFloatDummy()` for sequential `{0, 1, 2, ...}` inputs will
     have their range overwritten.
6. **Follow the Float-First `UnpackToFloat` / `PackFromFloat` Pattern in `Reference()`**:
   - When `Reference<Op>` in `core/model/ops/` operates on `float*` buffers (the
     standard LiteRT convention for math, reduction, activation, convolution,
     and matmul ops), unpack inputs (`float`, `tflite::half`, etc.) to
     `std::vector<float>` using `UnpackToFloat(in.data)`, run the `float`
     reference kernel, and pack back using
     `PackFromFloat(absl::MakeConstSpan(out_f32), out.data)`.
   - When `Reference<Op>` is a pure data-movement or comparison op templated
     directly on `T` (e.g., `ReferenceSlice<T>`, `ReferenceSelect<T>`,
     `ReferenceTranspose<T>`, `ReferenceTile<T>`, `ReferenceConcatenation<T>`),
     pass `in.data.data()` and `out.data.data()` directly.
7. **Choose the Right `GetConformanceSpec()` Comparator**:

   | Op Category | `ConformanceComparatorKind` | Required Spec Fields | Examples |
   | :--- | :--- | :--- | :--- |
   | Data movement / indexing / comparison | `kExact` | `spec.comparator_kind = ConformanceComparatorKind::kExact;` | `Slice`, `SelectV2`, `Transpose`, `ExpandDims`, `Squeeze` |
   | Elementwise float / simple composite | `kFloatElementwise` | Default (`ConformanceSpec{}`) | `Unary`, `SwiGLU`, `QkvNormRope` |
   | Reductions, norms, matmuls, attention | `kFloatAccumulationAware` | `spec.accumulation_depth = <reduction_elements>;` | `Mean`, `Softmax`, `BatchMatmul`, `FullyConnected`, `Sdpa` |
   | Quantized int8 / uint8 ops | `kQuantizedBucket` | `spec.bucket_tolerance = 1;` | Quantized `FullyConnected` |

Add the `cc_library` target for `<op>` in
[test/generators/BUILD](litert/test/generators/BUILD).

---

### Step 2: Export in `generators.h` & `test/generators/BUILD`

1. Add the `#include` (in alphabetical order) to
   [generators.h](litert/test/generators/generators.h):
   ```cpp
   #include "litert/test/generators/<op>.h"  // IWYU pragma: export
   ```
2. Add `":<op>"` to the `deps` of `cc_library(name = "generators", ...)` in
   [test/generators/BUILD](litert/test/generators/BUILD).

---

### Step 3: Register the Op in ATS (`litert/ats/`)

1. **Create `litert/ats/register_<op>.h` and `register_<op>.cc`**:
   - Declare and define both `AtsInferenceTest::Capture&` and
     `AtsCompileTest::Capture&` overloads of
     `Register<Op>(const AtsConf& options, size_t& test_id, size_t iters, ...)`.
   - Use `RegisterCombinations<Fixture, <Op>, ...>(iters, test_id, options, cap);`.
   - Add `cc_library(name = "register_<op>", ...)` in
     [ats/BUILD](litert/ats/BUILD).
2. **Wire into the Appropriate Registration Bundle**:
   - **Primitive TFLite Ops ->
     [register_single_ops.cc](litert/ats/register_single_ops.cc)**:
     - **Used for all primitive single ops.** Linked into **all** hardware
       backend ATS targets (`:ats`, `:cpu_ats`, `:gpu_ats`, `:qualcomm_ats`,
       `:metal_macos_ats`, `:webgpu_macos_ats`) and
       `:register_ops_contract_test`.
     - **Automatic `"CoreSingleOp"` vs. `"SingleOp"` Classification**:
       [`TestNames::Create`](litert/ats/common.h)
       automatically inspects the op's `LiteRtOpCode` against
       `IsCoreSingleOp(LiteRtOpCode)` in
       [ats/common.h](litert/ats/common.h). If
       `IsCoreSingleOp` returns true, the test suite receives the
       `"CoreSingleOp"` prefix; otherwise it receives `"SingleOp"`.
   - **StableHLO Composite Ops ->
     [register_composite_ops.cc](litert/ats/register_composite_ops.cc)**:
     - **Required for StableHLO composite ops** (`TestNames::Create`
       automatically assigns `"CompositeOp"` when the top-level op is
       `kLiteRtOpCodeShloComposite`). Linked into all backend ATS targets and
       `:register_ops_contract_test`.
     - **Important**: Set `/*iters=*/` in `register_composite_ops.cc` to
       `>= kGridSize` (e.g., `16` or `20`) so every entry in your
       `kStratifiedGrid` is executed!
3. **Add `":register_<op>"` to `deps`** of `:register_single_ops` (for primitive
   ops) or `:register_composite_ops` (for composite ops) in
   [ats/BUILD](litert/ats/BUILD).

---

### Step 4: Run Mandatory Verification Gates & Configure Backend Exclusions

Always run these verification commands using
`/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents`:

1. **Registration Contract Test** (instantiates `Create(rng)` across all
   registered combinations and verifies suite prefix, uniqueness, and signature
   formatting):
   ```bash
   /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test //litert/ats:register_ops_contract_test
   ```
2. **Targeted ATS Execution for the New Op**:
   ```bash
   /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents run //litert/ats:builtin_ats -- --do_register=".*<OpName>.*"
   /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents run //litert/ats:ats -- --do_register=".*<OpName>.*"
   ```
3. **Full Host CPU (Built-in, XNNPACK & YNNPACK) ATS Regression Checks**:
   ```bash
   /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test \
     //litert/ats:builtin_ats \
     //litert/ats:ats \
     //litert/ats:ynnpack_ats
   ```
   - **Why `:builtin_ats` Is Essential**: `:builtin_ats` (and
     `:builtin_cpu_ats`) runs the TFLite built-in CPU kernels
     (`--cpu_kernel_mode=builtin`, no delegate) directly against your
     generator's `Reference()`, validating graph construction and reference
     outputs end-to-end even when an op is excluded in `XNNPACK_DONT_REGISTER`
     on `:ats` (note that `BUILTIN_DONT_REGISTER` excludes `_f16` because TFLite
     built-in CPU kernels do not support `FLOAT16` tensors).
   - **Handling Unsupported Ops/Types on CPU Delegates (3 Exclusion Lists)**:
     - `:ats` and `:cpu_ats` run against the **XNNPACK** CPU delegate
       (`backend = "cpu"`), `:cpu_macos_ats` has its own inline `dont_register`
       list mirroring `XNNPACK_DONT_REGISTER`, and `:ynnpack_ats` /
       `:ynnpack_cpu_ats` use `YNNPACK_DONT_REGISTER` in
       [ats/BUILD](litert/ats/BUILD). Neither
       delegate supports every primitive op, data type, or composite op (for
       example, `OneHot`, `SelectV2`, `tfl.floor_div`, `Swiglu`, `QkvNormRope`,
       or `SdpaTransposed`).
     - If `:ats` or `:ynnpack_ats` fails because XNNPACK or YNNPACK does not
       support your new op or specific template combinations, add a narrow regex
       with an explanatory comment to all applicable CPU exclusion lists in
       [ats/BUILD](litert/ats/BUILD):
       1. `XNNPACK_DONT_REGISTER`
       2. `:cpu_macos_ats`'s inline `dont_register` list (keep in sync with
          `XNNPACK_DONT_REGISTER`)
       3. `YNNPACK_DONT_REGISTER`
     - Re-run `:builtin_ats`, `:ats`, and `:ynnpack_ats` to confirm all three
       are green, and run `build_cleaner` on
       `litert/test/generators/BUILD` and
       `litert/ats/BUILD` if needed.
