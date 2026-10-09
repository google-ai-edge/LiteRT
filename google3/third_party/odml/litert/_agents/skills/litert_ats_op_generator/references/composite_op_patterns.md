# StableHLO Composite Op Generator Patterns (`CompositeOp`)

Use these patterns when authoring a generator for a high-level StableHLO
composite operator (such as LLM attention, normalization, positional embedding,
or gated MLP blocks) in `litert/test/generators/<op>.h`.

Exemplars in the codebase:

- [swiglu.h](litert/test/generators/swiglu.h)
  (single output, 3 dynamic inputs `x, w_gate, w_up`, no flexbuffer attributes,
  `kFloatElementwise`)
- [qkv_norm_rope.h](litert/test/generators/qkv_norm_rope.h)
  (3 outputs `q_out, k_out, v_out` via `std::make_tuple`, 4 dynamic inputs +
  static `OwningCpuBuffer::CopyAs` / `Copy` constants, Flexbuffer
  `composite_attributes`)
- [sdpa.h](litert/test/generators/sdpa.h) and
  [sdpa_transposed.h](litert/test/generators/sdpa_transposed.h)
  (conditional `WithMask` input list via `std::conditional_t`, `SoftCap` trait,
  `kFloatAccumulationAware`)

---

## 1. Stratified Grid (`kStratifiedGrid`) for Realistic LLM Shapes

Unlike primitive ops that sample arbitrary random dimensions, composite ops use
a **stratified grid** (`kStratifiedGrid[]`) of 15–16 realistic LLM workload
configurations covering:

- **Decode (`seq_len = 1`)**: MHA, GQA, MQA, large head count, small head dim
- **Chunked Decode (`seq_len = 4..16`)**: speculative decoding / multi-token
  draft shapes
- **Prefill (`seq_len == kv_len`)**: short and medium context prefill
- **Batched (`batch > 1`)**: multi-stream decode and batched prefill

In `Create(Rng& rng)`:

```cpp
template <typename Rng>
static Expected<Ptr> Create(Rng& rng) {
  static size_t call_count = 0;
  size_t idx = call_count++ % kGridSize;
  (void)rng;  // Deterministic stratified sampling across iterations
  Params p = kStratifiedGrid[idx];
  return Create(p);
}
```

> [!IMPORTANT]
> When wiring `Register<CompositeOp>(options, test_id, /*iters=*/..., cap)` into
> [register_composite_ops.cc](litert/ats/register_composite_ops.cc),
> always set `iters >= kGridSize` (e.g., `16` or `20`) so every configuration in
> `kStratifiedGrid` is exercised.

---

## 2. Building the `StableHLOComposite` Graph

### Single-Output Composite (`swiglu.h` / `sdpa.h` pattern)

```cpp
static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
  auto element_type = litert::tensor::ApiType<T>::value;

  auto x = litert::tensor::Create(
      std::string(kInputNames[0]), element_type,
      {params.batch, params.seq_len, params.d_model});
  auto w_gate = litert::tensor::Create(
      std::string(kInputNames[1]), element_type,
      {params.d_model, params.d_ff});
  auto w_up = litert::tensor::Create(
      std::string(kInputNames[2]), element_type,
      {params.d_model, params.d_ff});

  // Optional: Build Flexbuffer composite_attributes and composite options
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    // fbb.Float("epsilon", 1e-6f);
  });
  fbb.Finish();

  litert::tensor::StableHLOCompositeOptions options;
  options.name = "odml.my_composite";
  options.version = 1;
  options.composite_attributes = fbb.GetBuffer();

  auto decompose = [](TensorTf x_in, TensorTf w_gate_in, TensorTf w_up_in) {
    auto gate = litert::tensor::BatchMatmul(x_in, w_gate_in);
    auto sig_gate = litert::tensor::Logistic(gate);
    auto swish = litert::tensor::Mul(gate, sig_gate);
    auto up = litert::tensor::BatchMatmul(x_in, w_up_in);
    return litert::tensor::Mul(swish, up);
  };

  auto out = litert::tensor::StableHLOComposite(
      options, decompose, x, w_gate, w_up);
  out.SetName(std::string(kOutputNames[0]));

  return SaveTensorGraph({out});
}
```

### Static Constant Weight & Scalar Tensors (`OwningCpuBuffer::CopyAs`)

When a composite op (such as a normalization layer) requires learned parameter
tensors (like `gamma` scale weights) or scalar constants (`eps`) to be **static
constants** baked into the flatbuffer rather than dynamic subgraph inputs (see
`qkv_norm_rope.h` for `OwningCpuBuffer::CopyAs` usage):

1. **Sample and store the constant weights in `Params`** (inside `Create(Rng&)`)
   and **omit them from `TestLogicTraits` `InputTypes` and `MakeInputs()`**:
   ```cpp
   // Only dynamic `x` is in InputTypes; static `gamma` is stored in Params.
   using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
   ```
2. **Create constant `TensorTf` buffers with `OwningCpuBuffer::CopyAs`** in
   `BuildGraph()`, and pass them into `StableHLOComposite`:
   ```cpp
   auto scale_type = litert::tensor::ApiType<ScaleType>::value;
   auto gamma_buf = litert::tensor::OwningCpuBuffer::CopyAs(
       scale_type, absl::MakeConstSpan(params.gamma));
   auto gamma = litert::tensor::Create(
       "gamma", scale_type, {params.dim}, std::move(gamma_buf));

   auto decompose = [eps = params.epsilon, element_type](
                        TensorTf x_in, TensorTf gamma_in) {
     std::array<T, 1> eps_val = {static_cast<T>(eps)};
     auto eps_buf = litert::tensor::OwningCpuBuffer::CopyAs(
         element_type, absl::MakeConstSpan(eps_val));
     auto eps_cst = litert::tensor::Create(
         "eps", element_type, {}, std::move(eps_buf));
     // ...
   };

   auto out = litert::tensor::StableHLOComposite(options, decompose, x, gamma);
   ```
   [`ReferenceEvaluator::EvaluateCompositeReference`](litert/test/generators/reference_evaluator.cc)
   automatically binds constant composite inputs from the graph's buffers when
   evaluating the decomposition subgraph.

### Multi-Output Composite (`qkv_norm_rope.h` pattern)

When a composite op produces multiple outputs, return
`std::make_tuple(out0, out1, ...)` from the `decompose` lambda and unpack with
structured bindings:

```cpp
auto decompose = [eps](TensorTf q_in, TensorTf k_in, ...) {
  // ...
  return std::make_tuple(q_out, k_out);
};

litert::tensor::StableHLOCompositeOptions composite_options{
    .name = "odml.qkv_norm_rope",
    .composite_attributes = fbb.GetBuffer(),
};

auto [q_res, k_res] = litert::tensor::StableHLOComposite(
    composite_options, decompose, q, k, q_norm_w, k_norm_w, rope_pos);

q_res.SetName(std::string(kOutputNames[0]));
k_res.SetName(std::string(kOutputNames[1]));

return SaveTensorGraph({q_res, k_res});
```

### Conditional Inputs (`WithMask` in `sdpa.h` pattern)

If a template parameter toggles an optional runtime input (e.g., `WithMask`),
use `std::conditional_t` in `TestLogicTraits`:

```cpp
using InputTypes = std::conditional_t<
    kWithMask,
    TypeList<T, T, T, T>,  // Q, K, V, Mask
    TypeList<T, T, T>>;    // Q, K, V
using Traits = TestLogicTraits<InputTypes, TypeList<T>, Params>;
```

---

## 3. Evaluating Reference Outputs via `ReferenceEvaluator`

Composite ops delegate `Reference()` directly to
[`ReferenceEvaluator::EvaluateCompositeReference`](litert/test/generators/reference_evaluator.h),
which traverses the decomposed subgraph inside `Graph()` and executes each
primitive op in topological order:

```cpp
Expected<void> Reference(const VarBuffers& inputs,
                         VarBuffers& outputs) const override {
  return ReferenceEvaluator::EvaluateCompositeReference(
      Graph(), inputs, outputs);
}
```

### Adding Missing Primitive Ops to `ReferenceEvaluator`

Inspect
[`ReferenceEvaluator::RegisterStandardOps()`](litert/test/generators/reference_evaluator.cc)
to verify that every `kLiteRtOpCodeTfl*` emitted by your composite op's
`decompose` lambda has a registered handler. If any primitive op is not yet
registered in `RegisterStandardOps()`:

1. Ensure `litert::internal::Reference<Op>` exists in
   `litert/core/model/ops/`.
2. Register the `kLiteRtOpCodeTfl*` handler inside
   `ReferenceEvaluator::RegisterStandardOps()` in
   [reference_evaluator.cc](litert/test/generators/reference_evaluator.cc).
3. Add a unit test in
   [reference_evaluator_test.cc](litert/test/generators/reference_evaluator_test.cc)
   and run:
   ```bash
   /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test //litert/test/generators:reference_evaluator_test
   ```

---

## 4. Registering a Composite Op in `litert/ats/`

1. In `litert/ats/register_<op>.cc`, call
   `RegisterCombinations` (`TestNames::Create` in `ats/common.h` automatically
   assigns the `"CompositeOp"` suite prefix when the top-level op is
   `kLiteRtOpCodeShloComposite`):
   ```cpp
   RegisterCombinations<
       Fixture,
       MyComposite,
       TypeList<float, tflite::half>>
     (iters, test_id, options, cap);
   ```
2. Wire `RegisterMyComposite(options, test_id, /*iters=*/16, cap);` into
   [register_composite_ops.cc](litert/ats/register_composite_ops.cc)
   and add `":register_my_composite"` to `deps` of `:register_composite_ops` in
   [ats/BUILD](litert/ats/BUILD).
3. If the XNNPACK or YNNPACK CPU delegates do not support your new composite op
   (like `SdpaTransposed`, `QkvNormRope`, or `Swiglu`), add its family name or
   unsupported signature pattern to all 3 CPU exclusion lists in
   [ats/BUILD](litert/ats/BUILD)
   (`XNNPACK_DONT_REGISTER`, `:cpu_macos_ats`'s inline `dont_register`, and
   `YNNPACK_DONT_REGISTER`), then run the verification gates (do **not** create
   standalone `<op>_test.cc` files in `test/generators/`; `:builtin_ats`
   executes the decomposed graph on TFLite built-in CPU kernels even when the op
   is excluded on XNNPACK/YNNPACK):
   ```bash
   /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test \
     //litert/ats:register_ops_contract_test \
     //litert/ats:builtin_ats \
     //litert/ats:ats \
     //litert/ats:ynnpack_ats
   ```
