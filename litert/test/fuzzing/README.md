<!--
Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# LiteRT runtime fuzzing

This directory contains FuzzTest-based tests for LiteRT model verification,
shape inference, model loading, and operator execution across built-in CPU
kernels and delegates. Its main purpose is to find memory-safety and robustness
bugs in code reached while a model is verified, prepared, allocated, and
invoked—especially calculations that turn model-controlled data into shapes,
allocation sizes, sub-byte packed byte counts, indexes, offsets, strides, and
loop bounds.

All fuzzing tests in this directory must be authored with the
[Google FuzzTest framework](https://github.com/google/fuzztest), using
`FUZZ_TEST` and FuzzTest domains. Do not introduce another fuzzing or
property-testing framework. Keeping one framework gives these tests a common
execution model, domain vocabulary, Bazel integration, seed/smoke mode, and
coverage-guided workflow.

The tests employ both **byte-based** and **property-based** fuzzing. While
byte-based fuzzing mutates raw byte streams to stress parser and schema ingress
(such as serialized model loading in `model_fuzzer_test.cc`), **per-operator
kernel fuzzing tests should almost exclusively use property-based fuzzing with
strictly separated valid and malformed cases to achieve deep code coverage**.
Byte-based mutation rarely penetrates deep operator kernels because arbitrary
bytes are rejected immediately by FlatBuffer schema verification. Property-based
domains, by contrast, construct structurally valid models that exercise deep
execution paths (shape calculation, buffer allocation, inner SIMD loops, and
delegate dispatch), while isolated malformed properties verify clean rejection
of specific invariants.

The governing security assumptions are documented in
[the LiteRT Runtime Threat Model and Fuzzing Policy](../../../docs/instructions/security.md).
Read that document before treating a fuzzer finding as a security issue.

## Scope and security boundary

The general LiteRT/TFLite security model treats models as programs and expects
untrusted models to run inside a sandbox. The stricter threat model applies to
WebNN-reachable CPU (built-in reference/optimized kernels, XNNPACK, YNNPACK) and
MLDrift (GPU) runtime paths through LiteRT's `CompiledModel` API. On that
untrusted / bring-your-own-model (BYOM) path, a model must pass mandatory
ingress validation before it reaches the runtime, and model metadata relied upon
after verification must remain immutable.

Fuzzers in this directory should therefore identify the contract they test:

*   model verification, shape inference, and loading;
*   a built-in CPU reference kernel;
*   a built-in CPU optimized or multithreaded kernel; or
*   an explicitly selected backend or delegate such as XNNPACK, YNNPACK, or
    MLDrift.

The distinction matters during triage. A crash reachable only through a model
that required ingress validation rejects is a robustness bug, not a vulnerability
across the WebNN security boundary. A crash or undefined behavior that remains
reachable after required checks pass is security-relevant when it can affect
memory access, allocation, tensor resizing, loop bounds, or control flow.

Operator fuzzing here is intentionally **not** restricted to the current list of
WebNN-supported operators. Tests for other operators improve general runtime
robustness and provide coverage in advance of possible WebNN expansion. Their
presence does not by itself expand the current threat model or change how a
finding is classified.

`litert::VerifyModel(...)` (and its boolean wrapper `litert::Verify(...)` in
`litert/core/model/verifier.h`) is the self-contained ingress verifier—it no
longer delegates to `tflite::Verify(...)`. It performs FlatBuffer schema and
version verification, subgraph and operator consistency checks, opcode range
validation, overflow-safe bounds checks on appended `Buffer.offset`/`size` and
`large_custom_options_offset`/`size`, constant buffer validation, `int32_t`
positive-dimension-product bounds checks, quantization and sparsity metadata
validation, LiteRT model unpacking, and per-op static validation through
`ShapeInferenceEngine::InferOpShapes`.

Note current verifier boundaries when authoring properties or triaging findings:

*   `litert::VerifyModel(...)` does not yet validate separate external weight
    metadata (`Tensor.external_buffer` / `external_buffers` /
    `external_buffer_groups`), does not yet have registered shape inferrers for
    `SCATTER_ND` or `SPLIT_V`, and enforces only partial per-op element-type
    rules.
*   A property claiming WebNN reachability must also keep generated models
    within the operator, type, rank, layout, quantization, constant-buffer, and
    output-metadata contracts enforced by WebNN lowering. Other properties may
    deliberately go beyond that surface for general runtime robustness, and
    should be labeled accordingly during triage.

## Directory structure

The tests fall into three groups:

*   `model_fuzzer_test.cc` exercises both raw serialized-model fuzzing through
    `litert::LiteRtVerifier`, `FlatBufferModel::VerifyAndBuildFromBuffer`, and
    `InterpreterBuilder` (`ModelFuzzerTest`) and structured fuzzing of
    `ValidateFlatBufferOperatorCodes`, `ValidateFlatBufferTensors` /
    `ValidateFlatBufferTensorQuantization`, and
    `ShapeInferenceEngine::InferOpShapes`.
*   `simple_fuzzer_test.cc` is a small value-only example that repeatedly
    invokes an `ADD` model with fuzzed input bytes.
*   `*_fuzz_test.cc` files construct minimal one-operator models and exercise
    operator preparation and execution through selected kernel implementations
    or delegates.

The shared one-op infrastructure consists of:

*   `one_op_fuzz_model.h` and `one_op_fuzz_model.cc`, which build, verify, load,
    allocate, populate, and optionally invoke a one-node model;
*   `fuzzing_util.h`, which provides checked arithmetic, tensor storage and
    alignment helpers, structured value generation, and the silent error
    reporter; and
*   `fuzzer_quota_allocator.h` and `fuzzer_quota_allocator.cc`, which bound live
    allocations made by the interpreter during one fuzz iteration via
    `TfLiteAllocator` / `Interpreter::SetAllocator`.

`BUILD` defines normal built-in targets and separate delegate-enabled targets
where linking a delegate should be explicit. Operator-specific fuzzers use the
`*_fuzz_test.cc` naming convention; see `BUILD` for the current targets and
delegate-enabled variants.

## One-op harness lifecycle

`BuildAndRunOneOpModel()` deliberately goes through public runtime machinery
instead of calling a kernel's internal `Prepare()` or `Eval()` function
directly. That preserves important parts of the real attack surface:

1.  Build a minimal FlatBuffer containing one operator.
2.  Check model-buffer and constant-buffer alignment.
3.  Run `litert::VerifyModel(..., run_spec.verify_options)`.
4.  Register only the operator and version range under test.
5.  Construct an `Interpreter` and attach a `FuzzerQuotaAllocator`.
6.  Resize runtime input tensors and set up any persistent read-only tensors.
7.  Run the `pre_allocate` hook, normally to apply a delegate.
8.  Call `AllocateTensors()`, which runs kernel preparation and shape handling.
9.  Run the `post_allocate` hook when a check requires allocated tensors (for
    example, proving delegation occurred).
10. Validate runtime tensor alignment and exact byte sizes, then copy input
    data.
11. Call `Invoke()` when execution is part of the tested contract.

The runner returns one of three results:

<!-- mdformat off(reason: standard GFM table rendering) -->

| Result | Meaning |
| :--- | :--- |
| `kSuccess` | Every requested stage completed successfully. |
| `kRejected` | Verification, construction, resizing, preparation, allocation, a hook, or invocation rejected the case cleanly. |
| `kHarnessFailure` | The fuzz harness itself was inconsistent or could not establish a required testing invariant. |

<!-- mdformat on -->

`kHarnessFailure` is never an acceptable model outcome; it means the harness
must be fixed. A property should normally require exactly `kSuccess` or exactly
`kRejected`, rather than merely asserting that the result was not a harness
failure.

The runner deliberately collapses clean rejection from all stages (model
verification, delegate graph partitioning, tensor resizing, kernel preparation,
allocation, hooks, and invocation) into `kRejected`. This achieves two core
goals:

*   **Backend reusability:** Different backends (built-in CPU reference and
    optimized kernels, XNNPACK/YNNPACK CPU delegates, or hardware delegates like
    MLDrift) validate constraints at different points in their lifecycle. For
    example, a delegate may reject an unsupported op, layout, or parameter
    during graph partitioning in `pre_allocate`, while a CPU kernel validates
    shapes in `Prepare()`, and an accelerator delegate may detect memory or
    dispatch constraints during `Invoke()`. Unifying clean rejection into
    `kRejected` provides a backend-agnostic contract so the same semantic cases
    can be reused across CPU, XNNPACK, YNNPACK, MLDrift, and future backends.
*   **Production security alignment:** Untrusted-model deployments are expected
    to combine ingress validation with bounded allocation. Under this threat
    model, clean rejection by any layer—FlatBuffer verification, shape
    inference, delegate partitioning, op shape validation, memory quota limits,
    or runtime parameter checks—is a valid defense that prevents unsafe
    execution.

When a property is intended to test a specific kernel or backend invariant,
ensure its domain, smoke tests, seeds, and coverage demonstrate that the case
reaches that stage instead of being rejected earlier by the harness or model
verifier.

## Separate valid and malformed properties

To achieve **deep code coverage** across operator kernels, valid and malformed
inputs must use different domains and different fuzz properties.

A valid property has a strong oracle:

```cpp
void OpExecutesValidCases(const OpCase& test_case) {
  ASSERT_EQ(RunOpCase(test_case), RunResult::kSuccess);
}
```

A malformed property deliberately violates a documented invariant and requires
clean rejection:

```cpp
void OpRejectsInvalidAxis(const OpCase& test_case) {
  ASSERT_EQ(RunOpCase(test_case), RunResult::kRejected);
}
```

This split is important for several reasons:

*   **Deep code coverage:** Valid inputs pass verification and preparation and
    spend fuzzing time in deeper execution paths (inner SIMD loops,
    multithreading, memory addressing, and delegate dispatch). If most generated
    cases are malformed, early rejection dominates the corpus and kernel code
    receives little coverage.
*   **A precise oracle:** Rejection fails a valid property and success fails a
    malformed property. A mixed property ("success or rejection is acceptable")
    can pass even if every valid case is rejected or every malformed case is
    accepted.
*   **Stable minimization:** A reproducer stays on the same side of the operator
    contract while FuzzTest shrinks it. In a mixed domain, shrinking can turn a
    deep execution failure into an unrelated early verifier rejection.
*   **Actionable failures:** The property name states which invariant was under
    test, such as an invalid axis, malformed padding matrix, channel mismatch,
    or overflowing shape product.
*   **Performance and signal:** Expected rejection is common in malformed
    properties. Keeping it separate makes it safe to suppress routine error
    messages without hiding unexpected failures in valid execution.

An invalid domain should not be arbitrary garbage. Start from an otherwise valid
case and violate one relationship at a time—for example, moving an axis just out
of range, changing one filter channel count, setting a stride just larger than
the input dimension (`stride = input_size + 1`), using a padding matrix with the
wrong number of columns, or changing a reshape target by one element. This keeps
the mutation close to the boundary so the intended validation code executes.

**Never clamp a valid domain in isolation without a paired malformed property.**
When a `Valid*Domain` restricts or clamps a parameter relationship (for example,
clamping `stride_height = std::min(stride_height, input_shape[1])` so valid
properties stay on the valid side of `litert::VerifyModel` or kernel checks),
pair it with a corresponding `Malformed*Domain` / `Rejects*` property that
explicitly crosses that boundary (such as `stride_height = input_shape[1] + 1`).
Tightening a valid domain alone hides the out-of-range region from the fuzzer
and leaves both the rejection check and any fallback path untested.

Some boundary-stress tests need a conditional oracle—for example, when a
generated shape is valid per the operator contract but intentionally exceeds the
fuzzer's allocation budget. Keep these properties distinct from ordinary valid
execution, compute the expected outcome with overflow-safe arithmetic, and
always reject `kHarnessFailure`.

## Empty tensor handling

An **empty tensor** is a tensor with one or more dimensions where at least one
dimension is zero (resulting in zero total elements). Handling of empty tensors
across backends follows these rules:

*   **Built-in CPU and XNNPACK backends allow empty tensor shapes:** The LiteRT
    core runtime, built-in CPU kernels (reference and optimized), and the
    XNNPACK delegate allow empty tensor shapes (`dims[i] == 0`) when the
    operator contract supports them. For example, the batch dimension of a
    `CONV_2D` input tensor may be zero in dynamic-batching pipelines, or an
    initial past-KV cache input to `CONCATENATION` may have a zero sequence
    dimension during batched prefill. The XNNPACK delegate binds a dummy
    non-null pointer (`&dummy_data_`) for zero-byte external values and skips
    zero-element operator executions at runtime. Support still depends on the
    operand and dimension (for example, `CONV_2D` allows an empty input batch
    dimension while requiring a non-empty filter tensor and non-zero channel
    counts).
*   **MLDrift (GPU) and other hardware delegates may reject empty tensors:**
    Hardware runtimes, device drivers, and shader microkernels frequently assume
    non-zero buffer allocations, valid strides, or non-zero divisor dimensions;
    such delegates should cleanly reject graphs containing unsupported empty
    tensor shapes during delegate partitioning, `Prepare()`, or `Invoke()`.

### Implications for fuzzing tests

Fuzz testing in this directory must follow these rules:

1.  **Built-in CPU and XNNPACK fuzz properties must test empty-shape support and
    rejection boundaries:**
    *   Where an operator contract supports empty tensors (e.g., a batch or data
        dimension of zero), valid fuzz domains for built-in CPU and XNNPACK
        targets should include zero dimensions to verify that execution succeeds
        (`RunResult::kSuccess`), allocates without memory leaks, and completes
        cleanly.
    *   If an operator does not support empty tensors along specific operands or
        dimensions (e.g., filter spatial dimensions, divisor dimensions, or
        channel counts of zero), the verifier, delegate, or kernel must reject
        them cleanly (`RunResult::kRejected`).
2.  **MLDrift and other hardware delegates that do not support empty tensors
    must reject them cleanly:** Any unsupported empty tensor shape must cleanly
    yield `RunResult::kRejected`.
3.  **No memory-safety or arithmetic failures on zero-element tensors:** In all
    backends, zero-element tensors must **never** cause undefined behavior,
    integer division by zero in stride/offset calculations, null-pointer
    dereferences (including `memcpy(nullptr, ..., 0)`), undersized buffer
    allocations, memory-safety violations, or crashes.

## Designing a per-operator fuzzer

### 1. State the contract

Decide whether the property covers loading, preparation, invocation, or a
specific implementation. Identify which fields are controlled by the model and
which values can arrive dynamically at invocation time. Checks for constant
parameter tensors may run during `Prepare()`; dynamic parameter values must also
be tested through the runtime path where they become available.

Do not assume that valid individual tensors imply a valid operation. Fuzz the
relationships the kernel depends on: ranks, matching dimensions, channel
divisibility, axis ranges, element-count equality, quantization compatibility,
and output-shape relationships.

### 2. Define a semantic case type

Use a small struct containing shapes, operator options, tensor types, parameter
values, execution-path choices, and bounded byte overlays. The struct should
describe an operation, not a serialized byte stream.

Prefer structured FuzzTest domains:

*   use `Map()` to derive values that must be related;
*   use small dimension ranges for normal execution;
*   include meaningful boundaries explicitly with `OneOf()` and `Just()`;
*   include constant and dynamic forms of shape, axis, padding, or index
    tensors;
*   include zero-element tensors when the operator contract permits them (see
    [Empty tensor handling](#empty-tensor-handling));
*   include negative axes or sentinel values only where the contract defines
    them; and
*   keep malformed transformations narrow and named.

Cover LiteRT's rank ceiling (`LITERT_TENSOR_MAX_RANK == 8`, which is also
Chromium's current WebNN untrusted-tensor limit) as well as any stricter
backend-specific limit (such as MLDrift's rank `5` limit) or
application-configured `VerifyOptions::max_rank`. Higher-rank direct TFLite
cases remain useful robustness coverage, but they are not WebNN security cases
unless an untrusted entry point can reach them. Do not add a broad defensive
rank limit to a built-in kernel merely to match one delegate or integration.

Arbitrary byte overlays are useful for tensor values after a valid shape and
storage size have been established. Apply central representation invariants,
such as canonicalizing Boolean values to `0` or `1`
(`ApplyCentralTensorInputInvariants`). Do not let a byte overlay accidentally
change the semantic property being tested.

### 3. Keep each iteration small

Per-op fuzzers are coverage tools, not bulk allocation stress tests. Small
tensors allow many more iterations per second and reach more control-flow
states. Use checked arithmetic before materializing any tensor and impose an
explicit element or byte limit.

Every one-op run must set `max_live_allocation_bytes`. Real-world deployments
handling untrusted models are expected to bound allocations; setting this limit
exercises `FuzzerQuotaAllocator` (backed by `TfLiteAllocator` /
`Interpreter::SetAllocator`) while protecting developer and CI machines.

Because the quota allocator bounds live memory, the primary security concerns to
fuzz around allocation are:

*   **Integer overflow and 32-bit truncation before the allocator:** The most
    dangerous allocation vulnerability occurs when large dimensions wrap in
    32-bit integer arithmetic (e.g., `shape_product * sizeof(T)` overflowing
    `int32_t` or `size_t`, or sub-byte `(count + 1) / 2` / `(count + 3) / 4`
    wrapping near `SIZE_MAX` on 32-bit targets). If an overflowing calculation
    produces an artificially small byte count, the request can pass the quota
    check, allocate an undersized buffer, and cause heap out-of-bounds access
    during kernel execution. Exercise 32-bit-sensitive boundaries near
    `INT32_MAX` ($2^{31} - 1$) and `UINT32_MAX` ($2^{32} - 1$) to ensure
    overflow is detected before allocation.
*   **Graceful unwinding on allocation failure:** When the quota allocator
    refuses an allocation (returning `nullptr` or an error status), every
    backend must check for failure and cleanly unwind without null-pointer
    dereferences, memory leaks, or double-frees during cleanup.
*   **Unmanaged allocations:** Verify that delegate backends and kernels do not
    bypass the quota allocator with untracked `malloc`, `new`, or external
    driver allocations.

For huge-dimension arithmetic boundaries, fuzz a small internal shape or byte
helper directly when possible. If an end-to-end model is necessary, avoid
populating or invoking tensors whose valid allocation would exceed the test
budget.

Exercise 32-bit-sensitive boundaries even when developing on a 64-bit host.
Element counts and byte counts may use `size_t`, while dimensions and many
kernel indexes use `int`. Promote before arithmetic, check every shape product,
and validate each narrowing conversion. A zero total element count must not hide
an invalid or overflowing nonzero dimension, stride, or offset (such as
`{65536, 65536, 0}`).

### 4. Preserve real model invariants

Build constants with aligned FlatBuffer buffers and make runtime buffers match
their tensor's exact byte size. Use `MakeValues()`, `MakeIntegerValues()`, and
`StorageBytesForElements()` instead of typed access to arbitrary unaligned
bytes. Packed sub-byte types (`INT4`, `UINT4`, `INT2`) require their packed
storage size rather than `element_count * sizeof(T)`.

Run the model verifier before interpreter construction. A per-op fuzzer may
generate models directly and use `litert::VerifyModel(...)` as the global
ingress check; it does not need to invoke a converter or WebNN lowering
pipeline.

Valid-case domains should satisfy the static invariants assigned to model
generation and ingress validation by the threat model. Malformed properties may
violate those invariants, but their results should be described as clean
rejection or general robustness coverage rather than automatically as a
WebNN-reachable security finding.

### 5. Cover execution implementations explicitly

Reference, optimized, multithreaded, and delegated implementations (XNNPACK,
YNNPACK, MLDrift) can have different scratch buffers, integer ranges, and
indexing code. Cover every registered implementation relevant to the operator.
Using shared semantic case types and the unified `kSuccess`/`kRejected` contract
keeps test properties reusable across CPU kernels and delegate backends. Put
built-in variants in the fuzz case/domain (or in clearly named properties over
the same semantic domains) so coverage gaps remain visible.

Delegate-enabled targets are separate Bazel targets in this directory. Apply a
delegate in `pre_allocate`, because normal delegated inference installs the
delegate before `AllocateTensors()`. A delegate property must also inspect the
execution plan (typically in `post_allocate`) and prove that a delegate node was
created. Successful fallback to a built-in kernel is not delegate coverage.

Delegate domains may be narrower than built-in domains when the delegate has a
real documented constraint. Keep those constraints local to the delegate
property; do not impose them on the built-in operator contract.

### 6. Add smoke tests and seeds

Add ordinary `TEST` cases for important paths, especially:

*   proof that invocation actually occurs;
*   proof that a delegate target really delegates;
*   constant and dynamic parameter tensors;
*   zero-element behavior;
*   the highest supported rank or a meaningful rank boundary; and
*   deterministic regressions found by fuzzing.

Seed coverage-guided properties with known boundary shapes when mutation is
unlikely to discover them efficiently. When promoting a fuzz-found bug to a
normal unit test near the affected implementation, verify red-green efficacy by
confirming the new test fails when the production fix is temporarily reverted.

### 7. Keep expected errors quiet

Malformed properties can reject thousands of inputs per second. The shared
`SilentErrorReporter` suppresses those expected diagnostics so they do not flood
stderr or dominate execution time. When debugging locally, temporarily restore
printing in that reporter or add a focused trace around the failing case. Do not
enable routine error printing in committed coverage-guided runs.

## What counts as a fuzzing failure?

Treat these outcomes as bugs and investigate their reachability:

*   a crash, sanitizer finding, out-of-bounds access, use-after-free, memory
    leak, or null dereference;
*   undefined behavior in shape, allocation, sub-byte packing, index, offset,
    stride, loop-bound, or control-flow arithmetic;
*   a valid property returning `kRejected`;
*   a malformed property returning `kSuccess`;
*   `kHarnessFailure` from any generated case;
*   a path advertised as delegated that did not actually delegate; or
*   inconsistent validation across execution paths when the difference can lead
    to unsafe execution.

Integer overflow in ordinary tensor-value arithmetic is normally a correctness
issue rather than a security issue when the value cannot influence allocation,
tensor resizing, indexing, loop bounds, or control flow. Likewise, numerical
precision and NaN/Inf differences are not memory-safety findings by themselves.
Follow the threat model's triage guide instead of classifying a result solely
because a sanitizer reported it.

## Building and running

Run a target in its short GoogleTest/FuzzTest smoke/seed mode while iterating:

```bash
bazelisk test //litert/test/fuzzing:pad_fuzz_test
```

Run one property in coverage-guided mode:

```bash
bazelisk run //litert/test/fuzzing:pad_fuzz_test -- \
  --fuzz=PadFuzzTest.PadExecutesValidCases \
  --fuzz_for=5m
```

Current delegate-enabled targets in `BUILD` include:

*   `convolution_xnnpack_fuzz_test`
*   `pad_xnnpack_fuzz_test`
*   `reduce_xnnpack_fuzz_test`
*   `reshape_xnnpack_fuzz_test`
*   `slice_xnnpack_fuzz_test`

On Windows, run one fuzz property per process (for example,
`bazelisk run -c fastbuild //litert/test/fuzzing:pad_fuzz_test --
--fuzz=PadFuzzTest.PadExecutesValidCases --fuzz_for=5m`).

Coverage-guided fuzzing and local regression runs should enable AddressSanitizer
(ASan) and UndefinedBehaviorSanitizer (UBSan) (using your environment's
`--config=asan` / `--config=ubsan` Bazel configs, or passing
`-fsanitize=address,undefined` compiler and linker flags). Running with ASan and
UBSan is critical for catching subtle heap out-of-bounds reads/writes,
use-after-free bugs, integer overflows in pointer/index calculations, and
unhandled allocation failures that might not crash in uninstrumented binaries.
When changing shape, index, packing, or allocation code, also validate in a
32-bit build/test configuration. CI runs fuzz tests in short smoke mode; local
or scheduled fuzzing campaigns must run long enough to exercise the structured
domain and deep implementation paths.

## New-fuzzer checklist

Before submitting a new per-op fuzzer, verify that:

*   it uses Google FuzzTest and does not introduce another fuzzing framework;
*   it uses property-based testing with separate valid and malformed domains to
    achieve deep code coverage;
*   the tested loading/kernel/delegate contract is clear;
*   valid and malformed cases use separate domains and exact outcomes;
*   malformed domains violate a specific documented invariant;
*   any constraint clamped in a valid domain (e.g., `std::min`) is paired with a
    corresponding malformed property that crosses that boundary;
*   both constant and dynamic parameter paths are covered where applicable;
*   all materialization and expected-outcome arithmetic is checked;
*   tensor sizes and live allocations are bounded (`max_live_allocation_bytes`);
*   32-bit overflow, sub-byte packing, and checked-narrowing boundaries are
    represented;
*   model, constant, and runtime buffers satisfy alignment and representation
    requirements;
*   empty tensor (zero-element) handling conforms to the CPU and XNNPACK support
    and per-operand rejection contracts (and hardware-delegate rejection where
    applicable);
*   relevant reference, optimized, multithreaded, and delegate (XNNPACK,
    YNNPACK, MLDrift) paths are observable and covered;
*   smoke tests prove that intended deep paths and delegation are reached;
*   ASan and UBSan have been enabled for a meaningful local run; and
*   stable fuzz-found bugs have focused ordinary unit-test regressions verified
    with a red-green check (failing when the fix is reverted).
