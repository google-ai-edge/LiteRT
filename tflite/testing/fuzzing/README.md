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

# TFLite runtime fuzzing

This directory contains FuzzTest-based tests for TFLite model loading and CPU
operator execution. Its main purpose is to find memory-safety and robustness
problems in code reached while a model is verified, prepared, allocated, and
invoked. The tests emphasize the calculations that turn model-controlled data
into shapes, allocation sizes, indexes, offsets, strides, and loop bounds.

All fuzzing tests in this directory must be authored with the
[Google FuzzTest framework](https://github.com/google/fuzztest), using
`FUZZ_TEST` and FuzzTest domains. Do not introduce another fuzzing or
property-testing framework. Keeping one framework gives these tests a common
execution model, domain vocabulary, Bazel integration, seed/smoke mode, and
coverage-guided workflow.

The tests employ both **byte-based** and **property-based** fuzzing. While byte-based
fuzzing mutates raw byte streams to stress parser and schema ingress (such as
model loading in `model_fuzzer_test.cc`), **per-operator kernel fuzzing tests
should almost exclusively use property-based fuzzing with strictly separated
valid and invalid cases to achieve deep code coverage**. Byte-based mutation
cannot penetrate deep operator kernels because arbitrary bytes are rejected
immediately by FlatBuffer schema verification. Property-based domains, by
contrast, construct structurally valid models that exercise deep execution paths
(shape calculation, buffer allocation, inner SIMD loops, and delegate dispatch),
while isolated malformed properties verify clean rejection of specific invariants.

The governing security assumptions are documented in
[the TFLite CPU runtime threat model](../../../g3doc/instructions/security.md).
Read that document before treating a fuzzer finding as a security issue.

## Scope and security boundary

The general TFLite security model treats models as programs and expects
untrusted models to run inside a sandbox. The stricter threat model applies to
the CPU runtime paths that WebNN can reach through LiteRT. On that path, a model
must pass the required ingress checks before it reaches the runtime, and model
metadata relied upon after verification must remain immutable.

Fuzzers in this directory should therefore identify the contract they test:

- model parsing and verification;
- a built-in reference kernel;
- a built-in optimized or multithreaded kernel; or
- an explicitly selected backend or delegate such as XNNPACK or MLDrift.

The distinction matters during triage. A crash reachable only through a model
that the required ingress validation rejects is a robustness bug, but it is not
reachable through the WebNN security boundary as currently defined. A crash or
undefined behavior that remains reachable after the required checks is
security-relevant when it can affect memory access, allocation, loop bounds, or
control flow.

Operator fuzzing here is intentionally **not** restricted to the current list
of WebNN-supported operators. Tests for other operators improve general TFLite
robustness and provide coverage in advance of possible WebNN expansion. Their
presence does not by itself expand the current threat model or change how a
finding is classified.

`tflite::Verify(...)` is one required ingress check, but it does not currently
establish every application-specific WebNN invariant. A property that claims
WebNN reachability must also keep its generated models within the relevant
operator, type, rank, layout, quantization, constant-buffer, and output-metadata
contracts expected from WebNN lowering. Other properties may deliberately go
beyond that surface for general runtime robustness, but should say so during
triage.

## Directory structure

The tests fall into three groups:

- `model_fuzzer_test.cc` mutates a serialized model and exercises verified
  model loading and interpreter construction.
- `simple_fuzzer_test.cc` is a small value-only example that repeatedly invokes
  an `ADD` model with fuzzed input bytes.
- `*_fuzz_test.cc` files construct minimal one-operator models and exercise
  operator preparation and execution through selected kernel implementations.

The shared one-op infrastructure consists of:

- `one_op_fuzz_model.h` and `one_op_fuzz_model.cc`, which build, verify, load,
  allocate, populate, and optionally invoke a one-node model;
- `fuzzing_util.h`, which provides checked arithmetic, tensor storage and
  alignment helpers, structured value generation, and the silent error
  reporter; and
- `fuzzer_quota_allocator.h` and `fuzzer_quota_allocator.cc`, which bound live
  allocations made by the interpreter during one fuzz iteration.

`BUILD` defines normal built-in targets and separate delegate-enabled targets
where linking a delegate should be explicit.

Operator-specific fuzzers use the `*_fuzz_test.cc` naming convention. See
`BUILD` for the current targets and delegate-enabled variants.

## One-op harness lifecycle

`BuildAndRunOneOpModel()` deliberately goes through public runtime machinery
instead of calling a kernel's internal `Prepare` or `Eval` function directly.
That preserves important parts of the real attack surface:

1. Build a minimal FlatBuffer containing one operator.
2. Check model-buffer and constant-buffer alignment.
3. Run `tflite::Verify(...)`.
4. Register only the operator and version range under test.
5. Construct an `Interpreter` with a quota allocator.
6. Resize runtime input tensors.
7. Run the pre-allocation hook, normally to install a delegate.
8. Call `AllocateTensors()`, which runs kernel preparation and shape handling.
9. Run the post-allocation hook when a check requires allocated tensors.
10. Validate runtime tensor alignment and exact byte sizes, then copy input
    data.
11. Call `Invoke()` when execution is part of the tested contract.

The runner returns one of three results:

| Result | Meaning |
| --- | --- |
| `kSuccess` | Every requested stage completed successfully. |
| `kRejected` | Verification, construction, resizing, preparation, allocation, a hook, or invocation rejected the case cleanly. |
| `kHarnessFailure` | The fuzz harness itself was inconsistent or could not establish a required testing invariant. |

`kHarnessFailure` is never an acceptable model outcome. It means the harness
must be fixed. A property should normally require exactly `kSuccess` or exactly
`kRejected`, rather than merely asserting that the result was not a harness
failure.

The runner deliberately collapses clean rejection from all stages (model
verification, delegate graph partitioning, tensor resizing, kernel preparation,
allocation, hooks, and invocation) into `kRejected`. This design achieves two
core goals:

- **Backend reusability:** Different backends (built-in CPU reference and
  optimized kernels, graph delegates like XNNPACK, or hardware delegates like
  MLDrift) validate constraints at different points in their execution lifecycle.
  For instance, a delegate may reject an unsupported op, layout, or parameter
  during graph partitioning in `pre_allocate`, while a CPU kernel validates
  shapes in `Prepare()`, and certain accelerator delegates detect memory or
  dispatch constraints during `Invoke()`. Unifying clean rejection into
  `kRejected` provides a backend-agnostic contract, allowing the exact same fuzz
  properties and test cases to be reused across CPU, XNNPACK, MLDrift, and
  future backends.
- **Production security alignment:** Real-world deployments handling models from
  untrusted sources (such as WebNN or sandboxed runtime environments) are expected
  to employ a quota-enabled allocator as a first-class line of defense. In this
  threat model, clean rejection by any layer—whether FlatBuffer verification,
  delegate partitioning, op shape validation, memory quota limits, or parameter
  bounds checks—is a valid and successful defense that prevents unsafe execution.

When a property is intended to test a specific kernel or backend invariant,
ensure its domain, smoke tests, seeds, and coverage demonstrate that the case
reaches that stage instead of being rejected earlier by the harness or FlatBuffer
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

- **Deep code coverage:** valid inputs get through verification and preparation
  and spend fuzzing time in the deeper execution path (such as inner SIMD loops,
  multithreading, memory addressing, and delegate dispatch). If most generated
  cases are malformed, early rejection dominates the corpus and kernel code
  receives little to no coverage.
- **A precise oracle:** rejection is a failure for a valid property and success
  for a malformed property. A mixed property such as "success or rejection is
  acceptable" can pass even if every valid case is incorrectly rejected or
  every malformed case is incorrectly accepted.
- **Stable minimization:** a reproducer stays on the same side of the operator
  contract while FuzzTest shrinks it. In a mixed domain, shrinking can turn a
  deep execution failure into an unrelated early verifier rejection.
- **Actionable failures:** the property name states which invariant was under
  test, such as an invalid axis, malformed padding matrix, channel mismatch, or
  overflowing shape product.
- **Performance and signal:** expected rejection is common in malformed
  properties. Keeping it separate makes it safe to suppress routine error
  messages without hiding unexpected failures in valid execution.

An invalid domain should not be arbitrary garbage. Start with an otherwise
valid case and violate one relationship at a time. Examples include moving an
axis just out of range, changing one filter channel count, using a padding
matrix with the wrong number of columns, or changing a reshape target by one
element. This keeps the mutation close to the boundary being tested and makes
it more likely that the intended validation code executes.

Some boundary-stress tests need a conditional oracle. For example, a generated
shape may be valid according to the operator contract but intentionally exceed
the fuzzer's allocation budget. Keep these properties distinct from ordinary
valid execution, compute the expected outcome with overflow-safe arithmetic,
and always reject `kHarnessFailure`.

## Empty tensor handling

An **empty tensor** is a tensor with one or more dimensions where at least one
dimension is zero (resulting in zero total elements). Handling of empty tensors
differs fundamentally between hardware/graph delegates and built-in CPU kernels:

- **All delegates should reject empty tensors:** All TFLite delegates (such as
  XNNPACK, GPU, or MLDrift) should reject graphs containing empty tensor shapes
  that are statically known in the graph during delegate partitioning /
  `Prepare`. Hardware runtimes, device drivers, and microkernels frequently
  assume non-zero allocations, valid strides, or non-zero divisor dimensions;
  delegating zero-element operations risks division-by-zero, invalid hardware
  buffer bindings, or unrecoverable driver errors.
- **Built-in CPU kernels may provide compatibility support:** Built-in CPU
  reference and optimized kernels may provide partial or full support for empty
  tensors where existing applications and production models rely on it. For
  example, the batch size dimension of a Conv2D input tensor could be zero in
  dynamic-batching or filtering pipelines. If a CPU kernel was already written
  to handle such shapes gracefully, we should continue having that support to
  avoid breaking compatibility with existing applications.

### Implications for fuzzing tests

Fuzz testing in this directory must be constructed according to these rules:

1. **Delegate fuzz properties must expect rejection:** When testing
   delegate-enabled targets (e.g., `*_xnnpack_fuzz_test`), any static graph
   containing an empty tensor shape must be treated as an unsupported/malformed
   configuration and verified to cleanly yield `RunResult::kRejected`.
2. **Built-in CPU fuzz properties must test compatibility without regressions:**
   - Where an operator contract or existing CPU implementation supports empty
     tensors (e.g., a batch dimension of zero), valid fuzz domains should
     include zero dimensions to verify that execution succeeds
     (`RunResult::kSuccess`), allocates without memory leaks, and completes
     cleanly.
   - If an operator does not support empty tensors along specific non-batch
     dimensions (e.g., filter spatial dimensions or channel counts of zero),
     the kernel must reject them cleanly (`RunResult::kRejected`) during
     `Prepare` or `Invoke`.
   - In all cases, zero-element tensors must **never** cause undefined
     behavior, integer division by zero in stride/offset calculations,
     undersized buffer allocations, memory safety violations, or crashes.

## Designing a per-operator fuzzer

### 1. State the contract

Decide whether the property covers loading, preparation, invocation, or a
specific implementation. Identify which fields are controlled by the model and
which values can arrive dynamically at invocation time. Checks for constant
parameter tensors may run during `Prepare`; dynamic parameter values must also
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

- use `Map()` to derive values that must be related;
- use small dimension ranges for normal execution;
- include meaningful boundaries explicitly with `OneOf()` and `Just()`;
- include constant and dynamic forms of shape, axis, padding, or index tensors;
- include zero-element tensors when the operator contract permits them (see
  [Empty tensor handling](#empty-tensor-handling));
- include negative axes or sentinel values only where the contract defines
  them; and
- keep malformed transformations narrow and named.

Cover the current WebNN rank boundary (currently rank 8 for untrusted tensors)
and any stricter limit imposed by a real implementation. Higher-rank cases are
useful robustness coverage, but they are not WebNN security cases unless a
supported untrusted entry point can reach them. Do not add a broad rank limit
to a built-in kernel merely to match one delegate or integration.

Arbitrary byte overlays are useful for tensor values after a valid shape and
storage size have been established. Apply central representation invariants,
such as canonicalizing Boolean values to `0` or `1`. Do not let a byte overlay
accidentally change the semantic property being tested.

### 3. Keep each iteration small

Per-op fuzzers are coverage tools, not bulk allocation stress tests. Small
tensors allow many more iterations per second and reach more control-flow
states. Use checked arithmetic before materializing any tensor and impose an
explicit element or byte limit.

Every one-op run must set `max_live_allocation_bytes`. Real-world environments
handling models from untrusted sources are expected to deploy with a
quota-enabled allocator to defend against memory exhaustion. Setting this limit
directly exercises that production defense while protecting developer and CI
machines.

Because the quota allocator bounds live memory, the primary security concerns to
fuzz around allocation are:

- **Integer overflow and 32-bit truncation before the allocator:** The most
  dangerous allocation vulnerability occurs when large dimensions wrap around in
  32-bit integer arithmetic (e.g., `shape_product * sizeof(T)` overflowing `int32_t`
  or `size_t`). If an overflowing calculation produces an artificially small byte
  count, the request will pass the quota check, allocate an undersized buffer,
  and lead to severe heap out-of-bounds access during kernel execution. Always
  exercise 32-bit-sensitive boundaries (such as dimension products near $2^{31}-1$
  or $2^{32}-1$) to ensure overflow is detected before allocation.
- **Graceful unwinding on allocation failure:** When the quota allocator refuses
  an allocation (returning `nullptr` or an error status), every backend must
  check for failure and cleanly unwind without null-pointer dereferences, memory
  leaks, or double-frees during error cleanup.
- **Unmanaged allocations:** Verify that delegate backends and kernels do not
  bypass the quota allocator with untracked `malloc`, `new`, or external driver
  allocations.

For huge-dimension arithmetic boundaries, fuzz a small internal shape helper
directly when possible. If an end-to-end model is necessary, avoid populating or
invoking tensors whose valid allocation would exceed the test budget.

Exercise 32-bit-sensitive boundaries even when developing on a 64-bit host.
Element counts and byte counts may use `size_t`, while dimensions and many
kernel indexes use `int`. Promote before arithmetic, check every shape product,
and validate each narrowing conversion. A zero total element count must not
hide an invalid or overflowing nonzero dimension, stride, or offset.

### 4. Preserve real model invariants

Build constants with aligned FlatBuffer buffers and make runtime buffers match
their tensor's exact byte size. Use `MakeValues()`, `MakeIntegerValues()`, and
`StorageBytesForElements()` instead of typed access to arbitrary unaligned
bytes. Packed types such as `INT4` require their packed storage size rather than
`element_count * sizeof(T)`.

Run the model verifier before interpreter construction. A per-op fuzzer may
generate models directly and use `tflite::Verify(...)` as the global ingress
check; it does not need to invoke a converter or WebNN lowering pipeline.

Valid-case domains should satisfy the static invariants assigned to model
generation and ingress validation by the threat model. Malformed properties
may violate those invariants, but their results must be described as clean
rejection or general robustness coverage rather than automatically as a
WebNN-reachable security finding.

### 5. Cover execution implementations explicitly

Reference, optimized, multithreaded, and delegated implementations (e.g.,
XNNPACK, MLDrift) can have different scratch buffers, integer ranges, and
indexing code. Cover every registered implementation that is relevant to the
operator. Using shared semantic case types and the unified `kSuccess`/`kRejected`
contract ensures test properties remain directly reusable across CPU kernels and
delegate backends. Put the variant in the fuzz case/domain when practical, or use
clearly named properties over the same semantic domains so coverage gaps remain
visible.

Delegate-enabled targets are separate Bazel targets in this directory. Apply a
delegate in `pre_allocate`, because normal delegated inference installs the
delegate before `AllocateTensors()`. A delegate property must also inspect the
execution plan and prove that a delegate node was created. Successful fallback
to the built-in kernel is not delegate coverage.

Delegate domains may be narrower than built-in domains when the delegate has a
real documented constraint. Keep those constraints local to the delegate
property; do not impose them on the built-in operator contract.

### 6. Add smoke tests and seeds

Add ordinary `TEST` cases for important paths, especially:

- proof that invocation actually occurs;
- proof that a delegate target really delegates;
- constant and dynamic parameter tensors;
- zero-element behavior;
- the highest supported rank or a meaningful rank boundary; and
- deterministic regressions found by fuzzing.

Seed coverage-guided properties with known boundary shapes when mutation is
unlikely to discover them efficiently. A small stable reproducer for a bug
should also become a normal kernel unit test near the affected implementation.

### 7. Keep expected errors quiet

Malformed properties can reject thousands of inputs per second. The shared
`SilentErrorReporter` suppresses those expected diagnostics so they do not
flood stderr or dominate execution time. When debugging locally, temporarily
restore printing in that reporter or add a focused trace around the failing
case. Do not enable routine error printing in committed coverage-guided runs.

## What counts as a fuzzing failure?

Treat these outcomes as bugs and investigate their reachability:

- a crash, sanitizer finding, out-of-bounds access, use-after-free, memory leak,
  or null dereference;
- undefined behavior in shape, allocation, index, offset, stride, loop-bound,
  or control-flow arithmetic;
- a valid property returning `kRejected`;
- a malformed property returning `kSuccess`;
- `kHarnessFailure` from any generated case;
- a path advertised as delegated that did not actually delegate; or
- inconsistent validation between CPU implementations when the difference can
  lead to unsafe execution.

Integer overflow in ordinary tensor-value arithmetic is normally a correctness
issue rather than a security issue when the value cannot influence allocation,
indexing, loop bounds, or control flow. Likewise, numerical precision and
NaN/Inf differences are not memory-safety findings by themselves. Follow the
threat model's triage guide instead of classifying a result solely because a
sanitizer reported it.

## Building and running

Run a target in its short GoogleTest/FuzzTest seed mode while iterating:

```bash
bazelisk test //third_party/tensorflow/lite/testing/fuzzing:pad_fuzz_test
```

Run one property in coverage-guided mode:

```bash
bazelisk run //third_party/tensorflow/lite/testing/fuzzing:pad_fuzz_test -- \
  --fuzz=PadFuzzTest.PadExecutesValidCases \
  --fuzz_for=5m
```

Delegate targets have names such as `pad_xnnpack_fuzz_test`,
`reduce_xnnpack_fuzz_test`, `reshape_xnnpack_fuzz_test`, and
`slice_xnnpack_fuzz_test`.

On Windows, use the repository's `windows_fuzztest` Bazel configuration and run
one property per process:

```powershell
bazelisk run -c fastbuild --config=windows_fuzztest `
  //third_party/tensorflow/lite/testing/fuzzing:pad_fuzz_test -- `
  --fuzz=PadFuzzTest.PadExecutesValidCases `
  --fuzz_for=5m
```

Coverage-guided fuzzing and local regression runs should usually have
AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan) enabled (e.g.,
passing `--config=asan` and `--config=ubsan`, or `--config=asan-ubsan` depending
on your Bazel configuration). Running with ASan and UBSan is critical for
catching subtle heap out-of-bounds reads/writes, use-after-free bugs, integer
overflows in pointer calculations, and unhandled allocation failures that might
not trigger an immediate crash in uninstrumented binaries. CI runs fuzz tests in
short smoke mode; a local or scheduled fuzzing campaign must run long enough to
exercise the structured domain and important implementation paths.

## New-fuzzer checklist

Before submitting a new per-op fuzzer, verify that:

- it uses Google FuzzTest and does not introduce another fuzzing framework;
- it uses property-based testing with separate valid and malformed domains to
  achieve deep code coverage;
- the tested loading/kernel/delegate contract is clear;
- valid and malformed cases use separate domains and exact outcomes;
- malformed domains violate a specific documented invariant;
- both constant and dynamic parameter paths are covered where applicable;
- all materialization and expected-outcome arithmetic is checked;
- tensor sizes and live allocations are bounded;
- 32-bit overflow and checked-narrowing boundaries are represented;
- model, constant, and runtime buffers satisfy alignment and representation
  requirements;
- empty tensor (zero-element) handling conforms to the delegate-rejection and
  CPU backward-compatibility contracts;
- relevant reference, optimized, multithreaded, and delegate (e.g.,
  XNNPACK, MLDrift) paths are observable and covered;
- smoke tests prove that intended deep paths are reached;
- ASan and UBSan have been enabled for a meaningful local run; and
- stable fuzz-found bugs have focused ordinary unit-test regressions.
