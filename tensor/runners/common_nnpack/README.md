# Shared NNPACK runner

`litert::tensor::NnpackRunner` provides the input/output API, buffer management,
and execution lifecycle shared by NNPACK-style runtime adapters. It owns an
already constructed graph and calls backend hooks to reshape, bind, and execute
it. [XnnpackRunner](../xnnpack/README.md) is the concrete implementation in this
tree. Graph construction and operator lowering belong to
[the tensor backends](../../backends/README.md).

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| [runner.h](runner.h) | Public input/output API, typed convenience overloads, graph/buffer accessors, and the abstract backend contract. |
| [runner.cc](runner.cc) | Validation, allocation and growth of external buffers, lazy preparation, and execution order. |
| [runner_test.cc](runner_test.cc) | Tests the shared implementation with an identity backend and injectable errors. |
| [runner_test_suite.h](runner_test_suite.h) | Reusable Google Test typed suite that exercises real tensor expressions through concrete runners. |
| [BUILD](BUILD) | `:runner`, `:runner_test`, and the test-only `:runner_test_suite` library. |

## Graph and runtime ownership

The constructor takes a `std::unique_ptr<NnpackGraph>`. The
[graph](../../backends/common_nnpack/graph.h) holds value metadata, backend IDs,
external-input/output flags, tensor lookup information, and constant storage.
The runner separately retains external buffers in a map from backend value ID
to `std::shared_ptr<Buffer>`. A concrete subclass owns and releases the backend
runtime and any execution resources.

Lookup follows `TensorHandle` → `NnpackGraph::Lookup()` → value index →
`NnpackValue::id` → external buffer. A value index and a backend ID are not
interchangeable. Lookup uses the underlying graph tensor's identity; creating
a different tensor with the same name does not identify the original input.

`graph()` and `external_buffers()` expose this state for inspection. Their
mutable counterparts bypass setter validation; changes do not automatically
rebuild a prepared runtime. The runner is movable and non-copyable. It has no
internal synchronization around its graph, buffer map, or execution state, so
serialize calls on a runner.

## Execution lifecycle

`PrepareRuntime()` calls `CreateRuntime(num_threads_)` once after a successful
creation. A failure leaves preparation retryable. `Run()` calls it lazily, so
explicit preparation is useful when runtime creation should happen before the
first invocation. Preparation alone does not bind buffers or propagate shapes.

Every `Run()` then performs the same sequence:

1. Convert each external input's current shape to backend dimensions, set the
   runtime input shape, require a bound buffer, and check its capacity.
2. Call `ReshapeRuntime()` to propagate input shapes through the graph.
3. Query each external output's shape, update its runner metadata, and reserve
   output storage. Missing outputs receive owning CPU buffers automatically.
4. Call `SetupExternalValues()` to bind the current buffers and collect locks.
5. Call `InvokeRuntime()` while those locks remain alive.

The first non-OK status ends the call. Later invocations reuse the prepared
runtime but repeat reshaping and binding. A failure can occur after metadata or
buffers have changed; execution is not transactional.

The base `SetNumThreads()` stores a count, initially one, for runtime creation.
It does not invalidate an existing runtime. Configure threads and backend
options before preparation, and consult the concrete runner's rules for any
changes after preparation.

## Inputs, outputs, and buffer lifetimes

| Workflow | API and behavior |
| --- | --- |
| Borrow input memory | `SetInput(tensor, bytes)` retains a read-only view with an exact byte count for the current runner shape. The mutable-byte overload also creates a read-only view. The sequence overload additionally checks the element type and rejects rvalue sequences. |
| Copy input memory | `SetInput(tensor, bytes, true)` or `SetInputAsCopy(tensor, sequence)` creates owning CPU storage. The sequence form checks the element type. |
| Bind another tensor's storage | `SetInput(tensor, external_tensor)` checks types and source-buffer presence, adopts the source shape, and shares its buffer when it has enough capacity. If an owning source buffer is too small, the runner instead allocates a larger buffer and copies the old bytes; the source tensor itself is unchanged. |
| Supply output storage | `SetOutput(tensor, bytes)` retains a writable view whose size must exactly match the currently recorded output shape. Omit it to allow automatic output allocation. |
| Change input shape | `ReshapeInput(tensor, shape)` updates runner metadata and grows an already bound owning input if needed. The next `Run()` propagates the shape and checks non-owning capacity. |
| Update part of an input | `WriteInput(tensor, offset_bytes, data)` requires a bound mutable buffer and a write within its capacity. Typed overloads convert to bytes; the offset is always in bytes and these overloads do not check the tensor element type. |
| Read an output | `ReadOutput()` returns a locked byte view of the bound output. `ReadOutputAs<T>()` also checks the requested element type. Neither method invokes the runtime. |

For a complete tensor-expression usage example, see the
[XNNPACK runner guide](../xnnpack/README.md).

### Shapes and allocation

The private `Reserve()` helper allocates missing storage, keeps sufficient
storage, and replaces an undersized `OwningCpuBuffer`. It cannot resize a
non-owning view or another buffer implementation. It never shrinks allocations.

`ReshapeInput()` changes the graph metadata stored by the runner, not the
original tensor handle. When it grows an owning input, it does **not** preserve
the old contents; initialize the entire new logical input before running.
For borrowed inputs, reshape and then bind a view with the new exact size.
Shape validity is not fully checked by the setter; for example, conversion to
runtime dimensions rejects negative dimensions during `Run()`.

Output shapes become current after runtime reshaping. A previously supplied
output view cannot grow, so an output that exceeds its capacity causes `Run()`
to fail. Automatic output allocation is simpler for changing output shapes.
After a shrink, the buffer may still have its previous capacity, and
`ReadOutput()` returns that whole buffer. Use the output's current
`graph().values()[index].info` to determine its logical shape and size.

### Locks and lifetimes

[Buffer](../../buffer.h) distinguishes ownership from CPU accessibility.
`LockedBufferSpan` keeps an access lock alive; its ownership behavior depends on
the buffer implementation:

- An `OwningCpuBuffer` lock retains the allocation, including when the runner
  later replaces that buffer. It is still a view of mutable storage, so another
  invocation can overwrite it while the runner reuses the same allocation.
- A `SpanCpuBuffer` or `MutableSpanCpuBuffer` lock is a no-op view. The caller
  must keep the underlying memory valid; retaining the wrapper or lock does
  not extend that memory's lifetime. Other buffer implementations may also
  provide access without owning the data.

Keep a named lock alive while using pointers obtained from it. Copy results
that must remain independent of later invocations. To use `WriteInput()`, bind
an owning copy or attach a `MutableSpanCpuBuffer` through an external tensor;
ordinary span/sequence input binding is read-only.

Unknown handles report `NotFound`. Wrong roles, incompatible types or sizes,
immutable writes, and undersized views report `InvalidArgument`. Operations
that require an absent input/output buffer report `FailedPrecondition`.
Backend errors propagate without translation by the shared runner.

## Backend implementation contract

A subclass supplies role bits through `FlagExternalInput()` and
`FlagExternalOutput()`, matching those used by its graph builder. Its
`CreateRuntime()` owns runtime creation, including cleanup or replacement of
partial state if creation is retried.

The remaining hooks implement the execution sequence:
`SetExternalValueShape()`, `ReshapeRuntime()`, `GetExternalValueShape()`,
`SetupExternalValues()`, and `InvokeRuntime()`. Shape hooks use backend IDs.
Setup receives all graph values and must select the external ones, bind their
buffers, and retain the necessary locks in the supplied vector or other
storage that lasts through invocation. Invocation must finish accessing those
buffers before returning because `Run()` then releases its local locks.

Use [the XNNPACK adapter](../xnnpack/runner.cc) as the production example and
`TestRunner` in [runner_test.cc](runner_test.cc) as a small identity backend.
Keep shared buffer policy and execution ordering here, backend API calls in
the concrete runner, and operation conversion in the backend graph code.

## Tests and builds

`runner_test.cc` checks copy/view behavior, binding from another tensor,
mutable writes, shape changes, output allocation and reads, validation errors,
move construction, and failure propagation from every execution hook. Add
backend-independent buffer and lifecycle cases there.

The typed suite in `runner_test_suite.h` covers representative FP32 arithmetic,
activations, convolutions, fully connected and batch matrix multiplication,
shape operations, reductions, resize, constants, runtime inputs, moves, and
buffer growth/errors. Its traits provide `Tag`, `Runner`, external role flags,
and support flags for convolution variants and resize. See
[XnnpackTestTraits](../xnnpack/runner_test.cc) for an instantiation. Register new
shared cases in `REGISTER_TYPED_TEST_SUITE_P` as well as defining the test.
Check the individual capability gates: nearest-neighbor rejection cases run
even when successful resize cases are disabled.

From the LiteRT repository root, following the repository's
[build environment instructions](../../../g3doc/instructions/BUILD_INSTRUCTIONS.md):

```bash
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" test \
  //tensor/runners/common_nnpack:runner_test \
  //tensor/runners/xnnpack:runner_test \
  --test_output=errors
```

The implementation library is `//tensor/runners/common_nnpack:runner`.
`:runner_test_suite` contains test definitions rather than an executable; the
XNNPACK test target instantiates them. For standalone Linux unit tests and
Android cross-compilation/device execution using `ANDROID_HOME`, use the
[tensor build and test guide](../../standalone/README.md), also linked from the
[top-level tensor README](../../README.md).
