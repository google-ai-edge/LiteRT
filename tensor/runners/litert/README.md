<!--
Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LiteRT compiled-model runners

This directory connects the [Tensor API](../../README.md) to LiteRT's
[`CompiledModel`](../../../litert/cc/litert_compiled_model.h) runtime. It supports
both graphs authored with `Tensor<TfLiteMixinTag>` and existing TFLite models,
with named inputs/outputs and shared `TensorBuffer` storage between stages.
Accelerator selection and compilation are controlled by the caller's LiteRT
`Environment` and `Options`. See the [runner overview](../README.md) for the
other execution paths.

## Components and runner choices

All implementation in this package is in headers; [BUILD](BUILD) declares the
corresponding libraries and tests.

| File | Responsibility and typical use |
| --- | --- |
| [lambda_model_runner.h](lambda_model_runner.h) | `CreateLambdaRunner` builds a graph from a map of input prototypes and a lambda returning named outputs. `CreateStaticRunner` accepts already-authored input/output maps. Both use `CompiledModelRunner` and expose input/output binding, synchronous execution, and reset. |
| [compiled_model_runner.h](compiled_model_runner.h) | Templated graph-to-runtime implementation. Evaluates a model functor, serializes its output graph, creates a LiteRT compiled model and I/O buffers, and supports typed data access, feedback, and intermediate graph probes. Its convenience I/O methods operate on signature 0. |
| [litert_dynamic_runner.h](litert_dynamic_runner.h) | `LitertDynamicRunner::Create` loads a model file or serialized byte span. Allocates buffers for every signature and provides name/index access and execution per signature. Overloads without a signature use the first signature. It also exposes existing WebGPU buffer handles. |
| [litert_buffer.h](litert_buffer.h) | `LitertBuffer` adapts a reference-counted LiteRT `TensorBuffer` to the Tensor API's `Buffer` interface, including scoped read/write locks and runtime type checks. |
| [feedback_loop_config.h](feedback_loop_config.h) | A pair of input/output names identifying state carried from one invocation to the next. |
| [lambda_model_runner_test.cc](lambda_model_runner_test.cc) | CPU tests for lambda/static graph construction, loading from memory, binary inputs, feedback buffer identity, and reset. |
| [litert_buffer_test.cc](litert_buffer_test.cc) | Host-buffer read/write locking, packed byte size, and buffer type conversion tests. |

`LitertDynamicRunner` loads models dynamically; it does not expose input
resizing or automatic buffer reallocation for new shapes. Its buffers are
created from the compiled signatures during initialization. Graph authoring
and TFLite operator conversion live in
[`backends/tflite`](../../backends/tflite/), outside this package.

## Construction and execution

For an authored graph, the lambda/static factories adapt named tensor maps to
`CompiledModelRunner`'s `Inputs`/`Outputs` interface. The model functor runs
once during construction to author the graph. Output map keys become tensor
names, while input prototypes must carry the intended input names themselves.
[`ModelFactory`](../../backends/tflite/tflite_flatbuffer_conversion.h) traverses
the output graph, emits a TFLite FlatBuffer, and supplies it to
`CompiledModel::Create`. The runner retains the serialized model bytes and
releases its saved output graph after allocating runtime buffers.

After construction, the usual sequence is `SetInput`, `Run`, then `GetOutput`.
`Run` invokes the compiled model synchronously; it does not rerun the authoring
lambda. The dynamic runner follows the same execution sequence after loading
an existing model, and keeps separate buffer vectors and feedback state for
each signature.

Automatic graph compilation uses `ABSL_CHECK_OK`, so construction failures
terminate rather than returning a status. For explicit error handling or
probing, construct `CompiledModelRunner` with `build_model_now=false` and
finish compilation before using runtime methods. `BuildModel` needs the
desired output tensors; its default empty list does not recover the saved
outputs automatically.

`AddTensorsAsOutputs` is a pre-build probe path: it uses
[`GraphProbe::StableTensorId`](../../internal/graph_probe.h) to find graph
outputs, inserts `Probe` operations for selected non-leaf tensors, and builds
the model. A nonempty selection becomes the compiled output set; an empty
selection uses the original outputs. Call it before the first build, while
the saved graph still exists.

## Buffers and lifetimes

`GetInput` and `GetOutput` return handles backed by `LitertBuffer`. Duplicating
a LiteRT buffer shares its underlying storage; it does not snapshot tensor
data. A later invocation can overwrite that storage. Read results before
reuse, or copy them when persistent values are needed. The lower-level
compiled runner also offers copying float, int32, and bool output getters.
Handle metadata conversion recognizes FP32, int32, int8, and bool, except that
the compiled runner's `GetInput` currently omits int8. Other element types are
reported as `Type::kUnknown`; copying bytes does not extend this mapping.

| Binding API | Current behavior |
| --- | --- |
| `SetInput(name, TensorHandle)` | Shares storage when the source is a `LitertBuffer`; otherwise locks the source and copies its bytes into the current input buffer. Both runner implementations support this. |
| `SetOutput(name, TensorHandle)` | Replaces the output binding with shared `LitertBuffer` storage. Available on the compiled/lambda runner; it requires a `LitertBuffer` source. |
| Compiled/lambda `SetInput(name, Span<const std::byte>)` | Temporarily wraps sufficiently aligned host memory when supported, otherwise copies the input. |
| Compiled/lambda `SetOutput(name, Span<std::byte>)` | Attempts temporary host-memory wrapping, otherwise schedules a copy into the span after execution. The current host-support helper queries **input** buffer requirements by name, including for outputs, so output wrapping is not guaranteed. |
| Dynamic `SetInput(..., Span<const uint8_t>)` | Copies bytes into the selected input buffer. |

Shared `TensorHandle` bindings persist until replaced or swapped by feedback.
For one temporary byte-span binding per slot outside feedback, a **successful**
compiled runner `Run` restores the saved bindings and performs output copy-back.
Keep external memory valid through that operation. Avoid rebinding a temporary
slot before running or using temporary spans on feedback slots: restoration
processes saved bindings in insertion order, and feedback swaps can move a
temporary binding to another slot. Those combinations can leave external
storage installed even after success. An early run/copy error can also leave
pending bindings installed; `Reset` does not clean them up.

Host wrapping checks 64-byte alignment, but callers must also satisfy the underlying
[`TensorBuffer` requirements](../../../litert/cc/litert_tensor_buffer.h),
including any required padding and accelerator-specific constraints.

Keep the `Environment` alive for the runners and buffers that use it.
`CompiledModelRunner` also stores its `Options` by reference. The dynamic
runner's byte-span factory borrows serialized model memory, which must remain
valid for the model's lifetime under the
[model-buffer API contract](../../../litert/c/litert_model.h).

`LitertBuffer::ByteSize()` reports packed bytes, matching the span exposed by
`Lock()`/`LockMutable()` rather than padded allocation size. Each locked span
retains a duplicate buffer handle and unlocks it on destruction. Lock failures
use checked assertions. Release locks before execution; sharing a buffer does
not provide scheduling or synchronization between independently used runners.
The WebGPU accessors return existing handles without creating new buffers.

## Feedback and pipeline wiring

Register feedback before starting a session. The first invocation reads the
initial input. Before every subsequent invocation, the runner swaps each
configured input/output buffer pair, so the previous output becomes the next
input without a tensor-data copy. `Reset` restores the original buffer
orientation and first-run flag; it does **not** zero or reinitialize state.
Set the next session's initial inputs after reset.

Registration resolves names but does not validate matching shapes, element
types, storage compatibility, or overlapping feedback pairs. The caller must
provide compatible, unambiguous pairs. Dynamic factory feedback applies to
the default signature; `RegisterFeedbackLoop(signature, input, output)`
supports another signature. Feedback constructors also configure GPU external
tensor patterns before compilation when GPU options are available. Registering
a loop after compilation does not perform that setup.

For a pipeline without feedback, an authored stage can write directly into a
loaded model's input buffer. Given initialized runners with compatible tensor
types, layouts, and backend buffer requirements, this fragment binds the
storage and executes the stages in order:

```cpp
LITERT_ASSIGN_OR_RETURN(auto core_input, core_runner.GetInput("input"));
LITERT_RETURN_IF_ERROR(pre_runner.SetOutput("processed", core_input));
LITERT_RETURN_IF_ERROR(pre_runner.SetInput("raw", raw_tensor));
LITERT_RETURN_IF_ERROR(pre_runner.Run());
LITERT_RETURN_IF_ERROR(core_runner.Run());
```

The fragment belongs in a function returning a compatible status type. See
[the unit tests](lambda_model_runner_test.cc) for complete CPU construction
and feedback examples. Because feedback changes buffer identity, reacquire
handles when following the current input/output binding across invocations.

## Build, tests, and changes

Use the repository's
[build instructions](../../../g3doc/instructions/BUILD_INSTRUCTIONS.md) and the
[runner test commands](../README.md#build-and-test).
Library labels match the five header basenames, for example
`//tensor/runners/litert:lambda_model_runner` and
`//tensor/runners/litert:litert_dynamic_runner`. The supported test labels are:

- `//tensor/runners/litert:lambda_model_runner_test`
- `//tensor/runners/litert:litert_buffer_test`

These tests require the LiteRT runtime and are outside the native-XNNPACK-only
[standalone build](../../standalone/README.md). The current local tests cover
CPU behavior; they do not establish GPU/WebGPU behavior, multi-signature
feedback, or temporary external-span error recovery.

Change the lambda adapter for graph-authoring ergonomics, the compiled runner
for graph serialization/probes and authored-model binding, and the dynamic
runner for loaded-model/signature behavior. Changes to buffer ownership or
feedback usually need corresponding updates in both runner implementations
and regression coverage in `lambda_model_runner_test.cc`. Add storage/locking
behavior in `LitertBuffer` and its tests; add operator lowering in the TFLite
backend rather than in these execution wrappers.
