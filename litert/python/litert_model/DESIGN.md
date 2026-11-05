# litert_model: design

`litert_model` is a Python LiteRT IR backed by the C++ model IR
(`LiteRtModelT`).
A model is loaded once into C++, and every query and edit works on that
in-memory IR. Flatbuffers are only parsed by `Model.load_*` and written by
`Model.serialize`. Python compiler plugins and tools use it to inspect and
rewrite models without parsing or rebuilding TFLite flatbuffers themselves.

## Where it fits

```mermaid
flowchart LR
  tflite[".tflite file or bytes"] --> load["Model.load_from_file / load_from_bytes"]
  load --> ir["litert_model.Model (owns LiteRtModelT)"]
  ir --> edit["Python passes: inspect, edit, infer shapes"]
  edit --> outline["outline_to_dispatch_op + register_buffer + attach_asset"]
  outline --> ser["Model.serialize(bytecode_alignment)"]
  ser --> out[".tflite with appended assets"]
```

Upstream, the package reuses the LiteRT C++ core: `litert/core/model/`
(`model.h`, `model_load.h`, `model_serialize.h`, `shape_inference.h`,
`graph_validation.h`), `litert/core/build_stamp.h` and
`litert/core/dispatch_op_schema.h`. Downstream clients are Python compiler
plugins (for example vendor backends under `litert/python/aot/vendors/`)
and model tools. The dispatch helpers follow the same steps as the C++
`ApplyPluginWithPartition` in `litert/compiler/plugin/compiler_plugin.h`.

## Layout

| File | Role |
|---|---|
| `litert_model.py` | Public Python API: `Model`, `Subgraph`, `Op`, `Tensor`, `Signature`, `ElementType`, `OpCode`, quantization dataclasses, `DispatchOpOptions`, built-in options round trip. |
| `_pywrap_litert_model.cc` | pybind11 module `_pywrap_litert_model`: thin handle classes, argument checks, error translation. Internal API. |
| `_pywrap_litert_model.pyi` | Type stub of the pybind module, used for strict type checking of `litert_model.py`. |
| `pywrap_handles.h` | Ownership model: `ModelState`, the `PySubgraph`/`PyOp`/`PyTensor`/`PySignature` handles, `IrError`, `FreedObjectTracker`. |
| `litert_model_lib.h`, `litert_model_lib.cc` | Graph editing, validation and dispatch helpers on `LiteRtModelT`. Errors are returned as `litert::Expected`; no exceptions and no pybind11. |
| `litert_model_lib_test.cc` | C++ unit tests of `litert_model_lib`. |
| `litert_model_test.py` | Python API tests: enums, loading, def-use, weights, quantization, built-in options, mutations, stale handles, serialization, subgraph pruning. |
| `litert_model_dispatch_test.py` | Shape inference, specialization, outlining, assets, build stamp, and a compile-flow stress test. |
| `litert_model_numerics_test.py` | Runs edited models with the LiteRT compiled-model runtime and checks the numbers. |
| `BUILD` | `litert_model_lib` (C++), `_pywrap_litert_model` (pybind extension), `litert_model` (Python library), and the tests. |

## Key concepts and data flow

**Three layers.** `litert_model.py` wraps every pybind object in a Python class
(`Subgraph(h)`, `Op(h)`, ...) and adds the Python-only parts: enums, numpy
conversion, quantization dataclasses, and built-in options packing through the
TFLite schema object API (`schema_py_generated`). `_pywrap_litert_model.cc`
checks arguments and handles, calls into C++, and turns failures into Python
exceptions. `litert_model_lib` holds the graph logic so it can be unit tested in
C++ without Python.

**Ownership.** One `ModelState` owns one `std::unique_ptr<LiteRtModelT>`.
`PyModel` and every handle hold a `std::shared_ptr<ModelState>`, so the model
lives as long as any handle or weight view (the `_WeightsView` buffer object
holds the state too). A handle stores the raw IR pointer plus a registration
id from `ModelState::Register`. `Get()` on a handle checks two things first:
the model was not serialized, and `(pointer, id)` is still registered.
Handles of the same object compare and hash equal, so they work as dict keys.

**Invalidation.** Bindings that deallocate IR objects tell the state to
forget them: `remove_op`, `remove_tensor`, `remove_signature`, `dce` and
`outline_to_dispatch_op` (through `FreedObjectTracker`), and
`prune_unreachable_subgraphs` (which compares all subgraphs, ops and tensors
before and after). A stale handle raises `LiteRtError("... has been removed
from the model")`. Because ids are never reused, a new object at a recycled
address does not match an old handle.

**Consumed model.** `Model.serialize` first runs
`ValidateModel(kSerializable)`. On failure it raises `ValidationError` and the
model stays usable. Otherwise it marks the state serialized, then moves the
model into `internal::SerializeModel`. After that, every access to the model
or its handles raises `LiteRtError("model has been serialized")`, even if
serialization failed. The `LiteRtModelT` object stays allocated, so weight
views taken earlier remain readable.

**Buffers and weights.** `set_weights` and `register_buffer` copy the data
into `ModelState::KeepBuffer` (a `std::list` that never moves its elements)
and register it with the model's `BufferManager` as a non-owned buffer.
`BufferManager` keeps owned buffers in a vector that copies and frees them
when it grows, which would leave zero-copy views dangling. Every copy is kept
until the model is destroyed, including copies that a later `set_weights`
replaced. `Tensor.weights` and `weights_bytes` are zero-copy, read-only views
when the data is a kept buffer or lies inside the flatbuffer the model was
loaded from. Otherwise they return a copy.

**Main types.**

- `Model`: subgraphs, signatures, metadata, `validate`, `serialize`,
  `infer_shapes`, `specialize_subgraph`, `prune_unreachable_subgraphs`,
  buffers and assets, build stamp.
- `Subgraph`: inputs and outputs, tensors, ops in topological order,
  `add_tensor`, `add_op`, `remove_op`, `remove_tensor`, `dce`,
  `replace_all_uses`, `outline_to_dispatch_op`.
- `Op`: code, custom code and options, inputs and outputs, `set_input`,
  `set_output`, `builtin_options` (a fresh `schema_fb.*OptionsT` copy; call
  `set_builtin_options` to apply changes), dispatch queries.
- `Tensor`: name, shape (`-1` for dynamic dims, `()` for scalars, `None` if
  unranked), element type, quantization, weights, `defining_op`, `users`.
- `Signature`: key, subgraph, names and tensors of the inputs and outputs. A
  model without signatures gets a placeholder signature from the loader, with
  key `DEFAULT_SIGNATURE_KEY` (`"<placeholder signature>"`), and it is
  serialized like any other signature.
- Quantization: `PerTensorQuantization`, `PerChannelQuantization` (read-only
  numpy `scales` and `zero_points`), and `BlockWiseQuantization` (scales and
  zero points are tensors of the same subgraph).
- Enums: `ElementType` and `OpCode` mirror `LiteRtElementType` and
  `LiteRtOpCode`. Op codes of a loaded model that LiteRT does not know yet
  appear as `UNKNOWN_<value>`.

**Shape inference.** `Model.infer_shapes`, `Subgraph.infer_shapes` and
`Model.specialize_subgraph` call `ShapeInferenceEngine` from
`litert/core/model/shape_inference.h`. Failures raise `ShapeInferenceError`
with `status` and `op` (the failing `Op`, when known). The Python layer
appends the op name to the message. `specialize_subgraph` appends a copy of
the subgraph with static input shapes. No signature refers to the copy, and
it is dropped again if inference fails.

**Serialization and dispatch ops.** `outline_to_dispatch_op` replaces all
ops of a subgraph with one `CUSTOM` op whose custom code is
`DISPATCH_OP_CUSTOM_CODE`. The op reads the subgraph inputs and writes the
subgraph outputs. Each output position gets its own dispatch output: a
constant, a forwarded input or a repeated tensor is replaced by a fresh
tensor with the same name, type and quantization. Signature outputs are
remapped by position. `register_buffer` returns a buffer id, and
`attach_asset(op, buffer_id, name)` attaches it to a dispatch op as entry
point `name`. `serialize(bytecode_alignment=N)` appends each asset once,
aligned to `N`, and writes its offset and size into the op's custom options.
`Op.dispatch_options()` reads them back, but only on a model that was
serialized and loaded again. `set_build_stamp` stores the target SoC
manufacturer and model under `BUILD_STAMP_METADATA_KEY`.

**Use by compiler plugins.** A Python vendor compiler plugin typically loads
or receives a model, lowers each signature's subgraph to its own bytecode,
then calls `outline_to_dispatch_op`, `register_buffer` and `attach_asset` for
each subgraph. It may call `prune_unreachable_subgraphs` to drop
decomposition subgraphs that are no longer referenced, calls
`set_build_stamp`, and finally calls `serialize`.
`DispatchTest.test_compile_flow_stress_with_live_views` exercises this flow.

## Design decisions and invariants

- **Edits keep the graph consistent.** Every mutation in `litert_model_lib`
  keeps op inputs and outputs consistent with tensor users and defining ops,
  and refuses edits that would put `Subgraph.ops` out of topological order.
  Transient states, such as a tensor that is used but not produced yet, are
  allowed. `validate` reports them.
- **Two validation levels.** `kSerializable`, used by `serialize`, checks
  only what `SerializeModel` needs, so everything the C++ loader produces
  serializes. `kFull`, used by `validate`, also checks topological order,
  that every used tensor has a source, ranks and unique signature keys.
  Signature names may repeat, because the placeholder signature repeats the
  name of a repeated output.
- **Error mapping.** Edit preconditions raise Python built-ins:
  `kLiteRtStatusErrorIndexOOB` becomes `IndexError`, and
  `InvalidArgument`, `InvalidGraphInvariant`, `NotFound` and `AlreadyExists`
  become `ValueError` (`ThrowEditError`). Missing signature keys raise
  `KeyError`. Other C++ failures, stale handles and use after `serialize`
  raise `LiteRtError` (a `RuntimeError` with `status` and `status_code`).
  Its subclasses are `ShapeInferenceError` and `ValidationError`.
- **No omitted inputs in the middle.** The C++ loader drops TFLite operand
  index `-1`, so `Op.inputs` never contains `None` and the inputs after an
  omitted one shift left. `add_op` accepts `None` only as trailing inputs,
  which it drops.
- **Operator code table.** `GetOrAddTflOpCodeIndex` reuses the
  lowest-version matching entry or appends a version 1 entry. `BROADCAST_TO`
  gets version 2, because the TFLite built-in op resolver registers it from
  version 2.
- **Enum lists in three places.** The C enums
  (`litert/c/litert_model_types.h`, `litert/c/litert_op_code.h`), the X-macro
  lists in `_pywrap_litert_model.cc` (linked to the C enums by `IfChange` /
  `ThenChange` lint comments), and the Python `ElementType`/`OpCode` enums.
  `EnumTest` fails if they drift.
- **`load_from_bytes` verifies first.** It copies the data and verifies the
  flatbuffer before `LoadModelFromBuffer`, which aborts on invalid input.

## Extending

To expose a new IR capability:

1. Add an `Expected`-returning helper to `litert_model_lib.h`/`.cc` that keeps
   def-use links and topological order, and test it in
   `litert_model_lib_test.cc`.
2. Bind it in `_pywrap_litert_model.cc`. Resolve every handle with `Get()`,
   check that handle arguments belong to the same model or subgraph, report
   errors with `CheckEdit`/`EditValueOrThrow` (precondition errors) or
   `ThrowIfError`/`ValueOrThrow` (`LiteRtError`), and forget any IR object
   you free (`FreedObjectTracker` or `ModelState::Forget`).
3. Add the method to `_pywrap_litert_model.pyi`.
4. Wrap it in `litert_model.py`: convert enums and unwrap or wrap handles.
5. Add Python tests, including the behavior of stale handles.
6. Add a `BUILD` dependency for every new header you include. Each target
   must depend directly on the headers it includes.

New element types or op codes must be added to the C header, the X-macro
list in `_pywrap_litert_model.cc` and the Python enum together. New shape
inference rules belong in `litert/core/model/shape_inference.*`, not here.

Agent skills: `_agents/skills/`.

## Testing

```sh
bazel test //litert/python/litert_model:litert_model_lib_test \
  //litert/python/litert_model:litert_model_test \
  //litert/python/litert_model:litert_model_dispatch_test \
  //litert/python/litert_model:litert_model_numerics_test
```

- `litert_model_lib_test` (C++): byte sizes and overflow, op code table,
  `AddOp`, built-in options, DCE, `ReplaceAllUses`, signature sync,
  validation levels, outlining, `RemoveTensor`, pruning, and validation of
  every test model.
- `litert_model_test` reads `litert/c/litert_model_types.h` and
  `litert/c/litert_op_code.h` as data for the enum check.
- `litert_model_dispatch_test`: shape inference, outlining and serialization.
- `litert_model_numerics_test` needs the compiled-model runtime and is tagged
  `no_oss`.

Test models come from `litert/test/testdata/` (`TESTDATA_MODELS` in
`BUILD`). Python tests locate them with `common.get_resource` from
`litert/python/aot/core/common.py`. No target is platform specific.

## Related

- C++ model IR: `litert/core/model/`.
- C++ compiler plugin flow: `litert/compiler/plugin/`.
- Python AOT compilation: `litert/python/aot/`.
