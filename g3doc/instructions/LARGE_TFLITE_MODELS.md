# Large `.tflite` Models and Weight Storage Options

LiteRT (previously known as TensorFlow Lite) provides multiple storage
mechanisms for models whose weights exceed the FlatBuffer size limit (~2 GiB).
Use this guide to choose the appropriate weight storage format for your
deployment target—mobile, desktop, or web—and to verify support across model
loading, execution, inspection, and serialization APIs.

--------------------------------------------------------------------------------

## Overview & Why This Guide Exists

The TFLite FlatBuffer structure must fit below approximately 2 GiB due to 32-bit
internal offset representations in FlatBuffers. Embedding all constant tensor
weights directly inside `Buffer.data` can exceed that limit for modern deep
learning models. To address that, the framework supports two mechanisms to move
constant payloads outside the core FlatBuffer structure, alongside traditional
inline storage:

| Storage Model | Deployment | References in the Model | Max Payload Size |
| :--- | :--- | :--- | :--- |
| **Inline Buffers** | Standard single `.tflite` file | `Tensor.buffer` indexes `Model.buffers`; `Buffer.data` is present | < 2 GiB total FlatBuffer |
| **`buffer_offset` Mode** | Single `.tflite` file with appended constants | `Buffer.offset` and `Buffer.size`; `Operator.large_custom_options_offset`/`size` | Total file can exceed 2 GiB |
| **External Buffers** | `.tflite` plus separate weight files or application-provided memory | `Tensor.external_buffer`, `Model.external_buffers`, `Model.external_buffer_groups` | Total weights can exceed 2 GiB across storage sources |

Both `buffer_offset` and external buffers allow the total payload to exceed the
FlatBuffer limit. Neither removes the ~2 GiB limit on the FlatBuffer
graph/metadata structure itself or the target device's physical memory limits.
External buffers describe immutable tensor data; appended payloads can also hold
custom-op options. See the
[FlatBuffer schema](../../tflite/converter/schema/schema.fbs).

**Default to the same `.tflite` file across web, desktop, and mobile.** For an
ordinary model such as ResNet-50, use inline buffers when they fit; web deployment
alone is not a reason to externalize weights or produce a different model.
Preserve an existing model's storage layout when it works on all intended targets.
Operator and tensor-type support must also be checked for each backend.

Use `buffer_offset` when appended storage is needed and one deployable file is
preferred; use external buffers when weights need separate storage or loading the
complete file exceeds a target's memory limits. Change packaging for a target only
when a concrete limitation requires it. If your pipeline inspects or rewrites
models, first check the
[Public API Support](#public-api-support-loading-inspection-and-serialization)
table: the current LiteRT serializer does not preserve separate-weight metadata.
See [Web Deployment](#web-deployment--streamed-loading-litertjs--webgpu) for the
ordinary single-file loader and the separate-weight streaming alternative.

--------------------------------------------------------------------------------

## Option 1: `buffer_offset` Mode (Single-File Deployment)

In `buffer_offset` mode, the model remains packaged as a single `.tflite` file
on disk. The graph structure, op definitions, and metadata reside in a valid
FlatBuffer at the beginning of the file, while constant tensor payloads are
appended immediately after the FlatBuffer.

### Representation

For each appended constant:

-   `Tensor.buffer` still indexes `Model.buffers`.
-   `Buffer.data` is **absent** (omit the data vector).
-   `Buffer.offset` and `Buffer.size` are 64-bit unsigned integers. The offset
    is relative to the beginning of the `.tflite` file and must be greater
    than 1. Offset values `0` and `1` are reserved sentinels and are not valid
    appended-payload offsets.
-   Appended custom-op options use the operator's `large_custom_options_offset`
    and `large_custom_options_size` fields with the same offset convention.

The exporter aligns appended constants to 16-byte boundaries and records a
`buffer_location` metadata entry with the string value `outside flatbuffers`.
Preserve this metadata when rewriting or transforming a model: converter
utilities and `BuildFromModel` rely on it to recognize this storage mode. The
payload locations themselves are determined by the `Buffer.offset` fields.

### Converter Guidance

-   **Python TFLite Converter**: Set
    `converter._experimental_use_buffer_offset = True` before invoking
    `convert()`.
-   **C++ Converter Flags**: Set `converter_flags.set_use_buffer_offset(true)`.
-   **MLIR Translation**: Pass the command-line flag `--use-buffer-offset`.

The [FlatBuffer exporter](../../tflite/converter/flatbuffer_export.cc) also
enables this mode proactively when the estimated module size exceeds the
FlatBuffer limit minus 512 MiB. If an initial export fails due to the FlatBuffer
size limit, it automatically retries once with offsets enabled. The
[StableHLO Python conversion pipeline](../../tflite/converter/python/stablehlo_tfl_pipeline.cc)
explicitly enables buffer offsets even for smaller models.

### Runtime Guidance

Use allocation-based loading APIs and supply the complete file, including appended
bytes. Support for inspecting or rewriting the loaded model is described separately
in the [API support table](#public-api-support-loading-inspection-and-serialization).

-   **TFLite C++**: Use `FlatBufferModel::BuildFromFile(...)` or
    `FlatBufferModel::VerifyAndBuildFromFile(...)`, then construct the
    interpreter from the `FlatBufferModel`.
-   **TFLite Python**: Use `Interpreter(model_path="model.tflite")`.
-   **LiteRT C++**: Use `Model::CreateFromFile(env, ...)` or the filename
    overload of `CompiledModel::Create(env, ..., options)`.
-   **LiteRT file descriptors**: `Model::CreateFromFd(env, fd, offset, size)`
    requires a region containing the complete `.tflite` model. Model offsets
    are relative to the start of that region, not the containing file. This
    API requires mmap support in the runtime build.
-   **In-Memory Loading**: APIs like
    `tflite::FlatBufferModel::BuildFromBuffer(...)`, `model_content=...`, or
    LiteRT buffer overloads require the **entire** `.tflite` byte sequence
    including appended payloads, along with its full length. The caller must keep
    the backing memory alive for the duration of model execution.
-   **Avoid partial allocations**: Passing only the FlatBuffer prefix loses the
    appended weights.
-   **Avoid `FlatBufferModel::BuildFromModel(const tflite::Model*)`**: This
    pointer-only overload rejects models marked with `buffer_location`.
-   **Raw-model builder**: The raw-model `InterpreterBuilder` overload needs an
    explicit backing `Allocation` to resolve appended offsets.
-   **Model rewriting**: A standard FlatBuffer unpack/repack does not preserve
    appended payloads or relocate their offsets. Rewriters must handle both
    constant buffers and large custom-op options explicitly.

LiteRT's native file, buffer, and file-descriptor APIs retain an allocation and
avoid the pointer-only `BuildFromModel` path. See
[LiteRT loading](../../litert/core/util/flatbuffer_tools.cc),
[the C entry points](../../litert/c/litert_model.cc), and
[TFLite interpreter parsing](../../tflite/core/interpreter_builder.cc).

--------------------------------------------------------------------------------

## Option 2: External Buffers (Separate Weight Storage)

Use external buffers when weight assets need separate packaging, sharding across
multiple files, or dynamic supply from host application memory. The model
records group names and byte slices; the runtime loader supplies the backing
storage.

### Emitting FlatBuffers Directly

For each external constant:

-   Set `Tensor.external_buffer` to a nonzero `ExternalBuffer.id`. This is an
    ID, not an index into `Model.external_buffers`.
-   Set `Tensor.buffer = 0`, and keep `Model.buffers[0]` empty. Do not attach
    inline or appended constant data to the same tensor, or mark it as variable.
-   Add an `ExternalBuffer` with a unique ID, a `group` index into
    `Model.external_buffer_groups`, and 64-bit byte `offset` and `length`
    fields.
-   Give the referenced `ExternalBufferGroup` a `name` identifying its backing
    storage (e.g. `weights.bin`). Group index `0` is valid; it is not a
    sentinel.

The exporter assigns IDs following `0x80000000 | external_buffer_index`, setting
the high bit to distinguish them from standard TFLite buffer indices. Follow
that convention when generating models for the same runtime/delegate paths.

The `packing` string records layout information. The built-in loader exposes it
to consumers but does not decode arbitrary compression or packing formats. For
CPU execution, provide bytes in the tensor's expected representation; the
externalization tool writes `packing = "unpacked"`.

For host access, the resolved tensor address must satisfy
`LITERT_HOST_MEMORY_BUFFER_ALIGNMENT` (currently 64 bytes). Align file slice
offsets accordingly:

-   For a packed file, align `section.offset + buffer.offset`.
-   For in-memory groups, align `group_base + buffer.offset`.
-   The length must cover the tensor's required byte size. These requirements
    are enforced by
    [host tensor buffer validation](../../litert/runtime/tensor_buffer.cc).

### Emitting MLIR

Use `tfl.external_const` with an `external_buffer` attribute specifying named
fields:

```mlir
%weights = "tfl.external_const"() <{
  external_buffer = #tfl.external_buffer<
    group_name = "weights.bin", offset = 0, length = 64, packing = "unpacked">
}> : () -> tensor<4x4xf32>
```

The exporter materializes the groups, buffers, and tensor references. The
converter or pipeline must separately write the referenced weight bytes to disk.

*(Note: `tfl.external_const` with only a `buffer_index` references an existing
FlatBuffer constant rather than creating a separate weight file. See
[op definition](../../tflite/converter/ir/tfl_ops.td),
[attribute definition](../../tflite/converter/ir/tfl_op_enums.td), and
[exporter](../../tflite/converter/flatbuffer_export.cc)).*

### Externalizing an Existing `.tflite` Model

The tool
[`litert/tools/externalize_tflite_flatbuffer.py`](../../litert/tools/externalize_tflite_flatbuffer.py)
extracts weights from `model.tflite` and writes a separate weight blob into an
output directory. In a Python environment with `flatbuffers`, NumPy, and the
generated `litert.python.schema_py_generated` module:

```bash
python3 -m litert.tools.externalize_tflite_flatbuffer \
  --input_model=/path/to/input.tflite \
  --output_dir=/path/to/output \
  --group_name=tflite_weights \
  --num_elements_threshold=256
```

Generate the Python schema bindings with `flatc` before running:

```bash
flatc --python --gen-onefile --gen-object-api \
  --filename-suffix _py_generated -o litert/python \
  tflite/converter/schema/schema.fbs
```

**Scope and Behavior of the Tool:**

-   Selects constant tensors at input 1 of `FULLY_CONNECTED`, `CONV_2D`,
    `DEPTHWISE_CONV_2D`, or `EMBEDDING_LOOKUP` exceeding the element count
    threshold.
-   Skips subgraph inputs, bias tensors, variables, and tensors already using
    external buffers.
-   Deduplicates identical externalized payloads and aligns slice offsets to 64
    bytes.
-   Resolves appended `Buffer.offset/size` payloads prior to processing;
    unselected payloads (such as biases) are repacked inline.
-   Does not stream arbitrarily large files (reads input into memory) and does
    not relocate `Operator.large_custom_options_*`.

The remaining inline model must fit in a FlatBuffer. Preserve any existing
external weight files separately: the Python `externalize(...,
existing_weights=...)` argument can copy an existing blob, but the CLI does not
expose that argument.

--------------------------------------------------------------------------------

## Web Deployment & Streamed Loading (litert.js & WebGPU)

Start with the same complete `.tflite` file used on desktop and mobile. The normal
`loadAndCompile` path accepts that file's bytes; it does not require a web-specific
graph or external weights. This applies to inline models and to files with
appended payloads, subject to backend support and available memory. For example,
a compatible ResNet-50 file can be served by URL on web and loaded from a
filesystem path on native targets without changing its contents.

The two JavaScript loading APIs have different memory behavior:

-   [`loadAndCompile(model, options)`](../../litert/js/packages/core/src/litert_web.ts)
    collects the complete model and copies it into Wasm memory, including any
    appended payloads. Passing a `ReadableStreamDefaultReader` does not enable
    separate-weight streaming. That input path's
    [collection helper](../../litert/js/packages/core/src/load_utils.ts) currently
    caps the model at 2,000,000,000 bytes.
-   [`loadModelAndWeights(modelData, weightsStream, options)`](../../litert/js/packages/core/src/streamed_loading.ts)
    loads the FlatBuffer into Wasm memory and supplies external weights through
    a separate `ReadableStream<Uint8Array>`. It requires a WebGPU device; select
    `accelerator: 'webgpu'` and a runtime/browser configuration that supports
    the delegate's streaming callback.

Choose separate-weight streaming when the application already uses external
weights or when measurements show that loading the shared complete file exceeds
the target's limits. When this representation is needed, prefer sharing the same
`.tflite` graph and weight files across platforms as well: native applications can
open the weight file, while web applications supply its bytes as a stream.

The current streamed-loading API accepts **one weight stream**, with
external-buffer offsets interpreted within that stream. It does not expose the
native loader's group-to-file or group-to-section maps, or fetch URLs from group
names. The application must supply the stream and arrange compatible offsets.
Applications using container bundles must extract the FlatBuffer and weight
stream themselves; this API does not take a `.litertlm` container as its model
argument.

External weights avoid storage in the Wasm heap on this WebGPU path, but loading
still uses JavaScript memory. The current callback accumulates enough bytes for
each requested tensor before calling `GPUQueue.writeBuffer`, then discards
processed data. Account for this staging memory and GPU allocations when choosing
tensor sizes and testing large models.

When using separate-weight streaming, keeping the FlatBuffer and remaining inline
constants small reduces Wasm memory usage. There is no 50 MiB graph-size threshold
in these APIs, and a particular graph size does not guarantee browser or GPU
compatibility. Compilation consumes
the weight stream; the public streamed-loading helper does not provide a
weight-free dry-run mode. Validate the intended browser, runtime build, model,
and GPU together with a known-good inference.

--------------------------------------------------------------------------------

## Runtime Execution & Weight Resolution

### Core TFLite Interpreter

The [interpreter builder](../../tflite/core/interpreter_builder.cc) parses and
stores `Tensor.external_buffer` IDs, but does not resolve external buffer groups
into filesystem files or load their bytes. Therefore,
the Python `Interpreter(model_path=...)` does not automatically load separate
weight files. Use LiteRT's compiled-model runtime or supply custom logic to map
external weights into tensor memory before `Prepare`/`AllocateTensors`.

### LiteRT Compiled-Model Runtime

The [LiteRT compiled-model runtime](../../litert/runtime/compiled_model.cc)
automatically instantiates a
[weight loader](../../weight_loader/external_weight_loader_litert.cc) unless the
client supplies one explicitly.

-   **CPU Execution**: The runtime maps host access and restores external tensor
    pointers as immutable `kTfLiteMmapRo` data **before applying delegates**,
    enabling CPU kernels and XNNPACK to execute directly over external weights.
-   **Hardware Accelerators**: The loader is passed to accelerator options.
    GPU-only loading depends on backend delegate integration. On Web, CPU
    pointer restoration runs only when CPU is requested without GPU/NPU to
    facilitate streaming weights directly into WebGPU accelerator buffers
    without intermediate CPU heap allocation.

For a model and its weight file in the same directory, loading is automatic. For
example, in C++:

```cpp
LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
LITERT_RETURN_IF_ERROR(
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu));
LITERT_ASSIGN_OR_RETURN(
    auto compiled_model,
    litert::CompiledModel::Create(env, "/models/model.tflite", options));
```

If the model references group name `weights.bin`, this automatically loads
`/models/weights.bin`.

### Public API Support: Loading, Inspection, and Serialization

The following describes the current native C APIs and their C++ wrappers.
`CreateFromFile`, `CreateFromBuffer`, and `CreateFromFd` refer to the corresponding
`Model` factories and `LiteRtCreateModelFrom*` functions. Loading retains the
original FlatBuffer; the compiled-model runtime resolves separate weights later.

| API or Operation | Appended Tensor Buffers (`Buffer.offset/size`) | Separate Weights (`Tensor.external_buffer`) |
| :--- | :--- | :--- |
| `CreateFromFile` | Reads the complete file allocation | Retains references; records the model directory for later weight resolution |
| `CreateFromBuffer` / `CreateFromFd` | Requires the complete model buffer or file region, including appended bytes | Retains references; no model directory, so use explicit storage mappings or filesystem paths valid from the working directory |
| `CompiledModel::Create` using the original FlatBuffer | Supplies the full allocation to the TFLite interpreter | Uses the weight loader; execution support depends on the requested backend |
| `LiteRtGetTensorWeights` followed by `LiteRtGetWeightsBytes` | Returns the appended tensor bytes | Returns empty weights for external constants with `Tensor.buffer = 0`; does not resolve external sources |
| `LiteRtSerializeModel` / `LiteRtSerializeModelWithSignatures` | Re-emits appended tensor data and recalculates offsets; validate output alignment | Does not preserve external-buffer IDs, groups, or slices |

These differences follow from the [model importer](../../litert/core/model/model_load.cc),
[public getters](../../litert/c/litert_model.cc), and
[serializer](../../litert/core/model/model_serialize.cc). Restoring weights into
the compiled interpreter does not populate the original `LiteRtModel`'s weight
objects. A model can therefore execute successfully while its inspection APIs
report empty weights.

**Appended custom-op options have a separate limitation.** The importer logs
that `large_custom_options_*` is unsupported in `litert::Model` and reads only
inline `custom_options`. `LiteRtGetCustomOptions` does not expose the appended
payload, and the serializer does not preserve it. Direct execution can still
use the original FlatBuffer allocation through TFLite's interpreter parser,
provided the custom op itself is supported.

**Do not use successful loading as evidence of a safe serialization round trip.**
For separate weights or appended custom-op options, use tooling that explicitly
preserves that representation, or materialize the data in a supported format
before rewriting. Compiler-plugin paths that
[reserialize the model](../../litert/runtime/compiled_model.cc) inherit these
limitations. Also, the public serialization API returns
`kLiteRtStatusErrorUnsupported` in builds with `LITERT_DISABLE_NPU`.

### How Group Names Resolve

The built-in weight loader resolves each group in the following order of
precedence:

Priority | Resolution Source                                               | Meaning of `ExternalBuffer.offset`
:------- | :-------------------------------------------------------------- | :---------------------------------
**1**    | Matching entry in `Options::SetWeightInMemoryMap(...)`          | Byte offset into the group's host-memory span
**2**    | Matching section in `Options::SetExternalWeightScopedFile(...)` | Offset within section (`section.offset + buffer.offset` in file)
**3**    | `ExternalBufferGroup.name` as a filesystem path                 | Offset from the start of that file

-   **Filesystem fallback**: Absolute paths are loaded directly. Relative paths
    resolve against the model file's directory when loaded from a path. When
    loaded from memory or a file descriptor, relative paths resolve against the
    current process working directory. This fallback does not fetch URLs or
    open Android assets by group name.
-   **Scoped packed files (`SetExternalWeightScopedFile`)**: Multiple groups can
    be packed into a single container file:

    ```cpp
    LITERT_ASSIGN_OR_RETURN(auto weight_file,
                           litert::ScopedFile::Open("/models/weights.pack"));
    litert::Options::ScopedWeightSectionMap sections;
    sections.emplace("weights.bin", litert::ScopedWeightSection{4096, 8192});
    LITERT_RETURN_IF_ERROR(
        options.SetExternalWeightScopedFile(weight_file, std::move(sections)));
    ```

    Call the setter before `CompiledModel::Create`. It moves ownership of
    `weight_file`; the handle is invalid after a successful call. Sections must
    have positive lengths and fit in the packed file; each tensor slice must
    fit in its section. See the [Options header](../../litert/cc/litert_options.h).
-   **Application memory (`SetWeightInMemoryMap`)**: Borrows client memory for
    group data; both the map and its backing memory must remain valid for the
    lifetime of the compiled model.
-   **CLI Runner**: The [`run_model` tool](../../litert/tools/run_model.cc)
    exposes section mapping via `--scoped_weight_file`, `--scoped_weight_group`,
    `--scoped_weight_offset`, and `--scoped_weight_length`.

### API Availability Across Languages

-   **Native C++** exposes `SetExternalWeightScopedFile`, `SetWeightInMemoryMap`,
    and `SetWeightLoader`. `SetWeightInMemoryMap` is unavailable when
    `LITERT_NO_ABSL` is defined. A client-supplied `WeightLoader` is borrowed and
    must outlive the compiled model. See
    [C++ options](../../litert/cc/litert_options.h).
-   **Native C** supports automatic filesystem weight resolution during
    compilation, but its public options API has no equivalent setters for a
    group-memory map, scoped weight file, or custom weight loader. See
    [C options](../../litert/c/litert_options.h).
-   **Python `CompiledModel`** uses the native file or buffer loading paths, but
    its [options](../../litert/python/litert_wrapper/compiled_model_wrapper/options.py)
    do not expose those three C++ weight-source setters. A filename preserves
    the model directory; an in-memory model does not. See
    [the Python wrapper](../../litert/python/litert_wrapper/compiled_model_wrapper/compiled_model_wrapper.cc).
-   **Kotlin/Android** also does not expose those setters. File loading retains
    the model path. Asset loading copies the complete asset into memory and
    performs one `AAsset_read` with an `int` result, so that path cannot complete
    a read larger than `INT_MAX` bytes. It does not automatically load weight
    groups from neighboring APK assets. Use extracted filesystem files and the
    file-loading API for that deployment. See
    [the JNI loaders](../../litert/kotlin/src/main/jni/litert_compiled_model_jni.cc).
-   **JavaScript** loads a complete `.tflite` file with `loadAndCompile`, including
    inline or appended weights within memory and backend limits. It also has a
    separate API for one weight stream and WebGPU execution; see
    [Web Deployment & Streamed Loading](#web-deployment--streamed-loading-litertjs--webgpu).

### External Input Bindings Are a Different Mechanism

`Options::AddExternalTensorBinding` and `LiteRtAddExternalTensorBinding` bind
caller-owned memory to a **named signature input** through
[`SetCustomAllocationForInputTensor`](../../litert/runtime/tfl_utils.cc). They do
not resolve arbitrary `Tensor.external_buffer` IDs or replace the weight-source
setters above. The input's size and alignment requirements still apply.

The current C API and [C++ runtime proxy](../../litert/cc/internal/litert_runtime_proxy.h)
use `int` for the binding size, although the C++ options setter accepts `size_t`.
This is not a supported path for binding a tensor larger than `INT_MAX` bytes.

--------------------------------------------------------------------------------

## Checklist for Model Authors, Converters, and Tooling

When generating, rewriting, or validating models with large weight storage:

-   [ ] **Shared Deployment Artifact**: Try the same model file on all intended
    targets first. Introduce different packaging only for an identified storage,
    backend, or memory limitation; validate each required variant.
-   [ ] **FlatBuffer Bounds**: Ensure graph structure, tensor metadata, and
    remaining inline buffers stay below the ~2 GiB FlatBuffer limit.
-   [ ] **Appended Payloads (`buffer_offset`)**:
    -   Ensure `buffer_location` metadata is set to `"outside flatbuffers"`.
    -   Validate that `Buffer.offset > 1` (offsets `0` and `1` are invalid
        sentinels).
    -   Ensure 16-byte alignment and verify that offsets do not exceed total
        file length.
    -   Recompute and preserve offsets for both constant buffers and custom-op
        options after any graph transformation.
-   [ ] **External Buffers (When Used)**:
    -   Ensure `Tensor.buffer == 0` (empty sentinel) and `Model.buffers[0]` is
        empty.
    -   Verify each `Tensor.external_buffer` is nonzero and resolves to an
        `ExternalBuffer` entry. IDs must be unique across those entries.
    -   Ensure referenced `group` indices exist in
        `Model.external_buffer_groups`.
    -   Verify tensor address alignment meets
        `LITERT_HOST_MEMORY_BUFFER_ALIGNMENT` (64 bytes).
    -   If separate-weight streaming is needed on web, use `loadModelAndWeights`
        and ensure every slice resolves within the supplied weight stream.
    -   Measure Wasm, JavaScript staging, and GPU memory use on the target device;
        a compact graph alone does not establish compatibility.
-   [ ] **API and Rewrite Compatibility**:
    -   Check that the chosen language API exposes the required weight source.
    -   Before using getters or serialization, check the
        [API support table](#public-api-support-loading-inspection-and-serialization).
    -   After rewriting, compare the output's weight references and payloads,
        including custom-op options, against the input. Do not rely solely on
        FlatBuffer verification or a successful return status.
-   [ ] **Slice Bounds & Overflow Protection**:
    -   Prevent integer overflow when computing byte bounds: require `offset <=
        total_size` and `length <= total_size - offset`.
-   [ ] **Runtime Verification**:
    -   Always run end-to-end inference verification on target hardware.
        FlatBuffer schema validation alone verifies structure, not weight file
        presence or mathematical integrity.
