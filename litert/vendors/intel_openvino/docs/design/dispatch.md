<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
# Intel OpenVINO Dispatch Design

## Purpose and boundary

The dispatch library implements LiteRT's vendor dispatch API for compiled
OpenVINO model blobs produced by the
[OpenVINO compiler plugin](compiler.md), including blobs packaged
with a shared weight pool. It manages device and invocation contexts, imports
compiled OpenVINO models, binds LiteRT tensor buffers, and executes inference.
It advertises basic dispatch capabilities, not LiteRT's asynchronous dispatch
API.

```text
LiteRT runtime -> dispatch device context -> invocation context per partition
    -> select model blob and device -> import OpenVINO model
    -> register/attach tensors -> infer -> outputs
```

## Contexts and model loading

[dispatch_api.cc](../../dispatch/dispatch_api.cc) exposes the dispatch entry points and
translates errors to LiteRT status codes. A
[device context](../../dispatch/device_context.h) owns the tensor-handle registry, per-model
shared weight banks, and a registry of invocation contexts. OpenVINO core state
is obtained from [openvino_shared_core.h](../../dispatch/openvino_shared_core.h).

An [invocation context](../../dispatch/invocation_context.h) represents one compiled
partition. Creation retains a view of the bytecode module and its function
name; the owning LiteRT model must remain alive while that view is in use.
Import is deferred until the first attachment or invocation, so failures such
as an unavailable device or invalid bytecode can surface at first use rather
than at context creation.

For a shared `OVGLOBAL` container, dispatch selects the subgraph by function
name and imports its payload. If the name is missing or unmatched, it falls
back only when the container has exactly one subgraph; otherwise it returns an
error. The backend is taken from the payload header when valid, otherwise from
the container entry; a standalone blob without a valid header defaults to NPU.
The selected OpenVINO device must be available. The process-wide shared core
caches the available-device list and the current device's remote context.

At import, dispatch forwards only runtime-supported entries from the model-wide
configuration map, plus the performance-mode hint from the options when no
explicit hint was provided. The allowlist is defined in
[utils.h](../../utils.h); compile-time options and per-graph configuration maps
are not generally forwarded to `import_model`.

## Shared weights and execution

The compiler and dispatcher share the container definition in
[global_graph.h](../../compiler/global_graph.h). Parsing borrows the container's
bytes, including the pool and payload spans. On GPU,
[weight_bank_runtime.h](../../dispatch/weight_bank_runtime.h) allocates a shared host-USM
pool and binds views to named weight parameters. On NPU it stages the shared
pool to a per-device-context file and imports the weightless model blob with
that file as its weights path. Staging normally copies in chunks, using
short-lived mappings where supported for file-backed models, with a
memory-backed fallback. `LITERT_OV_WEIGHTS_PATH` can instead supply a
pre-staged file; it must contain the matching pool layout and is not deleted by
dispatch. A temporary file created by dispatch is removed with its bank.
The file or GPU pool survives individual partition imports and releases.
Each bank initializes once per device context, so contexts must not reuse a
bank for unrelated pools.

## Tensor buffers

The invocation context reports packed-size requirements for OpenVINO custom
buffers, DMA-BUF, and Android hardware buffers (AHWB); actual DMA-BUF/AHWB
registration depends on platform build support. Explicit tensor strides are
rejected. The device context registers buffers under handles and converts them
to OpenVINO tensors; invocation contexts bind those handles by input/output
index. The backing buffer must remain valid until it is unregistered and no
longer in use.

[openvino_tensor_buffer.h](../../dispatch/openvino_tensor_buffer.h) implements the custom
buffer. Allocation records its type and shape, while the actual tensor is
created and zeroed on first lock or registration. NPU/GPU use a host tensor
from the current OpenVINO remote context; CPU uses a regular host tensor. An
imported custom buffer is a non-owning view, so its source must outlive the
view. DMA-BUF uses an NPU remote tensor; AHWB registration maps the backing
DMA-BUF into a host tensor and releases the mapping when unregistered.

## Deferred import and execution

For multi-context models, the compiler records a KV tier in each shared
subgraph. A foreground import for a known, positive tier attempts to release
idle imported contexts from other known tiers before importing; unknown tier
zero is excluded. Contexts in use or mid-import are skipped without waiting.
After import, other contexts in the same tier may be prefetched in the
background. A prefetch that finishes after the active tier changes attempts to
release itself. Previously attached buffers are recorded and rebound after a
later re-import. The device context must outlive its invocation contexts;
destruction unregisters an invocation context and waits for its prefetch.

The public dispatch API is synchronous even though the implementation starts
an OpenVINO asynchronous infer request and waits up to 10 seconds. On Android,
when enabled and available, the NPU HAL hook gates execution and can resolve a
model priority from LiteRT scheduling information. Access denial and errors
after inference starts are returned rather than retried; an error before start
can fall back to direct execution. The HAL integration can be disabled with
`--//litert/vendors/intel_openvino/dispatch:enable_npu_hal=false`.

## Source map and tests

- [dispatch_api.cc](../../dispatch/dispatch_api.cc): LiteRT dispatch API and tensor-buffer
  callbacks.
- [device_context.cc](../../dispatch/device_context.cc): registered tensors, shared banks,
  and cross-tier import coordination.
- [invocation_context.cc](../../dispatch/invocation_context.cc): model blob selection,
  import, attachment, inference, and deferred-import lifecycle.
- [openvino_tensor_buffer.cc](../../dispatch/openvino_tensor_buffer.cc): custom OpenVINO
  tensor-buffer implementation.
- [npu_kv_cache_policy.cc](../../dispatch/npu_kv_cache_policy.cc): cross-tier release policy.
- [npu_hal_wrapper.h](../../dispatch/npu_hal_wrapper.h): optional Android scheduling hook.
- [dispatch_api_openvino_test.cc](../../dispatch/dispatch_api_openvino_test.cc) and
  [openvino_tensor_buffer_test.cc](../../dispatch/openvino_tensor_buffer_test.cc): dispatch
  and buffer tests; see [BUILD](../../dispatch/BUILD) for targets and platform requirements.

The buffer test is host-only but still requires the OpenVINO runtime. The
dispatch API test is marked manual and depends on suitable runtime/test data;
both are tagged `no_oss` in the Bazel build.