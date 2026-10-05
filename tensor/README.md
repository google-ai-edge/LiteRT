# LiteRT Tensor API

**Author, run and chain ML graphs in C++ — one API from model authoring to
zero-copy on-device pipelines, on CPU, GPU and the browser.**

The LiteRT Tensor API is a lightweight, tensor-centric C++ library that sits on
top of the [LiteRT](https://github.com/google-ai-edge/LiteRT) runtime. You write
a graph the way you would write math — `Add(a, b)`, `Softmax(x)`,
`FullyConnected(x, w, b)` — and the same code can be:

*   **serialized to a `.tflite` model** (no converter, no Python in the loop),
*   **run directly on XNNPACK** (CPU, no model file in between), or on GPU /
    NPU through LiteRT's accelerators,
*   **wired into a pipeline** where pre-processing, models and post-processing
    share hardware buffers with zero CPU copies.

It is the library behind the
[SAM 2 video segmentation web demo](https://github.com/google-ai-edge/litert-samples/tree/main/samples/web_demos/src/sam2)
(the whole pipeline — Hiera encoder, memory attention, mask decoder, frame
pre-processing and compositing — is authored with the Tensor API, compiled to
WebAssembly and tracks an object at ~30 fps on WebGPU) and the Gemma 3 /
Gemma 4 examples in this directory.

The library lives in the
[LiteRT repository](https://github.com/google-ai-edge/LiteRT/tree/main/tensor)
under `tensor/`; headers are included as `tensor/...` and Bazel targets are
`//tensor/...`.

---

## Contents

1.  [Why the Tensor API](#why-the-tensor-api)
2.  [Concepts in two minutes](#concepts-in-two-minutes)
3.  [Quick start](#quick-start)
4.  [Building the library](#building-the-library)
5.  [Integrating into an existing app](#integrating-into-an-existing-app)
6.  [Verifying backends](#verifying-backends)
7.  [Examples](#examples)
8.  [Repository layout](#repository-layout)
9.  [Further reading](#further-reading)

---

## Why the Tensor API

| Problem | What the Tensor API gives you |
|---|---|
| You need a model that no converter produces — a custom attention layer, an image pre-processor, a mask compositor, a sampler. | Author it in C++ with 90+ ops (element-wise, activations, reshape/transpose/gather, reductions, matmul, conv, normalization, quantization…) and serialize it with `ModelFactory::Save`. The output is an ordinary `.tflite` that any LiteRT runtime (C++, Android, iOS, LiteRT.js) can run. |
| Pre- and post-processing are hand-written loops that copy data in and out of the accelerator. | Express them as graphs and run them next to the core model; `ModelChain` and `LitertBuffer` keep activations on the device between stages. |
| One graph, several targets. | `Tensor<Mixins...>` is templated on a backend tag. The same function builds a TFLite graph with `TfLiteMixinTag` (CPU, GPU or NPU through LiteRT) or an XNNPACK graph with `XnnpackMixinTag` — or both at once. |
| Is the GPU result right? | `GraphProbe` exposes any intermediate tensor as a model output so an optimized run can be compared against a CPU reference; a numerical test suite covers the op set per backend. |
| Multiple entry points and dynamic shapes. | Multi-signature models (`AddSignature` per entry point, weights shared), dynamic runners, feedback loops for KV caches and recurrent state. |

Concretely, in the SAM 2 demo the whole model is **11 signatures in one
164 MB file** (encode, `prompt{1..8}`, `track{2,7}`) with every weight shared,
and the per-video frame model is authored and compiled **in the page in ~0.2 s**
for the exact video geometry.

## Concepts in two minutes

```
          author                     serialize / lower                 run
┌────────────────────┐    ┌──────────────────────────────┐    ┌────────────────────────┐
│ Tensor<Mixins...>  │    │ TfLiteMixinTag  → ModelFactory│    │ LiteRT CompiledModel   │
│ arithmetic.h ops   │ →  │                   (.tflite)   │ →  │ (CPU / GPU / NPU)      │
│ (lazy graph build) │    │ XnnpackMixinTag → XNNPACK     │    │ XnnpackRunner (CPU)    │
└────────────────────┘    └──────────────────────────────┘    └────────────────────────┘
                                   ModelChain: stage → stage, shared LitertBuffers
```

*   **`TensorHandle` / `Tensor<Mixins...>`** (`tensor.h`) — a node in a lazily
    built graph: name, `Type`, `Shape`, optional buffer or constant. Creating
    tensors and calling ops never computes anything; it records a DAG.
*   **Ops** (`arithmetic.h`) — free functions and operator overloads over
    tensors. See [tensor_api.md › Core Op APIs](tensor_api.md#core-op-apis) for
    the full list.
*   **Backends** (`backends/`) — a *mixin tag* selects how the graph is
    lowered. The two tags can be combined (`Tensor<TfLiteMixinTag,
    XnnpackMixinTag>`) when a graph must run on both backends.

    | Tag | Header | Lowers to | Runs with |
    |---|---|---|---|
    | `TfLiteMixinTag` | `backends/tflite/arithmetic_tflite.h` | TFLite flatbuffer (`ModelFactory`) | any LiteRT `CompiledModel` on CPU, GPU or NPU; `runners/litert/*` |
    | `XnnpackMixinTag` | `backends/xnnpack/arithmetic.h` | XNNPACK subgraph | `runners/xnnpack:runner` (CPU) |

*   **Buffers** (`buffer.h`, `runners/litert/litert_buffer.h`) — `Buffer` /
    `MutableBuffer` with RAII `Lock()`; `OwningCpuBuffer`, `SpanCpuBuffer`
    for host data; `LitertBuffer` wraps a LiteRT `TensorBuffer` (host, AHWB,
    OpenCL, WebGPU) so stages can share device memory.
*   **Runners** (`runners/litert/`) — `CompiledModelRunner` (build a graph and
    compile it in one go), `LitertDynamicRunner` (load an existing `.tflite`,
    name-based I/O), `CreateLambdaRunner` (a graph given as a lambda).
*   **`ModelChain`** (`runners/model_chain.h`) — a DAG of `ModelStage`s
    (`CompiledModelStage` for `.tflite` files, `FunctionalModelStage` for host
    code). `Build()` negotiates buffer layouts at every boundary and
    pre-allocates shared `LitertBuffer`s; `Execute()` runs the stages in
    topological order with no intermediate CPU copies.

## Quick start

The same three-line graph, run two ways. The backend is chosen by the mixin
tag on `Tensor<...>`; everything else — ops, shapes, buffers — is identical.

### A. On CPU with XNNPACK (no model file, no LiteRT environment)

`XnnpackMixinTag` lowers the graph straight into an XNNPACK subgraph and
`XnnpackRunner` executes it. This is the lightest way to run Tensor API code
and the path the Gemma 4 example uses for prefill and decode.

```cpp
#include <iostream>
#include <vector>

#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

using namespace litert::tensor;
using XnnTensor = Tensor<XnnpackMixinTag>;

int main() {
  // 1. The graph. Constants carry their data; nothing runs yet.
  XnnTensor a({.name = "a", .type = Type::kFP32, .shape = {3}, .buffer = std::vector<float>{1, 2, 3}});
  XnnTensor b({.name = "b", .type = Type::kFP32, .shape = {3}, .buffer = std::vector<float>{4, 5, 6}});
  XnnTensor c = Add(a, b);

  // 2. A runner for the outputs you want: the graph is traced backwards from them.
  LRT_TENSOR_ASSIGN_OR_ABORT(XnnpackRunner runner, XnnpackRunner::Create({c}));
  LRT_TENSOR_ABORT_IF_ERROR(runner.Run());

  // 3. Read back through a typed, locked span.
  LRT_TENSOR_ASSIGN_OR_ABORT(auto out, runner.ReadOutputAs<float>(c));
  for (float v : out) std::cout << v << ' ';   // 5 7 9
}
```

```python
# BUILD
cc_binary(
    name = "add_xnnpack",
    srcs = ["add_xnnpack.cc"],
    deps = [
        "//tensor",
        "//tensor:arithmetic",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:macros",
    ],
)
```

### B. As a LiteRT model (`TfLiteMixinTag`)

The same graph serialized to a flatbuffer and run by LiteRT's `CompiledModel`
— the path to use when the graph has to run on GPU/NPU through LiteRT, be
shipped as a `.tflite`, or sit in a `ModelChain`.

```cpp
#include <iostream>
#include <vector>

#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "tensor/arithmetic.h"
#include "tensor/runners/litert/lambda_model_runner.h"
#include "tensor/tensor.h"

using namespace litert::tensor;

int main() {
  // 1. A LiteRT environment and the accelerator to use.
  auto env = std::move(*litert::Environment::Create({}));
  auto options = std::move(*litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);   // or kGpu

  // 2. The graph: input prototypes (name, type, shape) and a lambda over them.
  auto runner = CreateLambdaRunner(
      env, options,
      {{"a", Tensor<TfLiteMixinTag>({.name = "a", .type = Type::kFP32, .shape = {3}})},
       {"b", Tensor<TfLiteMixinTag>({.name = "b", .type = Type::kFP32, .shape = {3}})}},
      [](const auto& inputs) {
        Tensor c = Add(inputs.at("a"), inputs.at("b"));
        return absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>{{"c", c}};
      });

  // 3. Bind data and run. The graph is serialized and compiled on first use.
  runner.SetInput("a", Create("a", Type::kFP32, {3}, std::vector<float>{1, 2, 3}));
  runner.SetInput("b", Create("b", Type::kFP32, {3}, std::vector<float>{4, 5, 6}));
  runner.Run();

  // 4. Read the output through a locked buffer span.
  auto c = std::move(*runner.GetOutput("c"));
  auto span = c.GetBuffer().value()->Lock();
  const float* data = reinterpret_cast<const float*>(span.data());
  std::cout << data[0] << " " << data[1] << " " << data[2] << "\n";  // 5 7 9
}
```

```python
# BUILD
cc_binary(
    name = "add_litert",
    srcs = ["add_litert.cc"],
    deps = [
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/runners/litert:lambda_model_runner",
    ],
)
```

## Building the library

### Bazel, inside the LiteRT repository

The Tensor API is part of the LiteRT Bazel workspace; nothing extra to set up.

```bash
git clone https://github.com/google-ai-edge/LiteRT.git && cd LiteRT

bazel build //tensor/...                                   # library + examples
bazel test  //tensor:all //tensor/backends/tflite:all      # unit tests
bazel build //tensor/examples/segmentation:segmentation_example
```

Android cross-builds use the repository's `--config=android_arm64` (set
`ANDROID_NDK_HOME` etc. as described in
[`examples/segmentation/README.md`](examples/segmentation/README.md#bazel-on-android)).
GPU execution on device needs the matching accelerator shared library
(`libLiteRtClGlAccelerator.so` on Android, a Metal `.dylib` on macOS, a WebGPU
`.so`/`.dll` on Linux/Windows); the example targets bring it into runfiles via
`litert_gpu_accelerator_prebuilts()`.

### Bazel, from your own workspace

Add LiteRT as an external repository and depend on the `//tensor/...` targets.
[litert-samples](https://github.com/google-ai-edge/litert-samples/blob/main/WORKSPACE)
does exactly this (`@litert_archive`), and its
[SAM 2 `cc/BUILD`](https://github.com/google-ai-edge/litert-samples/blob/main/samples/web_demos/src/sam2/cc/BUILD)
is a complete real-world example of the deps an app needs:

```python
deps = [
    "@litert_archive//tensor",
    "@litert_archive//tensor:arithmetic",
    "@litert_archive//tensor/backends/tflite:arithmetic_tflite",
    "@litert_archive//tensor/backends/tflite:tflite_flatbuffer_conversion",  # ModelFactory
    "@litert_archive//tensor/runners:model_chain",
    "@litert_archive//tensor/runners/litert:litert_buffer",
    "@litert_archive//litert/cc:litert_compiled_model",
    "@litert_archive//litert/cc:litert_environment",
]
```

| You want | Depend on |
|---|---|
| Tensors and ops | `//tensor`, `//tensor:arithmetic`, `//tensor:buffer`, `//tensor:datatypes` |
| Serialize to `.tflite` (`ModelFactory`) | `//tensor/backends/tflite:tflite_flatbuffer_conversion`, `:arithmetic_tflite` |
| Run a `.tflite` by name (`LitertDynamicRunner`) | `//tensor/runners/litert:litert_dynamic_runner` |
| Build-and-run a graph (`CompiledModelRunner`, `CreateLambdaRunner`) | `//tensor/runners/litert:compiled_model_runner`, `:lambda_model_runner` |
| Multi-stage zero-copy pipelines | `//tensor/runners:model_chain`, `//tensor/runners/litert:litert_buffer` |
| XNNPACK directly (no flatbuffer) | `//tensor/backends/xnnpack:arithmetic`, `//tensor/runners/xnnpack:runner` |

### CMake

[`CMakeLists.txt`](CMakeLists.txt) is a subdirectory of the LiteRT CMake tree
(it links `litert_cc_internal`, `tensorflow-lite`, Abseil and FlatBuffers from
the parent project). It defines `litert_tensor_api_core`,
`litert_tensor_api_internal`, `litert_tensor_api_utils` and
`litert_tensor_api_tflite_backend`, and adds the segmentation and Gemma 3
examples when present. Configure from the LiteRT root:

```bash
cmake -S litert -B cmake_build/host -DCMAKE_BUILD_TYPE=Release -DLITERT_ENABLE_GPU=ON
cmake --build cmake_build/host --target litert_tensor_segmentation_example --parallel
```

For Android pass the NDK toolchain (`-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake
-DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26`); see
[`examples/segmentation/README.md`](examples/segmentation/README.md#cmake-on-android)
and [`examples/gemma3/README.md`](examples/gemma3/README.md). The CMake build
covers the core and the TFLite backend; the XNNPACK backend and runner are
Bazel-only today.

### WebAssembly

Author the pipeline in C++, compile it with Emscripten and hand the resulting
`.tflite` bytes to LiteRT.js (WebGPU) in the page. The SAM 2 demo's
[`wasm/CMakeLists.txt` and `build.sh`](https://github.com/google-ai-edge/litert-samples/tree/main/samples/web_demos/src/sam2/wasm)
show the full recipe: core + TFLite backend + `ModelChain` built with
`emcmake`, a small JS bridge that binds LiteRT.js `Tensor`s to
`LitertBuffer`s, and the browser doing all the GPU work.

## Integrating into an existing app

Four patterns, in increasing order of how much of your pipeline the Tensor API
owns. They compose: the SAM 2 demo uses all of them.

### 1. Author a model and ship the `.tflite`

Replace a converter step, or build a model that no converter produces. The
output is a plain flatbuffer; the app that runs it does not need the Tensor API.

```cpp
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/tensor.h"

using namespace litert::tensor;
using T = Tensor<TfLiteMixinTag>;

// ImageNet-style pre-processing: RGBA frame in [0,1] -> normalized RGB at SxS.
absl::Status WritePreprocessModel(int h, int w, int s, const std::string& path) {
  T frame({.name = "frame", .type = Type::kFP32, .shape = {1, h, w, 4}});
  T rgb = Slice(frame, {0, 0, 0, 0}, {1, h, w, 3});
  T resized = ResizeBilinear(rgb, {s, s});
  T mean({.type = Type::kFP32, .shape = {3}, .buffer = std::vector<float>{0.485f, 0.456f, 0.406f}});
  T stdv({.type = Type::kFP32, .shape = {3}, .buffer = std::vector<float>{0.229f, 0.224f, 0.225f}});
  T pixels = Div(Sub(resized, mean), stdv);
  pixels.SetName("pixels");

  ModelFactory factory;
  LRT_TENSOR_RETURN_IF_ERROR(factory.AddSignature({pixels}, "preprocess"));
  return factory.Save(path);           // or factory.CreateFlatbuffer() for bytes
}
```

*   Signature inputs/outputs are named after the tensors.
*   Call `AddSignature` once per entry point; tensors (weights) reachable from
    several signatures are stored once. The SAM 2 model packs encode, eight
    prompt variants and two tracking variants this way.
*   Constants can be large: `.buffer = OwningCpuBuffer::Copy<Type::kFP32>(...)`
    or a loaded weight span (`examples/utils/safetensor_loader.h` reads
    safetensors). Quantized constants set `TensorInit::quantization`
    (`PerChannelAffineQuantization`, `BlockwiseQuantization`, see `tensor.h`)
    on an int8/int4 buffer.
*   `AddSignature(inputs, outputs, name)` keeps declared inputs that are not
    connected to an output — useful for optional ports.

### 2. Put pre/post-processing next to an existing model, zero-copy

Keep your current `.tflite`; express the loops around it as graphs and bind
their outputs straight into the model's inputs.

```cpp
#include "tensor/runners/litert/lambda_model_runner.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"

auto pre = CreateLambdaRunner(env, options,
    {{"raw", Tensor<TfLiteMixinTag>({.name = "raw", .type = Type::kFP32, .shape = {1, 512, 512, 3}})}},
    [](const auto& in) {
      Tensor x = ResizeBilinear(in.at("raw"), {256, 256});
      x = Add(Mul(x, 2.0f), -1.0f);
      return absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>{{"normalized", x}};
    });

auto core = LitertDynamicRunner::Create(env, "core_model.tflite", options).value();

pre.SetOutput("normalized", core.GetInput(0).value());   // same buffer, no copy
pre.SetInput("raw", Create("raw", Type::kFP32, {1, 512, 512, 3}, std::move(frame)));
pre.Run();
core.Run();
```

`LitertDynamicRunner` addresses tensors by signature and name
(`SetInput("default", "input", t)`, `GetOutput("default", "output")`), so
nothing breaks when the model's tensor order changes. Feedback loops
(`FeedbackLoopConfig`) route an output back to an input between runs for KV
caches and recurrent state. The
[segmentation example](examples/segmentation/README.md) is this pattern end to
end — on a Samsung S25 the GPU zero-copy pipeline runs in **10 ms** against
**83 ms** for the CPU path.

### 3. Chain several models with `ModelChain`

When a pipeline has more than one model, let `ModelChain` own the plumbing: it
discovers each stage's I/O from the model signatures, negotiates a buffer
layout at every boundary and allocates the shared buffers once.

```cpp
#include "litert/cc/litert_environment.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/runners/model_chain.h"

using namespace litert::tensor;

auto env = std::make_shared<litert::Environment>(std::move(*litert::Environment::Create({})));
auto gpu = litert::HwAccelerators::kGpu;

// Stages: a .tflite file (or buffer / CompiledModel) + a signature.
auto pre    = CompiledModelStage::Create(env, "preprocess", "frame_model.tflite", gpu, /*signature_index=*/0).value();
auto encode = CompiledModelStage::Create(env, "encode",     "sam2.tflite",        gpu, /*signature_index=*/0).value();

ModelChain::Builder b;
b.WithEnvironment(env)
    .AddStage(pre)
    .AddStage(encode)
    .Connect("preprocess", "pixels", "encode", "pixels");   // output -> input
auto chain = b.Build().value();

// Entry input: any LitertBuffer (host, AHWB, OpenCL, WebGPU...). Allocate it
// from the stage's own descriptor so layout and alignment match.
auto desc  = pre->GetInputDescriptor("frame").value();
auto frame = LitertBuffer::CreateManagedHost(env, desc.shape, desc.element_type, desc.PackedBytes()).value();
chain.SetInputBuffer("frame", frame);

// Per frame: write pixels, Execute(), read / hand off outputs.
chain.Execute();
auto features = chain.GetOutputBuffer("encode", "feat_s0").value();  // stays on the GPU
```

*   A `FunctionalModelStage` runs host code (a lambda over locked input/output
    buffers) as a stage, for the bits that are not a model.
*   Stages added without `Connect` are linked in order by matching names.
*   `GetIntermediateBuffers()` exposes the shared buffers for inspection.
*   Because every memory slot of SAM 2's tracker is an input, feeding the
    memory bank is pure buffer binding: `SetInputBuffer("mem_3", older_output)`
    points a stage at a buffer an earlier frame wrote.

### 4. Run a graph directly on XNNPACK

For CPU paths where latency matters — LLM decode, samplers, audio front-ends —
skip the flatbuffer entirely: `XnnpackMixinTag` lowers the graph to an XNNPACK
subgraph and `XnnpackRunner` executes it with a thread pool and, optionally, a
weight cache. This is how `examples/gemma4` runs Gemma 4 prefill and decode.

Write the block once, templated on the mixins, so the same function also
serves the `.tflite` path:

```cpp
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

using namespace litert::tensor;

// Gated MLP (GeGLU), backend-agnostic. Weights are [out, in].
template <class... Mixins>
Tensor<Mixins...> GatedMlp(Tensor<Mixins...> x, Tensor<Mixins...> w_gate,
                           Tensor<Mixins...> w_up, Tensor<Mixins...> w_down) {
  Tensor gate = Gelu(FullyConnected(x, w_gate), /*approximate=*/true);
  Tensor up = FullyConnected(x, w_up);
  return FullyConnected(Mul(gate, up), w_down);
}

absl::Status Decode(const Weights& w, absl::Span<const std::vector<float>> steps) {
  using T = Tensor<XnnpackMixinTag>;

  // A tensor without a buffer is an external input: data is bound per run.
  T x({.name = "x", .type = Type::kFP32, .shape = {1, 8, 256}});
  // Tensors with a buffer are constants baked into the graph. SpanCpuBuffer
  // views memory you own (no copy); OwningCpuBuffer::Copy takes a copy.
  T w_gate({.name = "w_gate", .type = Type::kFP32, .shape = {1024, 256},
            .buffer = std::make_shared<SpanCpuBuffer>(w.gate)});
  T w_up({.name = "w_up", .type = Type::kFP32, .shape = {1024, 256},
          .buffer = std::make_shared<SpanCpuBuffer>(w.up)});
  T w_down({.name = "w_down", .type = Type::kFP32, .shape = {256, 1024},
            .buffer = std::make_shared<SpanCpuBuffer>(w.down)});
  T y = GatedMlp(x, w_gate, w_up, w_down);
  y.SetName("y");

  // Lower once. Outputs define the graph; everything they depend on is included.
  LRT_TENSOR_ASSIGN_OR_RETURN(XnnpackRunner runner, XnnpackRunner::Create({y}));
  runner.SetNumThreads(4);                      // pthreadpool for the kernels
  LRT_TENSOR_RETURN_IF_ERROR(runner.PrepareRuntime());   // optional: pay setup now, not on first Run()

  // Prefill: the full 8-token window.
  LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput(x, steps[0]));   // lvalue: bound by reference, no copy
  LRT_TENSOR_RETURN_IF_ERROR(runner.Run());

  // Decode: one token at a time. Reshape the input; the runtime re-plans.
  LRT_TENSOR_RETURN_IF_ERROR(runner.ReshapeInput(x, {1, 1, 256}));
  for (const std::vector<float>& token : steps.subspan(1)) {
    LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput(x, token));
    LRT_TENSOR_RETURN_IF_ERROR(runner.Run());
    LRT_TENSOR_ASSIGN_OR_RETURN(LockedBufferSpan<const float> out, runner.ReadOutputAs<float>(y));
    Sample(out);                                 // out.data(), out.size()
  }
  return absl::OkStatus();
}
```

Notes:

*   `SetInput(tensor, seq)` with an lvalue keeps a reference — the data must
    outlive the run. `SetInputAsCopy` copies; `SetInput(tensor, span_of_bytes)`
    binds raw memory. `SetOutput` lets you point an output at your own buffer.
*   `ReadOutputAs<T>` returns a `LockedBufferSpan`; it is a view into the
    runner's output, valid until the next `Run()`.
*   Large models: `runner.SetWeightsCache(xnn_weights_cache_t)` shares packed
    weights between runners (Gemma 4 uses one cache for its prefill and decode
    graphs and persists it with `--weight_cache`), and feeding the KV cache is
    `SetInput` on the cache tensors plus `ReshapeInput` as it grows — see
    `UpdateKvCache` in [`examples/gemma4/gemma4_runtime.h`](examples/gemma4/gemma4_runtime.h).
*   Custom ops: an XNNPACK lowering is a specialization of
    `OpMixin<YourOperation, XnnpackMixinTag>` with a `ToXnnpack()` method —
    [`examples/ops/transformer/transformer_ops_xnnpack.h`](examples/ops/transformer/transformer_ops_xnnpack.h)
    does this for RMSNorm, RoPE tables and attention masks.
*   To compare against LiteRT or export, instantiate the same `GatedMlp` with
    `Tensor<TfLiteMixinTag>` (pattern 1), or with both tags at once
    (`Tensor<TfLiteMixinTag, XnnpackMixinTag>`) and run the graph on either
    backend from one definition.

## Verifying backends

*   **`GraphProbe`** (`internal/graph_probe.h`) — register any intermediate
    tensor as an extra model output, run the graph once on the accelerator and
    once on CPU, and compare:

    ```cpp
    absl::flat_hash_map<GraphProbe::StableTensorId, std::string, GraphProbe::StableTensorIdHash> probes;
    probes[{attention_op_id, 0}] = "attn_probe";
    gpu_runner.AddTensorsAsOutputs(probes);
    cpu_runner.AddTensorsAsOutputs(probes);
    gpu_runner.Run(); cpu_runner.Run();
    EXPECT_THAT(gpu_runner.GetFloatOutput("attn_probe").value(),
                Pointwise(FloatNear(1e-3), cpu_runner.GetFloatOutput("attn_probe").value()));
    ```

*   **Numerical test suite** (`backends/testing/`) — a type-parameterized GTest
    suite that exercises the op set through a `TestBackendBridge`; implement
    the bridge for a backend to get 50+ op tests for free.
*   **Golden models** — the SAM 2 pipeline keeps a PyTorch reference and checks
    every signature against it (`tools/verify_chain.py` in the sample); the
    segmentation example compares its PNG to a golden image.

## Examples

| Example | Shows | Build / run |
|---|---|---|
| [SAM 2 video segmentation](https://github.com/google-ai-edge/litert-samples/tree/main/samples/web_demos/src/sam2) (litert-samples) | A full video model authored with the Tensor API (11 signatures, shared weights), a frame model built at run time, `ModelChain` with GPU-resident memory bank, wasm build, LiteRT.js on WebGPU; Gemma 4 in the browser turns language into prompts and tool calls. ~30 fps (one object, 384 px model, 720p video) in Chrome on an M4 Pro. | see its README |
| [`examples/gemma4`](examples/gemma4) | Gemma 4 (E2B / E4B) prefill + decode graphs on XNNPACK with KV cache, safetensors weights, weight cache, Perfetto tracing | `bazel build //tensor/examples/gemma4:gemma4_xnnpack_main` then `--weights=… --tokenizer=… --prompt=…` |
| [`examples/gemma3`](examples/gemma3/README.md) | Gemma 3 as a `.tflite` run through LiteRT `CompiledModel`, CPU/GPU, Bazel and CMake, Android | `bazel build //tensor/examples/gemma3:litert_main` |
| [`examples/segmentation`](examples/segmentation/README.md) | Pre/post-processing around an existing selfie-segmentation model, zero-copy GPU, golden check; 10 ms vs 83 ms CPU on S25 | `bazel build //tensor/examples/segmentation:segmentation_example` |
| [`examples/ops/transformer`](examples/ops/transformer) | Custom ops (RMSNorm, RoPE, attention masks) as `Operation`s with a graph-level definition for the `.tflite` path and a native XNNPACK lowering | `bazel build //tensor/examples/ops/transformer:transformer_ops_xnnpack` |

## Repository layout

```
tensor/
├── tensor.h, buffer.h, datatypes.h   core types: TensorHandle, Tensor<Mixins...>, buffers, Type/Shape
├── arithmetic.h                      the op set (free functions + operators)
├── backends/
│   ├── tflite/                       TfLiteMixinTag, ModelFactory (.tflite serialization)
│   ├── xnnpack/, common_nnpack/      XnnpackMixinTag, XNNPACK subgraph lowering
│   └── testing/                      cross-backend numerical test suite
├── runners/
│   ├── model_chain.h                 ModelChain, ModelStage, CompiledModelStage, FunctionalModelStage
│   ├── litert/                       CompiledModelRunner, LitertDynamicRunner, LambdaRunner, LitertBuffer
│   └── xnnpack/, common_nnpack/      XnnpackRunner
├── internal/                         graph IR, GraphProbe, shape inference, mixin registry
├── utils/                            status macros (LRT_TENSOR_RETURN_IF_ERROR, ...), matchers
├── examples/                         see above
├── tensor_api.md                     API reference
└── CMakeLists.txt                    CMake targets (core + TFLite backend)
```

## Further reading

*   [`tensor_api.md`](tensor_api.md) — API reference: tensors, buffers, the op
    list, LiteRT runners.
*   [`backends/testing/README.md`](backends/testing/README.md) — the numerical
    test suite.
*   [LiteRT](https://github.com/google-ai-edge/LiteRT) — the runtime, C++ API
    (`litert/cc`), accelerators.
*   [litert-samples](https://github.com/google-ai-edge/litert-samples) — SAM 2
    graphs (`models/sam2/sam2_hiera_tiny_video/tensor_api`) and the web demo.
