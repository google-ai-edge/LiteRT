# Gemma4 on the XNNPACK tensor runner

The safetensors example runs the text portion of Gemma4 E2B/E4B through tensor
expression graphs and XNNPACK CPU kernels. It reads safetensors directly;
no `.tflite` conversion or LiteRT interpreter is involved. Image/audio towers
and their inputs are outside this example.

The separate [native runner](native/README.md) executes the published-bundle coefficients through staged XNNPACK graphs, compact INT2 weights and active INT8 KV windows. These two entry points share tensor/model building blocks but have different cache and execution management.

## Components and execution

| Component | Responsibility |
| --- | --- |
| [xnnpack_main.cc](xnnpack_main.cc) | Command-line input, checkpoint loading, host embedding lookup, separate prefill/decode runners, masks and RoPE, growing KV buffers, greedy generation, timings, and logit dumps. |
| [gemma4_config.h](gemma4_config.h) | E2B/E4B dimensions, local/global attention pattern, RoPE parameters, and KV-sharing presets. |
| [gemma4_weights.cc](gemma4_weights.cc) | Hugging Face text-weight names to graph-weight names. |
| [gemma4_graph.h](gemma4_graph.h) | Embedding scaling, per-layer input projection, transformer stack and shared KV routing, final normalization, head projection, and logit softcap. |
| [helpers](helpers/README.md) | Attention, MLP, RMSNorm/RoPE integration, quantized embedding lookup, and mobile fully connected layers. |
| [reference_mobile.py](reference_mobile.py) | Independent Transformers CPU reference using the same mobile checkpoint, dequantized FP32 weights, static activation quantization, and lazy embedding rows. |
| [text_io.py](text_io.py) | Offline host tokenization and output decoding using the checkpoint's Hugging Face tokenizer. |
| [compare_logits.py](compare_logits.py) | Compare finite, complete logit vectors with matching token histories; report argmax, cosine, RMSE, and maximum absolute error. |
| `*_test.cc` and [helper references](helpers/reference/README.md) | Small numerical and metadata tests, including full prefill versus repeated cached decode. |

The driver maps the checkpoint, translates weight names, and creates token and
per-layer embedding accessors. It slices the combined per-layer projection into
one matrix per layer. Model variant detection uses the final normalization
width; this driver targets the supplied E2B/E4B presets. Actual loaded MLP
matrix shapes supply E2B's wider shared-layer MLPs.

Prefill receives all prompt embeddings, constant causal/sliding masks, and
position-zero RoPE tables. Its last-position logits predict the first token.
The driver then copies the owning layers' KV outputs into host buffers and
feeds one predicted token at a time to the decode runner. Each step binds new
embedding/mask/RoPE data, grows the bound cache shape, and appends new K/V
values. Shared layers reuse the last owning layer of the same attention type.
The current driver supports batch size one and retains growing FP32 KV caches,
including for sliding attention; it does not implement a ring buffer.

The native packed-weights cache is created before both runners, populated by
both `PrepareRuntime()` calls, and finalized before execution. It outlives the
runtimes. Source constants and embedding accessors also remain alive during
inference. Borrowed host input views are rebound for each call and their
storage remains live through `Run()`.

## Checkpoints and quantization

Google publishes mobile compressed-tensors checkpoints:

- [google/gemma-4-E2B-it-qat-mobile-ct](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-ct)
- [google/gemma-4-E4B-it-qat-mobile-ct](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-ct)

Use a local snapshot containing `model.safetensors`, `config.json`, and tokenizer
files. Keep `config.json` beside the weights, including on Android. The
[loader](../utils/README.md) resolves each module's exact/regex
quantization targets, physical packed shape, scale tensors, and activation
scales. These mobile checkpoints mix 2/4/8-bit weights. Two-bit weights expand
to signed INT4 storage without changing their quantized values; this is a
format adaptation, not another quantization pass. E2B's per-layer embedding is
4-bit groupwise; E4B's is 2-bit groupwise. Both use groups of 256 there.

Static INT8 input/output activation scales are applied to the corresponding
linear layers. Residuals, normalization, attention, softmax, and KV storage use
FP32. Ordinary BF16/FP16 weights used by those graph operations are converted
to FP32. An explicitly stored `lm_head.weight` is preferred over tied embedding
weights. The head preserves FP32 activations with a local XNNPACK FC extension;
its INT4 weights are converted to the unsigned nibble representation expected
by that kernel while remaining packed. The safetensors driver uses FP32 KV storage; the separate native runner uses an INT8 cache.

The native FP32 head path requires an even number of input channels for INT4
weights. It also requires at least two output channels and symmetric per-channel
weight scales on axis zero. Both supplied model presets satisfy these
constraints; other shapes are rejected during graph construction.

Google's mobile snapshots contain a BPE `tokenizer.json`, which is not a
SentencePiece model. Use `text_io.py` on the host with `--token_ids` in C++.
The retained `--tokenizer` path accepts a real, compatible SentencePiece
`tokenizer.model`; it has not been validated against these mobile snapshots.
`--token_ids` supplies the exact input sequence and adds no BOS or chat wrapper.


## Build and test

Use the [standalone build guide](../../standalone/README.md) for Linux and Android setup, dependency pinning, test commands and device selection. The standard safetensors binary has Bazel target //tensor/examples/gemma4:gemma4_xnnpack_main. The staged published-bundle binary is //tensor/examples/gemma4/native:gemma4_native_runner; its [README](native/README.md) describes the matched-bundle input and fixture format.

Small tests cover graph construction, full prefill versus cached decode, embedding validation, INT4 logical dimensions, static activation rounding/saturation and FP32-activation quantized FC. The shared Python reference fixtures and compare_logits.py provide numerical diagnostics; their scope is distinct from end-to-end model quality evaluation.

The [LiteRT-LM comparison guide](native/REPRODUCE.md) explains how to build the
native runner and compare CPU performance using the same model and workloads.
Model weights and runtime captures are separate from this source tree.

## Extension points

Change checkpoint indexing and compressed-tensors metadata in [the shared loader](../utils/README.md). Change mathematical model composition in gemma4_graph.h and [helpers](helpers/README.md), keeping small reference cases aligned. Change published-bundle staging, active KV windows and shared runtime resources in [native](native/README.md). Preserve the explicit distinction between safetensors checkpoint semantics and coefficients extracted from a published LiteRT-LM bundle.
