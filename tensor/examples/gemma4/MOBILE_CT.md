# Gemma 4 mobile compressed-tensors checkpoints

The [standard example](xnnpack_main.cc) can load local Gemma 4 E2B/E4B
safetensors and mobile compressed-tensors (CT) snapshots. It keeps the public
Gemma 4 graph and growing FP32 KV caches. For the optimized E2B runner with a
published-bundle-matched INT8 KV layout, compact INT2 constants and shared
workspace, use the separate [native runner](native/README.md).

## Loading and arithmetic

[SafetensorLoader](../utils/safetensor_loader.h) reads a safetensors file or a
directory, with optional CT configuration from its sibling `config.json`.
It resolves quantization groups and module targets, reads logical weight shapes,
and validates packed storage, channel scales and static activation scales.
The mobile path accepts 2-, 4- and 8-bit packed coefficients. Its 2-bit weights
are expanded losslessly into signed packed INT4 storage; this public loader does
not provide the native runner's compact static INT2 execution.

[MobileFullyConnected](helpers/mobile_fully_connected.h) applies the checkpoint's
static INT8 input/output scales around quantized linear operations. When a
channelwise INT4/INT8 projection has no activation scale metadata,
[FloatActivationFullyConnected](helpers/float_activation_fully_connected.h)
keeps its input in FP32. Missing activation metadata and static activation
quantization therefore select different arithmetic paths. The graph uses an
explicit `lm_head.weight` when supplied, otherwise it ties the output head to the
token embedding table.

[GemmaEmbeddingTable](helpers/quantized_embedding.h) treats per-channel INT4 shapes
as logical element dimensions, validates packed row boundaries, and decodes only
the requested BF16/FP16 embedding rows. Tensors retain the mapping or allocation
that backs their data. Loaded weights and the weight-cache provider outlive both
runners.

The first prediction comes from the final prompt position during prefill and
counts toward `--max_tokens`. A limit of one runs prefill alone. A single-token
prompt is a prefill with no cached history. Decode reuses its runner with growing
KV input shapes. Single-KV-head attention broadcasts directly in batched matrix
multiplication; the multi-KV-head path retains explicit tiling.

## Raw token IDs and numerical diagnosis

Build the normal example through the repository's Bazel setup:

```sh
bazelisk build //tensor/examples/gemma4:gemma4_xnnpack_main
```

Mobile snapshots commonly provide a Hugging Face `tokenizer.json` rather than a
SentencePiece model. [text_io.py](text_io.py) uses local tokenizer files to encode
text or apply the snapshot's chat template. It requires Transformers and the
checkpoint's tokenizer dependencies and does not download files:

```sh
python tensor/examples/gemma4/text_io.py --model /path/to/mobile-ct encode \
  --chat --prompt 'Write a short poem about coding.'
```

Pass the printed comma-separated IDs to `--token_ids`. This bypasses tokenizer
loading, BOS insertion and prompt wrapping. The usual `--tokenizer` and `--prompt`
SentencePiece path is also retained. For a short deterministic diagnostic:

```sh
mkdir -p /tmp/gemma4-ct-check
bazel-bin/tensor/examples/gemma4/gemma4_xnnpack_main \
  --weights=/path/to/mobile-ct \
  --token_ids=2,818,5279,529,7001,563 \
  --num_threads=4 --max_tokens=1 \
  --dump_logits=/tmp/gemma4-ct-check/cpp
```

Choose a fresh output prefix. The example writes `.prefill.f32`, subsequent
`.decode_NNNN.f32` files when requested, and a `.json` manifest containing exact
input/generated IDs, the stop reason and runtime flags. Logits are little-endian
FP32 and the full vocabulary is checked for finiteness before greedy selection.
`--dump_intermediates=true` additionally exports each layer's hidden states and
the final normalization output. These extra external outputs and file writes
change execution and timing; use the native fixed-history harness for performance
comparisons.

`--consistent_arithmetic=true` passes XNNPACK's
`XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC` runtime flag. This diagnostic mode is limited
to E2B's single KV head; it does not promise cross-platform bitwise equality.
The upstream `--weight_cache=/path/to/cache` (or `:auto`) file-cache interface and
`--perfetto_output=/path/to/trace` remain available.

[reference_mobile.py](reference_mobile.py) provides a separate Transformers FP32
execution reference with CT coefficients and static INT8 input/output
quantization. It requires PyTorch, safetensors, NumPy, and Transformers with Gemma
4 support. It uses FP32 KV caches, dequantizes active linear weights into FP32,
and can consume substantially more host memory than the C++ example. For the
same one-prediction comparison:

```sh
python tensor/examples/gemma4/reference_mobile.py \
  --model-dir /path/to/mobile-ct --input-ids 2,818,5279,529,7001,563 \
  --decode-steps 0 --threads 4 \
  --output-prefix /tmp/gemma4-ct-check/reference
python tensor/examples/gemma4/compare_logits.py \
  /tmp/gemma4-ct-check/reference /tmp/gemma4-ct-check/cpp --steps 1
```

[compare_logits.py](compare_logits.py) checks input histories, dimensions,
finiteness and recorded argmax, then reports cosine similarity, RMSE and maximum
absolute error. Later autoregressive steps are comparable only while preceding
generated IDs match. Its diagnostic thresholds are not a model-quality
acceptance test.

## Regression coverage and validation scope

The source includes regression tests for mixed CT loading and malformed metadata,
embedding dimensions and lazy rows, static activation rounding/saturation,
FP32-input quantized projections, explicit/tied output heads, singleton KV-head
broadcast, and cached decode versus full prefill. The focused Bazel targets are:

```sh
bazelisk test \
  //tensor/examples/utils:safetensor_loader_test \
  //tensor/examples/gemma4/helpers:quantized_embedding_test \
  //tensor/examples/gemma4/helpers:mobile_fully_connected_test \
  //tensor/examples/gemma4/helpers:float_activation_fully_connected_test \
  //tensor/examples/gemma4/helpers:attention_test \
  //tensor/examples/gemma4:gemma4_graph_test
```

The standalone build excludes only the loader's Perfetto scopes and the graph
test's TFLite file-cache-specific case via `LITERT_TENSOR_STANDALONE`; the regular
Bazel paths retain them. A portable in-memory weight-cache regression remains in
the standalone graph test. Consult the [native validation instructions](native/README.md)
for the consolidated build and its recorded test results. The raw CT commands
above are usage examples, not a claim that the migrated CLI or Transformers
reference has been run on a full checkpoint.
