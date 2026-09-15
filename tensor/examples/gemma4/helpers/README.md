# Gemma4 graph and host helpers

These components support the [Gemma4 graph and generation loop](../README.md).
Graph helpers assemble ordinary tensor expressions and
[transformer operations](../../ops/transformer/README.md); host helpers prepare
token embeddings and positional tables before the XNNPACK runner executes them.
The shared helpers include mobile activation quantization and embedding validation/conversion support.

## Component map

| File or directory | Responsibility |
| --- | --- |
| [attention.h](attention.h) | Weight lookup, Q/K/V projections, normalization, RoPE, grouped-query attention, masking, and output projection. Returns current K/V plus the combined K/V used for attention. |
| [feed_forward_network.h](feed_forward_network.h) | Up and gate projections, approximate GELU gating, and down projection. |
| [transformer.h](transformer.h) | Pre/post normalization, attention and feed-forward residual connections, optional per-layer embedding branch, and layer scaling. |
| [mobile_fully_connected.h](mobile_fully_connected.h) | Uses loaded scale metadata to select static INT8 activations, native FP32 activations with channelwise quantized weights, or ordinary floating fully connected lowering. |
| [float_activation_fully_connected.h](float_activation_fully_connected.h), [float_activation_fully_connected.cc](float_activation_fully_connected.cc) | Defines and lowers the native FP32 activation path used by the output head, retaining packed constant weights. |
| [rope.h](rope.h) | Computes cosine/sine values on the host for a position range, including partial rotary dimensions; also offers an allocating tensor overload. |
| [quantized_embedding.h](quantized_embedding.h), [quantized_embedding.cc](quantized_embedding.cc) | Validates embedding tables and provides batch, single-token, and per-layer row lookup. |
| [reference](reference/README.md) | NumPy mathematical fixtures used to produce small C++ numerical expectations. |
| `*_test.cc` | Focused numerical and validation cases alongside each helper, including RMS normalization implemented in the transformer operation directory. |

## Attention and cache flow

`TransformerLayer` normalizes its input, runs attention, adds the residual,
then applies the gated feed-forward block and its residual. An optional
per-layer embedding branch adds a gated projection before the final layer
scalar. The enclosing graph chooses local/global masks, head dimensions, and
which earlier layer supplies shared K/V.

Attention projects and reshapes Q/K/V into
`[batch, heads, sequence, head_dimension]` tensors, normalizes Q and K with
learned scales, normalizes V without a scale, and applies split-half
RoPE to Q/K. For an unshared layer, it concatenates existing caches with newly
computed K/V along the sequence axis. With one KV head, both attention matrix
products broadcast that head directly to the query heads. This avoids an
explicit Tile/broadcast node and allows E2B to run when consistent arithmetic
disables XNNPACK's broadcast rewrite. With multiple KV heads, grouped-query
attention still repeats each head through slicing, tiling, and concatenation.
Attention computes scores, applies optional softcapping and the supplied mask,
then runs softmax and the value product.

The returned `key_cache`/`value_cache` contain the new positions for an unshared
layer. `key_for_attn`/`value_for_attn` contain the combined cache and current
positions; a layer sharing K/V reuses these tensors. The caller owns persistent
cache storage and decides when to append data, reshape inputs, and bind updated
views. These graph helpers do not mutate a host cache in place.

`GetWeight` returns a matching tensor from the weight map, or creates an
unbuffered placeholder if the name is absent. Consequently, callers must load
required constants or bind those placeholders before execution. Preserve
canonical weight names when loading tensors: the static activation path derives
its metadata keys from the weight tensor's `.weight` suffix.

## Static activation scales

`MobileFullyConnected` recognizes paired `<module>.input_scale` and
`<module>.output_scale` entries. Both must be positive finite FP32 scalars.
With FP32 input and quantized INT4/INT8 weights it builds:

`FP32 → Cast to INT8(input scale, zero point 0) → FullyConnected → INT8(output scale, zero point 0) → Cast to FP32`.

These casts make activation rounding and saturation part of the graph. Missing
one scale or supplying incompatible types produces an error tensor. When
neither activation scale is present, channelwise quantized weights use
[FloatActivationFullyConnected](float_activation_fully_connected.h). This
example-local operation preserves FP32 activations instead of applying the
core backend's dynamic input quantization. The output head uses this path.
Its [lowering](float_activation_fully_connected.cc) converts signed INT4
nibbles into a private offset-binary buffer and defines the native XNNPACK
value with zero point 8, as required by the FP32 INT4 kernel. Packed storage
and scale owners remain attached to the built graph. Floating weights use
the ordinary floating path.

`FloatActivationFullyConnected` requires symmetric channelwise INT4/INT8
constant weights, one positive finite scale per output channel on axis zero,
and at least two output channels. INT4 input channels must be even for the
portable native kernel path; the helper rejects odd widths. Input and output
activations stay unquantized FP32, and the operation preserves leading input
dimensions.

## Embedding storage and lifetimes

`GemmaEmbeddingTable::Create` checks dimensions, backing byte capacity, and
quantization layout before lookup. It supports FP32, BF16, FP16, and signed
INT4/INT8 tables with per-row or blockwise scales. Packed INT4 rows must have
even logical widths; legacy byte-width shapes can be resolved using an explicit
expected dimension or blockwise metadata. Quantization uses axis zero, valid
group sizes, and matching scale/zero-point counts.

FP32 single-token lookup returns a view of the original table. BF16/FP16 lookup
converts only requested rows to FP32, and integer lookup dequantizes only those
rows. Single-token converted results own their row; batch lookup writes into
caller-provided storage. Per-layer lookup partitions each row into contiguous
layer segments. Invalid token IDs are logged and mapped to row zero.

The table retains its tensor and a buffer lock. A tensor backed by a borrowed
`SpanCpuBuffer` still requires its external owner to keep the storage alive.
The safetensor loader supplies the mapping ownership for file-backed tables.
See the [shared runner lifetime rules](../../../runners/common_nnpack/README.md#inputs-outputs-and-buffer-lifetimes)
when binding the resulting buffers to runtime inputs.

## Changes and validation

Change attention layout/cache handling in `attention.h`, transformer ordering
in `transformer.h`, activation-scale interpretation in
`mobile_fully_connected.h`, and storage formats in `quantized_embedding.cc`.
Update the associated small fixtures when changing numerical semantics. The
[reference README](reference/README.md) explains their scope; static integer
rounding/saturation and malformed embedding metadata are checked directly in
C++ tests.

Static-scale tests include exact positive and negative halfway accumulators,
nearest-even expectations, saturation, and separate single-row/multiple-row
execution. They do not establish full-model equivalence. The
[native runner guide](../native/README.md) explains the staged implementation
and its numerical contract; the [comparison recipe](../native/REPRODUCE.md)
covers performance against LiteRT-LM.

Use the [standalone guide](../../../standalone/README.md) for build/test setup. The helper libraries and tests also have targets in this [BUILD](BUILD) file. The staged [native model](../native/README.md) keeps its active-KV graph extensions separate from these shared helpers.
