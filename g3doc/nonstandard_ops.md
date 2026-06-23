# Non-Standard ODML Operators

LiteRT/TFLite supports a small set of named, non-standard operators. Some are
preserved `stablehlo.composite` ops from StableHLO MLIR, while others are
runtime `tfl.custom` / TFLite `CUSTOM` ops that require a custom registration.

These ops matter mostly for LLM conversion and runtime performance, especially
on the default CPU stack. They are not part of the StableHLO spec; their
meaning is defined by LiteRT/TFLite code and, in some cases, by shipped
published models.

This document is aimed at converter authors who need to emit these ops in a
form that LiteRT/TFLite recognizes.

## What Shipped Models Actually Use

As of this note, a survey of the non-NPU `.litertlm` bundles under
`/data/bt/models/litert-community` shows:

- surveyed bundles: 9
- most common composite name by far:
  - `odml.rms_norm`
- no `CUSTOM` nonstandard ops were seen in that surveyed set
- `odml.scaled_dot_product_attention` was not seen in that surveyed set
- `odml.runtime_bmm` and `odml.cache_update` were not seen in that surveyed
  set

A checked standalone published `.tflite` model shows the same broad pattern:

- `embeddinggemma-300M_seq256_mixed-precision.tflite`
  - signature: `embed_256`
  - no `STABLEHLO_COMPOSITE` ops
  - no `CUSTOM` nonstandard ops
  - attention/norm structure is fully decomposed into builtin ops

The one important exception is published Gemma4:

- the published Gemma4 bundle examined separately from the Hugging Face cache
  does contain:
  - `odml.runtime_bmm`
  - `odml.cache_update`
  - `odml.rms_norm`

So in practice, the ecosystem currently looks like this:

- public code and delegate support mention `odml.scaled_dot_product_attention`
- many shipped community bundles instead use decomposed attention plus
  preserved `odml.rms_norm`
- published Gemma4 is a more specialized bundle that also uses additional
  composite names

## Scope

This document mixes three different evidence levels:

- Publicly documented / code-recognized LiteRT behavior
- Behavior recognized by the XNNPACK delegate
- Additional composite names inferred from shipped published Gemma4 bundles

Important caveat:

- Presence in a published `.tflite` / `.litertlm` does not by itself imply that
  every backend has a dedicated fused execution path for that op.
- For several names below, the safest interpretation is:
  - the name is real
  - the serialized decomposition is real
  - backend-native handling may or may not exist

## If You Are Coming From `litert-torch`

This document is easier to read if you map it to the concepts used in
`~/src/litert-torch`.

In `litert-torch`, you usually encounter these ops in one of two ways:

- As ordinary PyTorch/Aten math such as:
  - `torch.nn.functional.scaled_dot_product_attention`
  - `aten.scaled_dot_product_attention.default`
  - `aten.rms_norm.default`
- As HLFB-marked regions created with
  `StableHLOCompositeBuilder(name="odml.<name>")`

Two public `litert-torch` examples are:

- `litert_torch.generative.layers.scaled_dot_product_attention_with_hlfb`
  - emits `odml.scaled_dot_product_attention`
- `litert_torch.generative.layers.normalization.rms_norm_with_hlfb`
  - emits `odml.rms_norm`

For a `litert-torch` reader, the main purpose of this file is:

- to explain what composite name a frontend should emit
- to explain the type/layout contract that LiteRT expects
- to explain when a published model contains additional composite names that are
  not part of the small public set most `litert-torch` users see directly

## Encoding Classes

There are two separate concepts that are easy to conflate:

- `stablehlo.composite` in MLIR, serialized as builtin
  `STABLEHLO_COMPOSITE` in the TFLite FlatBuffer
  - TFLite schema: `StableHLOCompositeOptions`
  - important fields:
    - `name`
    - `version`
    - `decomposition_subgraph_index`
    - `composite_attributes`
  - generic CPU fallback can execute the decomposition subgraph or inline it
  - backends may also recognize the composite name and replace it with a fused
    implementation
- `tfl.custom` in MLIR, serialized as `CUSTOM` in the TFLite FlatBuffer
  - `custom_code == <name>`
  - `custom_options` / `custom_initial_data` is typically a flexbuffers map
  - there is no `StableHLOCompositeOptions` object and no decomposition
    subgraph attached to this op
  - execution requires a registered custom kernel, or a delegate/backend that
    recognizes the custom op

Some source `stablehlo.composite` ops are intentionally lowered to
`tfl.custom` by the converter. After that legalization step, they are runtime
custom ops, not preserved composites. Their original composite decomposition is
not available through `StableHLOCompositeOptions`.

This document separates those cases:

- preserved `STABLEHLO_COMPOSITE` ops
- source composites that are legalized to runtime `CUSTOM` ops
- plain runtime `CUSTOM` ops that are not ODML StableHLO composites

## Signature Notation

The signatures below use symbolic dimensions rather than one concrete model's
numbers:

- `B`: batch size
- `T`: query/update sequence length, often `1` in decode
- `S`: key/value sequence length or cache capacity
- `Nq`: query head count
- `Nkv`: key/value head count
- `H`: per-head dimension
- `Hv`: value per-head dimension when it differs from the query/key head
  dimension
- `D`: hidden/channel dimension normalized by norm-like ops
- `C`: channel or feature dimension
- `G`: group count for group normalization
- `Kc`: causal-convolution kernel width
- `R`: recurrent/state-space state dimension
- `...`: leading batch-like dimensions that are preserved by the op

Unless stated otherwise, tensor order is the TFLite flatbuffer operand order.
Flatbuffer tensor names are not used for dispatch; the slot order and shapes are
the effective contract.

## Quick Catalog

| Name | Encoding In TFLite | Status In This Tree | Typical Role | PyTorch/Aten Mapping |
| --- | --- | --- | --- | --- |
| `odml.scaled_dot_product_attention` | Usually `STABLEHLO_COMPOSITE`; XNNPACK also recognizes `CUSTOM` with the same custom code | Publicly documented, XNNPACK-recognized, but not seen in the current surveyed non-NPU `litert-community` bundles | Float SDPA | `aten.scaled_dot_product_attention.default` |
| `odml.rms_norm` | `STABLEHLO_COMPOSITE` | Broadly used in surveyed published bundles; also present in testdata and vendor/compiler paths | RMSNorm semantic boundary | `aten.rms_norm.default` |
| `odml.group_norm` | `STABLEHLO_COMPOSITE` | Present in testdata and compiler/plugin paths | GroupNorm/LayerNorm-style semantic boundary | `aten.group_norm.default` / `torch.nn.GroupNorm` |
| `odml.l2_norm` | `STABLEHLO_COMPOSITE` | Present in testdata and vendor/compiler paths | L2 normalization semantic boundary | `torch.nn.functional.normalize(..., p=2)` |
| `odml.causal_conv_with_state_1d` | `STABLEHLO_COMPOSITE` | Native CPU-specialized composite implementation exists | Stateful causal depthwise 1-D convolution | No standard 1:1 Aten op |
| `odml.recurrent_linear_attention` | `STABLEHLO_COMPOSITE` | Native CPU-specialized composite implementation exists | Recurrent linear attention | No standard 1:1 Aten op |
| `odml.selective_state_space` | `STABLEHLO_COMPOSITE` | Native CPU-specialized composite implementation exists | Mamba-style selective state-space update | No standard 1:1 Aten op |
| `odml.runtime_bmm` | `STABLEHLO_COMPOSITE` | Observed in published Gemma4, but not in the current surveyed non-NPU `litert-community` bundles | Quantized-cache readback matmul | No standard 1:1 Aten op |
| `odml.cache_update` | `STABLEHLO_COMPOSITE` | Observed in published Gemma4, but not in the current surveyed non-NPU `litert-community` bundles | Quantized KV cache writeback/update | No standard 1:1 Aten op |
| `odml.update_kv_cache` | legalized from source composite to `CUSTOM` | Deprecated GenAI-style KV update path | Resource-backed KV update | No standard 1:1 Aten op |
| `odml.update_external_kv_cache` | legalized from source composite to `CUSTOM` | Deprecated external KV update path | Explicit-tensor KV update | No standard 1:1 Aten op |
| `odml.quantize_and_dequantize` | legalized from source composite to `CUSTOM` | Converter-recognized custom-legalized composite | Quantize/dequantize helper | No standard 1:1 Aten op |
| `odml.detector` | legalized from source composite to `CUSTOM` | Converter-recognized custom-legalized composite | Debug/detector helper | No standard 1:1 Aten op |

Not cataloged as numeric operator signatures:

- `odml.npu_call` and `odml.cpu_call`
  - compiler partition marker composites; their operand lists are whatever the
    outlined partition boundary requires
- test-only or negative-test names such as `odml.foo`, `odml.softmax`, and
  `odml.regular_composite`
  - these are fixtures unless a real runtime/delegate contract is added

## Preserved Composite Ops

### `odml.scaled_dot_product_attention`

#### Status

- Publicly documented in this tree
- Recognized by XNNPACK
- Can appear as either `STABLEHLO_COMPOSITE` or `CUSTOM`
- Not seen in the current surveyed non-NPU `litert-community` `.litertlm`
  bundles

#### Accepted Forms

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.scaled_dot_product_attention"`
- `CUSTOM` with
  `custom_code == "odml.scaled_dot_product_attention"`

These two encodings are not identical. The `STABLEHLO_COMPOSITE` form carries a
decomposition subgraph; the `CUSTOM` form relies on the `Register_SDPA()` custom
kernel or delegate handling.

#### Signature

Portable form:

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `query` | `FLOAT32` | `[B, T, Nq, H]` (`BTNH`) | Query projection. `T` is the query length. |
| input 1 | `key` | `FLOAT32` | `[B, S, Nkv, H]` (`BSNH`) | Key projection or key cache readback. `S` is the available KV length. |
| input 2 | `value` | `FLOAT32` | `[B, S, Nkv, H]` (`BSNH`) | Value projection or value cache readback. Current XNNPACK checks require the last dimension to match `query` and `key`. |
| input 3 | `attention_mask` | `FLOAT32` | rank-4, broadcastable to `[B, Nq, T, S]`; last dim must be `S` | Additive attention mask applied to logits before softmax. The reference custom op requires this input. |
| output 0 | `output` | `FLOAT32` | `[B, T, Nq, H]` (`BTNH`) | Attention result in the same external layout as `query`. |

Head-count constraints:

- `Nq` must be divisible by `Nkv`.
- `Nkv == Nq` is multi-head attention.
- `Nkv == 1` is multi-query attention.
- `1 < Nkv < Nq` is grouped-query attention.

XNNPACK's visitor can handle a missing mask input internally, but the custom
reference implementation in `tflite/experimental/genai/sdpa.cc` requires four
inputs. For a portable model, emit an explicit additive mask.

#### Where It Is Handled

- XNNPACK delegation logic:
  - `tflite/delegates/xnnpack/xnnpack_delegate.cc`
- Test helper:
  - `tflite/delegates/xnnpack/odml_sdpa_tester.cc`

#### Semantic Summary

This is the standard scaled-dot-product attention operator:

- scale query
- compute attention logits `Q x K^T`
- optionally add an attention mask
- softmax
- compute `attn x V`

Optional attributes:

- `scale`: float32 scalar
  - if absent, XNNPACK uses `1 / sqrt(head_dim)`
- `logit_cap`: float32 scalar
  - if present, XNNPACK applies tanh-based logit capping

#### Type Contract

Current XNNPACK path requires:

- `q`: `float32`
- `k`: `float32`
- `v`: `float32`
- optional `mask`: `float32`
- output: `float32`

#### Layout Contract

This is the most important source of converter mistakes.

XNNPACK's external SDPA contract expects rank-4 tensors in:

- query: `BTNH`
  - `[batch, query_seq, query_heads, head_dim]`
- key: `BSNH`
  - `[batch, kv_seq, kv_heads, head_dim]`
- value: `BSNH`
  - `[batch, kv_seq, kv_heads, head_dim]`
- output: `BTNH`
  - `[batch, query_seq, query_heads, head_dim]`

Mask expectations:

- mask is `float32`
- its last dimension must match `kv_seq`

#### Layout Difference Versus Common PyTorch Frontends

`aten.scaled_dot_product_attention.default` is a semantic op, not a fixed
layout op. In practice, many PyTorch LLM frontends naturally produce tensors in
head-major layouts such as:

- query: `BNTH`
  - `[batch, query_heads, query_seq, head_dim]`
- key/value: `BNSH`
  - `[batch, kv_heads, kv_seq, head_dim]`

That differs from the XNNPACK-facing LiteRT SDPA contract above:

- PyTorch/common frontend:
  - `BNTH` / `BNSH`
- XNNPACK-facing LiteRT composite:
  - `BTNH` / `BSNH`

This layout mismatch is real and performance-relevant. In this repo:

- preserving SDPA late in the converter often requires transpose wrappers
- those transposes can erase the expected benefit of the composite
- if SDPA is a primary target, layout should be chosen early in the model
  frontend/export path

XNNPACK also performs internal transposes inside its current SDPA path:

- query: `BTNH -> BNTH`
- key: `BSNH -> BNSH`
- output path ends by converting back to `BTNH`

So fixing converter-side layout removes extra graph scaffolding, but does not
necessarily eliminate all transpose work.

#### PyTorch / Aten Mapping

Direct semantic mapping exists:

- `aten.scaled_dot_product_attention.default`

How a `litert-torch` user usually encounters it:

- plain math path:
  - `torch.nn.functional.scaled_dot_product_attention`
- HLFB path:
  - `litert_torch.generative.layers.scaled_dot_product_attention_with_hlfb`
  - which marks a region with
    `StableHLOCompositeBuilder(name="odml.scaled_dot_product_attention")`

One local experimental frontend also uses a helper op to author the
backend-friendly layout directly:

- `torch.ops.litert_attention.sdpa_bsnh.default`

That helper is not part of public `litert-torch`; it is only an example of how
another frontend might choose to represent the same semantic op while forcing a
specific layout contract.

#### Notes For Converter Authors

- Serialize `scale` / `logit_cap` as flexbuffer float scalars, not strings.
- Do not assume a PyTorch frontend layout is automatically acceptable to
  XNNPACK.
- If a model already uses `BNTH` / `BNSH`, treat SDPA layout as a frontend
  authoring decision, not a late cleanup detail.
- Also do not assume that using this composite is required to match currently
  shipped public CPU bundles; several published community bundles instead ship
  decomposed attention and preserve only `odml.rms_norm`.

### `odml.rms_norm`

#### Status

- Broadly observed in shipped non-NPU `litert-community` bundles
- Also observed in published Gemma4 bundles
- Present in LiteRT testdata as `stablehlo.composite`
- Referenced by several vendor/compiler paths in this tree

#### Accepted Form

Observed and commonly used as:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.rms_norm"`

#### Signature

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `input` | usually `FLOAT32` | `[..., D]`, commonly `[B, T, D]` | Values to normalize. RMS is computed over the last dimension. |
| input 1 | `scale` / `gamma` | usually `FLOAT32` | `[D]`, or a shape broadcastable over `input`'s last dimension | Learned multiplicative scale applied after normalization. |
| output 0 | `output` | same logical type as `input` in the decomposition | same as `input` | Normalized and scaled tensor. |

Required composite attribute:

- `epsilon`: `FLOAT32` scalar. Added before `RSQRT`.

The local testdata uses `input = [1, 128, 2304]` and `scale = [2304]`.
Vendor builders in this tree also treat the normalization axis as the last
dimension.

#### Semantic Summary

Standard RMSNorm:

- compute RMS from the input
- add epsilon
- multiply by reciprocal square root
- apply learned scale

Observed decomposition families in published Gemma4 decode:

- common form:
  - `SUM`
  - `RSQRT`
  - `ADD`
  - `MUL` x4
- broadcast-adaptation form:
  - same as above plus `RESHAPE`

#### PyTorch / Aten Mapping

Direct mapping exists:

- `aten.rms_norm.default`

How a `litert-torch` user usually encounters it:

- plain math / module path:
  - `RMSNorm`
  - `aten.rms_norm.default`
- HLFB path:
  - `litert_torch.generative.layers.normalization.rms_norm_with_hlfb`
  - which marks a region with
    `StableHLOCompositeBuilder(name="odml.rms_norm")`

#### Notes For Converter Authors

- This is a good candidate to preserve as a semantic boundary.
- If decomposed, keep the epsilon and broadcast behavior exact.

### `odml.group_norm`

#### Status

- Present in LiteRT testdata as `stablehlo.composite`
- Listed in LiteRT `CompositeOptions`
- Recognized by the Google Tensor compiler plugin support list
- Handled by the Qualcomm composite builder path

#### Accepted Form

Observed and supported as:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.group_norm"`

#### Signature

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `input` | usually `FLOAT32` | tensor with channel dimension `C`; testdata uses `[B, C]` and `channel_axis = -1` | Values to normalize. The channel axis is identified by the `channel_axis` composite attribute. |
| input 1 | `scale` / `gamma` | usually `FLOAT32` | `[C]`, or a shape broadcastable over the channel axis | Learned multiplicative scale applied after normalization. |
| input 2 | `offset` / `beta` | usually `FLOAT32` | `[C]`, or a shape broadcastable over the channel axis | Learned additive offset applied after scaling. |
| output 0 | `output` | same logical type as `input` in the decomposition | same as `input` | Group-normalized output. |

Composite attributes used by current testdata/compiler paths:

- `epsilon`: `FLOAT32` scalar.
- `num_groups`: integer group count `G`.
- `channel_axis`: integer channel axis, commonly `-1`.

Shape constraints:

- `C` must be divisible by `G`.
- `scale` and `offset` must match or broadcast over the channel dimension.
- Qualcomm's builder maps `num_groups == 1` to a layer-norm-style builder and
  otherwise uses a group-norm builder.

#### Notes For Converter Authors

- Preserve the exact `channel_axis`, `num_groups`, and `epsilon` attributes.
- Treat non-last-channel layouts as backend-sensitive unless the target backend
  is known to consume `channel_axis` correctly.

### `odml.l2_norm`

#### Status

- Present in LiteRT testdata as `stablehlo.composite`
- Listed in LiteRT `CompositeOptions`
- Handled by vendor/compiler paths in this tree

#### Accepted Form

Observed and supported as:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.l2_norm"`

#### Signature

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `input` | usually `FLOAT32` | arbitrary tensor `X`, commonly `[..., D]` | Values to normalize. Current testdata normalizes over the last axis. |
| output 0 | `output` | same logical type as `input` in the decomposition | same as `input` | L2-normalized tensor. |

Composite attributes observed in testdata/compiler paths:

- `axis`: integer reduction axis. Current testdata uses `-1`.
- `epsilon`: `FLOAT32` scalar.

Semantic summary:

- `output = input / sqrt(sum(input * input, axis=axis, keep_dims=true) + epsilon)`

#### Notes For Converter Authors

- The builtin TFLite `L2_NORMALIZATION` op is a separate builtin op. This
  section is only about the preserved `odml.l2_norm` composite form.
- Some vendor builder paths consume only `epsilon`, so non-last-axis forms
  should be treated as backend-sensitive.

For the next three native CPU-specialized composite sections, "supported
quantized type" means `INT8`, `UINT8`, `INT16`, or `INT32` with usable
per-tensor or per-channel quantization metadata. Validate mixed-type models
against the target runtime before treating those combinations as portable.

### `odml.causal_conv_with_state_1d`

#### Status

- Native CPU-specialized implementation exists in
  `tflite/kernels/stablehlo_composite.cc`
- Falls back to the attached decomposition when the native special case is not
  selected

#### Accepted Form

Supported as:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.causal_conv_with_state_1d"`

#### Signature

Recommended four-input form:

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `input` | `FLOAT32` or supported quantized type | `[B, T, C]` | Current input sequence. |
| input 1 | `weight` | `FLOAT32` or supported quantized type | `[Kc, C]` or legacy channel-major `[C, Kc]` | Per-channel causal convolution weights. |
| input 2 | `bias` | optional `FLOAT32` or supported quantized type | `[C]` | Optional per-channel bias. |
| input 3 | `past_state` | optional `FLOAT32` or supported quantized type | `[B, Kc - 1, C]` for `[Kc, C]` weights; `[B, C, Kc - 1]` for legacy `[C, Kc]` weights | Previous state window. |
| output 0 | `output` | `FLOAT32` or supported quantized type | `[B, T, C]` | Convolution output for the current sequence. |
| output 1 | `present_state` | `FLOAT32` or supported quantized type | same state layout as `past_state` | Updated state window for the next invocation. |

Compatibility note:

- The native CPU path accepts two to four inputs. If only three inputs are
  provided, input 2 is interpreted as `bias` when it is rank 1; otherwise it is
  interpreted as `past_state`.
- Optional composite attribute `activation` accepts `"silu"` / `"swish"` in the
  native CPU path.

### `odml.recurrent_linear_attention`

#### Status

- Native CPU-specialized implementation exists in
  `tflite/kernels/stablehlo_composite.cc`
- Falls back to the attached decomposition when the native special case is not
  selected

#### Accepted Form

Supported as:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.recurrent_linear_attention"`

#### Signature

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `query` | `FLOAT32` or supported quantized type | rank-4 `[B, T, Nq, H]` or packed rank-3 `[B, T, Nq * H]` | Query features. |
| input 1 | `key` | `FLOAT32` or supported quantized type | rank-4 `[B, T, Nkv, H]` or packed rank-3 `[B, T, Nkv * H]` | Key features. |
| input 2 | `value` | `FLOAT32` or supported quantized type | rank-4 `[B, T, Nkv, Hv]` or packed rank-3 `[B, T, Nkv * Hv]` | Value features. |
| input 3 | `past_state` | optional `FLOAT32` or supported quantized type | `[B, Nkv, H, Hv]` | Previous recurrent state. |
| input 4 | `decay` | optional `FLOAT32` or supported quantized type | `[B, T, X]`, where `X` may be `1`, `Nkv`, `Nq`, `H`, or `Nkv * H` | Decay or delta-like control values, depending on `update_rule`. |
| input 5 | `beta` / `gate` | optional `FLOAT32` or supported quantized type | `[B, T, X]`, where `X` may be `1`, `Nkv`, or `Nq` | Gating values used by gated update rules. |
| output 0 | `output` | `FLOAT32` or supported quantized type | rank-4 `[B, T, Nq, Hv]` or packed rank-3 `[B, T, Nq * Hv]` | Recurrent attention output. |
| output 1 | `present_state` | `FLOAT32` or supported quantized type | `[B, Nkv, H, Hv]` | Updated recurrent state. |

Composite attributes consumed by the native CPU path:

- `q_num_heads`
- `kv_num_heads`
- `scale`
- `chunk_size`
- `use_chunked_prefill`
- `update_rule`: `"linear"`, `"gated"`, `"delta"`, or `"gated_delta"`

Shape constraints:

- `Nq >= Nkv`
- `Nq` must be divisible by `Nkv`
- rank-3 query/key/value use packed head dimensions; rank-4 tensors use
  explicit head dimensions

### `odml.selective_state_space`

#### Status

- Native CPU-specialized implementation exists in
  `tflite/kernels/stablehlo_composite.cc`
- Falls back to the attached decomposition when the native special case is not
  selected

#### Accepted Form

Supported as:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.selective_state_space"`

#### Signature

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `x` | `FLOAT32` or supported quantized type | rank-3 `[B, T, Nq]` or rank-4 `[B, T, Nq, H]` | Input sequence features. |
| input 1 | `delta` | `FLOAT32` or supported quantized type | rank-3 `[B, T, 1 or Nq]` or rank-4 `[B, T, 1 or Nq, 1 or H]` | Per-token step size before optional transform/clamp. |
| input 2 | `a` | `FLOAT32` or supported quantized type | `[Nq]`, `[Nq, R]`, or `[Nq, H, R]` | State transition parameter. |
| input 3 | `b` | `FLOAT32` or supported quantized type | rank-3 `[B, T, R]` or rank-4 `[B, T, G, R]` | Input projection/state update parameter. |
| input 4 | `c` | `FLOAT32` or supported quantized type | same rank pattern as `b`; last dimension `R` | Output projection parameter. |
| input 5 | `past_state` | optional `FLOAT32` | `[B, Nq, H, R]` | Previous recurrent state. Use `H = 1` when `x` is rank 3. |
| input 6 | `d` | optional `FLOAT32` or supported quantized type | `[Nq]` or `[Nq, H]` | Optional skip parameter added as `d * x`. |
| input 7 | `delta_bias` | optional `FLOAT32` or supported quantized type | `[Nq]` or `[Nq, H]` | Optional per-head/per-channel delta bias. |
| input 8 | `token_mask` | optional `BOOL`, `FLOAT32`, or supported quantized type | `[B, T]` | False/zero entries suppress output for that token. |
| input 9 | `reset_mask` | optional `BOOL`, `FLOAT32`, or supported quantized type | `[B, T]` | True/nonzero entries reset state before processing that token. |
| output 0 | `output` | `FLOAT32` or supported quantized type | same shape as `x` | Selective state-space output. |
| output 1 | `present_state` | `FLOAT32` | `[B, Nq, H, R]` | Updated recurrent state. |

Composite attributes consumed by the native CPU path:

- `num_groups`: optional expected group count `G`.
- `delta_transform`: `"softplus"` enables softplus on `delta`.
- `delta_softplus`: legacy boolean spelling for softplus.
- `delta_min` / `delta_max`: optional clamp bounds after delta transform.

Shape constraints:

- `b` and `c` must have the same rank and matching batch/sequence/group/state
  dimensions.
- `Nq` must be divisible by `G`.
- `present_state` is always rank 4; rank-3 `x` is treated as `H = 1`.

### `odml.runtime_bmm`

#### Status

- Observed in published Gemma4 bundles
- Not currently documented here as a public XNNPACK-recognized nonstandard op
- Not seen in the current surveyed non-NPU `litert-community` `.litertlm`
  bundles
- Treat this as an inferred compatibility note, not a public API guarantee

#### Accepted Form

Observed in the published Gemma4 decode section as:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.runtime_bmm"`

#### Signature

This signature is inferred from published Gemma4-style decompositions; this tree
does not currently expose a native public kernel contract for the op.

Core minimal form:

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `lhs` | `FLOAT32` | batch-matmul-compatible shape, usually `[..., M, K]` | Runtime left-hand operand, for example query states or attention probabilities. |
| input 1 | `rhs_quantized` | quantized, observed as cache-side integer storage | batch-matmul-compatible quantized RHS, usually `[..., N, K]` when transposed by the matmul or `[..., K, N]` otherwise | Quantized right-hand operand read from cache or a cache-derived slice. The decomposition starts by dequantizing it. |
| optional inputs | `shape_or_length_helpers` | usually integer tensors | scalar or small 1-D tensors | Published helper forms may carry live-length, slice, padding, or shape values around the core matmul. |
| output 0 | `output` | `FLOAT32` | batch-matmul result, usually `[..., M, N]` | Result of multiplying `lhs` by the dequantized RHS. |

Attention-oriented interpretation:

- key readback form: `lhs` is query-like, `rhs_quantized` is key-cache-like,
  and `output` is logits-like `[B, Nq, T, S_live]`.
- value readback form: `lhs` is probability/logit-like,
  `rhs_quantized` is value-cache-like, and `output` is activation-like
  `[B, Nq, T, H]`.

Do not rely on the optional helper inputs as a stable ABI. The published
decompositions show multiple helper families around the same core
`DEQUANTIZE + BATCH_MATMUL` meaning.

#### Semantic Summary

This is best understood as a quantized-RHS runtime batch matmul:

- `output = batch_matmul(lhs_fp32, dequantize(rhs_q))`

Observed decomposition families in published Gemma4 decode:

- minimal form:
  - `DEQUANTIZE`
  - `BATCH_MATMUL`
- prefix/live-length helper form:
  - `SLICE`
  - `ADD`
  - `SUB`
  - `MAXIMUM`
  - `RESHAPE`
  - `CONCATENATION`
  - `DEQUANTIZE`
  - `BATCH_MATMUL`
- padded/fixed-width helper form:
  - `SLICE`
  - `ADD`
  - `SUB`
  - `MAXIMUM`
  - `RESHAPE`
  - `CONCATENATION`
  - `DEQUANTIZE`
  - `BATCH_MATMUL`
  - `PACK`
  - `FILL`
  - `DYNAMIC_UPDATE_SLICE`

So there are really two layers of meaning:

- core meaning:
  - float lhs times quantized rhs
- optional decode-shape management:
  - slice live prefix
  - pad or write back into a fixed-width output shape

#### PyTorch / Aten Mapping

No standard 1:1 Aten op is known.

Closest decomposed Aten/math interpretation:

- prefix/layout helpers:
  - `aten.slice`
  - `aten.reshape`
  - `aten.cat`
  - `aten.maximum`
- then dequantization
- then `aten.bmm` / `aten.matmul`

How a `litert-torch` user should think about it:

- this is not a common public HLFB name in `litert-torch` today
- you are more likely to encounter the underlying idea indirectly via:
  - KV-cache-aware attention code
  - split-cache decode implementations
  - published LiteRT-LM bundles inspected after export
- mentally, this is closer to:
  - "read quantized cache, dequantize, then do the decode-time matmul"
  than to a single familiar public PyTorch op

#### Notes For Converter Authors

- Do not model this as "just batch matmul". The published decomposition shows
  that live-length and fixed-width decode shape handling are often part of the
  contract.
- The published decomposition explicitly contains `DEQUANTIZE`.

### `odml.cache_update`

#### Status

- Observed in published Gemma4 bundles
- Name also appears in LiteRT runtime magic-number handling
- Not seen in the current surveyed non-NPU `litert-community` `.litertlm`
  bundles
- Treat this section as an inferred compatibility note unless a public runtime
  contract is documented elsewhere

#### Accepted Form

Observed in published Gemma4 decode as:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.cache_update"`
- observed paired-KV contract in the published Gemma4 decode section:
  - `7` inputs
  - `2` outputs
- observed runtime-side compatibility expectations for the preserved quantized
  form:
  - `kv_cache_batch_size` (`INT32`) composite attr
  - `cache_size` (`INT32`) composite attr
  - `head_size` (`INT32`) composite attr
  - optional `scale_k` / `scale_v` (`FLOAT32`) composite attrs may also be
    present
- practical compatibility note:
  - a single-cache helper shaped like
    `QUANTIZE + DYNAMIC_UPDATE_SLICE -> 1 output`
    should not reuse the public name `odml.cache_update`
  - current delegate checks for the quantized public form assume the paired-KV
    contract above

#### Signature

This signature is inferred from the published Gemma4 paired-KV form; this tree
does not currently provide a standalone native kernel implementation for the
name.

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `k_slice` / fresh K fragment | `FLOAT32` | update-window tensor; commonly rank-4 with token/update length `T` and head size `H` | New key values for the current decode/prefill step before quantized cache storage. |
| input 1 | `v_slice` / fresh V fragment | `FLOAT32` | update-window tensor matching the K update's logical extent | New value values for the current decode/prefill step before quantized cache storage. |
| input 2 | `params` | `INT32` | scalar or small 1-D tensor | Auxiliary integer parameters used by the published decomposition. Treat this as decomposition-specific, not semantic model data. |
| input 3 | `k_cache` | `INT8` | fixed-capacity key cache; capacity dimension is `S == cache_size` | Existing quantized key cache to update. |
| input 4 | `v_cache` | `INT8` | fixed-capacity value cache; capacity dimension is `S == cache_size` | Existing quantized value cache to update. |
| input 5 | `k_start_indices` | `INT32` | 1-D index vector with length equal to the rank of `k_cache` | Start indices for the key-cache `DYNAMIC_UPDATE_SLICE` in the decomposition. |
| input 6 | `v_start_indices` | `INT32` | 1-D index vector with length equal to the rank of `v_cache` | Start indices for the value-cache `DYNAMIC_UPDATE_SLICE` in the decomposition. |
| output 0 | `updated_k_cache` | `INT8` | same shape as `k_cache` | Key cache after writing the quantized K fragment. |
| output 1 | `updated_v_cache` | `INT8` | same shape as `v_cache` | Value cache after writing the quantized V fragment. |

The public cache layout is model/backend-specific. The important portable facts
from the observed form are the paired K/V update, float source fragments,
quantized public cache tensors, and output shapes matching the corresponding
cache inputs.

#### Semantic Summary

This is best understood as a typed KV cache write/update op:

- take fresh float K/V fragments
- quantize them
- adapt layout as needed
- write into an existing fixed-capacity quantized cache tensor

Observed decomposition families in published Gemma4 decode:

- K-like layout variant:
  - `QUANTIZE` x2
  - `DYNAMIC_UPDATE_SLICE` x2
  - `TRANSPOSE`
- V-like layout variant:
  - `QUANTIZE` x2
  - `DYNAMIC_UPDATE_SLICE` x2
  - `RESHAPE`

#### Known Current Restriction

This is not a formal public ABI guarantee, but current nonstandard-op bring-up
results indicate an important portability restriction for the native GPU path:

- the observed working published path is Gemma-like
  - paired K/V update
  - `FLOAT32` source fragments
  - `INT8` public cache tensors
  - a compact/head-packed public cache layout where the delegate-facing cache
    update source effectively has `H = 1`
- converter experiments with Qwen3-style explicit multi-head public KV layouts
  have not been reliable on the native GPU `odml.cache_update` path
  - K cache exposed as `[B, H, T, D]`
  - V cache exposed as `[B, H, D, T]`
  - with `H > 1`
- until the backend contract is documented more explicitly, frontends should
  not assume `odml.cache_update` is portable across arbitrary explicit-head KV
  layouts just because the published Gemma path works
- if a model family needs explicit multi-head public KV tensors, prefer a
  builtin/fallback cache-update path unless the target backend is known to
  support that layout correctly

#### PyTorch / Aten Mapping

No standard 1:1 Aten op is known.

Closest decomposed Aten/math interpretation:

- quantization
- layout fixup
  - `transpose` or `reshape`
- dynamic slice update / scatter-style cache write

How a `litert-torch` user should think about it:

- this is not one of the small public HLFB names most `litert-torch` model
  authors call directly
- conceptually it is the runtime-side descendant of the cache-update logic in:
  - `litert_torch.generative.layers.kv_cache`
  - split-cache export paths
- mentally, this is closer to:
  - "write the new K/V slice into an existing fixed-capacity cache"
  than to a single standard Aten op

#### Notes For Converter Authors

- K and V may require different layout adaptation.
- Do not force a single K/V physical layout unless the target runtime contract
  explicitly requires it.
- Do not overload `odml.cache_update` with a private single-cache update helper.
  If the exported op updates only one cache tensor, lower it to builtins (for
  example `QUANTIZE + DYNAMIC_UPDATE_SLICE`) or use an internal name that is
  always decomposed before runtime.

## Composites Legalized To Runtime `CUSTOM`

Some StableHLO composite names are not preserved as
`STABLEHLO_COMPOSITE` in the TFLite dialect. Instead, the converter legalizes
them into `tfl.custom` with the same `custom_code`, and serializes composite
attributes into flexbuffers.

After this lowering, the FlatBuffer op is a TFLite `CUSTOM` op. It should be
handled like any other custom op: a runtime kernel or delegate must recognize
its `custom_code`. Generic `STABLEHLO_COMPOSITE` decomposition fallback no
longer applies to the legalized op.

Source of truth:

- `tflite/converter/stablehlo/transforms/legalize_stablehlo_composite_to_tfl_custom.cc`

Currently legalized composite names:

- `odml.update_kv_cache`
  - the pass also injects `num_layers` and `layer_index`
- `odml.update_external_kv_cache`
- `odml.quantize_and_dequantize`
- `odml.detector`

If your converter emits these as `stablehlo.composite`, make sure the composite
attributes are representable as flexbuffers.

### `odml.update_kv_cache`

#### Signature Before Legalization

The StableHLO composite form has five logical operands:

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `k_cache` | `FLOAT32` | `[1, S, Nkv, H]` | Existing full key cache. The current legalization pass drops this operand when creating the runtime `tfl.custom` op. |
| input 1 | `v_cache` | `FLOAT32` | `[1, S, Nkv, H]` | Existing full value cache. The current legalization pass drops this operand when creating the runtime `tfl.custom` op. |
| input 2 | `position` | `INT64` | `[T]` | Absolute token positions for the update slice. Length must match the update sequence length. |
| input 3 | `k_slice` | `FLOAT32` | `[1, T, Nkv, H]` | New key values to write into the resource-backed key cache. |
| input 4 | `v_slice` | `FLOAT32` | `[1, T, Nkv, H]` | New value values to write into the resource-backed value cache. |
| output 0 | `updated_k_cache` | `FLOAT32` | `[1, kv_cache_max, Nkv, H]` | Full key cache after update. |
| output 1 | `updated_v_cache` | `FLOAT32` | `[1, kv_cache_max, Nkv, H]` | Full value cache after update. |

#### Runtime `tfl.custom` Signature

After legalization, `tfl.custom(custom_code = "odml.update_kv_cache")` receives
only:

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `position` | `INT64` | `[T]` | First element is the first token position; the runtime writes `T` contiguous slots. |
| input 1 | `k_slice` | `FLOAT32` | `[1, T, Nkv, H]` | New key values. Batch size must be `1`. |
| input 2 | `v_slice` | `FLOAT32` | `[1, T, Nkv, H]` | New value values. Must have the same shape as `k_slice`. |
| output 0 | `updated_k_cache` | `FLOAT32` | `[1, kv_cache_max, Nkv, H]` | Resource-backed key cache for the selected layer. |
| output 1 | `updated_v_cache` | `FLOAT32` | `[1, kv_cache_max, Nkv, H]` | Resource-backed value cache for the selected layer. |

Custom options:

- `kv_cache_max`: maximum cache entries `S`.
- `num_layers`: injected by the legalization pass.
- `layer_index`: injected by the legalization pass.

The runtime implementation currently supports only rank-4 `[B, S, N, H]`
cache/slice tensors and enforces `B == 1`.

### `odml.update_external_kv_cache`

This custom op updates explicit cache tensors supplied as inputs and outputs.
It does not use the internal resource cache used by `odml.update_kv_cache`.

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `k_cache` | `FLOAT32` | `[1, S, Nkv, H]` | Existing key cache. |
| input 1 | `v_cache` | `FLOAT32` | `[1, S, Nkv, H]` | Existing value cache. Must have the same shape as `k_cache`. |
| input 2 | `position` | `INT32` | `[T]` | Per-token cache positions. Length must equal `k_slice.shape[1]`. |
| input 3 | `k_slice` | `FLOAT32` | `[1, T, Nkv, H]` | New key values. Batch size must be `1`. |
| input 4 | `v_slice` | `FLOAT32` | `[1, T, Nkv, H]` | New value values. Must have the same shape as `k_slice`. |
| output 0 | `updated_k_cache` | `FLOAT32` | same as `k_cache` | Key cache after update. |
| output 1 | `updated_v_cache` | `FLOAT32` | same as `v_cache` | Value cache after update. |

The implementation assumes positions are increasing; a lower position stops the
copy loop and marks exhaustion of update slices.

### `odml.quantize_and_dequantize`

This name is currently legalized to `tfl.custom`; this tree's legalization test
shows the following shape pattern:

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `input` | `FLOAT32` | arbitrary tensor `X`, for example `[4, 3]` | Source tensor to quantize and dequantize. |
| output 0 | `dequantized` | `FLOAT32` | same as `input` | Quantize/dequantize result. |
| output 1 | `quantized_as_float` or helper output | `FLOAT32` | same as `input` in current testdata | Auxiliary lowered result preserved by this custom form. |
| output 2 | `scale` | `FLOAT32` | per-axis scale shape, for example `[1, 3]` when `axis = 0` on `[4, 3]` test input | Quantization scale tensor. |

Custom options observed in tests:

- `axis`: integer axis attribute.
- `bits`: integer quantization bit width.

### `odml.detector`

This is a custom-legalized utility/debug-style composite in current tests, not
a numerically standardized ML operator.

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0..N | `input_i` | decomposition-defined | decomposition-defined | Tensor(s) observed by the detector custom op. |
| output 0..M | `output_i` | decomposition-defined | decomposition-defined | Detector output tensor(s), normally matching the decomposition result types. |

Custom options observed in tests:

- `name`: string identifier.
- `working_dir`: string path.

## Plain Runtime `CUSTOM` Ops

LiteRT also registers several non-ODML custom op names as accelerator stubs.
These are not `stablehlo.composite` ops and are not serialized with
`StableHLOCompositeOptions`. They appear as ordinary TFLite `CUSTOM` ops with a
`custom_code`, and they require backend/delegate handling.

Names currently added as accelerator-supported custom ops in
`litert/runtime/compiled_model.cc` include:

- `Convolution2DTransposeBias`
- `MaxPoolingWithArgmax2D`
- `MaxUnpooling2D`
- `Resampler`
- `custom_call.GroupNorm`
- `custom_call.LayerNorm`
- `custom_call.RmsNorm`
- `custom_call.PixelShuffle`

Do not document or emit these as `odml.*` StableHLO composites unless a
converter pass explicitly creates such a composite and preserves it. For these
plain custom ops, the signature source of truth is the custom op parser or the
delegate implementation that consumes the `custom_code`.

## How To Keep This File Current

There is no single authoritative registry file in this tree; the effective spec
is the code plus shipped models.

Useful commands:

```bash
# Composite names present in MLIR testdata.
rg -n --pcre2 'stablehlo\\.composite\\s+\"([^\"]+)\"' litert tflite

# Composite names recognized by XNNPACK.
rg -n 'kTfLiteBuiltinStablehloComposite' tflite/delegates/xnnpack/xnnpack_delegate.cc

# StableHLO composite names legalized into tfl.custom.
rg -n 'IsSupportedComposite\\(' tflite/converter/stablehlo/transforms/legalize_stablehlo_composite_to_tfl_custom.cc
```

For published-model-derived notes, re-check the exact shipped flatbuffer rather
than relying on memory or secondary notes.
