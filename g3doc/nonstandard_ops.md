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

### MLDrift GPU Delegate Coverage

The MLDrift GPU delegate implementation under
`ml_drift/delegate/composite/` is the source of truth for MLDrift-specific
support. It currently registers native parsers and GPU implementations for:

- `odml.cache_update` (`STABLEHLO_COMPOSITE`)
- `odml.runtime_bmm` (`STABLEHLO_COMPOSITE`)
- `moe` (plain TFLite `CUSTOM` op)

The other operators in this document may be implemented by other TFLite
kernels or delegates, but they are not implemented by that MLDrift composite
directory. In particular, the MLDrift support for `odml.runtime_bmm` and
`odml.cache_update` is native code, not merely a conclusion inferred from a
published model's decomposition.

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
| `odml.runtime_bmm` | `STABLEHLO_COMPOSITE` | Native MLDrift GPU parser and implementation; supports ordinary BMM and cache-backed external-weight paths | Runtime-bounded BMM / KV-cache readback matmul | No standard 1:1 Aten op |
| `odml.cache_update` | `STABLEHLO_COMPOSITE` | Native MLDrift GPU parser and implementation | Float or quantized KV cache writeback/update | No standard 1:1 Aten op |
| `moe` | `CUSTOM` | Native MLDrift GPU parser and implementation; also recognized by the opt-in XNNPACK MoE path | Routed GELU-gated mixture-of-experts block | No standard 1:1 Aten op |
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

- Implemented natively by the MLDrift GPU delegate
- Parsed by both the legacy `GraphFloat32` and newer `IrModel` paths
- Observed in published Gemma4 bundles

#### Accepted Form

For MLDrift delegation, the accepted form is:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.runtime_bmm"`
- exactly three **runtime** inputs in slots 0, 1, and 2; constant operands do
  not satisfy this count
- exactly one output

#### Signature

The canonical MLDrift form is rank 4. `B0` is an optional model batch and `B1`
is the batch dimension of the batched matmul:

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `lhs` | `FLOAT32` in current tests | `[B0, B1, M, K]` | Left operand. |
| input 1 | `rhs` | `FLOAT32` or `INT8` | `[B0, B1, N, K]` | Right operand in already-transposed storage. The last dimension must equal the last dimension of `lhs`. |
| input 2 | `runtime_params` | `INT32` | **exactly `[1, 1, 1, 7]`** | Runtime bounds. MLDrift uses element 2 as `active_tokens_aligned`; the other elements are not read by this op. |
| output 0 | `output` | normally `FLOAT32` | `[B0, B1, M, N]` | Result of `lhs * transpose(rhs)` with a runtime channel bound. |

The third input is part of the MLDrift delegation ABI, not an optional
decomposition helper. A rank-1 `[7]` tensor is not equivalent for MLDrift: its
parser compares the internal BHWC shape with `[1, 1, 1, 7]`, so the FlatBuffer
tensor must have the literal rank-4 shape to be delegated to MLDrift.

MLDrift's BMM kernel supports one internal batch dimension. When `B0 != 1`,
the MLDrift parser inserts reshapes around the operation:

```text
[B0, B1, M, K] -> [1, B0*B1, M, K]
[B0, B1, N, K] -> [1, B0*B1, N, K]
[1, B0*B1, M, N] -> [B0, B1, M, N]
```

`rhs` is always interpreted with `transpose_right = true` on the ordinary BMM
path. There is no attribute for selecting a non-transposed RHS.

#### Composite Attributes

The attributes are a flexbuffers map in
`StableHLOCompositeOptions.composite_attributes`:

| Attribute | Required | Meaning in MLDrift |
| --- | --- | --- |
| `is_global` | yes | Required by MLDrift validation, but its value is not otherwise consumed by the current MLDrift parser or GPU implementation. |
| `is_src` | yes | If true, `runtime_params[2]` bounds the source channels; if false, it bounds the destination channels. It also selects the V-cache versus K-cache quantization metadata when `rhs` comes from `odml.cache_update`. |
| `rhs_cache_update` | no | If true, force the cache/external-weights implementation path. Missing reads as false. |
| `scale` | required for a standalone `INT8` RHS | Its presence selects the external-weights path. For an `INT8` RHS not produced by `odml.cache_update`, it is the uniform dequantization scale. |

#### Semantic Summary

MLDrift chooses between two native implementations:

- Ordinary BMM path when `rhs_cache_update` is false, `scale` is absent, and
  `rhs` is not produced by `odml.cache_update`:
  - `output = batched_matmul(lhs, rhs, transpose_rhs=true)`
- External-weights fully-connected path when any of those conditions is not
  met:
  - treats the RHS as cache/external weights
  - supports float cache storage
  - supports `INT8` TFLite cache storage through MLDrift's `UINT8` packed
    representation plus a scale tensor

For an `INT8` RHS produced by `odml.cache_update`, the scale and scale-vector
length come from that producer:

- `is_src == true`: use `scale_v`, with `head_size` scale entries
- `is_src == false`: use `scale_k`, with `cache_size` scale entries

Consequently, a quantized `odml.cache_update` producer used this way must carry
the corresponding scale attribute. The MLDrift parser calls `.value()` on it.

#### PyTorch / Aten Mapping

No standard 1:1 Aten op is known.

Closest decomposed Aten/math interpretation:

- optional RHS dequantization
- `aten.bmm` / `aten.matmul` with the RHS transposed
- masking or slicing based on the runtime active-token bound

How a `litert-torch` user should think about it:

- this is not a common public HLFB name in `litert-torch` today
- you are more likely to encounter the underlying idea indirectly via:
  - KV-cache-aware attention code
  - split-cache decode implementations
  - published LiteRT-LM bundles inspected after export
- mentally, this is closer to:
  - "read quantized cache, dequantize, then do the decode-time matmul"
  than to a single familiar public PyTorch op

#### Notes For Converter Authors Targeting MLDrift

- Emit three runtime operands, even though only the first two are mathematical
  matmul operands.
- Store the RHS as `[..., N, K]`; MLDrift always transposes it logically.
- Emit the parameter tensor as rank 4 `[1, 1, 1, 7]`, and place the aligned
  active-token count at index 2.
- Always emit both `is_global` and `is_src`. `is_global` currently has no
  behavioral effect in MLDrift, but omitting it rejects MLDrift delegation.
- For standalone `INT8` RHS input, emit `scale`. Do not rely on an affine
  quantization scale being extracted automatically by the MLDrift parser.
- Use `rhs_cache_update=true` when the RHS has cache/external-weight storage
  semantics but its producer is not visible as a delegated
  `odml.cache_update` node.

### `odml.cache_update`

#### Status

- Implemented natively by the MLDrift GPU delegate
- Parsed by both the legacy `GraphFloat32` and newer `IrModel` paths
- Observed in published Gemma4 bundles

#### Accepted Form

For MLDrift delegation, the accepted form is:

- `STABLEHLO_COMPOSITE` with
  `StableHLOCompositeOptions.name == "odml.cache_update"`
- exactly `3` or `7` runtime inputs
- exactly `2` outputs
- required composite attributes:
  - `kv_cache_batch_size` (`INT32`)
  - `cache_size` (`INT32`)
  - `head_size` (`INT32`)
- optional attributes:
  - `scale_k` (`FLOAT32`)
  - `scale_v` (`FLOAT32`)

#### Signature

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `src_k` | `FLOAT32` or `FLOAT16` for a quantized cache | typically `[1, Bkv, T, H]` | Fresh K values. The width dimension is the number of update tokens. |
| input 1 | `src_v` | same as `src_k` | same logical shape as `src_k` | Fresh V values. |
| input 2 | `runtime_params` | `INT32` | at least two elements; current tests use `[2]` | Element 0 is `token_index_offset`; element 1 is `active_tokens`. |
| inputs 3..6 | decomposition-only inputs | decomposition-defined | decomposition-defined | Allowed only in the seven-input carrier. MLDrift intentionally ignores these slots; they exist so the CPU decomposition can express in-place dynamic slice updates. |
| output 0 | `updated_k_cache` | floating-point or `INT8` | backend-packed cache storage | K-cache destination. |
| output 1 | `updated_v_cache` | same storage class as K | backend-packed cache storage | V-cache destination. |

Unlike the MLDrift parser for `odml.runtime_bmm`, the MLDrift
`odml.cache_update` parser does not impose an exact shape on the runtime
parameter tensor. The MLDrift GPU kernel reads its first two `INT32` values.
MLDrift delegation requires the two-output paired-cache form.

#### Semantic Summary

For every update token `x`, MLDrift computes:

```text
token_index = runtime_params[0] + x
```

It writes the K/V values only when both
`token_index < cache_size` and `token_index < runtime_params[1]`. Source height
is broadcast across `kv_cache_batch_size` with modulo indexing.

K and V use different packed external-weight layouts because the two following
attention matmuls consume them in opposite orientations:

- K cache: output dimension is `cache_size`, input dimension is `head_size`
- V cache: output dimension is `head_size`, input dimension is `cache_size`

When the public outputs are `INT8`, MLDrift preserves their original TFLite
tensor references but represents their bytes internally as `UINT8`. The GPU
shader quantizes float source values using `scale_k` and `scale_v`; each scale
defaults to `1.0` when absent. These scales are also consumed by a downstream
quantized `odml.runtime_bmm` when the cache-update node remains its visible RHS
producer.

#### Current Implementation Limits

- The MLDrift `odml.cache_update` parser validates the operand count and
  attributes, but does not validate detailed K/V shape compatibility. The
  MLDrift GPU shader assumes matching source tensors and the packed K/V cache
  layouts described above.
- In the seven-input form, inputs 3 through 6 are not available to the native
  GPU operation. A frontend must not expect them to change GPU semantics.
- Quantized mode is selected from output 0 being `INT8`; the implementation
  assumes output 1 is the matching cache type.
- `scale_k` and `scale_v` must be nonzero when used for quantization.

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

#### Notes For Converter Authors Targeting MLDrift

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
- `moe`

### `moe`

`moe` is a plain TFLite `CUSTOM` op, not an `odml.*` preserved composite. The
MLDrift GPU delegate implements a complete routed, GELU-gated expert block. It
requires one output and has separate floating-point and symmetric-int8 weight
forms.

Use these dimensions:

- `T`: token/sequence count
- `D`: model dimension
- `F`: expert hidden dimension
- `E`: total number of experts
- `A`: active experts per token, with `0 < A <= E`

Common runtime inputs and output:

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 0 | `src` | floating-point | `[1, T, D]` or `[1, 1, T, D]` | Input token states. Batch must be 1. |
| input 1 | `top_weights` | floating-point | `[1, 1, T, A]` | Renormalized routing weights. |
| input 2 | `top_indices` | `INT32` | `[1, 1, T, A]` | Selected expert IDs. |
| output 0 | `output` | floating-point | same logical shape as `src` | Weighted sum of the selected expert outputs. |

The `weight_type == "fp32"` form has seven inputs total:

| Slot | Tensor | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| input 3 | `ff_gate_weight` | floating-point constant | `[F, E, 1, D]` | Per-expert gate projection. |
| input 4 | `ff1_weight` | floating-point constant | `[F, E, 1, D]` | Per-expert up projection. |
| input 5 | `linear_weight` | floating-point constant | `[D, E, 1, F]` | Per-expert down projection. |
| input 6 | `per_expert_scale` | `FLOAT32` or `FLOAT16` constant | `[1, 1, 1, E]` | Additional output scale selected by expert ID. |

The `weight_type == "int8"` form has ten inputs total. Each weight is a
constant, affine-quantized `INT8` tensor whose zero points must all be zero:

| Slot | Tensor | Type | Shape |
| --- | --- | --- | --- |
| input 3 | `ff_gate_weight` | symmetric `INT8` constant | `[F, E, 1, D]` |
| input 4 | `ff_gate_scale` | `FLOAT32` or `FLOAT16` constant | `[F, E, 1, 1]` |
| input 5 | `ff1_weight` | symmetric `INT8` constant | `[F, E, 1, D]` |
| input 6 | `ff1_scale` | `FLOAT32` or `FLOAT16` constant | `[F, E, 1, 1]` |
| input 7 | `linear_weight` | symmetric `INT8` constant | `[D, E, 1, F]` |
| input 8 | `linear_scale` | `FLOAT32` or `FLOAT16` constant | `[D, E, 1, 1]` |
| input 9 | `per_expert_scale` | `FLOAT32` or `FLOAT16` constant | `[1, 1, 1, E]` |

Custom options are a flexbuffers map with required keys `num_experts`,
`num_active_experts`, `model_dim`, `hidden_dim`, and `weight_type`. Optional
`activation`, when present, must be `"gelu"`; optional
`renormalized_top_weights`, when present, must be true. If the custom-options
buffer is completely absent, MLDrift can infer all five required properties
from the input count and tensor shapes. It does not use inference to repair a
present but incomplete or invalid map.

For token `t` and selected expert `e`, the implemented computation is:

```text
hidden = gelu(ff_gate_weight[e] * src[t])
         * (ff1_weight[e] * src[t])
expert_output = per_expert_scale[e] * linear_weight[e] * hidden
output[t] = sum(top_weights[t, route] * expert_output[route])
```

There are no expert biases in this ABI. Routing (`top_indices` and
`top_weights`) is computed outside the op.

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

# Nonstandard names recognized by MLDrift and their parser entry points.
rg -n 'odml\.cache_update|odml\.runtime_bmm|"moe"' ml_drift/delegate/composite

# StableHLO composite names legalized into tfl.custom.
rg -n 'IsSupportedComposite\\(' tflite/converter/stablehlo/transforms/legalize_stablehlo_composite_to_tfl_custom.cc
```

For published-model-derived notes, re-check the exact shipped flatbuffer rather
than relying on memory or secondary notes.
