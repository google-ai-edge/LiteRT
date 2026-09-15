# Transformer graph operations

This directory contains example-level tensor operations. The [Gemma4 helpers](../../gemma4/helpers/README.md)
use its RMS normalization and split-half RoPE when building expression graphs.
Execution is supplied by the existing [XNNPACK backend](../../../backends/xnnpack/README.md)
and [runner](../../../runners/xnnpack/README.md).

| File | Responsibility |
| --- | --- |
| [transformer_ops_graph.h](transformer_ops_graph.h) | Operation identities and attribute records in `litert::tensor::graph`. |
| [transformer_ops.h](transformer_ops.h) | Template builders that connect inputs/outputs and register backend mixins; also contains the composite `RoPE` builder. |
| [transformer_ops_xnnpack.h](transformer_ops_xnnpack.h) | Declares the available `OpMixin<Operation, XnnpackMixinTag>` specializations. Include it where these custom operations are instantiated. |
| [transformer_ops_xnnpack.cc](transformer_ops_xnnpack.cc) | Implements those specializations through ordinary tensor operations or host-created constant buffers. |

## Implemented XNNPACK paths

`RmsNorm` normalizes the last axis. Lowering shallow-clones its inputs, builds
`Square → Mean(keep_dims) → Add(epsilon) → Rsqrt → Mul(input)`, and optionally
multiplies by a scale tensor. `InlineImplementationGraphFor` connects this
implementation to the original operation's input and output value IDs. Gemma4
passes epsilon as a tensor and uses an invalid scale handle for unscaled value
normalization. The low-level operation also supports an epsilon attribute when
no epsilon input is supplied.

`RoPE` directly builds slices, negation, concatenation, multiplication, and
addition. It splits the final dimension into equal halves and computes
`x * cos + [-x2, x1] * sin`. Callers must provide a nonempty shape with an even
last dimension and compatible cosine/sine tensors.

`FillAttentionMask` and `FillRopeCosSin` materialize FP32 constants during graph
conversion. The mask implementation requires positive square sequence
dimensions and supports causal and sliding-window masking. The RoPE table
implementation generates duplicated halves for positions starting at zero.
Their constants do not update when runtime positions change. Gemma4's main
program instead supplies host-generated masks and RoPE tables for its growing
decode sequence.

The headers retain other operation builders, including cache mutation,
`RotaryEmbedding`, and `QkNorm`. They have no XNNPACK specialization here;
declaring a builder does not establish executable backend support.

## Extending and testing

Add an operation record, its tensor builder, and a backend specialization
together, or express the behavior using existing arithmetic operations as
`RoPE` does. Keep shape inference, optional inputs, and graph value mapping
consistent. The [Gemma4 helper tests](../../gemma4/helpers/README.md) cover the
normalization and rotation paths used by the model.

[Standalone example configuration](../../../standalone/examples.cmake) builds the lowering source and the model helper tests. See the [Gemma4 workflow](../../gemma4/README.md) and
[tensor build instructions](../../../README.md) for building and
running tests.
