# Example operation extensions

This directory holds tensor operations used by model examples. Its current
component is [transformer](transformer/README.md), which extends the tensor/XNNPACK build. It supplies RMS normalization and split-half
RoPE for the [Gemma4 helpers](../gemma4/helpers/README.md), plus constant mask and
positional-table lowering.

The extension pattern separates operation records and tensor builders from
backend mixin specializations. A model creates an expression graph through the
builder; the [XNNPACK backend](../../backends/xnnpack/README.md) discovers the
specialization during conversion, and the
[runner](../../runners/xnnpack/README.md) executes the resulting XNNPACK graph.
Some retained transformer builders have no specialization, so consult the
child README before adding one to an executable graph.

Keep model-specific extensions here and use the core
[arithmetic API](../../arithmetic.h) for operations already supported there.
When adding an extension, provide its shape/metadata handling, backend lowering,
and a focused numerical test together. The
[standalone example configuration](../../standalone/examples.cmake) currently links transformer
lowering into the Gemma4 graph library. See the
[Gemma4 workflow](../gemma4/README.md) and
[tensor build instructions](../../README.md) for validation.
