# Host Flash Attention Design

## Purpose

Host Flash Attention (HFA) in NPUW uses KV-cache history to skip empty cache
tiles during prefill. For LiteRT prefill graphs without `position_ids`, the
compiler prepares eligible attention and can tag a mask input; dispatch can
then read the bound mask and report history to OpenVINO before inference.

```text
LiteRT attention -> compiler HFA rewrite and mask tag -> compiled model
    -> dispatch reads bound mask -> OpenVINO history state -> NPUW HFA
```

## Compiler preparation

The NPU option `enable_hfa=true` enables SDPA fusion and GQA unfolding, then
decomposes eligible multi-token SDPA into MatMul, mask Add, Softmax, and
MatMul operations for HFA. Single-token decode attention remains fused. The
option is disabled by default and can be overridden per graph.

For decomposed attention, the compiler traces the mask back to a suitable
graph input and tags it `litert_hfa_history_mask`. If multiple inputs qualify,
it tags the widest mask so dispatch can read the full cache history. If no
suitable input is found, the compiler warns that history cannot be reported.

On the NPU weight-sharing path, a graph with decomposed attention that
references the shared weight pool uses the NPUW repeated-blocks pipeline with
attention isolated. The compiler sets `NPUW_ATTN=HFA` unless the configuration
already specifies another value. Explicit pipeline settings also win; a
pipeline other than `REP` leaves HFA unattached and the compiler warns.
Outside that path, decomposition alone does not automatically configure HFA.
See [options](../../README_options.md) for configuration details.

## Runtime history

After importing a model, dispatch looks for the tagged input and the OpenVINO
infer request's `npuw_history_size` state. The state is provided by OpenVINO,
not created by LiteRT. The reporter is recreated when a released model is
imported again.

Before each inference, including warm-up runs, the reporter reads the
currently bound additive mask. It must be contiguous `f32`, `f16`, or `bf16`
with shape `[1, 1, Q, past + Q]`, where `Q` and `past` are positive. The
reported history is one past the last visible cache column across the query
rows; uniformly negative padding rows contribute zero. This is a history
count, not a column index.

An untagged graph has no reporter. If the tagged input is unsupported or the
OpenVINO state is unavailable, dispatch logs a warning and cannot report
history. If updating an active reporter fails, the invocation returns an error.

## Source map

- [compiler/npu_optimizer.cc](../../compiler/npu_optimizer.cc): attention rewrite
  and mask tagging.
- [compiler/openvino_compile_context.cc](../../compiler/openvino_compile_context.cc):
  HFA options and NPUW configuration.
- [hfa_history.h](../../hfa_history.h) and [hfa_history.cc](../../hfa_history.cc): mask
  and history-state contract, calculation, and reporter.
- [dispatch/invocation_context.cc](../../dispatch/invocation_context.cc): reporter
  lifecycle and update before inference.
- [compiler/npu_optimizer_test.cc](../../compiler/npu_optimizer_test.cc) and
  [hfa_history_test.cc](../../hfa_history_test.cc): HFA rewrite and history tests.

See the [compiler design](compiler.md) and
[dispatch design](dispatch.md) for their broader responsibilities.