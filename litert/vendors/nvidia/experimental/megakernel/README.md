# The decode megakernel (experimental)

The decode step of Gemma 4 26B-A4B (30 layers, the head and its soft cap) in one
persistent kernel, behind a switch that is off by default.

## Enabling it

Set `LITERT_NVIDIA_MEGAKERNEL` when the model is compiled and run (with
`benchmark/bench.sh`: `--env LITERT_NVIDIA_MEGAKERNEL=1`). The model is the
export as it is; nothing is carved or converted.

| Value | The decode signature of a Gemma 4 mixture-of-experts export |
|---|---|
| unset, empty or `0` | the compiler's usual partitions and engines |
| `1` | one partition, one plugin layer, the megakernel: the step in one cooperative launch |
| `interpreted` | the same plugin layer running today's per-op launches (the megakernel's reference) |

Every other signature (prefill, verify) and every other model takes the
compiler's usual path with the switch set too.

## Isolation

With the switch unset none of this runs, and nothing the compiler or the
dispatch produces changes: the same partitions, the same engines byte for
byte, the same cache keys (the variable enters the key only when it is set).
Everything is in this directory except these hooks:

- `compiler/compiler_plugin.cc`: the variable's name in the list of switches
  that enter the cache key when set; at the top of the partition entry point,
  behind `MegakernelRequested()`, the selection of the whole decode subgraph;
  at compilation, `BuildMegakernelEngine` for a partition selected that way
  (`IsMegakernelPartition` is false unless one was).
- `compiler/tensorrt_graph_builder.cc`: `megakernel_lowering.inc` included as
  members of the builder class (the lowering reads the builder's in-place
  caches, MoE block plans and weight store); a call of `LowerMegakernel` and a
  `break` of the op-by-op loop, both behind `megakernel_body_`, which only
  `BuildMegakernel` sets; and `BuildMegakernelEngine`, the second entry point.
- `dispatch/dispatch_api.cc`: the plugin's registration, so that an engine
  that holds the plugin layer deserializes.
- `compiler/BUILD`, `dispatch/BUILD`: one dependency each.

The production kernels are not touched. The megakernel's attention and its
qkv norms and RoPE are copies of the per-op kernels' device code
(`decode_attention.cuh`, `qkv_norm_rope.cuh`, `int4_block_gemv.cuh`,
`block_sync.cuh` here, in the namespace `litert::nvidia::megakernel`), taken
when `trtllm/decode_attention.cu`, `qkv_norm_rope.cu` and `int4_block_gemv.cu`
were what they are at this directory's first commit: a later change of those
kernels is not followed here. The `interpreted` step and the lab's reference
call the production kernels through their C interfaces.

## Status

An experiment. On the RTX 5080 under WSL (2026-11-20, `bench.sh --model 26b
--workload short`, interleaved, 3 iterations each):

- Decode: 249 tokens/s with the megakernel against 229 for the default
  engine (1.09x) and 225 for `interpreted`; prefill is the default's in all
  three. Peak device memory: 14,922 MiB for both values against the default's
  14,894.
- Numerics: the teacher-forced comparison against the CPU passes for both
  values (`benchmark/teacher_forced.sh`, 642 steps at positions 19 to 1,078):
  top-1 agreement 0.983 and NLL +0.0099 nats a token with the megakernel,
  0.983 and +0.0094 with `interpreted`, 0.978 and +0.0099 for the default
  engine.
- Lab (synthetic step): 3.83 ms against the per-op launches' 4.29 ms at 1k
  tokens (1.12x), 4.20 against 4.44 ms at 8k (1.06x).
- Device memory: the kernel's bulk copies name their target as the CTA's
  shared memory. Named as the cluster's (`shared::cluster`), a bulk copy is a
  call into the driver's runtime on sm_120, and the driver then reserves that
  call's stack for every thread the device can hold: 1.65 GiB on the RTX 5080
  at the kernel's first launch, which the 26B does not leave free on a 16 GB
  card. Until 2026-11-20 the kernel used that form, the driver paged device
  memory out, and decode ran at about 130 to 150 tokens/s.
- Limits: a model the program does not describe fails its compilation with
  the reason (`megakernel: ...`); the kernel needs sm_90 or newer; the MSVC
  build has not been tried.
- What a faster kernel would take: a balanced attention (the attention part
  of a layer takes ~70 us for ~31 us of traffic), before it can approach the
  ~2.4 ms a step that the bytes allow.

## Files

- `megakernel_lowering.h`, `.cc`, `.inc`: the compiler's side. Which subgraph
  is taken whole, and the recognition of the step layer by layer (the
  attention from each `odml.cache_update`, the feed-forward block as the MoE
  lowering plans it, the tail) into a `MegakernelDescription`; the weights
  stay in the weight store and the experts in the MoE plugin's arenas.
- `megakernel_plugin.h`, `.cc`: the TensorRT plugin layer. It builds the
  program from the description and the engine's buffers and runs it, as the
  megakernel or as the per-op launches; native in-place cache writes follow.
- `megakernel_steps.h`, `.cu`: the steps the engine's graph runs as TensorRT
  layers (the embedding cast, the cache writes, the norm and residual after
  the attention, the final norm and the soft cap), for the per-op step.
- `program.h`: the program format. The caller describes the model as the
  per-op launches take it (`LiteRtNvidiaMegakernelModel`). The builder turns it
  into a list of instructions with a scratch per layer. A model without
  pointers builds the shapes alone, so a plugin can size its workspace first.
- `megakernel.h`, `megakernel_reference.cc`: building, launching (one
  cooperative launch per step) and the reference step: today's per-op
  launches.
- `megakernel.cu`: the kernel (sm_90 and newer). One CTA per SM, each walking
  every instruction: eight consumer warps run the instructions' work, and a
  producer warp streams every projection's weights into a ring in shared
  memory with bulk copies (TMA), in program order across instructions and
  layers, at most three units in flight. A projection's units are taken by the
  consumer warps in stream order; attention splits, combines and heads go to
  whole CTAs (the consumer warps, on a named barrier of their own). An
  instruction waits for the counters of the instructions it reads, and the
  counters carry the step's epoch.
- `megakernel_lab.cc`, `megakernel_lab_data.*`, `mktrace.*`: the lab and its
  trace (below).

## Numerics

FP32 accumulation everywhere and the same order every run, but not the
per-op kernels' order: projections, norms and reductions sum in the order that
suits the stream. The graph's BF16 rounding points are kept (x, h, the graph's
norms, the router's logits, the soft cap).

- The attention and the qkv norms and RoPE run this directory's copies of the
  per-op kernels' device code, operation for operation theirs; the attention
  reads chunks of 64 keys at every cache length, where the per-op kernel reads
  chunks of 32 below 2,688 keys for many rows.
- The lab reports how each layer differs from the reference: x and h, the
  caches' new rows (FP16), the routed experts, the output, the logits and
  their top 40. A layer whose router logits tie in BF16 at the top_k-th place
  can route to another expert; the synthetic step then diverges from there.
- Its first two versions kept the reference's order and matched it bit for
  bit; they ran at 0.80x of the per-op step.

## The lab

`megakernel_lab` runs the 26B's decode step on synthetic data with the real
shapes and the runtime's formats:

- INT4 projections with FP16 scales;
- the MoE plugin's expert-major arenas with compact scales, shared between
  layers (`--arenas`);
- FP16 caches as rows, filled to `--context`.

It runs the reference and the megakernel on the same inputs and compares them
layer by layer, and prints the device memory each one's first step takes. It
then times both as CUDA graphs, replayed in turn and interleaved: compare the
ratio, since times on WSL are rough.

    bazel build //litert/vendors/nvidia/experimental/megakernel:megakernel_lab
    megakernel_lab --context 1024 --replays 20 --trace step.json

## mktrace

With `--trace FILE.json` the lab makes one traced launch. Every CTA stamps
`%globaltimer` at each instruction in four places: when it reaches the
instruction, when its waits are satisfied, when its prologue is done, and when
its items are done (`LiteRtNvidiaMegakernelTraceBytes`). `mktrace.h` writes
this as Chrome trace JSON for Perfetto (ui.perfetto.dev) or chrome://tracing:

- one track per CTA, named with its SM;
- per instruction, a `wait` slice, a `prologue` slice and a slice of its items.

`SummarizeTrace` prints the share of the CTAs' time spent waiting, in prologues
and in items. Per opcode it prints the mean wait, prologue and items time per
CTA, and the span from the first CTA starting its items to the last one
finishing. CUPTI cannot see kernels on WSL, so this trace is the profiler.
