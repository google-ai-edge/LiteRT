# Gemma4 E2B native tensor runner

This package executes the published Gemma4 E2B model through LiteRT's tensor
API and XNNPACK, using an exported set of coefficients from a `.litertlm`
bundle. It builds tensor graphs directly and does not invoke the LiteRT-LM
executor or the TFLite interpreter during inference.

The implementation preserves the locally validated staged runner: compact
static INT2 MLP weights, published INT8 KV quantization, attention over active
cache rows, reusable prefill/decode graphs, a persistent worker pool, and shared
packed weights and scratch storage. It is specialized to the published E2B
bundle; it is not a general E4B or arbitrary checkpoint loader. Follow
[REPRODUCE.md](REPRODUCE.md) for the pinned model download, build and Android
performance recipe. The separate
[normal Gemma4 example](../README.md) supports the raw checkpoint workflow.

## Structure

| Component | Responsibility |
| --- | --- |
| [driver.cc](driver.cc) | Read fixed token histories, author and run graph stages, manage sessions, and write timings, logits and optional diagnostics. |
| [driver_support.h](driver_support.h) | Token parsing, finite/argmax checks, FP32 output writing, and input construction. These utilities replace the old inclusion of another driver's `.cc` file. |
| [stage_runner.h](stage_runner.h) | Adapt the common NNPACK runner to borrowed persistent threads, packed weights and an optional shared XNNPACK workspace. |
| [active_kv_bank.h](active_kv_bank.h) | Own stable token-major INT8 K/V storage and transactional appends for the 15 distinct KV owners. |
| [matched_bundle_loader.h](matched_bundle_loader.h) | Validate the fixed export schema, map coefficients and scales, and reconstruct genuine compact INT2 buffers. |
| [model/gemma4_graph.h](model/gemma4_graph.h) | Author the private matched-bundle graph and expose explicit stage boundaries. |
| [model/helpers/attention.h](model/helpers/attention.h) | Preserve E2B sharing, quantization, head layout and attention math while separating projection, cache append, attention and post-processing. |
| [model/helpers/bundle_matched_ops.h](model/helpers/bundle_matched_ops.h) | Preserve the dynamic QD8/QC2 LM head and the full per-layer projection's quantization boundary. |
| [model/helpers/static_int2_fully_connected.h](model/helpers/static_int2_fully_connected.h) | Define compact static QS8/QC2 fully connected operations and audit their lowering. |
| [model/helpers/int8_kv_cache.h](model/helpers/int8_kv_cache.h) | Shared owner metadata, published scales, and fixed-layout reference helpers used by the regression checks. |
| [active_runtime_audit.h](active_runtime_audit.h), [memory_snapshot.h](memory_snapshot.h) | Check actual attention operators and report diagnostic allocation/RSS information. |
| [tests](tests), [tools](tools), [fixtures](fixtures) | Operator/lifetime checks, reproducible export and comparison tools, and pinned fixed-history inputs. |

The graph/helpers use `litert::tensor::examples::gemma4::native` to keep their
matched-bundle behavior separate from the normal example. Config, RoPE table
construction, quantized embedding lookup and the FP32-activation FC helper are
shared with the parent example. The native LM head has its own operation; the
raw checkpoint helper's activation policy must not replace it.

## Execution and ownership

A `LiveRuntime` owns the worker pool and packed-weight cache, optionally owns a
shared workspace, and contains reusable prefill and decode stages. Its member
order destroys the stages before those shared resources. `StageRunner` borrows
the pool/cache; the pool's configured thread count cannot change after creation.
All stages sharing scratch run serially. External stage outputs retain their
own buffers, so growing the shared scratch arena must not invalidate a producer
output still needed by the next stage.

The prefill graph stops after the last distinct KV owner, because later layers
do not contribute additional cache entries needed by the prompt prefix. The
decode graph executes all 35 layers and produces complete-vocabulary logits.
The driver processes all prompt tokens except the last in reusable chunks,
then consumes the last prompt token through decode to obtain the first
prediction. Every subsequent forced token causes one decode call.

Each session allocates a fresh `ActiveKvBank`. Its 30 buffers store logical
`[capacity, head_dim]` K/V arrays; 8448 rows occupy 77,856,768 payload bytes plus
small kernel padding. `BeginAppend` reserves a logical append, each owner writes
its newly projected codes once, and `Commit` requires every owner to finish.
`Abort` and `Reset` change visibility; they do not promise to zero old bytes.
Readers must use the bank's visibility rules and mask any padded/stale rows.

Attention binds cache views directly. Global layers use the valid prefix;
local layers use the sliding window, with start/end alignment and masks that
preserve the original visible positions. FP32 queries and probabilities are
multiplied with INT8 K/V through XNNPACK's FP32/QCINT8 BMM fusion. There is no
dynamic activation quantization of attention in this path. Absolute position
inputs drive runtime FP32 RoPE arithmetic. The default 32-row alignment is part
of the validated numerical behavior, not just an allocation preference.

Sixty MLP matrices in layers 15–34 are reconstructed from their original INT2
provenance, reducing their coefficient storage from 566,231,040 widened bytes
to 283,115,520 compact bytes. Published activation scales are retained. The
loader rejects a tensor merely having small numeric values unless its source
metadata also identifies the expected original INT2 tensor.

## Run a fixed history

Build instructions and dependency setup live in the
[standalone build documentation](../../../standalone/README.md). The executable target is `gemma4_native_runner`; the Bazel label
is `//tensor/examples/gemma4/native:gemma4_native_runner`.

From the LiteRT root, after building and exporting the model:

```sh
"$NATIVE_RUNNER" \
  --bundle_dir="$MATCHED_BUNDLE" \
  --cases_file=tensor/examples/gemma4/native/fixtures/capacity_smoke_8_1.tsv \
  --output_dir="$NEW_OUTPUT_DIRECTORY" \
  --num_threads=4 --cache_capacity=8448 --prefill_chunk_rows=128 \
  --kv_alignment=32 --preserve_static_int2=true --share_workspace=true \
  --reuse_runtimes=true --warmup_runs=0 --measured_runs=1 \
  --memory_report=false --fixed_attention_extent=false \
  --dump_full_logits=true
```

The output directory must not already exist; its parent must exist. These flags
explicitly select the optimized configuration. For timing on an otherwise idle
target, use the `performance_*_64.tsv` files, one discarded warmup, repeated
measurements and `--dump_full_logits=false`. Choose the same thread count,
affinity, capacity and input history for every runtime being compared.

| Flag | Constraint or effect |
| --- | --- |
| `cache_capacity` | Positive, at most 8448, divisible by `kv_alignment`; must fit prompt plus forced inputs. This is storage capacity, not an instruction to attend over all rows. |
| `prefill_chunk_rows` | 128 or 1024; the final partial prefix chunk is padded and masked. |
| `kv_alignment` | Power of two from 1 to 128; default 32. Changing it can alter FP32 reduction order. |
| `preserve_static_int2` | Select the compact static QC2 MLP path; the driver checks that all 60 operators are present. |
| `share_workspace` | Share scratch among serialized stages. |
| `reuse_runtimes` | Retain compiled stages across sessions and repetitions; each session still has fresh KV storage. |
| `dump_full_logits` | Dump complete little-endian FP32 vectors after timing, only in the first measured repetition; also audit actual attention operators. |
| `dump_cache` | With full-logit dumping, also emit committed logical token-major INT8 caches. |
| `trace_position` | Dump an absolute token position at existing stage boundaries; default `-1` disables it. |
| `memory_report` | Record Linux/Android process memory and allocation snapshots; marks the capture ineligible for benchmark timing. |
| `fixed_attention_extent` | Diagnostic fixed-width attention control; requires alignment 1. |

The legacy arithmetic/cache compatibility flags remain constrained to their
defaults. Use the native trace/dump flags for diagnostics.

`elapsed_ms` for the prompt includes fresh KV allocation/zeroing, prefix
processing, the first complete prediction, and finite/argmax checks. Decode
elapsed time includes one forced-input call and its finite/argmax checks.
Model loading and graph creation are separate. `forward_ms` measures the sum
of stage calls and is narrower than elapsed time. Memory/trace captures and
warmups must not be mixed into ordinary timing results.

## Data and validation tools

Each TSV row has exactly three tab-separated fields:
`case_id`, comma-separated prompt IDs, and comma-separated forced input IDs
(`-` means none). IDs must be in `[0,262144)`, the prompt starts with BOS `2`,
and case IDs are unique filename-safe strings. There is no tokenizer, sampler,
chat-template application or EOS stopping in this harness. An N-token forced
continuation yields N+1 logit vectors, including the prompt's first prediction.

[tools/README.md](tools/README.md) describes the complete export pipeline.
The export consists of `manifest.json`, tensor bytes, FP32 scale arrays and
fixed constants, with source tensor indices and hashes. The C++ loader checks
the expected model identity, dimensions, metadata, files and values needed by
this specialized graph. The offline export validator additionally reads back
all exported weights and scales against the source bundle; a manifest hash
field alone is not a cryptographic verification of the mapped tensor bytes.

The source token manifest is pinned by SHA256. Regenerate the default 128,
1024 and 4096 prompt workloads, the short capacity smoke and the 4096 boundary
check with `python3 tensor/examples/gemma4/native/tools/prepare_fixtures.py`.
The repeated text workload measures execution; it is not a quality dataset.
The optional 8192 fixture is generated only when requested explicitly.

The six self-contained Bazel C++ test labels in this package are `stage_runner_smoke_test`,
`stage_workspace_test`, `active_kv_bank_test`, `active_attention_test`,
`active_extent_rounding_test` and `static_int2_test`. They cover borrowed
resources, output lifetimes across scratch growth, exact cache codes and
visibility, dynamic view/fusion behavior, numerical effects of rounded extents,
and static QC2/QC4 equivalence. The rounding test checks bounded numerical error;
its success does not claim every tested alignment is bitwise identical.
`static_int2_loader_test` is a manual binary taking one matched-bundle directory
argument; it verifies all 60 compact matrices and released widened mappings.
Standalone CMake targets and binaries add `litert_gemma4_native_` before each
short test name (including the manual loader test). For example, build or run
`litert_gemma4_native_active_attention_test` in the standalone configuration.

Use [tools/compare_live.py](tools/compare_live.py) to compare complete logits and,
when present, logical caches against a retained reference. Pass
`--require-bitwise` to make exact equality an explicit requirement. Merely
matching argmax IDs is weaker evidence. The [fixture manifest](fixtures/manifest.json)
records the fixed token histories and their hashes; full-model comparisons
must use the same histories and compatible quantization policies.

## Extending the implementation

Change graph staging in `driver.cc` and `active_graph_context.h`, cache ownership
or view rules in `active_kv_bank.h`, and model arithmetic in the private model
helpers. Keep the quantization boundaries, shared-owner mapping, absolute RoPE
positions, padding masks and first-logit schedule covered by regression tests.
Adding a new bundle requires updating the export/schema validation and checking
its coefficients and intermediate behavior against a reference; changing a
filename or model enum alone is insufficient.

The runtime and memory audits inspect internal XNNPACK structs and operator
types. Recheck them when updating the pinned dependency. They are local
diagnostic contracts, not stable public LiteRT interfaces.
