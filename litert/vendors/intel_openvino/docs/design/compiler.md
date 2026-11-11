# Intel OpenVINO Compiler Design

## Purpose and boundary

The compiler plugin selects supported LiteRT operations and converts the
resulting partitions into compiled OpenVINO model blobs, optionally packaged
with a shared weight pool. LiteRT owns partitioning and the compiled model;
this plugin supplies the selection, compilation, and compiled-result
interfaces. The companion [dispatch design](dispatch.md) describes
how the resulting blobs are loaded and run.

```text
LiteRT subgraph -> supported operations -> LiteRT partitions
    -> OpenVINO TFLite frontend -> OpenVINO model -> device compile/export
    -> standalone partition blobs OR one shared GlobalGraph container
    -> LiteRT compiled result -> OpenVINO dispatch
```

The plugin advertises NPU hardware support, but individual partitions can be
compiled for OpenVINO CPU, GPU, or NPU backends. Its entry points and supported
operation list are in [openvino_compiler_plugin.cc](../../compiler/openvino_compiler_plugin.cc).

## Partition selection and conversion

The partition callback rejects a subgraph if any ranked input has rank six or
greater. Otherwise it selects operations from the plugin's supported-op list;
unsupported operations are not selected. Selection is not a guarantee that
every possible shape, type, or configuration will compile on the chosen
OpenVINO device.

[graph_iterator.h](../../compiler/graph_iterator.h) adapts each LiteRT partition to the
OpenVINO TFLite frontend. [decoder.h](../../compiler/decoder.h) provides tensor metadata and
operation attributes, leaving OpenVINO to construct the corresponding nodes.
The iterator also converts packed signed 2-bit weights to unsigned 2-bit for
NPU/CPU or signed 4-bit for GPU.

For each partition, [openvino_compile_context.h](../../compiler/openvino_compile_context.h)
resolves the target and properties, applies applicable NPU SoC settings, and
runs NPU-specific graph optimizations. The default is NPU with latency mode;
each partition can override its backend, and per-graph configuration entries
override model-wide entries. Plugin-specific optimizer switches are consumed
locally; other configuration properties are passed to OpenVINO. NPU constant
folding is enabled by the compile context; additional rewrites, including
attention and MoE transformations, are controlled by options. See
[options](../../README_options.md) for the public configuration surface and
examples.

The plugin compiles the converted model with `ov::Core::compile_model`, exports
it, and associates a `Partition_<index>` call name with the compiled result.
OpenVINO exceptions become compilation errors. A selected operation can still
fail during conversion or device compilation.

## Cross-partition weights

Weight sharing is enabled by default, but is used only when there is more than
one partition and every partition targets the same GPU or the same NPU. A CPU
partition, mixed backends, a single partition, or an explicit opt-out uses
standalone blobs with baked weights instead.

[weight_bank.h](../../compiler/weight_bank.h) collects weights by LiteRT buffer ID so tensors
sharing storage are represented once. The shared path builds a pool before
compiling the partitions. On NPU it first converts and optimizes each partition
to harvest live and generated constants, prunes unused buffers, then rebuilds
the graphs for compilation. Rebuilding the graphs saves memory but requires
converting them twice. Both passes must produce consistent constants and pool
offsets.

- **GPU:** [weights_to_parameters.h](../../compiler/weights_to_parameters.h) replaces shared
  constants with parameters and records their names and buffer IDs for runtime
  binding.
- **NPU:** [alias_shared_constants.h](../../compiler/alias_shared_constants.h) aliases and
  tags constants with their pool offsets. The NPU weightless configuration uses
  a content-derived bank name to keep unrelated pools separate.

## Compiler/dispatch bytecode contract

Standalone output consists of a 20-byte versioned `LRTOVHDR` header followed
by the OpenVINO export. The header records the backend; its definition and
validation live in [bytecode_header.h](../../bytecode_header.h).

On the shared path, [global_graph.h](../../compiler/global_graph.h) defines an `OVGLOBAL`
container: a buffer directory and contiguous deduplicated pool, followed by
named subgraphs with device, KV tier, constant map, and
header-prefixed OpenVINO payload. The current writer produces version 3;
the parser also accepts version 2, for which the KV tier is zero. Pool entries
are laid out in ascending buffer-ID order. NPU weightless offsets must match
their offsets in the staged pool file. The layout is documented as
little-endian; the current reader and writer copy native integer
representations, so they assume little-endian hosts.

The shared container is serialized once as bytecode module 0, and every
partition call points to that module while retaining its own name. Standalone
compilation emits one module per partition. The dispatcher selects the named
subgraph and resolves its weights; both sides must agree on container and
header versions, offsets, names, and backing-storage lifetime.

## Source map and tests

- [openvino_compiler_plugin.cc](../../compiler/openvino_compiler_plugin.cc): plugin API,
  selection, conversion, compilation, and compiled-result construction.
- [openvino_soc_config.cc](../../compiler/openvino_soc_config.cc) and
  [npu_optimizer.cc](../../compiler/npu_optimizer.cc): SoC settings and NPU model rewrites.
- [global_graph_test.cc](../../compiler/global_graph_test.cc): container serialization and
  parsing without an OpenVINO runtime dependency.
- [openvino_compiler_plugin_test.cc](../../compiler/openvino_compiler_plugin_test.cc),
  [graph_iterator_test.cc](../../compiler/graph_iterator_test.cc), and
  [decoder_test.cc](../../compiler/decoder_test.cc): compiler and frontend adapter tests.
- [weight_bank_test.cc](../../compiler/weight_bank_test.cc),
  [weights_to_parameters_test.cc](../../compiler/weights_to_parameters_test.cc), and
  [alias_shared_constants_test.cc](../../compiler/alias_shared_constants_test.cc): shared
  weight preparation.

Other test targets are listed in [BUILD](../../compiler/BUILD); several are marked manual and
require OpenVINO dependencies or suitable hardware.