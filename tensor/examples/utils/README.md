# Checkpoint loading utilities

These utilities load local safetensors into the tensor runner's CPU storage
abstractions, including Google's Gemma4 mobile compressed-tensors checkpoints.

| Files | Responsibility |
| --- | --- |
| [safetensor_loader.h](safetensor_loader.h), [safetensor_loader.cc](safetensor_loader.cc) | Read a file or directory of shards, validate tensor ranges, retain mapped storage, decode quantization metadata, and translate checkpoint names into graph weight names. |
| [safetensor_loader_test.cc](safetensor_loader_test.cc) | Synthetic files covering storage lifetime, invalid ranges and metadata, mapped types, name mapping, and mobile packed-weight conversion. |
| [safetensors.h](safetensors.h), [safetensors.cc](safetensors.cc) | Low-level safetensors header, dtype, offsets, and memory-mapping support. |
| [minijson.h](minijson.h), [minijson.cc](minijson.cc) | JSON value/parser implementation used by the safetensors reader and companion model configuration. |

`SafetensorLoader::Load(path)` indexes the tensors in a file or all safetensors
files in a directory. Each tensor's metadata references shared storage for its
source file. Mapped `SpanCpuBuffer` owners retain that mapping, so returned
tensors can outlive the loader. Converted or repacked tensors own their bytes.

`LoadTensor(name)` resolves logical `.weight` names against mobile
`.weight_packed` storage when a matching compressed-tensors group exists in
adjacent `config.json`. Module targets select the bit width and channel/group
scales. Offset-encoded packed INT32 values are decoded into signed INT4
storage; two-bit values expand to INT4 without changing their values. INT8
weights retain signed byte storage. Group scales support lazy per-layer
embedding lookup, while channel scales feed XNNPACK fully connected layers.

The mobile loader converts ordinary BF16/FP16 graph weights and activation
scales to FP32. The large unquantized per-layer embedding table can retain its
mapped low-precision representation for selected-row lookup. Original
header-only quantization metadata remains a separate supported input path;
plain mapped tensors otherwise retain their storage dtype. Keep `config.json`
next to mobile weights when copying them to a device.

`LoadWeightsWithMapping()` accepts checkpoint-name to graph-name mappings and
also carries mobile input/output activation scales and an explicitly stored
`lm_head.weight`. The [Gemma4 driver](../gemma4/xnnpack_main.cc) supplies the
mapping, creates embedding accessors, and slices the combined per-layer
projection into individual graph weights. The
[mobile fully connected helper](../gemma4/helpers/mobile_fully_connected.h)
consumes activation scales; the loader itself performs no inference.

For a new checkpoint format, extend metadata validation and logical storage
conversion here, then add small synthetic loader cases before running the
[full model comparison](../gemma4/README.md). Preserve mapped owners and exact
logical shapes: packed storage dimensions are not graph tensor dimensions.
Use //tensor/examples/utils:safetensor_loader_test for the Bazel target, or the [standalone guide](../../standalone/README.md) for the supported CMake workflow.
