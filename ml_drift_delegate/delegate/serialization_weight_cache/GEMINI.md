# Gemini Context: Serialization Weight Cache

This directory contains code for building and managing caches for Large Language
Models (LLMs) in the ML Drift delegate.

## Purpose

When loading an LLM for the first time, the weights are often not in an optimal
shape for GPU execution. They need to be rearranged into a better shape (e.g.,
for texture layout or specific alignments). This process can be time-consuming.

To avoid repeating this overhead on subsequent loads, this component allows
caching the rearranged weights on disk. On the second and subsequent loads, the
cache is loaded directly from disk, significantly reducing startup time.

## Architecture Evolution: One File Per Model

Originally, the system generated up to one weight cache file per subgraph. This
caused file proliferation and management issues.

During this session, we refactored the system to use a **unified per-model
cache file format**. All subgraphs belonging to the same model now share a
single cache file.

### Key Changes:

- **Schema Update**: The FlatBuffers schema in `serialization_schema.fbs` was
  updated from a single `BufferList` root to a `ModelCache` root, which contains
  a list of `SubgraphBufferList` objects, each identified by a
  `unique_model_identifier`.
- **Shared Cache Instance**: The `SerializationWeightCache` instance is now
  stored in `MlDriftDelegateData`, making it accessible to all subgraphs during
  initialization.
- **Append Support**: `CacheBuilder` now supports opening the file in append
  mode (without `O_TRUNC`) and preserving existing subgraphs when a new subgraph
  adds its data.

## Implementation Details

### Multi-Subgraph Preservation

To prevent overwriting the FlatBuffer metadata when appending new weight data to
the file:

1.  In `StartBuildStep`, we read the existing `ModelCache` from disk and
    re-encode all *other* subgraphs into the current `FlatBufferBuilder`
    session.
2.  New weights are appended to the file.
3.  In `StopBuildStep`, the new subgraph and the preserved subgraphs are
    combined into a new `ModelCache` root and written back to the file.

### Memory Management

To avoid holding large temporary buffers in memory during the long model
execution phase:

*   The `CacheBuilder::data_` buffer (used as a temporary staging area for
    weights before writing to disk) is now explicitly freed at the end of
    `StopBuildStep()`.
*   This ensures that even if a large tensor was processed during building,
    its memory is released immediately after it is flushed to disk.

## Testing and Limitations

### Android / OpenCL Behavior

*   **Limitation**: The standard path for generating the weight cache via
    `PrepareWeightsInBatches` is currently only implemented for the **Metal**
    backend (used on iOS/macOS). For `OpenCL` (used on Android), this method is
    not implemented.
*   **User Insight / Workaround**: We discovered that passing
    `--convert_weights_on_gpu=false` to the binary successfully triggers weight
    cache generation even on Android with OpenCL!
*   **Performance Impact**: In our tests on a Pixel 9, using the weight cache
    reduced the `Init Executor` time from **~25 seconds down to ~1.8 seconds**
    (approx 13.6x speedup).

## Cache Invalidation

The cache invalidation strategy is automatic and robust.

### Implicit Invalidation (File Name)

The cache file name is derived from the model token and a suffix:
`${model_token}_mldrift_weight_cache.bin`. The `unique_model_identifier` is
used to look up the specific subgraph *inside* the file.

### Explicit Invalidation (Header Checks)

`SerializationWeightCache::Load` performs strict checks on the header:

- **Header Version**: Must match `kVersion` (incremented to `3` during this
  refactor).
- **Build Identifier**: A SHA-256 hash of the MLDrift source files generated at
  build time. Prevents using caches built with different code.
