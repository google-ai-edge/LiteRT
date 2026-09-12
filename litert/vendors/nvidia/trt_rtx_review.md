# NVIDIA integration review after permanent-worktree migration

Reviewed against LiteRT upstream `761d99cb9` plus the individually preserved
NVIDIA/12B commits, and LiteRT-LM upstream `007e7c76` plus its two
signature-selection commits. This review does not remove runtime behavior.
Source findings below are not measured savings unless explicitly stated.

## Follow-up opportunities, in priority order

### Isolate the observed cold-compilation timing difference

The new build's cold compiler/packing/persistence interval was approximately
379 seconds in two runs, versus 302 seconds for the same-day retained control.
Priming the source model did not close the gap. The independently compiled
artifact files are byte-identical, and warm throughput/memory agree, but those
checks do not prove cold-start equivalence. `trt_rtx_measurements.md` retains
the complete evidence and cache/memory-guard caveats. A mixed old/new LM-host
and plugin-binary comparison is the next isolating test; no cause is asserted
and no speculative runtime change is included in this migration.

### Serialized JIT must include configuration in its cache identity

`compiler/compiler_plugin.cc:427`, `BuildCompilerSdkVersion()`, returns the
plain SDK version when AOT is disabled. Only the AOT case appends the NVIDIA
environment/configuration fingerprint. LiteRT's compilation cache includes
that SDK-version string in its key (`litert/core/cache/compilation_cache.cc`).

Consequently, a persistent non-AOT compiler cache can have the same key after
changing NVIDIA environment options such as the FC weight cap or
`PREDEQUANTIZE_FC_WEIGHTS`. Selecting `JIT_HANDLE=1` also does not invalidate an
already cached serialized result through this version string. Fresh-cache
measurements are unaffected; this is a source-confirmed key omission, not a
completed end-to-end stale-cache reproduction.

Follow-up: use the compile-configuration identity for all cacheable compiler
results, not only AOT. Add a cache-key test that changes NVIDIA options with
AOT disabled. Preserve runtime-only options outside the key where safe.

### Runtime-cache naming rereads the entire engine

`dispatch/dispatch_api.cc:1839`, `RuntimeCachePath()`, hashes
`bytecode_.engine_data` through `bytecode_.engine_size` to derive a filename.
That happens even after an AOT mapping passes trusted file-identity validation.
The freshly built 12B decode engine contains 5,969,299,980 bytes, so this remains
a substantial linear read distinct from the avoided AOT fingerprint scan.

Follow-up: persist an engine-specific content fingerprint at compilation and
carry it through the bytecode format, then use it for runtime-cache naming.
Keep a content-hash fallback for older bytecode and in-memory JIT. An artifact
pathname alone is not a safe replacement key. Measure initialization latency,
file-backed RSS, and numerical/cache equivalence before accepting the change;
the current review does not attribute a numerical saving to it.

### Optional greedy-sampler API has no consumer in these two source trees

`dispatch/greedy_sampler_c_api.h` exports three NVIDIA sampler functions,
implemented in `dispatch/dispatch_api.cc` and backed by
`dispatch/greedy_sampler_kernel.cu`. Searches across both permanent worktrees
found definitions and tests but no LiteRT-LM caller of those exported names.

Follow-up: consider a separately linked optional component after checking
external consumers. It does not allocate sampler GPU resources until called,
so removal should not be sold as a multi-gigabyte runtime memory saving.

### Debug IO dumping contains an avoidable bounds hazard

`dispatch/dispatch_api.cc:1931`, `DumpDeviceBufferPrefix()`, copies a checksum
sample starting at `device_ptr + size / 2`, but caps the length by `size`, not
by the remaining `size - size / 2`. Some small buffers can be read past their
end, and the second `cudaMemcpy` result is ignored. The formal runs disable
this debug option.

Follow-up: cap by the remaining bytes and check the copy result, with a
small-buffer regression test, or remove this diagnostic if no longer useful.
This is not evidence that normal inference reads out of bounds.

### Profiling timestamp name is misleading

`memory_profile.cc:66` labels `absl::GetCurrentTimeNanos()` as `monotonic_ns`.
That is a wall-clock timestamp. The external sampler used for migration
measurements uses `time.perf_counter()` for elapsed time and documents this
distinction. Renaming the field or using a monotonic clock requires updating
the parsers together; it is a profiling correctness improvement, not an
inference memory optimization.

## Logic to retain

- `compiler/compiler_plugin.cc:1028`, `TryLoadAotManifest()`, reads the small
  manifest and checks artifact existence, type, and size. It does **not** hash
  all artifact contents. These checks allow a missing cache artifact to cause
  recompilation instead of a later dispatch failure.
- `dispatch/aot_artifact.cc:113`, `MappedAotArtifact::Open()`, opens without
  following symlinks, checks the opened file, and uses the read-only file's
  recorded identity for the cheap path. Changed or copied files fall back to
  content validation; mismatches fail closed. Removing this boundary permits
  same-size stale/corrupt artifacts. Both compiler and dispatch checks have
  a purpose; the normal path is already lazy.
- Old locator/bytecode versions and their content-validation fallback support
  existing persisted artifacts. Remove only under an explicit compatibility
  policy, not because the newest writer no longer emits them.
- The specialized external vocabulary-head path
  (`compiler/tensorrt_graph_builder.cc`, `MatchTensorRtLlmHead()`) requires
  hidden size 1536 and vocabulary size 262144, so the 12B hidden-size-3840 model
  does not use it. The fresh E2B compatibility run actually selected and
  initialized it (`external W2 head matched`, 100,663,296 weight bytes), then
  returned `PARIS` in agreement with CPU. This path and its softcap semantics
  are demonstrably still used, not dead code.
- The FP8/GEMM path remains necessary for prefill and cases outside the
  custom sub-byte GEMV plugin's supported shapes/types. Decode success does
  not justify deleting it.
- `EnsureSubbyteGemvPluginRegistered()` is an intentionally empty link anchor
  for the translation unit containing TensorRT's static plugin registration.
  Its empty body alone does not make it removable.
- LiteRT-LM's migrated logic forwards selected signatures and filters their
  buffer setup. It contains no second NVIDIA AOT artifact-validation layer.
  Small linear membership checks over the selected names do not warrant a
  new caching abstraction.

## Verification boundary

Builds and focused tests, real CPU/NPU generation, AOT miss/hit validation,
uninstrumented throughput, and externally sampled memory are tracked in
`trt_rtx_migration.md`. This review deliberately preserves the measured
implementation. Its proposed follow-ups need their own reproductions and
before/after acceptance checks.
