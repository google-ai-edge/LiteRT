# ML Drift & LiteRT-LM On-Device Probe Snippets

These temporary probes work without adding any build dependencies (using only
`<cstdio>`, `<cstdlib>`, and `<cmath>`) so they pass Bazel/Blaze
`layering_check`. Always revert them with `hg revert` once the root cause is
isolated.

---

## 1. Host Executor Prefill/Decode State Dumper (`llm_litert_compiled_model_executor.cc`)

Add inside an anonymous namespace in
`google3/third_party/odml/litert_lm/runtime/executor/llm_litert_compiled_model_executor.cc`
and call `DebugDumpBuffers("prefill_in", input_buffers)` /
`DebugDumpBuffers("prefill_out", output_buffers)` around `Run()` in
`BindTensorsAndRunPrefill` (forcing `async = false` when `DebugDumpEnabled()`):

```cpp
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>

bool DebugDumpEnabled() {
  static const bool enabled =
      std::getenv("LITERT_LM_DEBUG_DUMP_STATE") != nullptr;
  return enabled;
}

bool DebugDumpBudgetLeft() {
  static std::atomic<int> budget{6};
  return budget.fetch_sub(1) > 0;
}

template <typename T>
void DumpTypedSpan(const char* tag, const std::string& name,
                   absl::Span<const T> data) {
  double sum = 0.0, sumabs = 0.0;
  double min_v = data.empty() ? 0.0 : static_cast<double>(data[0]);
  double max_v = min_v;
  size_t nan_cnt = 0;
  for (T v : data) {
    double d = static_cast<double>(v);
    if (std::isnan(d)) {
      ++nan_cnt;
      continue;
    }
    sum += d;
    sumabs += std::abs(d);
    if (d < min_v) min_v = d;
    if (d > max_v) max_v = d;
  }
  fprintf(stderr,
          "[DBG] %s %s n=%zu sum=%.4g sumabs=%.4g min=%.4g max=%.4g nan=%zu "
          "head=[",
          tag, name.c_str(), data.size(), sum, sumabs, min_v, max_v, nan_cnt);
  for (size_t i = 0; i < std::min<size_t>(6, data.size()); ++i) {
    fprintf(stderr, "%s%.4g", (i ? ", " : ""), static_cast<double>(data[i]));
  }
  fprintf(stderr, "]\n");
}
```

---

## 2. In-Shader Runtime Parameter Readback Probe (`add_values_to_cache_kernel.cc`)

When `kv_cache_k_{N}` / `kv_cache_v_{N}` are unexpectedly zero or corrupted, use
the KV cache output buffers themselves as a readback channel for the shader's
`args.params` buffer in
`ml_drift/delegate/composite/add_values_to_cache_kernel.cc`:

```cpp
const bool probe_params = std::getenv("MLDRIFT_CACHE_PROBE") != nullptr;

// 1) In the ring/non-ring token_index branch: bypass the update_length gate
// when probe_params is set so thread X=0 always writes to token_index=0:
if (probe_params) {
  c += "  int update_length = args.update_width;\n";
  c += "  int token_index = X;\n";
}

// 2) Right after final_value_k and final_value_v are loaded (non-quantized path):
if (probe_params) {
  c += "  {\n";
  c += "    int p0 = args.params.Read(0);\n";
  c += "    int p1 = args.params.Read(1);\n";
  c += "    int p2 = args.params.Read(2);\n";
  c += "    int p3 = args.params.Read(3);\n";
  c += "    int p4 = args.params.Read(4);\n";
  c += "    int p5 = args.params.Read(5);\n";
  c += "    int p6 = args.params.Read(6);\n";
  c += "    final_value_k = " +
       ucl::Convert(ToUclDataType(op_def.src_tensors[0].GetDataType(), 4),
                    "ucl::Init<float4>(float(p0), float(p1), float(p2), float(p3))") +
       ";\n";
  c += "    final_value_v = " +
       ucl::Convert(ToUclDataType(op_def.src_tensors[1].GetDataType(), 4),
                    "ucl::Init<float4>(float(p4), float(p5), float(p6), 0.0f)") +
       ";\n";
  c += "  }\n";
}
```

With both `LITERT_LM_DEBUG_DUMP_STATE=1` and `MLDRIFT_CACHE_PROBE=1`, the host
dump prints `kv_cache_k_N head=[p0, p1, p2, p3, p0, p1]` and
`kv_cache_v_N head=[p4, p5, p6, 0, ...]`.

---

## 3. `StridedSlice` Storage-Type Descriptor Dumper (`strided_slice.cc`)

In `google3/third_party/ml_drift/common/kernels/strided_slice.cc` inside
`GetStridedSliceCode`, immediately after `AddSrcTensor` / `AddDstTensor`:

```cpp
if (std::getenv("MLDRIFT_SLICE_PROBE") != nullptr) {
  const auto& s = op_def.src_tensors[0];
  const auto& d = op_def.dst_tensors[0];
  if (s.GetBHWDCShape().c <= 16 && s.GetBHWDCShape().w <= 4 &&
      s.GetBHWDCShape().h <= 4) {
    fprintf(stderr,
            "[SLICEDBG] starts.c=%d ends.c=%d aligned=%d | "
            "src c=%d storage=%d dtype=%d layout=%d -> "
            "dst c=%d storage=%d dtype=%d\n",
            attr.starts.c, attr.ends.c, Is4Aligned(attr) ? 1 : 0,
            s.GetBHWDCShape().c, static_cast<int>(s.GetStorageType()),
            static_cast<int>(s.GetDataType()), static_cast<int>(s.GetLayout()),
            d.GetBHWDCShape().c, static_cast<int>(d.GetStorageType()),
            static_cast<int>(d.GetDataType()));
  }
}
```

`TensorStorageType` enum values (`tensor_desc.h`):

- `0 = UNKNOWN`
- `1 = BUFFER`
- `2 = IMAGE_BUFFER`
- `3 = TEXTURE_2D`
- `4 = TEXTURE_3D`
- `5 = TEXTURE_ARRAY`
- `6 = SINGLE_TEXTURE_2D`
