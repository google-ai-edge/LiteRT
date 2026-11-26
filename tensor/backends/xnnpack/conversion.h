/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/conversion.h"
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/type_id.h"
#include "tensor/tensor.h"

namespace litert::tensor {

class XnnpackBuildContext;

// Base class for XNNPACK operations.
class XnnpackOperation : public graph::BackendExtension {
 public:
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<XnnpackOperation>();
  }
  // Converts the operation to XNNPACK.
  virtual absl::Status ToXnnpack(const graph::Operation& op,
                                 XnnpackBuildContext& ctx) const = 0;
};

class XnnpackBuildContext : public NnpackBuildContext {
 public:
  using NnpackBuildContext::NnpackBuildContext;

  absl::string_view BackendName() const override { return "XNNPACK"; }
  uint32_t FlagExternalInput() const override {
    return XNN_VALUE_FLAG_EXTERNAL_INPUT;
  }
  uint32_t FlagExternalOutput() const override {
    return XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  xnn_subgraph_t subgraph() {
    return static_cast<XnnpackGraph&>(*graph_).GetSubgraph();
  }

  // Defines the constant int8 `weights` of a fully connected op, quantized per
  // tensor, as weights quantized per output channel (`xnn_datatype_qcint8`)
  // with the same scale and zero point for each channel: XNNPACK's fully
  // connected operator only takes channelwise quantized weights with a float
  // input. The value isn't shared with the other consumers of `weights`, which
  // keep the per-tensor quantized value.
  absl::StatusOr<uint32_t> DefinePerTensorQuantizedWeightsAsChannelwise(
      const graph::Tensor& weights);

  void SetWarmFcBuffers(absl::flat_hash_set<const Buffer*> warm_fc_buffers,
                        size_t max_warm_rows, size_t max_warm_blocks) {
    warm_fc_buffers_ = std::move(warm_fc_buffers);
    warm_max_rows_ = max_warm_rows;
    warm_max_blocks_ = max_warm_blocks;
  }

 protected:
  absl::Status EnsureInitialized() override;
  std::unique_ptr<NnpackGraph> CreateEmptyGraph() override;
  absl::Status CreateSubgraph(size_t external_value_ids,
                              uint32_t flags) override;
  absl::Status DefineTensorValue(const graph::Tensor& tensor,
                                 NnpackValue& value) override;
  absl::Status DefineConstantTensor(Type datatype,
                                    absl::Span<const size_t> shape,
                                    const void* data, uint32_t* id) override;
  absl::Status LowerOp(const graph::Operation& op) override;

 private:
  absl::flat_hash_set<const Buffer*> warm_fc_buffers_;
  size_t warm_max_rows_ = 0;
  size_t warm_max_blocks_ = 0;
  const float* warm_dummy_f32_scales_ = nullptr;
  const void* warm_dummy_bf16_scales_ = nullptr;
};

// Builds an XNNPACK graph from the given outputs.
absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
    std::vector<TensorHandle> outputs);

// Builds an XNNPACK graph assuming a warm `xnn_weights_cache_t`: constant 2D
// `FullyConnected` weight tensors with FP32/FP16 activations are defined with
// synthetic non-null pointers and dummy positive scales without locking or
// materializing lazy buffers.
absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraphWithWarmFcCache(
    std::vector<TensorHandle> outputs);

// Callback invoked when an `XnnpackGraph` is prepared against an
// `xnn_weights_cache_t`, forwarding constant weight/bias fingerprints to the
// cache implementation (`cache_context == cache->context`).
using XnnConstantFingerprintSink = void (*)(
    void* cache_context, absl::Span<const XnnConstantFingerprint> fingerprints);

// Associates `sink` with `cache` so `NotifyXnnWeightsCacheConstantFingerprints`
// forwards constant tensor fingerprints to `sink(cache->context, ...)`.
// `cache` is used as a map key and must call
// `UnregisterXnnWeightsCacheFingerprintSink(cache)` before destruction; `sink`
// is a plain function pointer. Thread-safe (guarded by an internal mutex).
void RegisterXnnWeightsCacheFingerprintSink(xnn_weights_cache_t cache,
                                            XnnConstantFingerprintSink sink);

// Removes any fingerprint sink registered for `cache`. Safe to call with a null
// or unregistered `cache`. Thread-safe (guarded by an internal mutex).
void UnregisterXnnWeightsCacheFingerprintSink(xnn_weights_cache_t cache);

// Dispatches `fingerprints` to the sink registered for `cache` (if non-null and
// non-empty), releasing the registry mutex before invoking `sink`. Thread-safe.
void NotifyXnnWeightsCacheConstantFingerprints(
    xnn_weights_cache_t cache,
    absl::Span<const XnnConstantFingerprint> fingerprints);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_
