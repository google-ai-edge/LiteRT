// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_TRANSPOSED_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_TRANSPOSED_PARSER_H_

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

constexpr const char kSdpaTransposedType[] = "sdpa_transposed";

struct SdpaTransposedAttributes {
  RuntimeCheckParams runtime_check;
  std::optional<float> softcap;
  ExternalWeightsAttributes bmm1_weights;
  ExternalWeightsAttributes bmm2_weights;
  bool is_prefill = false;
  // True if key and value caches were produced by `odml.cache_update`
  // (AddValuesToCache) with packed 4D GPU layouts.
  bool from_cache_update = false;
  // True if causal masking (key <= q_start + query_idx) should be enforced
  // implicitly in registers (e.g. when a causal BOOL mask is pruned for Flash
  // SDPA or `is_causal` attribute is set). When false and `mask` is absent,
  // attention is full (non-causal) over `[0, active_tokens)`.
  bool is_causal = false;
  // Dequantization scales of INT8 symmetric per-tensor quantized K and V
  // caches, broadcast to one entry per output channel of the matmul that reads
  // the cache (`scale_k`: one per cache slot, `scale_v`: one per head_dim
  // element). Set, together with kUint8 `bmm1_weights` / `bmm2_weights`
  // descriptors and `from_cache_update`, only when K and V are int8; absent
  // for float caches. The cache bytes must be in the packed layout and
  // encoding written by the GPU `AddValuesToCache` kernel, i.e. `int8 + 128`
  // (offset binary), which `FullyConnectedInt8ExternalWeights` decodes as
  // `(byte - 128) * scale` when no zero-point tensor is given.
  std::optional<
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::kFloat32>>
      scale_k;
  std::optional<
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::kFloat32>>
      scale_v;
};

// Returns true if `tensor` is an INT8 tensor with affine quantization.
bool IsInt8Tensor(const TfLiteTensor& tensor);

// Checks the K (input 1) and V (input 2) caches of an `odml.sdpa_transposed`
// node: either both are float, or both are INT8 with symmetric per-tensor
// quantization. Returns an Unavailable error otherwise so that the node is left
// to the CPU decomposition.
absl::Status CheckSdpaTransposedKvCacheTypes(const TfLiteContext* context,
                                             const TfLiteNode* tflite_node);

// Fills the INT8 KV cache fields of `attr` for int8 `k_tensor` / `v_tensor`:
// kUint8 external-weights descriptors of both matmuls and the broadcast
// `scale_k` / `scale_v` tensors. Callers must also set `from_cache_update`, as
// the int8 cache bytes are only meaningful in the layout written by
// `odml.cache_update`. `k_shape` / `v_shape` are the BHWC shapes the kernel
// consumes (after merging the model batch). The scales are taken from
// `cache_update_attr` when the caches are produced by an `odml.cache_update`
// node of the same graph (these are the scales the bytes were quantized with)
// and from the tensors' quantization parameters otherwise.
void SetSdpaTransposedInt8KvCacheWeights(
    const TfLiteTensor& k_tensor, const TfLiteTensor& v_tensor,
    const ::ml_drift::BHWC& k_shape, const ::ml_drift::BHWC& v_shape,
    const AddValuesToCacheAttributes* cache_update_attr,
    SdpaTransposedAttributes& attr);

class SdpaTransposedOperationParser : public TFLiteOperationParser {
 public:
  explicit SdpaTransposedOperationParser(
      const ::ml_drift::GpuInfo* gpu_info = nullptr)
      : gpu_info_(gpu_info) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;

 private:
  const ::ml_drift::GpuInfo* const gpu_info_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_TRANSPOSED_PARSER_H_
