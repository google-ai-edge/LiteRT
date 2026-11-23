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
  // Element type of quantized K/V caches written by `odml.cache_update`: kInt8
  // or packed kInt4. kUnknown for float caches.
  ::ml_drift::DataType kv_cache_dtype = ::ml_drift::DataType::kUnknown;
  // Dequantization scale of a symmetric per-tensor INT8 query (`in[0]`) when
  // the query arrives at the GPU graph as raw `DataType::kInt8` (e.g., an
  // external buffer input); absent when the query is float or already
  // dequantized by ML Drift's tensor reader.
  std::optional<float> scale_q;
  // Dequantization scales of the symmetric per-tensor quantized K and V caches.
  // Set, together with kUint8 `bmm1_weights` / `bmm2_weights` descriptors, a
  // quantized `kv_cache_dtype`, and `from_cache_update`, only when K and V are
  // quantized; absent for float caches.
  std::optional<float> scale_k;
  std::optional<float> scale_v;
};

// Returns true if `tensor` is an INT8 tensor with affine quantization.
bool IsInt8Tensor(const TfLiteTensor& tensor);

// Returns true if `tensor` is an INT4 tensor with affine quantization.
bool IsInt4Tensor(const TfLiteTensor& tensor);

// Returns the per-tensor scale of an affine quantized `tensor`.
float GetAffineTensorScale(const TfLiteTensor& tensor);

// Checks the query (input 0) and output (output 0) types of an
// `odml.sdpa_transposed` node: query must be float or symmetric per-tensor
// INT8, and output must be float. An INT8 query must be read by this node only,
// since a query entering the GPU partition directly is retagged to raw INT8 in
// place for all of its GPU consumers.
absl::Status CheckSdpaTransposedQueryAndOutputTypes(
    const TfLiteContext* context, const TfLiteNode* tflite_node);

// Checks the K (input 1) and V (input 2) caches of an `odml.sdpa_transposed`
// node: both must have matching types and be float, symmetric per-tensor INT8,
// or symmetric per-tensor INT4 (4D with head_dim % 8 == 0 and
// cache_size % 4 == 0). Quantized caches must have batch size 1, matching K and
// V heads, and be graph inputs or `odml.cache_update` outputs, i.e. hold the
// packed layout written by `odml.cache_update`. Returns an Unavailable error
// otherwise so that the node is left to the CPU decomposition.
absl::Status CheckSdpaTransposedKvCacheTypes(const TfLiteContext* context,
                                             const TfLiteNode* tflite_node);

// Fills the quantized KV cache fields of `attr` for INT8 or INT4 `k_tensor` /
// `v_tensor` (`kv_cache_dtype` kInt8 or kInt4): `kv_cache_dtype`, kUint8
// external-weights descriptors of both matmuls (over the logical unpacked cache
// shapes), and `scale_k` / `scale_v`. `k_shape` / `v_shape` are the BHWC shapes
// in the graph after merging the model batch (for kInt4, their head_dim axis is
// already packed by half via `PackInt4KvCacheShape`).
void SetSdpaTransposedQuantizedKvCacheWeights(
    const TfLiteTensor& k_tensor, const TfLiteTensor& v_tensor,
    const ::ml_drift::BHWC& k_shape, const ::ml_drift::BHWC& v_shape,
    ::ml_drift::DataType kv_cache_dtype,
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
