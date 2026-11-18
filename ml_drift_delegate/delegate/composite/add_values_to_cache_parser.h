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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_ADD_VALUES_TO_CACHE_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_ADD_VALUES_TO_CACHE_PARSER_H_

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

constexpr const char kAddValuesToCacheType[] = "add_values_to_cache";

struct AddValuesToCacheAttributes {
  int kv_cache_batch_size;
  int cache_size;
  int head_size;
  // Indices into the `params` buffer for the write offset and active-tokens
  // bound.
  int token_index_offset_param_idx = 0;
  int active_tokens_param_idx = 1;
  // Element type of quantized K/V caches, or kUnknown for float caches:
  // - kInt8: symmetric per-tensor INT8, stored as unsigned bytes (offset +128).
  // - kUint8: per-tensor UINT8 with zero point 128 (the same bytes as kInt8).
  // - kInt4: symmetric per-tensor INT4, packed two nibbles per byte in a
  //   half-size kUint8 buffer (offset +8, low nibble = head slice 2*S, high
  //   nibble = head slice 2*S + 1).
  ::ml_drift::DataType kv_cache_dtype = ::ml_drift::DataType::kUnknown;
  // quantized kv cache case
  std::optional<float> scale_k;
  std::optional<float> scale_v;
  // Local Attention Ring buffer case
  std::optional<bool> is_ring_buffer;
  // Index of the time axis of each 4D tensor ([B, H, *, *], so 2 or 3). The
  // defaults are the layouts the op had before these attributes existed:
  // K cache [B, H, S, D], V cache [B, H, D, S], and both updates [B, H, T, D].
  int k_cache_ts_idx = 2;
  int v_cache_ts_idx = 3;
  int k_update_ts_idx = 2;
  int v_update_ts_idx = 2;
};

// Reads the optional `*_ts_idx` layout attributes of `odml.cache_update` into
// `attr` (absent attributes keep their defaults), and returns an Unavailable
// error for layouts the kernel does not implement so that the node is left to
// the CPU decomposition instead of being computed with the wrong layout.
absl::Status ReadAddValuesToCacheLayout(const flexbuffers::Map& attributes,
                                        AddValuesToCacheAttributes& attr);

// Validates the K/V cache output tensor types, quantization parameters, and
// alignment constraints for `odml.cache_update`.
absl::Status CheckAddValuesToCacheOutputs(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const AddValuesToCacheAttributes& attr);

// Populates `kv_cache_dtype` and falls back to `TfLiteAffineQuantization`
// scales on `out_k` / `out_v` when `scale_k` / `scale_v` were not set in
// FlexBuffers.
void PopulateAddValuesToCacheQuantAttributes(const TfLiteTensor& out_k,
                                             const TfLiteTensor& out_v,
                                             AddValuesToCacheAttributes& attr);

// The producer and the number of consumers of a tensor in the TFLite graph.
struct TfLiteTensorUsage {
  // Node that writes the tensor, or nullptr for graph inputs and constants.
  const TfLiteNode* producer = nullptr;
  const TfLiteRegistration* producer_registration = nullptr;
  // Number of nodes that read the tensor.
  int num_consumers = 0;
};

// Scans the execution plan of `context` for the producer and the consumers of
// tensor `tensor_idx`.
absl::StatusOr<TfLiteTensorUsage> GetTfLiteTensorUsage(
    const TfLiteContext* context, int tensor_idx);

// Returns true if `node` is a `STABLEHLO_COMPOSITE` named `name`.
bool IsStablehloComposite(const TfLiteNode* node,
                          const TfLiteRegistration* registration,
                          absl::string_view name);

// Halves the `head_size` axis of a 4D INT4 KV cache BHWC shape so that a
// `DataType::kUint8` `kBuffer` descriptor allocates 0.5 bytes per logical
// element (matching `TfLiteTensor::bytes`):
// - `cache_ts_idx == 2` (`[B, K, S, H]`): returns `[B, K, S, H / 2]`.
// - `cache_ts_idx == 3` (`[B, K, H, S]`): returns `[B, K, H / 2, S]`.
::ml_drift::BHWC PackInt4KvCacheShape(const ::ml_drift::BHWC& logical_shape,
                                      int cache_ts_idx);

class AddValuesToCacheOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_ADD_VALUES_TO_CACHE_PARSER_H_
