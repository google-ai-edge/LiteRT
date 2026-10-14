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

#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr int kQuery = 0;
constexpr int kKeyCache = 1;
constexpr int kValueCache = 2;
constexpr int kKeyNew = 3;
constexpr int kValueNew = 4;
constexpr int kMask = 5;
constexpr int kParam = 6;
constexpr int kNumInputs = 7;

bool IsFloatType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat16;
}

// Returns the 4D dims of `tensor`, or an error if it is not rank 4.
absl::Status Get4dDims(const TfLiteTensor& tensor, const char* name,
                       int32_t dims[4]) {
  if (tensor.dims == nullptr || tensor.dims->size != 4) {
    return absl::UnimplementedError(
        absl::StrCat("odml.fused_sdpa_cache_update expects a 4D ", name, "."));
  }
  for (int i = 0; i < 4; ++i) dims[i] = tensor.dims->data[i];
  return absl::OkStatus();
}

int64_t NumElements(const TfLiteTensor& tensor) {
  int64_t count = 1;
  for (int i = 0; tensor.dims != nullptr && i < tensor.dims->size; ++i) {
    count *= tensor.dims->data[i];
  }
  return count;
}

}  // namespace

absl::Status FusedSdpaCacheUpdateOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (tflite_node->inputs->size != kNumInputs ||
      GetNumberOfRuntimeInputsForNode(context, tflite_node) != kNumInputs) {
    return absl::UnavailableError(
        "odml.fused_sdpa_cache_update expects 7 runtime inputs.");
  }
  if (tflite_node->outputs->size != 1 && tflite_node->outputs->size != 3) {
    return absl::UnavailableError(
        "odml.fused_sdpa_cache_update expects 1 or 3 outputs.");
  }
  for (int i = 0; i < kNumInputs; ++i) {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

  auto input = [&](int i) -> const TfLiteTensor& {
    return context->tensors[tflite_node->inputs->data[i]];
  };
  auto output = [&](int i) -> const TfLiteTensor& {
    return context->tensors[tflite_node->outputs->data[i]];
  };

  int32_t q[4], kc[4], vc[4], kn[4], vn[4], mask[4], out[4];
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kQuery), "query", q));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kKeyCache), "key cache", kc));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kValueCache), "value cache", vc));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kKeyNew), "new key", kn));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kValueNew), "new value", vn));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kMask), "mask", mask));
  ABSL_RETURN_IF_ERROR(Get4dDims(output(0), "output", out));

  for (int i : {kQuery, kKeyCache, kValueCache, kKeyNew, kValueNew}) {
    if (!IsFloatType(input(i).type)) {
      return absl::UnimplementedError(
          "odml.fused_sdpa_cache_update supports only float Q/K/V tensors.");
    }
  }

  const int heads = kc[1];
  const int cache_size = kc[2];
  const int head_dim = kc[3];
  const int new_len = kn[2];
  if (q[0] != 1 || kc[0] != 1 || vc[0] != 1 || kn[0] != 1 || vn[0] != 1) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update supports only batch size 1.");
  }
  if (heads <= 0 || cache_size <= 0 || head_dim <= 0 || new_len <= 0) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update has an empty dimension.");
  }
  if (q[1] != heads || q[3] != head_dim || q[2] % new_len != 0 ||
      vc[1] != heads || vc[2] != head_dim || vc[3] != cache_size ||
      kn[1] != heads || kn[3] != head_dim || vn[1] != heads ||
      vn[2] != head_dim || vn[3] != new_len) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update has inconsistent Q/K/V shapes.");
  }
  for (int i = 0; i < 4; ++i) {
    if (out[i] != q[i]) {
      return absl::InvalidArgumentError(
          "odml.fused_sdpa_cache_update output must have the query shape.");
    }
  }
  // The packed KV cache layouts group the cache and head dims by 4.
  if (cache_size % 4 != 0 || head_dim % 4 != 0) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update requires cache size and head dim to be "
        "multiples of 4.");
  }
  if (input(kMask).type != kTfLiteBool && !IsFloatType(input(kMask).type)) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update supports only bool or float masks.");
  }
  if (mask[0] != 1 || mask[1] != 1 || mask[2] != new_len ||
      mask[3] != cache_size + new_len) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects a [1, 1, T, W + T] mask.");
  }
  if (input(kParam).type != kTfLiteInt32 || NumElements(input(kParam)) < 2) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update expects an int32 param tensor with at "
        "least 2 elements.");
  }
  if (tflite_node->outputs->size == 3) {
    for (int i : {1, 2}) {
      const TfLiteTensor& cache_out = output(i);
      const TfLiteTensor& cache_in = input(i == 1 ? kKeyCache : kValueCache);
      if (cache_out.type != cache_in.type || cache_out.dims == nullptr ||
          cache_out.dims->size != 4) {
        return absl::InvalidArgumentError(
            "odml.fused_sdpa_cache_update cache outputs must match inputs.");
      }
      for (int d = 0; d < 4; ++d) {
        if (cache_out.dims->data[d] != cache_in.dims->data[d]) {
          return absl::InvalidArgumentError(
              "odml.fused_sdpa_cache_update cache outputs must match inputs.");
        }
      }
    }
  }
  return absl::OkStatus();
}

void FusedSdpaCacheUpdateOperationParser::Parse(
    const TfLiteNode* tflite_node, const TfLiteRegistration*,
    ::ml_drift::GraphFloat32* graph, ObjectReader* reader) {
  ::ml_drift::Node* node = graph->NewNode();
  node->operation.type = kFusedSdpaCacheUpdateType;
  for (int i = 0; i < kNumInputs; ++i) {
    reader->AddInput(node, i);
  }
  reader->AddOutputs(node);

  FusedSdpaCacheUpdateAttributes attr;
  attr.update_cache = tflite_node->outputs->size == 3;
  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  if (params != nullptr && params->attributes != nullptr) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (!flexbuffer_map["softcap"].IsNull()) {
      attr.softcap = flexbuffer_map["softcap"].AsFloat();
    }
  }
  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
