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

#include "litert/vendors/intel_openvino/compiler/encoded_weights_store.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/compiler/cc/litert_model.h"

namespace litert::openvino {

namespace {
constexpr uint8_t kI2ToU2XorMask = 0xAA;
}  // namespace

std::optional<WeightEncoding> EncodedWeightStore::EncodingFor(
    const litert::compiler::Tensor& tensor) {
  if (static_cast<LiteRtElementType>(tensor.ElementType()) ==
          kLiteRtElementTypeInt2 &&
      tensor.QTypeId() != kLiteRtQuantizationNone) {
    return WeightEncoding::kI2ToU2;
  }
  return std::nullopt;
}

std::vector<uint8_t> EncodedWeightStore::Encode(
    WeightEncoding encoding, absl::Span<const uint8_t> bytes) {
  std::vector<uint8_t> out(bytes.begin(), bytes.end());
  switch (encoding) {
    case WeightEncoding::kI2ToU2:
      for (auto& byte : out) byte ^= kI2ToU2XorMask;
      break;
  }
  return out;
}

std::optional<absl::Span<const uint8_t>> EncodedWeightStore::GetOrEncode(
    const litert::compiler::Tensor& tensor, WeightEncoding encoding) {
  const int32_t buffer_id = tensor.Weights().BufferId();
  if (buffer_id < 0) return std::nullopt;
  const auto key = std::make_pair(buffer_id, encoding);
  if (auto it = index_.find(key); it != index_.end()) return it->second;
  auto& buf = buffers_.emplace_back(Encode(encoding, tensor.Weights().Bytes()));
  return index_.emplace(key, absl::MakeConstSpan(buf)).first->second;
}

}  // namespace litert::openvino
