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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ENCODED_WEIGHTS_STORE_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ENCODED_WEIGHTS_STORE_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/compiler/cc/litert_model.h"

namespace litert::openvino {

enum class WeightEncoding {
  // XOR flips each 2-bit element's MSB, which is equivalent to adding 2 mod 4
  // ([-2..1] -> [0..3]). The byte layout (4 elements per byte) is unchanged, so
  // the zero points must be shifted by kI2ToU2ZeroPointOffset to keep the
  // dequantized values the same.
  kI2ToU2,
};

inline constexpr int64_t kI2ToU2ZeroPointOffset = 2;

// Holds converted copies of weight buffers, keyed by buffer id and encoding.
// The graph iterator feeds OpenVINO the converted bytes and the weight bank
// pools the very same ones, so each buffer is converted once and the shared
// pool always matches the compiled constants.
class EncodedWeightStore {
 public:
  EncodedWeightStore() = default;
  EncodedWeightStore(const EncodedWeightStore&) = delete;
  EncodedWeightStore& operator=(const EncodedWeightStore&) = delete;

  static std::optional<WeightEncoding> EncodingFor(
      const litert::compiler::Tensor& tensor);

  static std::vector<uint8_t> Encode(WeightEncoding encoding,
                                     absl::Span<const uint8_t> bytes);

  std::optional<absl::Span<const uint8_t>> GetOrEncode(
      const litert::compiler::Tensor& tensor, WeightEncoding encoding);

  size_t Size() const { return index_.size(); }

 private:
  std::deque<std::vector<uint8_t>> buffers_;
  std::map<std::pair<int32_t, WeightEncoding>, absl::Span<const uint8_t>>
      index_;
};

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ENCODED_WEIGHTS_STORE_H_
