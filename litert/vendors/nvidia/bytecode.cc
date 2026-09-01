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

#include "litert/vendors/nvidia/bytecode.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "tsl/platform/fingerprint.h"

namespace litert::nvidia {
namespace {

constexpr uint32_t kMagic = 0x4e52544c;  // "LTRN", little endian.
constexpr uint32_t kAotLocatorMagic =
    0x414e524c;  // "LRNA" (LiteRT NVIDIA AOT), little endian.
constexpr uint32_t kAotManifestMagic =
    0x494e524c;  // "LRNI" (LiteRT NVIDIA index), little endian.

template <typename T>
void AppendScalar(std::vector<uint8_t>& out, T value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

Expected<void> AppendString(std::vector<uint8_t>& out,
                            const std::string& value) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    return Error(kLiteRtStatusErrorInvalidArgument, "String is too large");
  }
  AppendScalar<uint32_t>(out, static_cast<uint32_t>(value.size()));
  out.insert(out.end(), value.begin(), value.end());
  return {};
}

template <typename T>
Expected<T> ReadScalar(const uint8_t*& cur, const uint8_t* end) {
  if (static_cast<size_t>(end - cur) < sizeof(T)) {
    return Error(kLiteRtStatusErrorInvalidArgument, "Truncated bytecode");
  }
  T value;
  std::memcpy(&value, cur, sizeof(T));
  cur += sizeof(T);
  return value;
}

Expected<std::string> ReadString(const uint8_t*& cur, const uint8_t* end) {
  LITERT_ASSIGN_OR_RETURN(uint32_t size, ReadScalar<uint32_t>(cur, end));
  if (static_cast<size_t>(end - cur) < size) {
    return Error(kLiteRtStatusErrorInvalidArgument, "Truncated string");
  }
  std::string value(reinterpret_cast<const char*>(cur), size);
  cur += size;
  return value;
}

Expected<void> AppendStringVector(std::vector<uint8_t>& out,
                                  const std::vector<std::string>& values) {
  if (values.size() > std::numeric_limits<uint32_t>::max()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Too many TensorRT tensor names");
  }
  AppendScalar<uint32_t>(out, static_cast<uint32_t>(values.size()));
  for (const auto& value : values) {
    LITERT_RETURN_IF_ERROR(AppendString(out, value));
  }
  return {};
}

Expected<std::vector<std::string>> ReadStringVector(const uint8_t*& cur,
                                                    const uint8_t* end) {
  LITERT_ASSIGN_OR_RETURN(uint32_t count, ReadScalar<uint32_t>(cur, end));
  // Every encoded string needs at least its uint32 length field. Bound the
  // allocation before reserve() so malformed bytecode cannot request an
  // attacker-controlled amount of host memory.
  if (count > static_cast<size_t>(end - cur) / sizeof(uint32_t)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT bytecode string count exceeds payload");
  }
  std::vector<std::string> values;
  values.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto value, ReadString(cur, end));
    values.push_back(std::move(value));
  }
  return values;
}

Expected<void> ValidateTrtLlmHead(
    const TensorRtLlmHead& head, const std::vector<std::string>& output_names) {
  if (head.hidden_output_port >= output_names.size() ||
      head.logits_output_port >= output_names.size() ||
      head.hidden_output_port == head.logits_output_port) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT-LLM head output ports");
  }
  if (output_names[head.hidden_output_port].empty() ||
      !output_names[head.logits_output_port].empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT-LLM head output bindings");
  }
  if (head.k == 0 || head.n == 0 || !std::isfinite(head.soft_cap) ||
      head.soft_cap <= 0.0f) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT-LLM head dimensions or soft cap");
  }
  const uint64_t num_elements =
      static_cast<uint64_t>(head.k) * static_cast<uint64_t>(head.n);
  uint64_t expected_weight_bytes = 0;
  switch (head.weight_format) {
    case TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved:
      if (head.k % 64 != 0 || head.n % 64 != 0) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Invalid TensorRT-LLM W4 head dimensions");
      }
      expected_weight_bytes = (num_elements + 1) / 2;
      break;
    case TensorRtLlmHeadWeightFormat::kInt2TfliteRowMajor:
      if (head.k % 16 != 0) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Invalid TensorRT-LLM W2 head dimensions");
      }
      expected_weight_bytes = (num_elements + 3) / 4;
      break;
    default:
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Invalid TensorRT-LLM head weight format");
  }
  const uint64_t expected_scale_bytes =
      static_cast<uint64_t>(head.n) * sizeof(uint16_t);
  if (expected_weight_bytes > std::numeric_limits<size_t>::max() ||
      expected_scale_bytes > std::numeric_limits<size_t>::max() ||
      head.packed_weights == nullptr ||
      head.packed_weights_size != static_cast<size_t>(expected_weight_bytes) ||
      head.bf16_scales == nullptr ||
      head.bf16_scales_size != static_cast<size_t>(expected_scale_bytes)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT-LLM head weight or scale payload");
  }
  return {};
}

Expected<size_t> TensorRtWeightBytes(TensorRtWeightDataType data_type,
                                     uint64_t count) {
  if (count == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT refit weight is empty");
  }
  uint64_t bytes = 0;
  switch (data_type) {
    case TensorRtWeightDataType::kFloat:
    case TensorRtWeightDataType::kInt32:
      if (count > std::numeric_limits<uint64_t>::max() / 4) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT refit weight size overflows");
      }
      bytes = count * 4;
      break;
    case TensorRtWeightDataType::kHalf:
    case TensorRtWeightDataType::kBf16:
      if (count > std::numeric_limits<uint64_t>::max() / 2) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT refit weight size overflows");
      }
      bytes = count * 2;
      break;
    case TensorRtWeightDataType::kInt64:
      if (count > std::numeric_limits<uint64_t>::max() / 8) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT refit weight size overflows");
      }
      bytes = count * 8;
      break;
    case TensorRtWeightDataType::kInt8:
    case TensorRtWeightDataType::kBool:
    case TensorRtWeightDataType::kUint8:
    case TensorRtWeightDataType::kFp8:
    case TensorRtWeightDataType::kE8m0:
      bytes = count;
      break;
    case TensorRtWeightDataType::kInt4:
    case TensorRtWeightDataType::kFp4:
      // Avoid overflowing when rounding an odd uint64_t count upward.
      bytes = count / 2 + count % 2;
      break;
    default:
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unknown TensorRT refit weight type");
  }
  if (bytes > std::numeric_limits<size_t>::max()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT refit weight is too large");
  }
  return static_cast<size_t>(bytes);
}

void AppendTypedHead(std::vector<uint8_t>& out, const TensorRtLlmHead& head) {
  AppendScalar<uint32_t>(out, head.hidden_output_port);
  AppendScalar<uint32_t>(out, head.logits_output_port);
  AppendScalar<uint32_t>(out, head.k);
  AppendScalar<uint32_t>(out, head.n);
  AppendScalar<float>(out, head.soft_cap);
  AppendScalar<uint32_t>(out, static_cast<uint32_t>(head.weight_format));
  AppendScalar<uint64_t>(out, static_cast<uint64_t>(head.packed_weights_size));
  out.insert(out.end(), head.packed_weights,
             head.packed_weights + head.packed_weights_size);
  AppendScalar<uint64_t>(out, static_cast<uint64_t>(head.bf16_scales_size));
  out.insert(out.end(), head.bf16_scales,
             head.bf16_scales + head.bf16_scales_size);
}

Expected<TensorRtLlmHead> ReadTypedHead(
    const uint8_t*& cur, const uint8_t* end,
    const std::vector<std::string>& output_names) {
  TensorRtLlmHead head;
  LITERT_ASSIGN_OR_RETURN(head.hidden_output_port,
                          ReadScalar<uint32_t>(cur, end));
  LITERT_ASSIGN_OR_RETURN(head.logits_output_port,
                          ReadScalar<uint32_t>(cur, end));
  LITERT_ASSIGN_OR_RETURN(head.k, ReadScalar<uint32_t>(cur, end));
  LITERT_ASSIGN_OR_RETURN(head.n, ReadScalar<uint32_t>(cur, end));
  LITERT_ASSIGN_OR_RETURN(head.soft_cap, ReadScalar<float>(cur, end));
  LITERT_ASSIGN_OR_RETURN(uint32_t weight_format,
                          ReadScalar<uint32_t>(cur, end));
  head.weight_format = static_cast<TensorRtLlmHeadWeightFormat>(weight_format);
  LITERT_ASSIGN_OR_RETURN(uint64_t weight_size, ReadScalar<uint64_t>(cur, end));
  if (weight_size > static_cast<uint64_t>(end - cur)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Truncated TensorRT-LLM head weights");
  }
  head.packed_weights = cur;
  head.packed_weights_size = static_cast<size_t>(weight_size);
  cur += head.packed_weights_size;
  LITERT_ASSIGN_OR_RETURN(uint64_t scale_size, ReadScalar<uint64_t>(cur, end));
  if (scale_size > static_cast<uint64_t>(end - cur)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Truncated TensorRT-LLM head scales");
  }
  head.bf16_scales = cur;
  head.bf16_scales_size = static_cast<size_t>(scale_size);
  cur += head.bf16_scales_size;
  LITERT_RETURN_IF_ERROR(ValidateTrtLlmHead(head, output_names));
  return head;
}

Expected<TensorRtBytecode> ParseSharedWeightBundle(const uint8_t*& cur,
                                                   const uint8_t* end,
                                                   const char* function_name) {
  struct SharedWeightView {
    TensorRtWeightDataType data_type;
    uint64_t count;
    const uint8_t* data;
    size_t size;
  };

  LITERT_ASSIGN_OR_RETURN(uint32_t shared_count,
                          ReadScalar<uint32_t>(cur, end));
  if (shared_count > static_cast<size_t>(end - cur) /
                         (sizeof(int32_t) + sizeof(uint64_t) * 2)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT shared weight count exceeds payload");
  }
  std::vector<SharedWeightView> shared_weights;
  shared_weights.reserve(shared_count);
  for (uint32_t i = 0; i < shared_count; ++i) {
    LITERT_ASSIGN_OR_RETURN(int32_t data_type_value,
                            ReadScalar<int32_t>(cur, end));
    const auto data_type = static_cast<TensorRtWeightDataType>(data_type_value);
    LITERT_ASSIGN_OR_RETURN(uint64_t count, ReadScalar<uint64_t>(cur, end));
    LITERT_ASSIGN_OR_RETURN(uint64_t size_u64, ReadScalar<uint64_t>(cur, end));
    LITERT_ASSIGN_OR_RETURN(size_t expected_size,
                            TensorRtWeightBytes(data_type, count));
    if (size_u64 != expected_size ||
        size_u64 > static_cast<uint64_t>(end - cur)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Invalid TensorRT shared weight payload size");
    }
    shared_weights.push_back(
        {data_type, count, cur, static_cast<size_t>(size_u64)});
    cur += static_cast<size_t>(size_u64);
  }

  LITERT_ASSIGN_OR_RETURN(uint32_t entry_count, ReadScalar<uint32_t>(cur, end));
  if (entry_count == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT shared weight bundle has no engines");
  }
  const bool select_only_entry = function_name == nullptr && entry_count == 1;
  if (function_name == nullptr && !select_only_entry) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT bundle engine name is required");
  }

  TensorRtBytecode selected;
  bool found = false;
  std::unordered_set<std::string> function_names;
  for (uint32_t i = 0; i < entry_count; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto entry_function_name, ReadString(cur, end));
    if (!function_names.insert(entry_function_name).second) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Duplicate TensorRT bundle engine name");
    }
    LITERT_ASSIGN_OR_RETURN(auto input_names, ReadStringVector(cur, end));
    LITERT_ASSIGN_OR_RETURN(auto output_names, ReadStringVector(cur, end));
    LITERT_ASSIGN_OR_RETURN(uint64_t engine_size_u64,
                            ReadScalar<uint64_t>(cur, end));
    if (engine_size_u64 == 0 ||
        engine_size_u64 > static_cast<uint64_t>(end - cur)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Invalid TensorRT bundle engine payload");
    }
    const uint8_t* engine_data = cur;
    const size_t engine_size = static_cast<size_t>(engine_size_u64);
    cur += engine_size;

    LITERT_ASSIGN_OR_RETURN(uint32_t has_head, ReadScalar<uint32_t>(cur, end));
    if (has_head > 1) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Invalid TensorRT bundle head marker");
    }
    std::optional<TensorRtLlmHead> head;
    if (has_head != 0) {
      LITERT_ASSIGN_OR_RETURN(head, ReadTypedHead(cur, end, output_names));
    }

    LITERT_ASSIGN_OR_RETURN(uint32_t refit_count,
                            ReadScalar<uint32_t>(cur, end));
    if (refit_count > static_cast<size_t>(end - cur) / (sizeof(uint32_t) * 2)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "TensorRT refit weight count exceeds payload");
    }
    const bool is_selected =
        select_only_entry || entry_function_name == function_name;
    std::vector<TensorRtRefitWeight> refit_weights;
    if (is_selected) {
      refit_weights.reserve(refit_count);
    }
    std::unordered_set<std::string> refit_names;
    for (uint32_t j = 0; j < refit_count; ++j) {
      LITERT_ASSIGN_OR_RETURN(auto refit_name, ReadString(cur, end));
      LITERT_ASSIGN_OR_RETURN(uint32_t shared_index,
                              ReadScalar<uint32_t>(cur, end));
      if (shared_index >= shared_weights.size()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT refit weight index is out of range");
      }
      if (is_selected) {
        if (!refit_names.insert(refit_name).second) {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "Duplicate TensorRT refit weight name");
        }
        const auto& shared = shared_weights[shared_index];
        refit_weights.push_back({std::move(refit_name), shared.data_type,
                                 shared.count, shared.data, shared.size});
      }
    }

    if (is_selected) {
      if (found) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT bundle engine selection is ambiguous");
      }
      found = true;
      selected.version = kTensorRtBytecodeVersionWithSharedWeights;
      selected.function_name = std::move(entry_function_name);
      selected.input_names = std::move(input_names);
      selected.output_names = std::move(output_names);
      selected.engine_data = engine_data;
      selected.engine_size = engine_size;
      selected.trtllm_head = std::move(head);
      selected.refit_weights = std::move(refit_weights);
    }
  }
  if (cur != end) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Unexpected trailing TensorRT bundle data");
  }
  if (!found) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT bundle engine was not found");
  }
  return selected;
}

}  // namespace

Expected<std::vector<uint8_t>> PackTensorRtBytecode(
    const std::string& function_name, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const void* engine_data,
    size_t engine_size, const TensorRtLlmHead* trtllm_head) {
  if (engine_data == nullptr || engine_size == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT engine bytecode is empty");
  }
  if (engine_size > std::numeric_limits<uint64_t>::max()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT engine too large");
  }
  if (trtllm_head != nullptr) {
    LITERT_RETURN_IF_ERROR(ValidateTrtLlmHead(*trtllm_head, outputs));
  }

  uint32_t version = kTensorRtBytecodeVersion;
  if (trtllm_head != nullptr) {
    version = trtllm_head->weight_format ==
                      TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved
                  ? kTensorRtBytecodeVersionWithTrtLlmHead
                  : kTensorRtBytecodeVersionWithTypedHead;
  }

  std::vector<uint8_t> out;
  out.reserve(
      sizeof(uint32_t) * 5 + function_name.size() + inputs.size() * 32 +
      outputs.size() * 32 + engine_size +
      (trtllm_head == nullptr
           ? 0
           : sizeof(uint32_t) *
                     (version == kTensorRtBytecodeVersionWithTypedHead ? 6
                                                                       : 5) +
                 sizeof(uint64_t) * 2 + trtllm_head->packed_weights_size +
                 trtllm_head->bf16_scales_size));
  AppendScalar<uint32_t>(out, kMagic);
  AppendScalar<uint32_t>(out, version);
  LITERT_RETURN_IF_ERROR(AppendString(out, function_name));
  LITERT_RETURN_IF_ERROR(AppendStringVector(out, inputs));
  LITERT_RETURN_IF_ERROR(AppendStringVector(out, outputs));
  AppendScalar<uint64_t>(out, static_cast<uint64_t>(engine_size));
  const auto* engine_bytes = static_cast<const uint8_t*>(engine_data);
  out.insert(out.end(), engine_bytes, engine_bytes + engine_size);
  if (trtllm_head != nullptr) {
    AppendScalar<uint32_t>(out, trtllm_head->hidden_output_port);
    AppendScalar<uint32_t>(out, trtllm_head->logits_output_port);
    AppendScalar<uint32_t>(out, trtllm_head->k);
    AppendScalar<uint32_t>(out, trtllm_head->n);
    AppendScalar<float>(out, trtllm_head->soft_cap);
    if (version == kTensorRtBytecodeVersionWithTypedHead) {
      AppendScalar<uint32_t>(out,
                             static_cast<uint32_t>(trtllm_head->weight_format));
    }
    AppendScalar<uint64_t>(
        out, static_cast<uint64_t>(trtllm_head->packed_weights_size));
    out.insert(out.end(), trtllm_head->packed_weights,
               trtllm_head->packed_weights + trtllm_head->packed_weights_size);
    AppendScalar<uint64_t>(
        out, static_cast<uint64_t>(trtllm_head->bf16_scales_size));
    out.insert(out.end(), trtllm_head->bf16_scales,
               trtllm_head->bf16_scales + trtllm_head->bf16_scales_size);
  }
  return out;
}

Expected<std::vector<uint8_t>> PackTensorRtSharedWeightBundle(
    const std::vector<TensorRtSharedWeight>& shared_weights,
    const std::vector<TensorRtBundleEntry>& entries) {
  if (shared_weights.size() > std::numeric_limits<uint32_t>::max() ||
      entries.empty() ||
      entries.size() > std::numeric_limits<uint32_t>::max()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT shared weight bundle size");
  }

  size_t reserve_size = sizeof(uint32_t) * 4;
  auto add_reserve_size = [&](size_t bytes) -> bool {
    if (bytes > std::numeric_limits<size_t>::max() - reserve_size) {
      return false;
    }
    reserve_size += bytes;
    return true;
  };
  for (const auto& weight : shared_weights) {
    LITERT_ASSIGN_OR_RETURN(
        size_t expected_size,
        TensorRtWeightBytes(weight.data_type, weight.count));
    if (weight.data.size() != expected_size ||
        !add_reserve_size(sizeof(int32_t) + sizeof(uint64_t) * 2) ||
        !add_reserve_size(weight.data.size())) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Invalid TensorRT shared weight");
    }
  }

  std::unordered_set<std::string> function_names;
  for (const auto& entry : entries) {
    if (entry.function_name.empty() ||
        !function_names.insert(entry.function_name).second ||
        entry.engine_data == nullptr || entry.engine_size == 0 ||
        entry.engine_size > std::numeric_limits<uint64_t>::max() ||
        entry.refit_weights.size() > std::numeric_limits<uint32_t>::max()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Invalid TensorRT bundle engine entry");
    }
    if (entry.trtllm_head != nullptr) {
      LITERT_RETURN_IF_ERROR(
          ValidateTrtLlmHead(*entry.trtllm_head, entry.output_names));
    }
    std::unordered_set<std::string> refit_names;
    for (const auto& ref : entry.refit_weights) {
      if (ref.name.empty() || !refit_names.insert(ref.name).second ||
          ref.shared_weight_index >= shared_weights.size()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Invalid TensorRT bundle refit reference");
      }
      if (!add_reserve_size(sizeof(uint32_t) * 2 + ref.name.size())) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT bundle size overflows");
      }
    }
    if (!add_reserve_size(sizeof(uint32_t) * 5 + sizeof(uint64_t)) ||
        !add_reserve_size(entry.function_name.size()) ||
        !add_reserve_size(entry.engine_size)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "TensorRT bundle size overflows");
    }
    for (const auto& name : entry.input_names) {
      if (!add_reserve_size(sizeof(uint32_t) + name.size())) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT bundle size overflows");
      }
    }
    for (const auto& name : entry.output_names) {
      if (!add_reserve_size(sizeof(uint32_t) + name.size())) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT bundle size overflows");
      }
    }
    if (entry.trtllm_head != nullptr) {
      const auto& head = *entry.trtllm_head;
      if (!add_reserve_size(sizeof(uint32_t) * 6 + sizeof(uint64_t) * 2) ||
          !add_reserve_size(head.packed_weights_size) ||
          !add_reserve_size(head.bf16_scales_size)) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT bundle size overflows");
      }
    }
  }

  std::vector<uint8_t> out;
  out.reserve(reserve_size);
  AppendScalar<uint32_t>(out, kMagic);
  AppendScalar<uint32_t>(out, kTensorRtBytecodeVersionWithSharedWeights);
  AppendScalar<uint32_t>(out, static_cast<uint32_t>(shared_weights.size()));
  for (const auto& weight : shared_weights) {
    AppendScalar<int32_t>(out, static_cast<int32_t>(weight.data_type));
    AppendScalar<uint64_t>(out, weight.count);
    AppendScalar<uint64_t>(out, static_cast<uint64_t>(weight.data.size()));
    out.insert(out.end(), weight.data.begin(), weight.data.end());
  }
  AppendScalar<uint32_t>(out, static_cast<uint32_t>(entries.size()));
  for (const auto& entry : entries) {
    LITERT_RETURN_IF_ERROR(AppendString(out, entry.function_name));
    LITERT_RETURN_IF_ERROR(AppendStringVector(out, entry.input_names));
    LITERT_RETURN_IF_ERROR(AppendStringVector(out, entry.output_names));
    AppendScalar<uint64_t>(out, static_cast<uint64_t>(entry.engine_size));
    const auto* engine_bytes = static_cast<const uint8_t*>(entry.engine_data);
    out.insert(out.end(), engine_bytes, engine_bytes + entry.engine_size);
    AppendScalar<uint32_t>(out, entry.trtllm_head != nullptr ? 1 : 0);
    if (entry.trtllm_head != nullptr) {
      AppendTypedHead(out, *entry.trtllm_head);
    }
    AppendScalar<uint32_t>(out,
                           static_cast<uint32_t>(entry.refit_weights.size()));
    for (const auto& ref : entry.refit_weights) {
      LITERT_RETURN_IF_ERROR(AppendString(out, ref.name));
      AppendScalar<uint32_t>(out, ref.shared_weight_index);
    }
  }
  return out;
}

TensorRtArtifactFingerprint FingerprintTensorRtArtifact(const void* data,
                                                        size_t size) {
  if (data == nullptr || size == 0) {
    return {};
  }
  const auto fingerprint = tsl::Fingerprint128(
      absl::string_view(static_cast<const char*>(data), size));
  return {fingerprint.low64, fingerprint.high64};
}

Expected<std::vector<uint8_t>> PackTensorRtAotLocator(
    const TensorRtAotLocator& locator) {
  if (locator.path.empty() || locator.path.front() != '/' ||
      locator.path.find('\0') != std::string::npos ||
      locator.artifact_size == 0 ||
      locator.path.size() > std::numeric_limits<uint32_t>::max()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT AOT artifact locator");
  }
  std::vector<uint8_t> out;
  out.reserve(sizeof(uint32_t) * 3 + sizeof(uint64_t) * 3 +
              locator.path.size());
  AppendScalar<uint32_t>(out, kAotLocatorMagic);
  AppendScalar<uint32_t>(out, kTensorRtAotLocatorVersion);
  AppendScalar<uint64_t>(out, locator.artifact_size);
  AppendScalar<uint64_t>(out, locator.fingerprint.low);
  AppendScalar<uint64_t>(out, locator.fingerprint.high);
  LITERT_RETURN_IF_ERROR(AppendString(out, locator.path));
  return out;
}

Expected<std::optional<TensorRtAotLocator>> TryParseTensorRtAotLocator(
    const void* data, size_t size) {
  if (data == nullptr || size < sizeof(uint32_t)) {
    return std::optional<TensorRtAotLocator>();
  }
  const auto* cur = static_cast<const uint8_t*>(data);
  const auto* end = cur + size;
  LITERT_ASSIGN_OR_RETURN(uint32_t magic, ReadScalar<uint32_t>(cur, end));
  if (magic != kAotLocatorMagic) {
    return std::optional<TensorRtAotLocator>();
  }
  LITERT_ASSIGN_OR_RETURN(uint32_t version, ReadScalar<uint32_t>(cur, end));
  if (version != kTensorRtAotLocatorVersion) {
    return Error(kLiteRtStatusErrorUnsupportedCompilerVersion,
                 "Unsupported TensorRT AOT locator version");
  }
  TensorRtAotLocator locator;
  LITERT_ASSIGN_OR_RETURN(locator.artifact_size,
                          ReadScalar<uint64_t>(cur, end));
  LITERT_ASSIGN_OR_RETURN(locator.fingerprint.low,
                          ReadScalar<uint64_t>(cur, end));
  LITERT_ASSIGN_OR_RETURN(locator.fingerprint.high,
                          ReadScalar<uint64_t>(cur, end));
  LITERT_ASSIGN_OR_RETURN(locator.path, ReadString(cur, end));
  if (locator.path.empty() || locator.path.front() != '/' ||
      locator.path.find('\0') != std::string::npos ||
      locator.artifact_size == 0 || cur != end) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT AOT artifact locator");
  }
  return std::optional<TensorRtAotLocator>(std::move(locator));
}

Expected<std::vector<uint8_t>> PackTensorRtAotManifest(
    const TensorRtAotManifest& manifest) {
  if (manifest.locators.empty() ||
      manifest.locators.size() > std::numeric_limits<uint32_t>::max() ||
      manifest.call_infos.empty() ||
      manifest.call_infos.size() > std::numeric_limits<uint32_t>::max() ||
      manifest.call_infos.size() != manifest.bytecode_indices.size()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT AOT manifest cardinality");
  }
  size_t reserve_size = sizeof(uint32_t) * 4 + sizeof(uint64_t) * 2;
  for (const auto& locator : manifest.locators) {
    if (locator.empty() ||
        locator.size() > std::numeric_limits<uint32_t>::max() ||
        locator.size() > std::numeric_limits<size_t>::max() - reserve_size) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Invalid TensorRT AOT manifest locator");
    }
    LITERT_ASSIGN_OR_RETURN(auto parsed, TryParseTensorRtAotLocator(
                                             locator.data(), locator.size()));
    if (!parsed.has_value()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "TensorRT AOT manifest contains non-locator bytecode");
    }
    reserve_size += sizeof(uint32_t) + locator.size();
  }
  for (size_t i = 0; i < manifest.call_infos.size(); ++i) {
    if (manifest.call_infos[i].empty() ||
        manifest.bytecode_indices[i] >= manifest.locators.size() ||
        manifest.call_infos[i].size() > std::numeric_limits<uint32_t>::max() ||
        manifest.call_infos[i].size() >
            std::numeric_limits<size_t>::max() - reserve_size) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Invalid TensorRT AOT manifest call entry");
    }
    reserve_size += sizeof(uint32_t) * 2 + manifest.call_infos[i].size();
  }

  std::vector<uint8_t> out;
  out.reserve(reserve_size);
  AppendScalar<uint32_t>(out, kAotManifestMagic);
  AppendScalar<uint32_t>(out, kTensorRtAotManifestVersion);
  AppendScalar<uint64_t>(out, manifest.cache_key.low);
  AppendScalar<uint64_t>(out, manifest.cache_key.high);
  AppendScalar<uint32_t>(out, static_cast<uint32_t>(manifest.locators.size()));
  for (const auto& locator : manifest.locators) {
    AppendScalar<uint32_t>(out, static_cast<uint32_t>(locator.size()));
    out.insert(out.end(), locator.begin(), locator.end());
  }
  AppendScalar<uint32_t>(out,
                         static_cast<uint32_t>(manifest.call_infos.size()));
  for (size_t i = 0; i < manifest.call_infos.size(); ++i) {
    LITERT_RETURN_IF_ERROR(AppendString(out, manifest.call_infos[i]));
    AppendScalar<uint32_t>(out, manifest.bytecode_indices[i]);
  }
  return out;
}

Expected<TensorRtAotManifest> ParseTensorRtAotManifest(const void* data,
                                                       size_t size) {
  if (data == nullptr || size == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT AOT manifest is empty");
  }
  const auto* cur = static_cast<const uint8_t*>(data);
  const auto* end = cur + size;
  LITERT_ASSIGN_OR_RETURN(uint32_t magic, ReadScalar<uint32_t>(cur, end));
  if (magic != kAotManifestMagic) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT AOT manifest magic");
  }
  LITERT_ASSIGN_OR_RETURN(uint32_t version, ReadScalar<uint32_t>(cur, end));
  if (version != kTensorRtAotManifestVersion) {
    return Error(kLiteRtStatusErrorUnsupportedCompilerVersion,
                 "Unsupported TensorRT AOT manifest version");
  }
  TensorRtAotManifest manifest;
  LITERT_ASSIGN_OR_RETURN(manifest.cache_key.low,
                          ReadScalar<uint64_t>(cur, end));
  LITERT_ASSIGN_OR_RETURN(manifest.cache_key.high,
                          ReadScalar<uint64_t>(cur, end));
  LITERT_ASSIGN_OR_RETURN(uint32_t locator_count,
                          ReadScalar<uint32_t>(cur, end));
  if (locator_count == 0 ||
      locator_count > static_cast<size_t>(end - cur) / sizeof(uint32_t)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT AOT manifest locator count");
  }
  manifest.locators.reserve(locator_count);
  for (uint32_t i = 0; i < locator_count; ++i) {
    LITERT_ASSIGN_OR_RETURN(uint32_t locator_size,
                            ReadScalar<uint32_t>(cur, end));
    if (locator_size == 0 || locator_size > static_cast<size_t>(end - cur)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Truncated TensorRT AOT manifest locator");
    }
    std::vector<uint8_t> locator(cur, cur + locator_size);
    cur += locator_size;
    LITERT_ASSIGN_OR_RETURN(auto parsed, TryParseTensorRtAotLocator(
                                             locator.data(), locator.size()));
    if (!parsed.has_value()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "TensorRT AOT manifest contains non-locator bytecode");
    }
    manifest.locators.push_back(std::move(locator));
  }
  LITERT_ASSIGN_OR_RETURN(uint32_t call_count, ReadScalar<uint32_t>(cur, end));
  if (call_count == 0 ||
      call_count > static_cast<size_t>(end - cur) / (sizeof(uint32_t) * 2)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT AOT manifest call count");
  }
  manifest.call_infos.reserve(call_count);
  manifest.bytecode_indices.reserve(call_count);
  for (uint32_t i = 0; i < call_count; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto call_info, ReadString(cur, end));
    LITERT_ASSIGN_OR_RETURN(uint32_t bytecode_index,
                            ReadScalar<uint32_t>(cur, end));
    if (call_info.empty() || bytecode_index >= manifest.locators.size()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Invalid TensorRT AOT manifest call entry");
    }
    manifest.call_infos.push_back(std::move(call_info));
    manifest.bytecode_indices.push_back(bytecode_index);
  }
  if (cur != end) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Unexpected trailing TensorRT AOT manifest data");
  }
  return manifest;
}

Expected<TensorRtBytecode> ParseTensorRtBytecode(const void* data, size_t size,
                                                 const char* function_name) {
  if (data == nullptr || size == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument, "Bytecode is empty");
  }
  const auto* cur = static_cast<const uint8_t*>(data);
  const auto* end = cur + size;

  LITERT_ASSIGN_OR_RETURN(uint32_t magic, ReadScalar<uint32_t>(cur, end));
  if (magic != kMagic) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT bytecode magic");
  }
  LITERT_ASSIGN_OR_RETURN(uint32_t version, ReadScalar<uint32_t>(cur, end));
  if (version != kTensorRtBytecodeVersion &&
      version != kTensorRtBytecodeVersionWithTrtLlmHead &&
      version != kTensorRtBytecodeVersionWithTypedHead &&
      version != kTensorRtBytecodeVersionWithSharedWeights) {
    return Error(kLiteRtStatusErrorUnsupportedCompilerVersion,
                 "Unsupported TensorRT bytecode version");
  }
  if (version == kTensorRtBytecodeVersionWithSharedWeights) {
    return ParseSharedWeightBundle(cur, end, function_name);
  }

  TensorRtBytecode bytecode;
  bytecode.version = version;
  LITERT_ASSIGN_OR_RETURN(bytecode.function_name, ReadString(cur, end));
  LITERT_ASSIGN_OR_RETURN(bytecode.input_names, ReadStringVector(cur, end));
  LITERT_ASSIGN_OR_RETURN(bytecode.output_names, ReadStringVector(cur, end));
  LITERT_ASSIGN_OR_RETURN(uint64_t engine_size_u64,
                          ReadScalar<uint64_t>(cur, end));
  if (engine_size_u64 > static_cast<uint64_t>(end - cur)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Truncated TensorRT engine payload");
  }
  bytecode.engine_data = cur;
  bytecode.engine_size = static_cast<size_t>(engine_size_u64);
  cur += bytecode.engine_size;
  if (version == kTensorRtBytecodeVersionWithTrtLlmHead ||
      version == kTensorRtBytecodeVersionWithTypedHead) {
    TensorRtLlmHead head;
    LITERT_ASSIGN_OR_RETURN(head.hidden_output_port,
                            ReadScalar<uint32_t>(cur, end));
    LITERT_ASSIGN_OR_RETURN(head.logits_output_port,
                            ReadScalar<uint32_t>(cur, end));
    LITERT_ASSIGN_OR_RETURN(head.k, ReadScalar<uint32_t>(cur, end));
    LITERT_ASSIGN_OR_RETURN(head.n, ReadScalar<uint32_t>(cur, end));
    LITERT_ASSIGN_OR_RETURN(head.soft_cap, ReadScalar<float>(cur, end));
    if (version == kTensorRtBytecodeVersionWithTrtLlmHead) {
      head.weight_format =
          TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved;
    } else {
      LITERT_ASSIGN_OR_RETURN(uint32_t weight_format,
                              ReadScalar<uint32_t>(cur, end));
      head.weight_format =
          static_cast<TensorRtLlmHeadWeightFormat>(weight_format);
    }
    LITERT_ASSIGN_OR_RETURN(uint64_t weight_size,
                            ReadScalar<uint64_t>(cur, end));
    if (weight_size > static_cast<uint64_t>(end - cur)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Truncated TensorRT-LLM head weights");
    }
    head.packed_weights = cur;
    head.packed_weights_size = static_cast<size_t>(weight_size);
    cur += head.packed_weights_size;
    LITERT_ASSIGN_OR_RETURN(uint64_t scale_size,
                            ReadScalar<uint64_t>(cur, end));
    if (scale_size > static_cast<uint64_t>(end - cur)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Truncated TensorRT-LLM head scales");
    }
    head.bf16_scales = cur;
    head.bf16_scales_size = static_cast<size_t>(scale_size);
    cur += head.bf16_scales_size;
    LITERT_RETURN_IF_ERROR(ValidateTrtLlmHead(head, bytecode.output_names));
    bytecode.trtllm_head = head;
  }
  if (cur != end) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Unexpected trailing TensorRT bytecode data");
  }
  return bytecode;
}

}  // namespace litert::nvidia
