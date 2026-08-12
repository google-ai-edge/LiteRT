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
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::nvidia {
namespace {

constexpr uint32_t kMagic = 0x4e52544c;  // "LTRN", little endian.

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

Expected<TensorRtBytecode> ParseTensorRtBytecode(const void* data,
                                                 size_t size) {
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
      version != kTensorRtBytecodeVersionWithTypedHead) {
    return Error(kLiteRtStatusErrorUnsupportedCompilerVersion,
                 "Unsupported TensorRT bytecode version");
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
