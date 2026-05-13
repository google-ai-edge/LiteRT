// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/samsung/schema/samsung_byte_code.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/samsung/schema/litert_samsung_header_generated.h"

namespace litert::samsung {

std::string GenerateSignature(const absl::Span<const char>& weights) {
  auto seed = weights.size();
  for (const auto& x : weights) {
    seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  LITERT_LOG(LITERT_INFO, "Seed %lu", seed);

  static char item[] = "0123456789abcedf";
  std::string signature;
  for (int i = 0; i < sizeof(seed) * 2 /*= 8/4*/; i++) {
    signature += item[int(seed & 0x0F)];
    seed = seed >> 4;
  }

  return signature;
}

Expected<uint64_t> UpdateBufferOffset(uint64_t start_offset,
                                      const absl::Span<const char>& data,
                                      schema::BufferSection *const buf) {
  start_offset = ForceAlignSize(start_offset);
  uint64_t end_offset = start_offset + data.size();
  if (!buf->mutate_start_offset(start_offset) ||
      !buf->mutate_end_offset(end_offset)) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Fail to update buffer section's offset");
  }

  return end_offset;
}

Expected<SamsungByteCode::ByteCodeT> SamsungByteCode::CreateImpl(
    const absl::Span<const char>& dispatch_binary,
    const std::vector<std::string>& signatures,
    const std::vector<absl::Span<const char>> *weights) {
  // Don't use external weight if signature not provided.
  bool use_external_weights = !signatures.empty();

  if (weights != nullptr) {
    // Ensure valid data exist in weight
    for (auto& weight : *weights) {
      if (weight.data() == nullptr || weight.size() == 0) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Invalid separated weights.");
      }
    }
    if (signatures.size() != weights->size()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Require number of signatures match weights' number.");
    }
  }

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<flatbuffers::String>> fb_signatures;
  std::vector<flatbuffers::Offset<schema::SeparatedWeights>>
      fb_separated_weights;
  if (use_external_weights) {
    for (auto& sig : signatures) {
      fb_signatures.emplace_back(builder.CreateString(sig.c_str()));

      if (weights != nullptr) {
        fb_separated_weights.emplace_back(schema::CreateSeparatedWeightsDirect(
            builder, schema::CreateBufferSection(builder, 1, 1), sig.c_str()));
      }
    }
  }

  auto fb_dispatch_binary = schema::CreateDispatchBinaryDirect(
      builder, schema::CreateBufferSection(builder, 1, 1), use_external_weights,
      use_external_weights ? &fb_signatures : nullptr);

  auto header_root = schema::CreateLiteRTSamsungHeaderDirect(
      builder, SAMSUNG_BYTE_CODE_HEADER_MAJOR_VERSION, fb_dispatch_binary,
      !fb_separated_weights.empty() ? &fb_separated_weights : nullptr);
  builder.Finish(header_root);

  uint8_t *buffer = builder.GetBufferPointer();
  size_t header_size = builder.GetSize();
  // Update the offset
  auto mutable_header = schema::GetMutableLiteRTSamsungHeader(buffer);

  uint64_t cursor = header_size;
  LITERT_ASSIGN_OR_RETURN(
      cursor, UpdateBufferOffset(
                  cursor, dispatch_binary,
                  mutable_header->mutable_dispatch_binary()->mutable_buf()));

  if (weights != nullptr && !weights->empty()) {
    auto mutable_separated_weights =
        mutable_header->mutable_separated_weights();
    for (int32_t index = 0; index < weights->size(); index++) {
      LITERT_ASSIGN_OR_RETURN(
          cursor,
          UpdateBufferOffset(cursor, weights->at(index),
                             mutable_separated_weights->GetMutableObject(index)
                                 ->mutable_buf()));
    }
  }

  // Copy to buffer
  auto byte_code_buf = ByteCodeT(cursor);
  memcpy(byte_code_buf.data(), buffer, header_size);
  memcpy(byte_code_buf.data() +
             mutable_header->dispatch_binary()->buf()->start_offset(),
         dispatch_binary.data(), dispatch_binary.size());
  if (weights != nullptr && !weights->empty()) {
    for (int32_t index = 0; index < weights->size(); index++) {
      auto separated_weight = mutable_header->separated_weights()->Get(index);
      memcpy(byte_code_buf.data() + separated_weight->buf()->start_offset(),
             weights->at(index).data(), weights->at(index).size());
    }
  }

  return byte_code_buf;
}

Expected<SamsungByteCode::UniquePtr>
SamsungByteCode::Create(const absl::Span<const char>& dispatch_binary,
                        const std::vector<absl::Span<const char>>& weights) {
  auto dispatch_binary_size = dispatch_binary.size();
  if (dispatch_binary.data() == nullptr || dispatch_binary_size == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Dispatch binary compiled from subgraph is invalid.");
  }

  std::vector<std::string> signatures;
  for (auto& weight : weights) {
    signatures.emplace_back(GenerateSignature(weight));
  }
  LITERT_ASSIGN_OR_RETURN(auto byte_code_buf,
                          CreateImpl(dispatch_binary, signatures, &weights));

  return SamsungByteCode::UniquePtr(
      new SamsungByteCode(std::move(byte_code_buf), signatures));
}

Expected<SamsungByteCode::UniquePtr> SamsungByteCode::Create(
    const absl::Span<const char> &dispatch_binary,
    const std::vector<std::string> &external_weight_signatures) {
  auto dispatch_binary_size = dispatch_binary.size();
  if (dispatch_binary.data() == nullptr || dispatch_binary_size == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Dispatch binary compiled from subgraph is invalid.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto byte_code_buf,
      CreateImpl(dispatch_binary, external_weight_signatures));
  return SamsungByteCode::UniquePtr(new SamsungByteCode(
      std::move(byte_code_buf), external_weight_signatures));
}

SamsungByteCode::SamsungByteCode(
    ByteCodeT&& byte_code, const std::vector<std::string>& weight_signatures)
    : byte_code_(byte_code), external_weight_signatures_(weight_signatures) {}

bool SamsungByteCode::HasExternalWeights() const {
  return !external_weight_signatures_.empty();
}

LiteRtStatus SamsungByteCode::GetWeightSignatures(
    std::vector<std::string>& signatures) const {
  signatures = external_weight_signatures_;
  return kLiteRtStatusOk;
}

LiteRtStatus SamsungByteCode::Dump(std::vector<char>& output) {
  output.swap(byte_code_);
  byte_code_.clear();

  return kLiteRtStatusOk;
}
} // namespace litert::samsung
