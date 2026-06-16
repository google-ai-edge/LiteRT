// Copyright 2024 Google LLC.
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

#include "litert/core/dispatch_op_schema.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"

namespace litert {
namespace internal {
namespace {

static constexpr const char kBytecodeSizeKey[] = "bytecode_size";
static constexpr const char kBytecodeOffsetKey[] = "bytecode_offset";
static constexpr const char kNameKey[] = "name";

}  // namespace

OwningBufferRef<uint8_t> MakeDispatchOpOptions(DispatchOpOptions options) {
  flexbuffers::Builder fbb;

  // Set maximum width for scalars to 64 bits. This prevents any upsizing of
  // the buffer when updating the bytecode size and offset in place.
  fbb.ForceMinimumBitWidth(flexbuffers::BIT_WIDTH_64);

  auto start = fbb.StartMap();

  fbb.UInt(kBytecodeSizeKey, options.bytecode_size);
  fbb.UInt(kBytecodeOffsetKey, options.bytecode_offset);
  fbb.String(kNameKey, options.name);

  fbb.EndMap(start);
  fbb.Finish();

  auto buf = fbb.GetBuffer();
  OwningBufferRef<uint8_t> res;
  res.Assign(buf.data(), buf.size());

  return res;
}

bool UpdateDispatchOpOptionsInPlace(DispatchOpOptions options,
                                    MutableBufferRef<uint8_t> buffer) {
  if (!buffer.Data() ||
      !flexbuffers::VerifyBuffer(buffer.Data(), buffer.Size())) {
    return false;
  }
  auto root = flexbuffers::GetRoot(buffer.Data(), buffer.Size());
  if (!root.IsMap()) {
    return false;
  }
  auto opts = root.AsMap();

  // Update name if same len.
  auto name = opts[kNameKey];
  auto size = opts[kBytecodeSizeKey];
  auto offset = opts[kBytecodeOffsetKey];
  if (!name.IsString() || !size.IsUInt() || !offset.IsUInt()) {
    return false;
  }
  const auto name_ok = name.MutateString(options.name);

  // Update bytecode size and offset. Since min scalar bit width is set to max
  // possible value, it shouldn't fail in theory.
  const auto size_ok = size.MutateUInt(options.bytecode_size);
  const auto offset_ok = offset.MutateUInt(options.bytecode_offset);

  return name_ok && size_ok && offset_ok;
}

Expected<DispatchOpOptions> GetDispatchOpOptions(BufferRef<uint8_t> buffer) {
  if (!buffer.Data() ||
      !flexbuffers::VerifyBuffer(buffer.Data(), buffer.Size())) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid dispatch op options flexbuffer");
  }
  const auto root = flexbuffers::GetRoot(buffer.Data(), buffer.Size());
  if (!root.IsMap()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Dispatch op options must be a map");
  }
  const auto opts = root.AsMap();

  const auto bytecode_size_ref = opts[kBytecodeSizeKey];
  const auto bytecode_offset_ref = opts[kBytecodeOffsetKey];
  const auto name_ref = opts[kNameKey];
  if (!bytecode_size_ref.IsUInt() || !bytecode_offset_ref.IsUInt() ||
      !name_ref.IsString()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Dispatch op options have invalid fields");
  }

  const uint64_t bytecode_size = bytecode_size_ref.AsUInt64();
  const uint64_t bytecode_offset = bytecode_offset_ref.AsUInt64();
  if (bytecode_size > std::numeric_limits<size_t>::max() ||
      bytecode_offset > std::numeric_limits<size_t>::max()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Dispatch op bytecode range exceeds size_t");
  }

  return DispatchOpOptions{
      static_cast<size_t>(bytecode_size),
      static_cast<size_t>(bytecode_offset),
      name_ref.AsString().str(),
  };
}

}  // namespace internal
}  // namespace litert
