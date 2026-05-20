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

#ifndef ODML_LITERT_VENDORS_SAMSUNG_SCHEMA_BYTE_CODE_H_
#define ODML_LITERT_VENDORS_SAMSUNG_SCHEMA_BYTE_CODE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::samsung {

#define SAMSUNG_BYTE_CODE_HEADER_MAJOR_VERSION 1
#define SAMSUNG_BYTE_CODE_HEADER_MINOR_VERSION 0

#define SAMSUNG_BYTE_CODE_ALIGNMENT 64

// Returns true if the content is a LiteRT-LM file.
//
// Args:
//   content: The content of the file to check.
//
// Returns:
//   True if the content is a LiteRT-LM file, false otherwise.
// bool IsSamsungByteCode(absl::string_view content);

inline size_t ForceAlignSize(size_t size) {
  return ((size + SAMSUNG_BYTE_CODE_ALIGNMENT - 1) /
          SAMSUNG_BYTE_CODE_ALIGNMENT) *
         SAMSUNG_BYTE_CODE_ALIGNMENT;
}

class SamsungByteCode {
 public:
  using UniquePtr = std::unique_ptr<SamsungByteCode>;
  using ByteCodeT = std::vector<char>;

  static Expected<UniquePtr> Create(
      const absl::Span<const char>& dispatch_binary,
      const std::vector<absl::Span<const char>>& weights);

  static Expected<UniquePtr> Create(
      const absl::Span<const char>& dispatch_binary,
      const std::vector<std::string>& external_weight_signatures);

  SamsungByteCode(SamsungByteCode&) = delete;
  SamsungByteCode& operator=(const SamsungByteCode&) = delete;

  LiteRtStatus GetWeightSignatures(std::vector<std::string>& signatures) const;
  bool HasExternalWeights() const;

  Expected<uint64_t> GetByteCodeSize() const { return byte_code_.size(); }

  // Dump byte code to given vector, and byte code in the class will be
  // expired, and no longer valid.
  LiteRtStatus Dump(std::vector<char>& output);

 private:
  SamsungByteCode(ByteCodeT&& byte_code,
                  const std::vector<std::string>& weight_signature);

  static Expected<ByteCodeT> CreateImpl(
      const absl::Span<const char>& dispatch_binary,
      const std::vector<std::string>& signatures,
      const std::vector<absl::Span<const char>>* weights = nullptr);

  ByteCodeT byte_code_;
  std::vector<std::string> external_weight_signatures_;
};

}  // namespace litert::samsung

#endif  // ODML_LITERT_VENDORS_SAMSUNG_SCHEMA_BYTE_CODE_H_
