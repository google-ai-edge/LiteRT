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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_APPLE_BYTECODE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_APPLE_BYTECODE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"

namespace litert::apple {

inline constexpr uint32_t kAppleMlxBytecodeVersion = 1;
inline constexpr uint32_t kAppleMlxMagic =
    0x584c4d41;  // 'AMLX' in little endian

struct MlxBytecode {
  std::vector<int64_t> weights_dims;
  LiteRtElementType weights_type;
  std::vector<uint8_t> weights_data;

  bool has_bias = false;
  std::vector<int64_t> bias_dims;
  LiteRtElementType bias_type;
  std::vector<uint8_t> bias_data;

  uint32_t activation = 0;  // LiteRtActivationFunctionType
};

Expected<std::vector<uint8_t>> PackMlxBytecode(const MlxBytecode& bytecode);
Expected<MlxBytecode> ParseMlxBytecode(const void* data, size_t size);

}  // namespace litert::apple

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_APPLE_BYTECODE_H_
