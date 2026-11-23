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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DECODE_SLOTS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DECODE_SLOTS_H_

#include <cstdint>
#include <string>

namespace litert::nvidia {

// A batched decode engine (compiler/slot_network.h) names slot `slot`'s copy
// of the decode engine's input or output `name` this way; the compiler and
// the dispatch agree on it.
inline std::string SlotTensorName(const std::string& name, int32_t slot) {
  return name + "__slot" + std::to_string(slot);
}

// A batched decode engine of n slots lies beside its decode engine's AOT
// artifact, at this path.
inline std::string SlotArtifactPath(const std::string& decode_artifact,
                                    int32_t slots) {
  return decode_artifact + ".slots" + std::to_string(slots);
}

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DECODE_SLOTS_H_
