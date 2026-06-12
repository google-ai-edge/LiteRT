// Copyright (C) 2026 Intel Corporation
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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_BYTECODE_HEADER_H_
#define LITERT_VENDORS_INTEL_OPENVINO_BYTECODE_HEADER_H_

#include <cstdint>
#include <cstring>
#include <string>

#include "litert/c/options/litert_intel_openvino_options.h"

namespace litert {
namespace openvino {

// Self-describing prefix that the OpenVINO compiler plugin prepends to each
// partition's exported bytecode.  The dispatcher inspects the magic bytes;
// if the prefix is present, the embedded graph type is used when calling
// `ov::Core::import_model`.  When the prefix is absent (e.g. an older
// bytecode), the dispatcher falls back to a default device for backward
// compatibility.
struct OpenVinoBytecodeHeader {
  static constexpr char kMagic[8] = {'L', 'R', 'T', 'O', 'V', 'H', 'D', 'R'};
  static constexpr uint32_t kCurrentVersion = 1;

  char magic[8];
  uint32_t version;
  // LiteRtIntelOpenVinoGraphBackend, stored as uint32_t to keep the layout
  // ABI-stable independent of enum size.
  uint32_t graph_backend;
  // Reserved for future use (must be zero on write).
  uint32_t reserved;
};

static_assert(sizeof(OpenVinoBytecodeHeader) == 20,
              "OpenVinoBytecodeHeader must be exactly 20 bytes for ABI "
              "stability across compiler and dispatcher.");

// Translates a LiteRtIntelOpenVinoGraphBackend to the OpenVINO device name
// string used by ov::Core (e.g. "NPU", "CPU").
inline std::string GraphBackendToString(LiteRtIntelOpenVinoGraphBackend t) {
  switch (t) {
    case kLiteRtIntelOpenVinoGraphBackendCPU:
      return "CPU";
    case kLiteRtIntelOpenVinoGraphBackendGPU:
      return "GPU";
    case kLiteRtIntelOpenVinoGraphBackendNPU:
      return "NPU";
    case kLiteRtIntelOpenVinoGraphBackendMax:
      break;
  }
  return "NPU";
}

// Writes a header for the given graph type into a freshly allocated string.
inline std::string MakeBytecodeHeader(
    LiteRtIntelOpenVinoGraphBackend graph_backend) {
  OpenVinoBytecodeHeader hdr{};
  std::memcpy(hdr.magic, OpenVinoBytecodeHeader::kMagic, sizeof(hdr.magic));
  hdr.version = OpenVinoBytecodeHeader::kCurrentVersion;
  hdr.graph_backend = static_cast<uint32_t>(graph_backend);
  hdr.reserved = 0;
  return std::string(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
}

// If `bytecode` starts with a valid OpenVinoBytecodeHeader, populates
// `*graph_backend` and `*payload_offset` (the offset of the actual OV bytecode
// past the header) and returns true.  Otherwise returns false and leaves
// the outputs untouched.
inline bool TryParseBytecodeHeader(
    const void* bytecode, size_t bytecode_size,
    LiteRtIntelOpenVinoGraphBackend* graph_backend, size_t* payload_offset) {
  if (bytecode == nullptr || bytecode_size < sizeof(OpenVinoBytecodeHeader)) {
    return false;
  }
  OpenVinoBytecodeHeader hdr;
  std::memcpy(&hdr, bytecode, sizeof(hdr));
  if (std::memcmp(hdr.magic, OpenVinoBytecodeHeader::kMagic,
                  sizeof(hdr.magic)) != 0) {
    return false;
  }
  if (hdr.version != OpenVinoBytecodeHeader::kCurrentVersion) {
    return false;
  }
  // Reject values that fall outside the known backend range so callers never
  // see an enum value the runtime cannot interpret.
  if (hdr.graph_backend >=
      static_cast<uint32_t>(kLiteRtIntelOpenVinoGraphBackendMax)) {
    return false;
  }
  if (graph_backend != nullptr) {
    *graph_backend =
        static_cast<LiteRtIntelOpenVinoGraphBackend>(hdr.graph_backend);
  }
  if (payload_offset != nullptr) {
    *payload_offset = sizeof(OpenVinoBytecodeHeader);
  }
  return true;
}

}  // namespace openvino
}  // namespace litert

#endif  // LITERT_VENDORS_INTEL_OPENVINO_BYTECODE_HEADER_H_
