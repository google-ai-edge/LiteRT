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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_AOT_ARTIFACT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_AOT_ARTIFACT_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "litert/cc/litert_expected.h"
#include "litert/vendors/nvidia/bytecode.h"

namespace litert::nvidia {

enum class AotArtifactValidation {
  kTrustedFileIdentity,
  kComputedFingerprint,
  kProcessCache,
};

const char* AotArtifactValidationName(AotArtifactValidation validation);

// Owns a read-only mapping of one content-addressed TensorRT AOT artifact.
// Locator v2 can avoid an eager full-file scan when the writer-recorded,
// read-only inode identity still exactly matches. A mismatch falls back to the
// content fingerprint so copied artifacts remain portable and changed bytes
// still fail closed. The identity fast path assumes the cache owner does not
// deliberately unseal and rewrite the same inode; set
// LITERT_NVIDIA_TENSORRT_AOT_FORCE_CONTENT_VALIDATION for a full content audit.
class MappedAotArtifact {
 public:
  static Expected<std::unique_ptr<MappedAotArtifact>> Open(
      const TensorRtAotLocator& locator);

  ~MappedAotArtifact();

  MappedAotArtifact(const MappedAotArtifact&) = delete;
  MappedAotArtifact& operator=(const MappedAotArtifact&) = delete;

  const uint8_t* data() const { return static_cast<const uint8_t*>(data_); }
  size_t size() const { return size_; }
  AotArtifactValidation validation() const { return validation_; }

 private:
  MappedAotArtifact(void* data, size_t size, AotArtifactValidation validation)
      : data_(data), size_(size), validation_(validation) {}

  void* data_ = nullptr;
  size_t size_ = 0;
  AotArtifactValidation validation_ =
      AotArtifactValidation::kComputedFingerprint;
};

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_AOT_ARTIFACT_H_
