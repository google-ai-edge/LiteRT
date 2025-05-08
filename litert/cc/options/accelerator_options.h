// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ACCELERATOR_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ACCELERATOR_OPTIONS_H_

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

// Builds an option object that can be passed to LiteRT model compilation.
//
// ```cpp
// ```
class GpuOptions : public litert::OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static Expected<GpuOptions> Create();
  static const char* GetPayloadIdentifier();

  LiteRtStatus EnableConstantTensorSharing(bool enabled);
  LiteRtStatus EnableInfiniteFloatCapping(bool enabled);
  LiteRtStatus EnableBenchmarkMode(bool enabled);
  LiteRtStatus EnableAllowSrcQuantizedFcConvOps(bool enabled);
  LiteRtStatus SetDelegatePrecision(LiteRtDelegatePrecision precision);
  LiteRtStatus SetBufferStorageType(LiteRtDelegateBufferStorageType type);
  LiteRtStatus SetPreferTextureWeights(bool prefer_texture_weights);
  LiteRtStatus SetSerializationDir(const char* serialization_dir);
  LiteRtStatus SetModelToken(const char* model_token);
  LiteRtStatus SetSerializeProgramCache(bool serialize_program_cache);
  LiteRtStatus SetSerializeExternalTensors(bool serialize_external_tensors);
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ACCELERATOR_OPTIONS_H_
