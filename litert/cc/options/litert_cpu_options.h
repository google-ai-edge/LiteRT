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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CPU_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CPU_OPTIONS_H_

#include <cstdint>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

class CpuOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static absl::string_view Identifier();

  static Expected<CpuOptions> Create();
  static Expected<CpuOptions> Create(OpaqueOptions& original);

  Expected<void> SetNumThreads(int num_threads);
  Expected<int> GetNumThreads() const;

  Expected<void> SetXNNPackFlags(uint32_t flags);
  Expected<uint32_t> GetXNNPackFlags() const;

  Expected<void> SetXNNPackWeightCachePath(const char* path);
  Expected<absl::string_view> GetXNNPackWeightCachePath() const;

  Expected<void> SetXNNPackWeightCacheFileDescriptor(int fd);
  Expected<int> GetXNNPackWeightCacheFileDescriptor() const;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CPU_OPTIONS_H_
