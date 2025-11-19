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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_

#include <string>
#include <utility>

#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::intel_openvino {

// Wraps a LiteRtIntelOpenVinoOptions object for convenience.
class IntelOpenVinoOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  IntelOpenVinoOptions() = delete;

  static const char* Discriminator();

  static Expected<IntelOpenVinoOptions> Create(OpaqueOptions& options);

  static Expected<IntelOpenVinoOptions> Create();

  void SetDeviceType(LiteRtIntelOpenVinoDeviceType device_type);

  LiteRtIntelOpenVinoDeviceType GetDeviceType() const;

  void SetPerformanceMode(LiteRtIntelOpenVinoPerformanceMode performance_mode);

  LiteRtIntelOpenVinoPerformanceMode GetPerformanceMode() const;

  void SetConfigsMapOption(const char* key, const char* value);

  int GetNumConfigsMapOptions() const;

  std::pair<std::string, std::string> GetConfigsMapOption(int index) const;

 private:
  LiteRtIntelOpenVinoOptions Data() const;
};

}  // namespace litert::intel_openvino

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
