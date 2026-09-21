// Copyright (C) 2023 Amlogic, Inc. All rights reserved.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_AML_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_AML_OPTIONS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h" // from @com_google_absl
#include "litert/c/options/litert_aml_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::aml
{

  // Wraps a LiteRtAmlOptions object for convenience.
  class AmlOptions : public OpaqueOptions
  {
  public:
    using OpaqueOptions::OpaqueOptions;

    static const char *Discriminator()
    {
      return LiteRtAmlOptionsGetIdentifier();
    }

    static Expected<AmlOptions> Create(OpaqueOptions &options);

    static Expected<AmlOptions> Create();

    void SetLogLevel(LiteRtAmlOptionsLogLevel log_level);
    LiteRtAmlOptionsLogLevel GetLogLevel() const;

    void SetPerformanceMode(LiteRtAmlOptionsPerfMode perf_mode);
    LiteRtAmlOptionsPerfMode GetPerformanceMode() const;

    void SetMemoryPolicy(LiteRtAmlOptionsMemoryPolicy policy);
    LiteRtAmlOptionsMemoryPolicy GetMemoryPolicy() const;

    void SetModelPath(const std::string &model_path);
    std::string GetModelPath() const;

    void SetModelName(const std::string &model_name);
    std::string GetModelName() const;

    void SetProfiling(LiteRtAmlOptionsProfiling profiling);
    LiteRtAmlOptionsProfiling GetProfiling() const;

    // void SetDumpTensorIds(const std::vector<std::int32_t> &ids);
    // std::vector<std::int32_t> GetDumpTensorIds();

    // void SetIrJsonDir(const std::string &ir_json_dir);
    // absl::string_view GetIrJsonDir();

  private:
    LiteRtAmlOptions Data() const;
  };

} // namespace litert::aml
#endif // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_AML_OPTIONS_H_
