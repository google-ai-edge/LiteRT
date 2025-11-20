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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"
namespace litert {
// RuntimeOptions is a wrapper around LiteRtRuntimeOptions. It is used to
// configure the runtime options of LiteRt runtime.
class RuntimeOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static absl::string_view Identifier();

  static Expected<RuntimeOptions> Create();
  static Expected<RuntimeOptions> Create(OpaqueOptions& original);

  Expected<void> SetEnableProfiling(bool enable_profiling);
  Expected<bool> GetEnableProfiling() const;
  Expected<void> SetErrorReporterMode(
      LiteRtErrorReporterMode error_reporter_mode);
  Expected<LiteRtErrorReporterMode> GetErrorReporterMode() const;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
