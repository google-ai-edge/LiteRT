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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_COMPILER_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_COMPILER_OPTIONS_H_

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

class CompilerOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static absl::string_view Discriminator() {
    return LiteRtGetCompilerOptionsIdentifier();
  };

  static Expected<CompilerOptions> Create();
  static Expected<CompilerOptions> Create(OpaqueOptions& original);

  Expected<void> SetPartitionStrategy(
      LiteRtCompilerOptionsPartitionStrategy partition_strategy);
  Expected<LiteRtCompilerOptionsPartitionStrategy> GetPartitionStrategy() const;

  Expected<void> SetDummyOption(bool dummy_option);
  Expected<bool> GetDummyOption() const;
};
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_COMPILER_OPTIONS_H_
