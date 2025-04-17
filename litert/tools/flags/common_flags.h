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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_COMMON_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_COMMON_FLAGS_H_

#include <string>
#include <vector>

#include "absl/flags/declare.h"  // from @com_google_absl

// Common flags that can be shared accross all the litert CLI tools.
// TODO: lukeboyer - Use these flags in all CLI tools.

ABSL_DECLARE_FLAG(std::string, model);

ABSL_DECLARE_FLAG(std::string, soc_manufacturer);

ABSL_DECLARE_FLAG(std::string, soc_model);

ABSL_DECLARE_FLAG(std::vector<std::string>, libs);

ABSL_DECLARE_FLAG(std::vector<std::string>, o);

ABSL_DECLARE_FLAG(std::string, err);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_COMMON_FLAGS_H_
