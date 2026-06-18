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
//
// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0

#include "litert/tools/flags/vendors/samsung_flags.h"

#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_samsung_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_samsung_options.h"
#include "litert/tools/flags/options_parser_registry.h"

ABSL_FLAG(std::string, samsung_soc_model, "E9965", "Target SoC model.");
ABSL_FLAG(bool, samsung_large_model_support, false,
          "Whether to enable large model support.");

namespace litert::samsung {

Expected<void> UpdateSamsungOptionsFromFlags(SamsungOptions& options) {
  bool large_model_support = absl::GetFlag(FLAGS_samsung_large_model_support);
  LITERT_RETURN_IF_ERROR(
      options.SetEnableLargeModelSupport(large_model_support));

  std::string soc_model_str = absl::GetFlag(FLAGS_samsung_soc_model);
  if (!soc_model_str.empty()) {
    LITERT_RETURN_IF_ERROR(options.SetSocModel(soc_model_str.c_str()));
  }
  return {};
}

}  // namespace litert::samsung

namespace litert::samsung {

LITERT_REGISTER_OPTIONS_PARSER([](Options& options) -> Expected<void> {
  LITERT_ASSIGN_OR_RETURN(auto& samsung_opts, options.GetSamsungOptions());
  return UpdateSamsungOptionsFromFlags(samsung_opts);
});

}  // namespace litert::samsung
