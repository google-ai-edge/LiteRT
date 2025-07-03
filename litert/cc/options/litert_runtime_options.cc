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

#include "litert/cc/options/litert_runtime_options.h"

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_runtime_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"

namespace litert {

absl::string_view RuntimeOptions::Identifier() {
  return LiteRtGetRuntimeOptionsIdentifier();
}

Expected<RuntimeOptions> RuntimeOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtCreateRuntimeOptions(&options));
  return RuntimeOptions(options, OwnHandle::kYes);
}

Expected<void> RuntimeOptions::SetShloCompositeInlining(
    bool shlo_composite_inlining) {
  LiteRtRuntimeOptions runtime_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindRuntimeOptions(Get(), &runtime_options));
  LITERT_RETURN_IF_ERROR(LiteRtSetRuntimeOptionsShloCompositeInlining(
      runtime_options, shlo_composite_inlining));
  return {};
}

Expected<bool> RuntimeOptions::GetShloCompositeInlining() const {
  LiteRtRuntimeOptions runtime_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindRuntimeOptions(Get(), &runtime_options));
  bool shlo_composite_inlining;
  LITERT_RETURN_IF_ERROR(LiteRtGetRuntimeOptionsShloCompositeInlining(
      runtime_options, &shlo_composite_inlining));
  return shlo_composite_inlining;
}

Expected<void> RuntimeOptions::SetEnableProfiling(
  bool enable_profiling) {
LiteRtRuntimeOptions runtime_options;
LITERT_RETURN_IF_ERROR(LiteRtFindRuntimeOptions(Get(), &runtime_options));
LITERT_RETURN_IF_ERROR(LiteRtSetRuntimeOptionsEnableProfiling(
    runtime_options, enable_profiling));
return {};
}

Expected<bool> RuntimeOptions::GetEnableProfiling() const {
LiteRtRuntimeOptions runtime_options;
LITERT_RETURN_IF_ERROR(LiteRtFindRuntimeOptions(Get(), &runtime_options));
bool enable_profiling;
LITERT_RETURN_IF_ERROR(LiteRtGetRuntimeOptionsEnableProfiling(
    runtime_options, &enable_profiling));
return enable_profiling;
}
}  // namespace litert
