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

#include "litert/cc/litert_options.h"

#include "litert/c/litert_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_gpu_options.h"

namespace litert {

Expected<GpuOptions&> Options::GetGpuOptions() {
  if (!gpu_options_) {
    LITERT_ASSIGN_OR_RETURN(gpu_options_, GpuOptions::Create());
  }
  return *gpu_options_;
}

Expected<void> Options::Build() {
  if (gpu_options_) {
    LITERT_RETURN_IF_ERROR(
        LiteRtAddOpaqueOptions(Get(), gpu_options_->Release()));
  }
  return {};
}

}  // namespace litert
