// Copyright 2026 Google LLC.
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

#include "litert/cc/options/litert_samsung_options.h"

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_samsung_options.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::samsung {

const char* SamsungOptions::Discriminator() {
  return LrtSamsungOptionsGetIdentifier();
}

Expected<SamsungOptions> SamsungOptions::Create() {
  LrtSamsungOptions options = nullptr;
  LITERT_RETURN_IF_ERROR(LrtCreateSamsungOptions(&options));
  return SamsungOptions(options);
}

LiteRtStatus SamsungOptions::GetOpaqueOptionsData(
    const char** identifier, void** payload,
    void (**payload_deleter)(void*)) const {
  return LrtGetOpaqueSamsungOptionsData(Get(), identifier, payload,
                                        payload_deleter);
}

Expected<void> SamsungOptions::SetEnableLargeModelSupport(
    bool large_model_support) {
  LITERT_RETURN_IF_ERROR(
      LrtSamsungOptionsSetEnableLargeModelSupport(Get(), large_model_support));
  return {};
}

Expected<bool> SamsungOptions::GetEnableLargeModelSupport() const {
  bool large_model_support;
  LITERT_RETURN_IF_ERROR(
      LrtSamsungOptionsGetEnableLargeModelSupport(Get(), &large_model_support));
  return large_model_support;
}

}  // namespace litert::samsung
