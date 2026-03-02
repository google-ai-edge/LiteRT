// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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
#include "litert/vendors/samsung/soc_model.h"

#include <cstdint>

#include "litert/c/internal/litert_logging.h"
#include "litert/cc/litert_expected.h"

namespace litert::samsung {

constexpr SocModel kSocModels[] = {
    {"E9965", 9965},
};

constexpr LiteRtParamIndex kNumOfSocModels =
    sizeof(kSocModels) / sizeof(kSocModels[0]);

Expected<int32_t> GetSocModelID(std::string_view soc_model) {
  for (LiteRtParamIndex index = 0; index < kNumOfSocModels; index++) {
    if (soc_model == kSocModels[index].soc_name) {
      return kSocModels[index].id;
    }
  }

  LITERT_LOG(LITERT_ERROR, "Unsupported soc model `%s`", soc_model.data());
  return Error(kLiteRtStatusErrorNotFound, "Fail to get soc id.");
}

} // namespace litert::samsung
