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
#ifndef ODML_LITERT_LITERT_VENDORS_SAMSUNG_SOC_MODEL_H_
#define ODML_LITERT_LITERT_VENDORS_SAMSUNG_SOC_MODEL_H_

#include <string_view>
#include <cstdint>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::samsung {

struct SocModel {
  const char *soc_name;
  int32_t id;

  constexpr SocModel(const char *p_soc_name, int32_t p_id)
      : soc_name(p_soc_name), id(p_id) {}
};

extern const SocModel kSocModels[];
extern const LiteRtParamIndex kNumOfSocModels;

Expected<int32_t> GetSocModelID(std::string_view soc_model);

} // namespace litert::samsung

#endif // ODML_LITERT_LITERT_VENDORS_SAMSUNG_SOC_MODEL_H_
