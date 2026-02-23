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
#ifndef ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_COMPILE_MODEL_H_
#define ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_COMPILE_MODEL_H_

#include <vector>

#include "litert/vendors/samsung/ai_litecore_manager.h"

namespace litert::samsung {

Expected<std::vector<char>> Compile(AiLiteCoreManager::Ptr ai_lite_core,
                                    const std::vector<char> &g_buffer);

}
#endif
