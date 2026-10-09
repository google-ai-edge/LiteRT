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
#include "litert/vendors/samsung/compiler/builders/op_wrapper.h"

#include <string>

#include "litert/c/litert_common.h"

namespace litert::samsung {

LiteRtStatus NameGenerator::Generate(OpWrapper& op_wrapper) {
  if (op_wrapper.op_name_.empty()) {
    const std::string& type = op_wrapper.op_type_;
    auto id = ++type_id_[type];
    op_wrapper.op_name_ = kNamePrefix + type + '_' + std::to_string(id);
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::samsung
