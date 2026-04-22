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
#ifndef ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_BUILDERS_ELEMENTWISE_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_BUILDERS_ELEMENTWISE_OP_BUILDER_H_

#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/samsung/compiler/builders/op_wrapper.h"

namespace litert::samsung {

Expected<OpWrapper> BuildAddOp(const Op& op);

Expected<OpWrapper> BuildMulOp(const Op& op);

Expected<OpWrapper> BuildDivOp(const Op& op);

Expected<OpWrapper> BuildExpOp(const Op& op);

Expected<OpWrapper> BuildGreaterOp(const Op& op);

Expected<OpWrapper> BuildGreaterEqualOp(const Op& op);

Expected<OpWrapper> BuildMaxOp(const Op& op);

Expected<OpWrapper> BuildMinOp(const Op& op);

Expected<OpWrapper> BuildCosOp(const Op& op);

Expected<OpWrapper> BuildSinOp(const Op& op);

Expected<OpWrapper> BuildRsqrtOp(const Op& op);

Expected<OpWrapper> BuildSqrtOp(const Op& op);

Expected<OpWrapper> BuildSubOp(const Op& op);

Expected<OpWrapper> BuildSquaredDifferenceOp(const Op& op);

Expected<OpWrapper> BuildAbsOp(const Op& op);

Expected<OpWrapper> BuildEqualOp(const Op& op);

Expected<OpWrapper> BuildCeilOp(const Op& op);

Expected<OpWrapper> BuildFloorOp(const Op& op);

Expected<OpWrapper> BuildFloorDivOp(const Op& op);

Expected<OpWrapper> BuildLessOp(const Op& op);

Expected<OpWrapper> BuildLogOp(const Op& op);

Expected<OpWrapper> BuildPowOp(const Op& op);

Expected<OpWrapper> BuildLogicalAndOp(const Op& op);

Expected<OpWrapper> BuildNotEqualOp(const Op& op);

}  // namespace litert::samsung

#endif  // ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_BUILDERS_ADD_OP_BUILDER_H_
