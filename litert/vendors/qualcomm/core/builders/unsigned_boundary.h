// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_UNSIGNED_BOUNDARY_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_UNSIGNED_BOUNDARY_H_

#include <vector>

#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

// Rewrites the ops in `ops` whose QNN op type only accepts unsigned quantized
// activations on `backend`, so a signed activation does not fail
// QnnBackend_validateOpConfig with error 3110. Each such op has its signed
// per-tensor quantized inputs and outputs swapped for unsigned equivalents
// (same scale, offset shifted by 128 or 32768, so the real values match) and
// gets a Convert op inserted on either side to reconnect it to the rest of the
// graph.
//
// `ops` is modified in place and grows by the number of inserted Converts.
void InsertUnsignedActivationBoundaries(BackendType backend,
                                        TensorPool& tensor_pool,
                                        std::vector<OpWrapper>& ops);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_UNSIGNED_BOUNDARY_H_
