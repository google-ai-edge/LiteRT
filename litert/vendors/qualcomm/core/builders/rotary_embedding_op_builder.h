// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_ROTARY_EMBEDDING_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_ROTARY_EMBEDDING_OP_BUILDER_H_

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper CreateRotaryEmbeddingOp(const TensorWrapper& token_embedding,
                                  const TensorWrapper& cos,
                                  const TensorWrapper& sin,
                                  const TensorWrapper& output,
                                  bool interleaved);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_ROTARY_EMBEDDING_OP_BUILDER_H_
