// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/rotary_embedding_op_builder.h"

#include "QnnOpDef.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"

namespace qnn {

OpWrapper CreateRotaryEmbeddingOp(const TensorWrapper& token_embedding,
                                  const TensorWrapper& cos,
                                  const TensorWrapper& sin,
                                  const TensorWrapper& output,
                                  bool interleaved) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ROTARY_EMBEDDING),
               QNN_OP_ROTARY_EMBEDDING, QnnOpCode::kRotaryEmbedding);
  op.AddInputTensor(token_embedding);
  op.AddInputTensor(cos);
  op.AddInputTensor(sin);
  op.AddOutputTensor(output);
  op.AddScalarParam<bool>(QNN_OP_ROTARY_EMBEDDING_PARAM_INTERLEAVED,
                          interleaved);
  return op;
}

}  // namespace qnn
