// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/unpack_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

OpWrapper CreateUnpackOp(const TensorWrapper& input_0,
                         const std::vector<ConstTensorWrapperRef>& outputs,
                         std::uint32_t axis) {
  auto name = GetUniqueOpName(QNN_OP_UN_PACK);
  OpWrapper op;
  op.SetName(std::move(name));
  op.SetType(QNN_OP_UN_PACK, QnnOpCode::kUnPack);
  op.AddInputTensor(input_0);
  for (const auto& output : outputs) {
    op.AddOutputTensor(output);
  }
  op.AddScalarParam<uint32_t>(QNN_OP_UN_PACK_PARAM_AXIS, axis);
  return op;
}

}  // namespace qnn
