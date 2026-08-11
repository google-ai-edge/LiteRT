// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/unsigned_boundary.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {

using OpBackendKey = std::pair<QnnOpCode, BackendType>;

const absl::flat_hash_set<OpBackendKey>& UnsignedOnlyOps() {
  static const auto* kSet = new absl::flat_hash_set<OpBackendKey>{
      {QnnOpCode::kElementWiseNeuron, BackendType::kHtpBackend},
      {QnnOpCode::kPad, BackendType::kHtpBackend},
      {QnnOpCode::kResizeNearestNeighbor, BackendType::kHtpBackend},
  };
  return *kSet;
}

bool IsSignedActivation(const TensorWrapper& tensor) {
  return tensor.IsQuantI8() || tensor.IsQuantI16();
}

Qnn_DataType_t UnsignedCounterpart(Qnn_DataType_t signed_dtype) {
  return signed_dtype == QNN_DATATYPE_SFIXED_POINT_8
             ? QNN_DATATYPE_UFIXED_POINT_8
             : QNN_DATATYPE_UFIXED_POINT_16;
}

std::int32_t SignedToUnsignedOffsetDiff(Qnn_DataType_t signed_dtype) {
  return signed_dtype == QNN_DATATYPE_SFIXED_POINT_8 ? 128 : 32768;
}

TensorWrapper* CreateUnsignedTwin(TensorPool& tensor_pool,
                                 const TensorWrapper& signed_tensor) {
  if (!IsSignedActivation(signed_tensor)) {
    return nullptr;
  }
  const auto* quant = std::get_if<ScaleOffsetQuantizeParamsWrapper>(
      &signed_tensor.GetQuantParams());
  if (quant == nullptr) {
    return nullptr;
  }
  const Qnn_DataType_t signed_dtype = signed_tensor.GetDataType();
  const Qnn_ScaleOffset_t scale_offset{
      quant->GetScale(),
      quant->GetOffset() - SignedToUnsignedOffsetDiff(signed_dtype)};
  QuantizeParamsWrapperVariant unsigned_quant;
  unsigned_quant.emplace<ScaleOffsetQuantizeParamsWrapper>(scale_offset);
  return &tensor_pool.CreateNativeTensor(UnsignedCounterpart(signed_dtype),
                                         unsigned_quant,
                                         signed_tensor.GetDimensions());
}

}  // namespace

void InsertUnsignedActivationBoundaries(BackendType backend,
                                        TensorPool& tensor_pool,
                                        std::vector<OpWrapper>& ops) {
  std::vector<OpWrapper> rewritten;
  rewritten.reserve(ops.size());

  for (auto& op : ops) {
    if (!UnsignedOnlyOps().contains({op.GetOpCode(), backend})) {
      rewritten.emplace_back(std::move(op));
      continue;
    }

    std::vector<TensorWrapper*> unsigned_inputs;
    std::vector<TensorWrapper*> unsigned_outputs;
    bool needs_rewrite = false;
    for (size_t i = 0; i < op.GetInputCount(); ++i) {
      unsigned_inputs.emplace_back(
          CreateUnsignedTwin(tensor_pool, op.GetInputTensor(i)));
      needs_rewrite |= unsigned_inputs.back() != nullptr;
    }
    for (size_t i = 0; i < op.GetOutputCount(); ++i) {
      unsigned_outputs.emplace_back(
          CreateUnsignedTwin(tensor_pool, op.GetOutputTensor(i)));
      needs_rewrite |= unsigned_outputs.back() != nullptr;
    }
    if (!needs_rewrite) {
      rewritten.emplace_back(std::move(op));
      continue;
    }

    std::vector<const TensorWrapper*> original_inputs;
    std::vector<const TensorWrapper*> original_outputs;
    for (size_t i = 0; i < op.GetInputCount(); ++i) {
      original_inputs.emplace_back(&op.GetInputTensor(i));
    }
    for (size_t i = 0; i < op.GetOutputCount(); ++i) {
      original_outputs.emplace_back(&op.GetOutputTensor(i));
    }

    op.ClearInputOutputTensors();
    for (size_t i = 0; i < original_inputs.size(); ++i) {
      if (unsigned_inputs[i] != nullptr) {
        rewritten.emplace_back(
            CreateConvertOp(*original_inputs[i], *unsigned_inputs[i]));
        op.AddInputTensor(*unsigned_inputs[i]);
      } else {
        op.AddInputTensor(*original_inputs[i]);
      }
    }
    for (size_t i = 0; i < original_outputs.size(); ++i) {
      op.AddOutputTensor(unsigned_outputs[i] != nullptr ? *unsigned_outputs[i]
                                                        : *original_outputs[i]);
    }
    rewritten.emplace_back(std::move(op));

    for (size_t i = 0; i < original_outputs.size(); ++i) {
      if (unsigned_outputs[i] != nullptr) {
        rewritten.emplace_back(
            CreateConvertOp(*unsigned_outputs[i], *original_outputs[i]));
      }
    }
  }

  ops = std::move(rewritten);
}

}  // namespace qnn
