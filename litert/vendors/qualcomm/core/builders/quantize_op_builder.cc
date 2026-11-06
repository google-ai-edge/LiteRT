// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"

#include <cstdint>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper CreateConvertOp(const TensorWrapper& input,
                          const TensorWrapper& output) {
  OpWrapper op(GetUniqueOpName(QNN_OP_CONVERT), QNN_OP_CONVERT,
               QnnOpCode::kConvert);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);
  return op;
}

OpWrapper CreateQuantizeOp(const TensorWrapper& input,
                           const TensorWrapper& output) {
  OpWrapper op(GetUniqueOpName(QNN_OP_QUANTIZE), QNN_OP_QUANTIZE,
               QnnOpCode::kQuantize);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);
  return op;
}

namespace {

std::vector<OpWrapper> BuildQuantizeOpLPAI(
    TensorPool&, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  if ((inputs[0].get().IsQuantI8() || inputs[0].get().IsQuantU8() ||
       inputs[0].get().IsQuantI16() || inputs[0].get().IsQuantU16()) &&
      (outputs[0].get().IsQuantI8() || outputs[0].get().IsQuantU8() ||
       outputs[0].get().IsQuantI16() || outputs[0].get().IsQuantU16())) {
    return MakeVector(CreateConvertOp(inputs[0], outputs[0]));
  }
  return MakeVector(CreateQuantizeOp(inputs[0], outputs[0]));
}

std::vector<OpWrapper> BuildQuantizeOpDefault(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  if (inputs[0].get().IsPerTensorQuantWithOffsetDiff(outputs[0].get())) {
    auto& op = CreateOpWrapper(res, QNN_OP_CAST);
    op.AddInputTensor(inputs[0]);
    op.AddOutputTensor(outputs[0]);
  } else if ((inputs[0].get().IsQuantI8() || inputs[0].get().IsQuantU8() ||
              inputs[0].get().IsQuantI16() || inputs[0].get().IsQuantU16()) &&
             (outputs[0].get().IsQuantI8() || outputs[0].get().IsQuantU8() ||
              outputs[0].get().IsQuantI16() || outputs[0].get().IsQuantU16())) {
    res.emplace_back(CreateConvertOp(inputs[0], outputs[0]));
    return res;
  } else {
    return MakeVector(CreateQuantizeOp(inputs[0], outputs[0]));
  }

  return res;
}

}  // namespace

std::vector<OpWrapper> BuildQuantizeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, BackendType backend_type) {
  if (backend_type == BackendType::kLpaiBackend) {
    return BuildQuantizeOpLPAI(tensor_pool, inputs, outputs);
  }
  return BuildQuantizeOpDefault(tensor_pool, inputs, outputs);
}

std::vector<OpWrapper> BuildDequantizeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  TensorWrapper& in = inputs[0].get();
  TensorWrapper& out = outputs[0].get();

  if (in.IsF16() && out.IsF32()) {
    // Fold constant fp16 -> fp32 cast into a static tensor.
    if (in.IsTensorStatic()) {
      const Qnn_ClientBuffer_t& buf = in.GetQnnTensor().v2.clientBuf;
      const uint32_t num_elements = buf.dataSize / sizeof(uint16_t);
      const uint16_t* src = reinterpret_cast<const uint16_t*>(buf.data);

      // fp16 has no native C++ type; bits are accessed as uint16_t.
      // Convert IEEE 754 fp16 bits to fp32: s[15] e[14:10] m[9:0].
      std::vector<float> fp32_data(num_elements);
      for (uint32_t i = 0; i < num_elements; ++i) {
        const uint32_t h = src[i];
        const uint32_t sign = (h & 0x8000u) << 16;
        const uint32_t exp = (h & 0x7C00u) >> 10;
        const uint32_t mantissa = h & 0x03FFu;
        uint32_t bits;
        if (exp == 0) {
          if (mantissa == 0) {
            bits = sign;
          } else {
            uint32_t e = 127 - 14;
            uint32_t m = mantissa << 1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
          }
        } else if (exp == 31) {
          bits = sign | 0x7F800000u | (mantissa << 13);  // Inf or NaN.
        } else {
          bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
        fp32_data[i] = *reinterpret_cast<const float*>(&bits);
      }

      out.GetQnnTensor().v2.type = QNN_TENSOR_TYPE_STATIC;
      out.SetTensorData<float>(absl::MakeSpan(fp32_data));
      return res;
    }

    auto& quantize_op = CreateOpWrapper(res, QNN_OP_CAST);
    quantize_op.AddInputTensor(in);
    quantize_op.AddOutputTensor(out);
    return res;
  }

  auto& quantize_op = CreateOpWrapper(res, QNN_OP_DEQUANTIZE);
  quantize_op.AddInputTensor(in);
  quantize_op.AddOutputTensor(out);
  return res;
}

}  // namespace qnn
