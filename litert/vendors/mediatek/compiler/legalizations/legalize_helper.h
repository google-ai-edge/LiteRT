// Copyright (c) 2025 MediaTek Inc.
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
#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_LEGALIZE_HELPER_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_LEGALIZE_HELPER_H_

#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"

#define CHECK_RT_STATUS(function_call, error_message) \
  do {                                                \
    auto status = function_call;                      \
    if (status != kLiteRtStatusOk) {                  \
      return Error(status, error_message);            \
    }                                                 \
  } while (0)

#define GET_ELEMENT_TYPE(op) ((op).RankedTensorType()->ElementType())

namespace litert::mediatek {

inline Expected<uint32_t> AddZeroBiasForConvBase(const Tensor& input_tensor,
                                          const Tensor& filter_tensor,
                                          int32_t num_element,
                                          OperandMap& operand_map) {
  std::vector<uint32_t> bias_shape = {static_cast<unsigned int>(num_element)};
  auto input_type = GET_ELEMENT_TYPE(input_tensor);
  auto bias_neuron_type = (input_type == ElementType::Float32)
                              ? NEURON_TENSOR_FLOAT32
                              : NEURON_TENSOR_INT32;
  auto bias_data_size =
      (input_type == ElementType::Float32) ? sizeof(float) : sizeof(int32_t);
  std::vector<uint8_t> bias_data(num_element * bias_data_size, 0);
  return operand_map.AddTensorByType(bias_neuron_type, bias_shape,
                                     bias_data.data(), bias_data.size());
}

//==============================================================================
// kLiteRtOpCodeTflSum
//==============================================================================
inline Expected<uint32_t> AddSumKeepDimsOption(const litert::Op& op,
                                               OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_SUM.
  bool keepdim = 0;
  CHECK_RT_STATUS(LiteRtGetSumKeepDimsOption(op.Get(), &keepdim),
                  "Fails to get SumKeepDims");
  return operand_map.AddScalarBool(keepdim);
}

//==============================================================================
// kLiteRtOpCodeTflConv2d
//==============================================================================
inline Expected<uint32_t> AddConv2dPaddingOption(const litert::Op& op,
                                                 OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  uint32_t padding = 0;
  CHECK_RT_STATUS(LiteRtGetConv2dPaddingOption(op.Get(), &padding),
                  "Fails to get Conv2dPadding");
  return operand_map.AddScalarInt32(padding);
}

inline Expected<uint32_t> AddConv2dStrideWOption(const litert::Op& op,
                                                 OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t stride_w = 0;
  CHECK_RT_STATUS(LiteRtGetConv2dStrideWOption(op.Get(), &stride_w),
                  "Fails to get Conv2dStrideW");
  return operand_map.AddScalarInt32(stride_w);
}

inline Expected<uint32_t> AddConv2dStrideHOption(const litert::Op& op,
                                                 OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t stride_h = 0;
  CHECK_RT_STATUS(LiteRtGetConv2dStrideHOption(op.Get(), &stride_h),
                  "Fails to get Conv2dStrideH");
  return operand_map.AddScalarInt32(stride_h);
}

inline Expected<uint32_t> AddConv2dFuseActivationOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  uint32_t fuse = 0;
  CHECK_RT_STATUS(LiteRtGetConv2dFusedActivationOption(op.Get(), &fuse),
                  "Fails to get Conv2dFuseActivation");
  return operand_map.AddScalarInt32(fuse);
}

inline Expected<uint32_t> AddConv2dDilationWOption(const litert::Op& op,
                                                   OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t dilation_w = 0;
  CHECK_RT_STATUS(LiteRtGetConv2dDilationWOption(op.Get(), &dilation_w),
                  "Fails to get Conv2dDilationW");
  return operand_map.AddScalarInt32(dilation_w);
}

inline Expected<uint32_t> AddConv2dDilationHOption(const litert::Op& op,
                                                   OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t dilation_h = 0;
  CHECK_RT_STATUS(LiteRtGetConv2dDilationHOption(op.Get(), &dilation_h),
                  "Fails to get Conv2dDilationH");
  return operand_map.AddScalarInt32(dilation_h);
}

inline Expected<uint32_t> AddConv2dDataOption(const litert::Op& op,
                                              OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  return operand_map.AddScalarBool(false);
}

//==============================================================================
// kLiteRtOpCodeTflDepthwiseConv2d
//==============================================================================
inline Expected<uint32_t> AddDepthwiseConv2dPaddingOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  uint32_t padding = 0;
  CHECK_RT_STATUS(LiteRtGetDepthwiseConv2dPaddingOption(op.Get(), &padding),
                  "Fails to get DepthwiseConv2dPadding");
  return operand_map.AddScalarInt32(padding);
}

inline Expected<uint32_t> AddDepthwiseConv2dStrideWOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t stride_w = 0;
  CHECK_RT_STATUS(LiteRtGetDepthwiseConv2dStrideWOption(op.Get(), &stride_w),
                  "Fails to get DepthwiseConv2dStrideW");
  return operand_map.AddScalarInt32(stride_w);
}

inline Expected<uint32_t> AddDepthwiseConv2dStrideHOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t stride_h = 0;
  CHECK_RT_STATUS(LiteRtGetDepthwiseConv2dStrideHOption(op.Get(), &stride_h),
                  "Fails to get DepthwiseConv2dStrideH");
  return operand_map.AddScalarInt32(stride_h);
}

inline Expected<uint32_t> AddDepthwiseConv2dDepthMultiplierOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t depth_multiplier = 0;
  CHECK_RT_STATUS(LiteRtGetDepthwiseConv2dDepthMultiplierOption(
                      op.Get(), &depth_multiplier),
                  "Fails to get DepthwiseConv2dDepthMultiplier");
  return operand_map.AddScalarInt32(depth_multiplier);
}

inline Expected<uint32_t> AddDepthwiseConv2dFuseActivationOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  uint32_t fuse = 0;
  CHECK_RT_STATUS(
      LiteRtGetDepthwiseConv2dFusedActivationOption(op.Get(), &fuse),
      "Fails to get DepthwiseConv2dFuseActivation");
  return operand_map.AddScalarInt32(fuse);
}

inline Expected<uint32_t> AddDepthwiseConv2dDilationWOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t dilation_w = 0;
  CHECK_RT_STATUS(
      LiteRtGetDepthwiseConv2dDilationWOption(op.Get(), &dilation_w),
      "Fails to get DepthwiseConv2dDilationW");
  return operand_map.AddScalarInt32(dilation_w);
}

inline Expected<uint32_t> AddDepthwiseConv2dDilationHOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t dilation_h = 0;
  CHECK_RT_STATUS(
      LiteRtGetDepthwiseConv2dDilationHOptions(op.Get(), &dilation_h),
      "Fails to get DepthwiseConv2dDilationH");
  return operand_map.AddScalarInt32(dilation_h);
}

inline Expected<uint32_t> AddDepthwiseConv2dDataOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  return operand_map.AddScalarBool(false);
}

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_LEGALIZE_HELPER_H_
