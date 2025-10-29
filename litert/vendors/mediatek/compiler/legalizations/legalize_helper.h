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

#include <cstdint>
#include <cstring>
#include <vector>

#include "neuron/api/NeuronAdapter.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"

namespace litert::mediatek {

inline ElementType GetElementType(const Tensor& tensor) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, tensor.RankedTensorType());
  return tensor_type.ElementType();
}

inline LiteRtStatus ConvertPaddingType(
    const uint32_t litert_padding, NeuronAdapterPaddingCode& neuron_padding) {
  switch (litert_padding) {
    case 0: {
      neuron_padding = NEURON_PADDING_SAME;
      break;
    }
    case 1: {
      neuron_padding = NEURON_PADDING_VALID;
      break;
    }
    default: {
      return kLiteRtStatusErrorUnsupported;
    }
  }
  return kLiteRtStatusOk;
}

inline Expected<uint32_t> AddZeroBiasForConvBase(const Tensor& input_tensor,
                                                 const Tensor& filter_tensor,
                                                 int32_t num_element,
                                                 OperandMap& operand_map) {
  std::vector<uint32_t> bias_shape = {static_cast<unsigned int>(num_element)};
  auto input_type = GetElementType(input_tensor);
  auto bias_neuron_type = (input_type == ElementType::Float32)
                              ? NEURON_TENSOR_FLOAT32
                              : NEURON_TENSOR_INT32;
  auto bias_data_size =
      num_element *
      ((input_type == ElementType::Float32) ? sizeof(float) : sizeof(int32_t));

  int32_t bias_extra_data_idx = -1;
  LITERT_ASSIGN_OR_RETURN(bias_extra_data_idx,
                          operand_map.RegisterExtraData(bias_data_size));
  memset(operand_map.GetExtraData(bias_extra_data_idx), 0, bias_data_size);
  return operand_map.AddTensorByType(
      bias_neuron_type, bias_shape,
      operand_map.GetExtraData(bias_extra_data_idx), bias_data_size);
}

//==============================================================================
// kLiteRtOpCodeTflSum
//==============================================================================
inline Expected<uint32_t> AddSumKeepDimsOption(const litert::Op& op,
                                               OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_SUM.
  bool keepdim = false;
  LITERT_RETURN_IF_ERROR(LiteRtGetSumKeepDimsOption(op.Get(), &keepdim))
      << "Fails to get SumKeepDims";
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
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dPaddingOption(op.Get(), &padding))
      << "Fails to get Conv2dPadding";
  NeuronAdapterPaddingCode neuron_padding = NEURON_PADDING_SAME;
  LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, neuron_padding))
      << "Fails to convert padding";
  return operand_map.AddScalarInt32(neuron_padding);
}

inline Expected<uint32_t> AddConv2dStrideWOption(const litert::Op& op,
                                                 OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t stride_w = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideWOption(op.Get(), &stride_w))
      << "Fails to get Conv2dStrideW";
  return operand_map.AddScalarInt32(stride_w);
}

inline Expected<uint32_t> AddConv2dStrideHOption(const litert::Op& op,
                                                 OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t stride_h = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideHOption(op.Get(), &stride_h))
      << "Fails to get Conv2dStrideH";
  return operand_map.AddScalarInt32(stride_h);
}

inline Expected<uint32_t> AddConv2dFuseActivationOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  uint32_t fuse = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dFusedActivationOption(op.Get(), &fuse))
      << "Fails to get Conv2dFuseActivation";
  return operand_map.AddScalarInt32(fuse);
}

inline Expected<uint32_t> AddConv2dDilationWOption(const litert::Op& op,
                                                   OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t dilation_w = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dDilationWOption(op.Get(), &dilation_w))
      << "Fails to get Conv2dDilationW";
  return operand_map.AddScalarInt32(dilation_w);
}

inline Expected<uint32_t> AddConv2dDilationHOption(const litert::Op& op,
                                                   OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t dilation_h = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dDilationHOption(op.Get(), &dilation_h))
      << "Fails to get Conv2dDilationH";
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
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dPaddingOption(op.Get(), &padding))
      << "Fails to get DepthwiseConv2dPadding";
  NeuronAdapterPaddingCode neuron_padding = NEURON_PADDING_SAME;
  LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, neuron_padding))
      << "Fails to convert padding";
  return operand_map.AddScalarInt32(neuron_padding);
}

inline Expected<uint32_t> AddDepthwiseConv2dStrideWOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t stride_w = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dStrideWOption(op.Get(), &stride_w))
      << "Fails to get DepthwiseConv2dStrideW";
  return operand_map.AddScalarInt32(stride_w);
}

inline Expected<uint32_t> AddDepthwiseConv2dStrideHOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t stride_h = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dStrideHOption(op.Get(), &stride_h))
      << "Fails to get DepthwiseConv2dStrideH";
  return operand_map.AddScalarInt32(stride_h);
}

inline Expected<uint32_t> AddDepthwiseConv2dDepthMultiplierOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t depth_multiplier = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dDepthMultiplierOption(
      op.Get(), &depth_multiplier))
      << "Fails to get DepthwiseConv2dDepthMultiplier";
  return operand_map.AddScalarInt32(depth_multiplier);
}

inline Expected<uint32_t> AddDepthwiseConv2dFuseActivationOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  uint32_t fuse = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dFusedActivationOption(op.Get(), &fuse))
      << "Fails to get DepthwiseConv2dFuseActivation";
  return operand_map.AddScalarInt32(fuse);
}

inline Expected<uint32_t> AddDepthwiseConv2dDilationWOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t dilation_w = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDilationWOption(op.Get(), &dilation_w))
      << "Fails to get DepthwiseConv2dDilationW";
  return operand_map.AddScalarInt32(dilation_w);
}

inline Expected<uint32_t> AddDepthwiseConv2dDilationHOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  int32_t dilation_h = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDilationHOptions(op.Get(), &dilation_h))
      << "Fails to get DepthwiseConv2dDilationH";
  return operand_map.AddScalarInt32(dilation_h);
}

inline Expected<uint32_t> AddDepthwiseConv2dDataOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_CONV2D.
  return operand_map.AddScalarBool(false);
}

//==============================================================================
// kLiteRtOpCodeTflAveragePool2d
//==============================================================================
inline Expected<uint32_t> AddAveragePool2dPaddingOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_AVERAGE_POOL_2D.
  uint32_t padding = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dPaddingOption(op.Get(), &padding))
      << "Fails to get AveragePool2dPadding";
  NeuronAdapterPaddingCode neuron_padding = NEURON_PADDING_SAME;
  LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, neuron_padding))
      << "Fails to convert padding";
  return operand_map.AddScalarInt32(neuron_padding);
}

inline Expected<uint32_t> AddAveragePool2dStrideWOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_AVERAGE_POOL_2D.
  int32_t stride_w = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dStrideWOption(op.Get(), &stride_w))
      << "Fails to get AveragePool2dStrideW";
  return operand_map.AddScalarInt32(stride_w);
}

inline Expected<uint32_t> AddAveragePool2dStrideHOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_AVERAGE_POOL_2D.
  int32_t stride_h = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dStrideHOption(op.Get(), &stride_h))
      << "Fails to get AveragePool2dStrideH";
  return operand_map.AddScalarInt32(stride_h);
}

inline Expected<uint32_t> AddAveragePool2dFuseActivationOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_AVERAGE_POOL_2D.
  uint32_t fuse = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dFusedActivationOption(op.Get(), &fuse))
      << "Fails to get AveragePool2dFuseActivation";
  return operand_map.AddScalarInt32(fuse);
}

inline Expected<uint32_t> AddAveragePool2dFilterWOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_AVERAGE_POOL_2D.
  int32_t filter_w = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dFilterWidthOption(op.Get(), &filter_w))
      << "Fails to get AveragePool2dFilterW";
  return operand_map.AddScalarInt32(filter_w);
}

inline Expected<uint32_t> AddAveragePool2dFilterHOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_AVERAGE_POOL_2D.
  int32_t filter_h = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dFilterHeightOption(op.Get(), &filter_h))
      << "Fails to get AveragePool2dFilterH";
  return operand_map.AddScalarInt32(filter_h);
}

//==============================================================================
// kLiteRtOpCodeTflMaxPool2d
//==============================================================================
inline Expected<uint32_t> AddMaxPool2dPaddingOption(const litert::Op& op,
                                                    OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_MAX_POOL_2D.
  uint32_t padding = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dPaddingOption(op.Get(), &padding))
      << "Fails to get MaxPool2dPadding";
  NeuronAdapterPaddingCode neuron_padding = NEURON_PADDING_SAME;
  LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, neuron_padding))
      << "Fails to convert padding";
  return operand_map.AddScalarInt32(neuron_padding);
}

inline Expected<uint32_t> AddMaxPool2dStrideWOption(const litert::Op& op,
                                                    OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_MAX_POOL_2D.
  int32_t stride_w = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dStrideWOption(op.Get(), &stride_w))
      << "Fails to get MaxPool2dStrideW";
  return operand_map.AddScalarInt32(stride_w);
}

inline Expected<uint32_t> AddMaxPool2dStrideHOption(const litert::Op& op,
                                                    OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_MAX_POOL_2D.
  int32_t stride_h = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dStrideHOption(op.Get(), &stride_h))
      << "Fails to get MaxPool2dStrideH";
  return operand_map.AddScalarInt32(stride_h);
}

inline Expected<uint32_t> AddMaxPool2dFilterWOption(const litert::Op& op,
                                                    OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_MAX_POOL_2D.
  int32_t filter_w = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFilterWidthOption(op.Get(), &filter_w))
      << "Fails to get MaxPool2dFilterW";
  return operand_map.AddScalarInt32(filter_w);
}

inline Expected<uint32_t> AddMaxPool2dFilterHOption(const litert::Op& op,
                                                    OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_MAX_POOL_2D.
  int32_t filter_h = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFilterHeightOption(op.Get(), &filter_h))
      << "Fails to get MaxPool2dFilterH";
  return operand_map.AddScalarInt32(filter_h);
}

inline Expected<uint32_t> AddMaxPool2dFuseActivationOption(
    const litert::Op& op, OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_MAX_POOL_2D.
  uint32_t fuse = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFusedActivationOption(op.Get(), &fuse))
      << "Fails to get MaxPool2dFuseActivation";
  return operand_map.AddScalarInt32(fuse);
}

//==============================================================================
// kLiteRtOpCodeTflReduceMax
//==============================================================================
inline Expected<uint32_t> AddReduceMaxKeepDimsOption(const litert::Op& op,
                                                     OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_REDUCE_MAX.
  bool keepdim = false;
  LITERT_RETURN_IF_ERROR(LiteRtGetReduceMaxKeepDimsOption(op.Get(), &keepdim))
      << "Fails to get ReduceMaxKeepDims";
  return operand_map.AddScalarBool(keepdim);
}

//==============================================================================
// kLiteRtOpCodeTflDiv
//==============================================================================
inline Expected<uint32_t> AddDivFuseActivationOption(const litert::Op& op,
                                                     OperandMap& operand_map) {
  // Note that return type should be same as the NEURON parameters needs for
  // NEURON_DIV.
  uint32_t fuse = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetDivFusedActivationOption(op.Get(), &fuse))
      << "Fails to get DivFuseActivation";
  return operand_map.AddScalarInt32(fuse);
}

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_LEGALIZE_HELPER_H_
