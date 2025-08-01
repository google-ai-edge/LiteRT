// Copyright (C) 2025 Intel Corporation
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

#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"

#include <string>

#include "litert/vendors/intel_openvino/utils.h"

namespace litert {
namespace openvino {

size_t GraphIteratorDelegate::size() const {
  return iterator_indices_.input_index_ + iterator_indices_.output_index_ +
         iterator_indices_.op_index_;
}

void GraphIteratorDelegate::reset() { node_index_ = 0; }

void GraphIteratorDelegate::next() { node_index_++; }

bool GraphIteratorDelegate::is_end() const {
  if (node_index_ == size())
    return true;
  else
    return false;
}

bool fill_tensor_meta(
    ov::frontend::tensorflow_lite::TensorMetaInfo& tensor_meta_info,
    const litert::Tensor& litert_tensor) {
  std::vector<int64_t> shape_vec;
  const ElementType type = litert_tensor.ElementType();
  ov::element::Type ov_element_type =
      MapLiteTypeToOV(static_cast<LiteRtElementType>(type));
  if (GetOVTensorShape(litert_tensor, shape_vec) != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "Unsupported tensor element shape");
  }
  if (litert_tensor.QTypeId() == kLiteRtQuantizationPerTensor) {
    const auto& quantization = litert_tensor.PerTensorQuantization();
    auto ov_quantization =
        std::make_shared<ov::frontend::tensorflow_lite::QuantizationInfo>();
    ov_quantization->set_scale({quantization.scale});
    ov_quantization->set_zero_point({quantization.zero_point});
    tensor_meta_info.m_quantization_info = ov_quantization;
  } else if (litert_tensor.QTypeId() == kLiteRtQuantizationPerChannel) {
    const auto& quantization = litert_tensor.PerChannelQuantization();
    auto ov_quantization =
        std::make_shared<ov::frontend::tensorflow_lite::QuantizationInfo>();
    std::vector<float> scale_vec(quantization.num_channels, 0);
    std::vector<int64_t> zero_point_vec(quantization.num_channels, 0);
    for (int i = 0; i < quantization.num_channels; i++) {
      scale_vec[i] = quantization.scales[i];
      zero_point_vec[i] = quantization.zero_points[i];
    }
    ov_quantization->set_scale(scale_vec);
    ov_quantization->set_zero_point(zero_point_vec);
    ov_quantization->set_axis(quantization.quantized_dimension);
    tensor_meta_info.m_quantization_info = ov_quantization;
  } else if (litert_tensor.QTypeId() != kLiteRtQuantizationNone) {
    LITERT_LOG(LITERT_ERROR, "Unsupported Quantization type %d ",
               litert_tensor.QTypeId());
    return false;
  }
  ov::PartialShape tensor_shape{shape_vec};
  tensor_meta_info.m_partial_shape = tensor_shape;
  tensor_meta_info.m_element_type = ov_element_type;
  tensor_meta_info.m_tensor_name = std::string(litert_tensor.Name());
  return true;
}

std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>
GraphIteratorDelegate::get_decoder() const {
  ov::frontend::tensorflow_lite::TensorMetaInfo tensor_meta_info;
  if (node_index_ < iterator_indices_.input_index_) {
    const auto& input_vec = subgraph_ptr_->Inputs();
    const auto& input = input_vec[node_index_];
    int64_t input_index = node_index_;
    int64_t output_index = -1;
    if (!fill_tensor_meta(tensor_meta_info, input)) {
      return nullptr;
    }

    return std::make_shared<litert::openvino::DecoderTensor>(
        tensor_meta_info, input_index, output_index);
  } else if (node_index_ >= iterator_indices_.input_index_ &&
             node_index_ < iterator_indices_.input_index_ +
                               iterator_indices_.output_index_) {
    const auto& output_vec = subgraph_ptr_->Outputs();
    const auto& output =
        output_vec[node_index_ - iterator_indices_.input_index_];
    int64_t input_index = -1;
    int64_t output_index = node_index_;
    if (!fill_tensor_meta(tensor_meta_info, output)) {
      return nullptr;
    }
    return std::make_shared<litert::openvino::DecoderTensor>(
        tensor_meta_info, input_index, output_index);
  } else {
    const auto& op_vec = subgraph_ptr_->Ops();
    const auto& op = op_vec[node_index_ - iterator_indices_.input_index_ -
                            iterator_indices_.output_index_];
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> input_meta_info;
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> output_meta_info;
    for (const auto& input : op.Inputs()) {
      tensor_meta_info = {};
      if (!fill_tensor_meta(tensor_meta_info, input)) {
        return nullptr;
      }
      if (input.HasWeights()) {
        LITERT_LOG(LITERT_VERBOSE, "Data is static or constant for op %d",
                   op.Code());
        tensor_meta_info.m_tensor_data = input.Weights().Bytes().data();
        if (op.Code() != LiteRtOpCode::kLiteRtOpCodeTflConv2d &&
            op.Code() != LiteRtOpCode::kLiteRtOpCodeTflDepthwiseConv2d &&
            op.Code() != LiteRtOpCode::kLiteRtOpCodeTflMul &&
            op.Code() != LiteRtOpCode::kLiteRtOpCodeTflAdd &&
            op.Code() != LiteRtOpCode::kLiteRtOpCodeTflFullyConnected)
          tensor_meta_info.m_quantization_info = nullptr;
      }
      input_meta_info.push_back(tensor_meta_info);
    }
    for (const auto& output : op.Outputs()) {
      tensor_meta_info = {};
      if (!fill_tensor_meta(tensor_meta_info, output)) {
        return nullptr;
      }
      output_meta_info.push_back(tensor_meta_info);
    }
    return std::make_shared<litert::openvino::DecoderOperation>(
        input_meta_info, output_meta_info, op, node_index_);
  }
}

}  // namespace openvino
}  // namespace litert
