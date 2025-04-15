// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "graph_iterator.h"

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

void fill_tensor_meta(ov::frontend::tensorflow_lite::TensorMetaInfo& tensor_meta_info,
                      std::vector<int64_t> shape_vec, ov::element::Type ov_element_type,
                      std::string name) {
    ov::PartialShape tensor_shape{shape_vec};
    tensor_meta_info.m_partial_shape = tensor_shape;
    tensor_meta_info.m_element_type = ov_element_type;
    tensor_meta_info.m_tensor_name = name;
}

std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase> GraphIteratorDelegate::get_decoder()
    const {
    ov::frontend::tensorflow_lite::TensorMetaInfo tensor_meta_info;
    std::vector<int64_t> shape_vec;
    if (node_index_ < iterator_indices_.input_index_) {
        const auto& input_vec = subgraph_ptr_->Inputs();
        const auto& input = input_vec[node_index_];
        const ElementType type = input.ElementType();
        ov::element::Type ov_element_type = MapLiteTypeToOV(type);
        if (GetOVTensorShape(input, shape_vec) != kLiteRtStatusOk) {
            LITERT_LOG(LITERT_INFO, "Unsupported tensor element shape");
        }
        int64_t input_index = node_index_;
        int64_t output_index = -1;
        fill_tensor_meta(tensor_meta_info, shape_vec, ov_element_type, std::string(input.Name()));

        return std::make_shared<litert::openvino::DecoderTensor>(tensor_meta_info, input_index, output_index);
    } else if (node_index_ >= iterator_indices_.input_index_ &&
               node_index_ < iterator_indices_.input_index_ + iterator_indices_.output_index_) {
        const auto& output_vec = subgraph_ptr_->Outputs();
        const auto& output = output_vec[node_index_ - iterator_indices_.input_index_];
        int64_t input_index = -1;
        int64_t output_index = node_index_;
        const ElementType type = output.ElementType();
        ov::element::Type ov_element_type = MapLiteTypeToOV(type);
        if (GetOVTensorShape(output, shape_vec) != kLiteRtStatusOk) {
            LITERT_LOG(LITERT_INFO, "Unsupported tensor element shape");
        }
        fill_tensor_meta(tensor_meta_info, shape_vec, ov_element_type, std::string(output.Name()));
        return std::make_shared<litert::openvino::DecoderTensor>(tensor_meta_info, input_index, output_index);
    } else {
        const auto& op_vec = subgraph_ptr_->Ops();
        const auto& op =
            op_vec[node_index_ - iterator_indices_.input_index_ - iterator_indices_.output_index_];
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> input_meta_info;
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> output_meta_info;
        for (const auto& input : op.Inputs()) {
            const ElementType type = input.ElementType();
            ov::element::Type ov_element_type = MapLiteTypeToOV(type);
            shape_vec.clear();
            if (GetOVTensorShape(input, shape_vec) != kLiteRtStatusOk) {
                LITERT_LOG(LITERT_INFO, "Unsupported tensor element shape for op creation");
            }
            fill_tensor_meta(tensor_meta_info, shape_vec, ov_element_type,
                             std::string(input.Name()));
            if (input.HasWeights()) {
                LITERT_LOG(LITERT_VERBOSE, "Data is static or constant");
                tensor_meta_info.m_tensor_data = input.Weights().Bytes().data();
            }
            input_meta_info.push_back(tensor_meta_info);
        }
        for (const auto& output : op.Outputs()) {
            const ElementType type = output.ElementType();
            ov::element::Type ov_element_type = MapLiteTypeToOV(type);
            shape_vec.clear();
            if (GetOVTensorShape(output, shape_vec) != kLiteRtStatusOk) {
                LITERT_LOG(LITERT_INFO, "Unsupported tensor element shape for op creation");
            }
            fill_tensor_meta(tensor_meta_info, shape_vec, ov_element_type,
                             std::string(output.Name()));
            output_meta_info.push_back(tensor_meta_info);
        }
        return std::make_shared<litert::openvino::DecoderOperation>(input_meta_info, output_meta_info, op,
                                                  node_index_);
    }
}

}  // namespace openvino
}  // namespace litert
