// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <unordered_set>
#include <vector>
#include <memory>

#include <openvino/frontend/tensorflow_lite/decoder.hpp>
#include <openvino/frontend/tensorflow_lite/graph_iterator.hpp>
#include <openvino/frontend/tensorflow_lite/quantization_info.hpp>
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_model.h"
#include "litert/core/model/model.h"

#include "decoder.h"

using namespace litert;

ov::element::Type MapLiteTypeToOV(const ElementType element_type) {
    ov::element::Type ov_type;
    switch (element_type) {
        case ElementType::Bool:
            ov_type = ov::element::boolean;
            break;
        case ElementType::Int4:
            ov_type = ov::element::i4;
            break;
        case ElementType::Int8:
            ov_type = ov::element::i8;
            break;
        case ElementType::Int16:
            ov_type = ov::element::i16;
            break;
        case ElementType::Int32:
            ov_type = ov::element::i32;
            break;
        case ElementType::Int64:
            ov_type = ov::element::i64;
            break;
        case ElementType::UInt8:
            ov_type = ov::element::u8;
            break;
        case ElementType::UInt16:
            ov_type = ov::element::u16;
            break;
        case ElementType::UInt32:
            ov_type = ov::element::u32;
            break;
        case ElementType::UInt64:
            ov_type = ov::element::u64;
            break;
        case ElementType::Float16:
            ov_type = ov::element::f16;
            break;
        case ElementType::Float32:
            ov_type = ov::element::f32;
            break;
        case ElementType::Float64:
            ov_type = ov::element::f64;
            break;
        case ElementType::BFloat16:
            ov_type = ov::element::bf16;
            break;
        default:
            ov_type = ov::element::undefined;
    }
    return ov_type;
}

LiteRtStatus GetOVTensorShape(const litert::Tensor& litert_tensor,
                              std::vector<int64_t>& ov_shape_vec) {
    if (litert_tensor.TypeId() != kLiteRtRankedTensorType) return kLiteRtStatusErrorInvalidArgument;

    const auto ranked_tensor_type = litert_tensor.RankedTensorType();
    if (!ranked_tensor_type) {
        LITERT_LOG(LITERT_ERROR, "%s", ranked_tensor_type.Error().Message().data());
        return ranked_tensor_type.Error().Status();
    }

    const auto tensor_layout = ranked_tensor_type->Layout();
    if (tensor_layout.Rank() == 0)
        return kLiteRtStatusErrorUnsupported;
    else {
        ov_shape_vec.resize(tensor_layout.Rank());
        for (int i = 0; i < ov_shape_vec.size(); i++)
            ov_shape_vec[i] = tensor_layout.Dimensions()[i];
    }
    return kLiteRtStatusOk;
}

namespace litert {
namespace openvino {

struct OVGraphIndices {
    int32_t input_index_ = 0;
    int32_t output_index_ = 0;
    int32_t const_index_ = 0;
    int32_t op_index_ = 0;
};

// GraphIteratorDelegate traverses through the graph/subgraph i/o's and ops.
// Objective of this class is to create TensorMetaInfo structures to pass to
// DecoderTensor to manage I/Os and fill the op specific information
// in DecoderOperation objects. OpenVINO tensorflow lite frontend takes
// the responsibility for creating OV op nodes.
class GraphIteratorDelegate : public ov::frontend::tensorflow_lite::GraphIterator {
public:
    GraphIteratorDelegate(const litert::Subgraph* graph) : subgraph_ptr_(graph) {
        for (const auto& input : subgraph_ptr_->Inputs()) {
            if (input.IsSubgraphInput()) {
                iterator_indices_.input_index_++;
            } else if (input.IsConstant()) {
                iterator_indices_.const_index_++;
            }
        }
        for (const auto& output : subgraph_ptr_->Outputs()) {
            if (output.IsSubgraphOutput()) {
                iterator_indices_.output_index_++;
            }
        }
        for (const auto& op : subgraph_ptr_->Ops()) {
            iterator_indices_.op_index_++;
        }
    }

    ~GraphIteratorDelegate() = default;

    /// \brief Get a number of operation nodes in the graph
    size_t size() const override;

    /// \brief Set iterator to the start position
    void reset() override;

    /// \brief Move to the next node in the graph
    void next() override;

    /// \brief Returns true if iterator goes out of the range of available nodes
    bool is_end() const override;

    /// \brief Return a pointer to a decoder of the current node
    std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase> get_decoder() const override;

    /// \brief Returns the number of sub-graphs that can be enumerated with
    /// get_subgraph
    size_t get_subgraph_size() const override { return 0; }

    /// \brief Returns iterator for a subgraph created on demand
    /// If there is no query for specific sub-graph iterator shouldn't be created
    /// idx should be in range 0..get_subgraph_size()-1
    std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> get_subgraph(
        size_t idx) const override{};

private:
    size_t node_index_ = 0;
    const litert::Subgraph* subgraph_ptr_;
    struct OVGraphIndices iterator_indices_;
};

}  // namespace openvino
}  // namespace litert
