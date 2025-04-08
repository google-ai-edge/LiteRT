// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "graph_iterator.h"
#include "openvino/frontend/tensorflow_lite/quantization_info.hpp"

namespace tflite {
namespace openvinodelegate {
size_t GraphIteratorDelegate::size() const {
    // TODO: implement this
}

void GraphIteratorDelegate::reset() { node_index_ = 0; }

void GraphIteratorDelegate::next() { node_index_++; }

bool GraphIteratorDelegate::is_end() const {
    if (node_index_ == size())
        return true;
    else
        return false;
}

std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase> GraphIteratorDelegate::get_decoder()
    const {
    // TODO: implement this
    // Iterate through all nodes and i/o tensors from the subgraph
    // to create nodes and Tensor in OV frontend expectation
}

size_t GraphIteratorDelegate::get_subgraph_size() const { return 0; }
}  // namespace openvinodelegate
}  // namespace tflite
