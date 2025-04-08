// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <unordered_set>

#include "litert/cc/litert_model.h"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"

namespace litert {
namespace openvino {

class GraphIteratorDelegate : public ov::frontend::tensorflow_lite::GraphIterator {
public:
    GraphIteratorDelegate(Subgraph graph);

    ~GraphIteratorDelegate() = default;

    std::vector<int> get_compute_inputs() { return input_nodes_; }
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
    size_t get_subgraph_size() const override;

    /// \brief Returns iterator for a subgraph created on demand
    /// If there is no query for specific sub-graph iterator shouldn't be created
    /// idx should be in range 0..get_subgraph_size()-1
    std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> get_subgraph(
        size_t idx) const override{};

private:
    size_t node_index_ = 0;
};

}  // namespace openvino
}  // namespace litert
