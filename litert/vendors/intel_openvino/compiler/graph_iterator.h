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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_COMPILER_GRAPH_ITERATOR_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_COMPILER_GRAPH_ITERATOR_H_

#include <memory>
#include <unordered_set>
#include <vector>

#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"
#include "openvino/frontend/tensorflow_lite/quantization_info.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/intel_openvino/compiler/decoder.h"
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
class GraphIteratorDelegate
    : public ov::frontend::tensorflow_lite::GraphIterator {
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
      iterator_indices_.output_index_++;
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
  std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase> get_decoder()
      const override;

  /// \brief Returns the number of sub-graphs that can be enumerated with
  /// get_subgraph
  size_t get_subgraph_size() const override { return 0; }

  /// \brief Returns iterator for a subgraph created on demand
  /// If there is no query for specific sub-graph iterator shouldn't be created
  /// idx should be in range 0..get_subgraph_size()-1
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> get_subgraph(
      size_t idx) const override {
    LITERT_LOG(LITERT_ERROR, "get_subgraph not implemented");
    return nullptr;
  };

 private:
  size_t node_index_ = 0;
  const litert::Subgraph* subgraph_ptr_;
  struct OVGraphIndices iterator_indices_;
};

}  // namespace openvino
}  // namespace litert

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_COMPILER_GRAPH_ITERATOR_H_
