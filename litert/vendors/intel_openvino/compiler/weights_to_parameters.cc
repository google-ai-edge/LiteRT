// Copyright 2026 Google LLC.
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

#include "litert/vendors/intel_openvino/compiler/weights_to_parameters.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

namespace litert::openvino {

// Constants at or below this size hold shape/control values that OpenVINO reads
// during compile-time shape inference (Transpose orders, axes, Reshape targets,
// scalars), so they must stay baked.
constexpr size_t kMinConvertBytes = 256;

size_t ConvertWeightsToParameters(const std::shared_ptr<ov::Model>& model,
                                  const WeightBank& bank,
                                  std::map<uint32_t, uint32_t>* const_map) {
  // Collect before mutating: replacing nodes while iterating get_ops() is unsafe.
  std::vector<std::shared_ptr<ov::op::v0::Constant>> to_promote;
  for (const auto& node : model->get_ops()) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) {
      continue;
    }
    if (!bank.BufferIdOfName(constant->get_friendly_name()).has_value()) {
      continue;
    }
    if (constant->get_byte_size() <= kMinConvertBytes) {
      continue;
    }
    to_promote.push_back(constant);
  }

  size_t promoted = 0;
  for (const auto& constant : to_promote) {
    const std::string name = constant->get_friendly_name();
    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        constant->get_element_type(), constant->get_output_partial_shape(0));
    parameter->set_friendly_name(name);
    parameter->get_output_tensor(0).set_names({name});
    ov::replace_node(constant, parameter);
    model->add_parameters({parameter});
    ++promoted;
  }

  // Map each promoted weight-Parameter to its BufferId by final input index.
  const auto& inputs = model->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(
        inputs[i].get_node_shared_ptr());
    if (!parameter) {
      continue;
    }
    const std::optional<int32_t> buffer_id =
        bank.BufferIdOfName(parameter->get_friendly_name());
    if (buffer_id.has_value()) {
      (*const_map)[static_cast<uint32_t>(i)] = static_cast<uint32_t>(*buffer_id);
    }
  }
  return promoted;
}

}  // namespace litert::openvino
