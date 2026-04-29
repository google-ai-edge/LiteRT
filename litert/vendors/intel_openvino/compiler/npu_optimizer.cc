// Copyright (C) 2026 Intel Corporation
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

#include "litert/vendors/intel_openvino/compiler/npu_optimizer.h"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace litert {
namespace openvino {

EliminateMatMulFakeQuantize::EliminateMatMulFakeQuantize() {
  namespace pattern = ov::pass::pattern;
  auto matmul_pattern = pattern::wrap_type<ov::op::v0::MatMul>(
      {pattern::any_input(), pattern::any_input()},
      pattern::consumers_count(1));
  auto fq_pattern = pattern::wrap_type<ov::op::v0::FakeQuantize>(
      {matmul_pattern, pattern::any_input(), pattern::any_input(),
       pattern::any_input(), pattern::any_input()});

  ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
    auto pattern_map = m.get_pattern_value_map();
    auto matmul = pattern_map[matmul_pattern];
    auto fq = pattern_map[fq_pattern].get_node_shared_ptr();

    ov::copy_runtime_info(fq, matmul.get_node_shared_ptr());
    ov::replace_node(fq, matmul.get_node_shared_ptr());
    return true;
  };

  auto m = std::make_shared<pattern::Matcher>(fq_pattern,
                                              "EliminateMatMulFakeQuantize");
  register_matcher(m, callback);
}

CastIntegerSignToFloat::CastIntegerSignToFloat() {
  namespace pattern = ov::pass::pattern;
  auto sign_pattern = pattern::wrap_type<ov::op::v0::Sign>(
      {pattern::any_input()},
      [](const ov::Output<ov::Node>& output) {
        const auto node = output.get_node_shared_ptr();
        // Match Sign nodes whose input element type is a non-real (integer)
        // type. Float types are already supported by the NPU plugin.
        return !node->get_input_element_type(0).is_real();
      });

  ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
    auto pattern_map = m.get_pattern_value_map();
    auto sign_node = std::dynamic_pointer_cast<ov::op::v0::Sign>(
        pattern_map[sign_pattern].get_node_shared_ptr());
    if (!sign_node) {
      return false;
    }

    const auto original_int_type = sign_node->get_output_element_type(0);
    auto to_fp32 = std::make_shared<ov::op::v0::Convert>(
        sign_node->input_value(0), ov::element::f32);
    auto sign_fp32 = std::make_shared<ov::op::v0::Sign>(to_fp32);
    auto cast_back =
        std::make_shared<ov::op::v0::Convert>(sign_fp32, original_int_type);

    cast_back->set_friendly_name(sign_node->get_friendly_name());
    ov::copy_runtime_info(sign_node, {to_fp32, sign_fp32, cast_back});
    ov::replace_node(sign_node, cast_back);
    return true;
  };

  auto m = std::make_shared<pattern::Matcher>(sign_pattern,
                                              "CastIntegerSignToFloat");
  register_matcher(m, callback);
}

void NpuOptimizer::Run(const std::shared_ptr<ov::Model>& model) const {
  ov::pass::Manager pass_manager;
  if (cast_integer_sign_to_float_) {
    pass_manager.register_pass<CastIntegerSignToFloat>();
  }
  if (eliminate_matmul_fq_) {
    pass_manager.register_pass<EliminateMatMulFakeQuantize>();
  }
  pass_manager.run_passes(model);
}

}  // namespace openvino
}  // namespace litert
