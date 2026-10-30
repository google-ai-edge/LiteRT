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

#include "litert/runtime/tfl_utils.h"

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_dispatch_delegate_vendor_options.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

TEST(TflUtilsTest, BuildDispatchDelegateVendorOptionsEmptySignatures) {
  LiteRtModelT model;
  LITERT_ASSERT_OK_AND_ASSIGN(
      DispatchDelegateVendorOptions options,
      BuildDispatchDelegateVendorOptions(model));
  EXPECT_TRUE(options.FunctionMappings().empty());
}

TEST(TflUtilsTest, BuildDispatchDelegateVendorOptionsSuccess) {
  LiteRtModelT model;

  // Subgraph 0: contains a non-custom op and a dispatch op with input/output
  // tensors.
  LiteRtSubgraphT& sg0 = model.EmplaceSubgraph();
  LiteRtOpT& non_custom_op = sg0.EmplaceOp();
  non_custom_op.SetOpCode(kLiteRtOpCodeTflAdd);

  LiteRtOpT& op0 = sg0.EmplaceOp();
  op0.SetOpCode(kLiteRtOpCodeTflCustom);
  op0.SetCustomCode(std::string(kLiteRtDispatchOpCustomName));
  DispatchOpOptions op_opts0{};
  op_opts0.name = "vision_func";
  op0.SetCustomOptions(MakeDispatchOpOptions(std::move(op_opts0)));

  LiteRtTensorT& in0 = sg0.EmplaceTensor();
  in0.SetName("raw_image_input");
  op0.Inputs().push_back(&in0);

  // An intermediate tensor not registered in the signature.
  LiteRtTensorT& in0_internal = sg0.EmplaceTensor();
  in0_internal.SetName("internal_temp_tensor");
  op0.Inputs().push_back(&in0_internal);

  LiteRtTensorT& out0 = sg0.EmplaceTensor();
  out0.SetName("raw_vision_output");
  op0.Outputs().push_back(&out0);

  model.EmplaceSignature(
      &sg0, /*input_names=*/std::vector<std::string>{"canonical_image_input"},
      /*input_tensors=*/std::vector<LiteRtTensor>{&in0},
      /*output_names=*/std::vector<std::string>{"canonical_vision_output"},
      /*output_tensors=*/std::vector<LiteRtTensor>{&out0}, "signature_vision");

  // Subgraph 1: contains another dispatch op with input/output tensors.
  LiteRtSubgraphT& sg1 = model.EmplaceSubgraph();
  LiteRtOpT& op1 = sg1.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflCustom);
  op1.SetCustomCode(std::string(kLiteRtDispatchOpCustomName));
  DispatchOpOptions op_opts1{};
  op_opts1.name = "audio_func";
  op1.SetCustomOptions(MakeDispatchOpOptions(std::move(op_opts1)));

  LiteRtTensorT& in1 = sg1.EmplaceTensor();
  in1.SetName("raw_audio_input");
  op1.Inputs().push_back(&in1);

  LiteRtTensorT& out1 = sg1.EmplaceTensor();
  out1.SetName("raw_audio_output");
  op1.Outputs().push_back(&out1);

  model.EmplaceSignature(
      &sg1, /*input_names=*/std::vector<std::string>{"canonical_audio_input"},
      /*input_tensors=*/std::vector<LiteRtTensor>{&in1},
      /*output_names=*/std::vector<std::string>{"canonical_audio_output"},
      /*output_tensors=*/std::vector<LiteRtTensor>{&out1}, "signature_audio");

  LITERT_ASSERT_OK_AND_ASSIGN(
      DispatchDelegateVendorOptions vendor_options,
      BuildDispatchDelegateVendorOptions(model));

  absl::Span<const FunctionMapping> mappings =
      vendor_options.FunctionMappings();
  ASSERT_EQ(mappings.size(), 2);

  EXPECT_EQ(mappings[0].function_name, "vision_func");
  EXPECT_EQ(mappings[0].signature_name, "signature_vision");
  // Only the canonical signature input is recorded; internal_temp_tensor at
  // port 1 is skipped.
  ASSERT_EQ(mappings[0].input_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[0].input_tensor_ports[0].tensor_name,
            "canonical_image_input");
  EXPECT_EQ(mappings[0].input_tensor_ports[0].port_index, 0);
  ASSERT_EQ(mappings[0].output_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[0].output_tensor_ports[0].tensor_name,
            "canonical_vision_output");
  EXPECT_EQ(mappings[0].output_tensor_ports[0].port_index, 0);

  EXPECT_EQ(mappings[1].function_name, "audio_func");
  EXPECT_EQ(mappings[1].signature_name, "signature_audio");
  ASSERT_EQ(mappings[1].input_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[1].input_tensor_ports[0].tensor_name,
            "canonical_audio_input");
  EXPECT_EQ(mappings[1].input_tensor_ports[0].port_index, 0);
  ASSERT_EQ(mappings[1].output_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[1].output_tensor_ports[0].tensor_name,
            "canonical_audio_output");
  EXPECT_EQ(mappings[1].output_tensor_ports[0].port_index, 0);
}

}  // namespace
}  // namespace litert::internal
