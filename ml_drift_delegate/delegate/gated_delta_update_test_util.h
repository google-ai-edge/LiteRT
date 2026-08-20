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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GATED_DELTA_UPDATE_TEST_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GATED_DELTA_UPDATE_TEST_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tflite/schema/schema_generated.h"
#include "tflite/version.h"

namespace litert::ml_drift {

// Programmatically builds an in-memory TFLite FlatBuffer containing a single
// gated_delta_update custom op node with the given tensor dimensions.
inline std::vector<uint8_t> CreateGatedDeltaUpdateModelBuffer(int B, int H,
                                                              int N, int D_k,
                                                              int D_v,
                                                              int mode = 0) {
  flatbuffers::FlatBufferBuilder builder;

  // 1. Operator code: Custom op "gated_delta_update"
  auto custom_code = builder.CreateString("gated_delta_update");
  auto opcode = tflite::CreateOperatorCode(
      builder,
      /*deprecated_builtin_code=*/tflite::BuiltinOperator_CUSTOM, custom_code,
      /*version=*/1,
      /*builtin_code=*/tflite::BuiltinOperator_CUSTOM);
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes = {opcode};
  auto opcodes_vec = builder.CreateVector(opcodes);

  // 2. Buffers: Buffer 0 is empty (required by TFLite)
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));

  // 3. Tensors:
  // Inputs:
  // 0: q [B, H, N, D_k]
  // 1: k [B, H, N, D_k]
  // 2: v [B, H, N, D_v]
  // 3: beta [B, H, N]
  // 4: g [B, H, N]
  // 5: initial_state [B, H, D_k, D_v]
  // Outputs:
  // 6: out [B, H, N, D_v]
  // 7: final_state [B, H, D_k, D_v]
  const std::vector<std::vector<int32_t>> shapes = {
      {B, H, N, D_k}, {B, H, N, D_k},   {B, H, N, D_v}, {B, H, N},
      {B, H, N},      {B, H, D_k, D_v}, {B, H, N, D_v}, {B, H, D_k, D_v},
  };

  const std::vector<std::string> names = {
      "q", "k", "v", "beta", "g", "initial_state", "out", "final_state"};

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  for (int i = 0; i < 8; ++i) {
    auto shape_vec = builder.CreateVector(shapes[i]);
    auto name_str = builder.CreateString(names[i]);
    tensors.push_back(tflite::CreateTensor(builder, shape_vec,
                                           tflite::TensorType_FLOAT32,
                                           /*buffer=*/0, name_str));
  }
  auto tensors_vec = builder.CreateVector(tensors);

  // 4. Custom options for operator (mode)
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("mode", mode); });
  fbb.Finish();
  auto custom_options_vec = builder.CreateVector(fbb.GetBuffer());

  // 5. Operator
  std::vector<int32_t> op_inputs = {0, 1, 2, 3, 4, 5};
  std::vector<int32_t> op_outputs = {6, 7};
  auto op_inputs_vec = builder.CreateVector(op_inputs);
  auto op_outputs_vec = builder.CreateVector(op_outputs);

  auto op = tflite::CreateOperator(builder, /*opcode_index=*/0, op_inputs_vec,
                                   op_outputs_vec, tflite::BuiltinOptions_NONE,
                                   /*builtin_options=*/0, custom_options_vec,
                                   tflite::CustomOptionsFormat_FLEXBUFFERS);
  std::vector<flatbuffers::Offset<tflite::Operator>> operators = {op};
  auto operators_vec = builder.CreateVector(operators);

  // 6. Subgraph
  std::vector<int32_t> subgraph_inputs = {0, 1, 2, 3, 4, 5};
  std::vector<int32_t> subgraph_outputs = {6, 7};
  auto sg_inputs_vec = builder.CreateVector(subgraph_inputs);
  auto sg_outputs_vec = builder.CreateVector(subgraph_outputs);

  auto subgraph = tflite::CreateSubGraph(builder, tensors_vec, sg_inputs_vec,
                                         sg_outputs_vec, operators_vec,
                                         builder.CreateString("main"));
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {subgraph};
  auto subgraphs_vec = builder.CreateVector(subgraphs);

  auto buffers_vec = builder.CreateVector(buffers);
  auto desc = builder.CreateString("GatedDeltaUpdate programmatic model");

  auto model = tflite::CreateModel(builder, TFLITE_SCHEMA_VERSION, opcodes_vec,
                                   subgraphs_vec, desc, buffers_vec);
  builder.Finish(model, tflite::ModelIdentifier());

  const uint8_t* buf = builder.GetBufferPointer();
  size_t size = builder.GetSize();
  return std::vector<uint8_t>(buf, buf + size);
}

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GATED_DELTA_UPDATE_TEST_UTIL_H_
