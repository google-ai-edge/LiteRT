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
inline std::vector<uint8_t> CreateGatedDeltaUpdateModelBuffer(
    int B, int H, int N, int D_k, int D_v, int mode = 0, int H_k = -1,
    const std::string& state_dtype = "float32", bool has_valid_len = false,
    int valid_len_size = -1) {
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
  // 0: q [B, H_k, N, D_k]
  // 1: k [B, H_k, N, D_k]
  // 2: v [B, H, N, D_v]
  // 3: beta [B, H, N]
  // 4: g [B, H, N]
  // 5: initial_state [B, H, D_k, D_v]
  // 6 (optional): valid_len [B] (INT32)
  // Outputs:
  // 6 (or 7): out [B, H, N, D_v]
  // 7 (or 8): final_state [B, H, D_k, D_v]
  int actual_H_k = (H_k > 0) ? H_k : H;
  std::vector<std::vector<int32_t>> shapes = {
      {B, actual_H_k, N, D_k},
      {B, actual_H_k, N, D_k},
      {B, H, N, D_v},
      {B, H, N},
      {B, H, N},
      {B, H, D_k, D_v},
  };
  std::vector<std::string> names = {"q",    "k", "v",
                                    "beta", "g", "initial_state"};
  std::vector<tflite::TensorType> dtypes(6, tflite::TensorType_FLOAT32);

  if (has_valid_len) {
    shapes.push_back({valid_len_size > 0 ? valid_len_size : B});
    names.push_back("valid_len");
    dtypes.push_back(tflite::TensorType_INT32);
  }

  const int32_t out_idx = static_cast<int32_t>(shapes.size());
  shapes.push_back({B, H, N, D_v});
  names.push_back("out");
  dtypes.push_back(tflite::TensorType_FLOAT32);

  const int32_t final_state_idx = static_cast<int32_t>(shapes.size());
  shapes.push_back({B, H, D_k, D_v});
  names.push_back("final_state");
  dtypes.push_back(tflite::TensorType_FLOAT32);

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  for (size_t i = 0; i < shapes.size(); ++i) {
    auto shape_vec = builder.CreateVector(shapes[i]);
    auto name_str = builder.CreateString(names[i]);
    tensors.push_back(tflite::CreateTensor(builder, shape_vec, dtypes[i],
                                           /*buffer=*/0, name_str));
  }
  auto tensors_vec = builder.CreateVector(tensors);

  // 4. Custom options for operator (mode, state_dtype)
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("mode", mode);
    fbb.String("state_dtype", state_dtype);
  });
  fbb.Finish();
  auto custom_options_vec = builder.CreateVector(fbb.GetBuffer());

  // 5. Operator
  std::vector<int32_t> op_inputs = {0, 1, 2, 3, 4, 5};
  if (has_valid_len) {
    op_inputs.push_back(6);
  }
  std::vector<int32_t> op_outputs = {out_idx, final_state_idx};
  auto op_inputs_vec = builder.CreateVector(op_inputs);
  auto op_outputs_vec = builder.CreateVector(op_outputs);

  auto op = tflite::CreateOperator(builder, /*opcode_index=*/0, op_inputs_vec,
                                   op_outputs_vec, tflite::BuiltinOptions_NONE,
                                   /*builtin_options=*/0, custom_options_vec,
                                   tflite::CustomOptionsFormat_FLEXBUFFERS);
  std::vector<flatbuffers::Offset<tflite::Operator>> operators = {op};
  auto operators_vec = builder.CreateVector(operators);

  // 6. Subgraph
  std::vector<int32_t> subgraph_inputs = op_inputs;
  std::vector<int32_t> subgraph_outputs = op_outputs;
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

// Programmatically builds an in-memory TFLite FlatBuffer containing a single
// block-mode `gated_delta_update` custom op node with constant `a_log`,
// `dt_bias`, and `norm_weight` parameter buffers.
inline std::vector<uint8_t> CreateGatedDeltaUpdateBlockModelBuffer(
    int T, int H_k, int H_v, int D_k, int D_v, const std::vector<float>& a_log,
    const std::vector<float>& dt_bias, const std::vector<float>& norm_weight,
    float eps = 1e-6f, float norm_eps = 1e-6f, bool has_valid_mask = false) {
  flatbuffers::FlatBufferBuilder builder;

  auto custom_code = builder.CreateString("gated_delta_update");
  auto opcode = tflite::CreateOperatorCode(
      builder, tflite::BuiltinOperator_CUSTOM, custom_code, /*version=*/1,
      tflite::BuiltinOperator_CUSTOM);
  auto opcodes_vec = builder.CreateVector({opcode});

  auto add_const_buffer = [&](const std::vector<float>& vals) {
    auto bytes =
        builder.CreateVector(reinterpret_cast<const uint8_t*>(vals.data()),
                             vals.size() * sizeof(float));
    return tflite::CreateBuffer(builder, bytes);
  };
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));  // 0: empty runtime buffer
  buffers.push_back(add_const_buffer(a_log));        // 1: a_log
  buffers.push_back(add_const_buffer(dt_bias));      // 2: dt_bias
  buffers.push_back(add_const_buffer(norm_weight));  // 3: norm_weight

  const int K_dim = H_k * D_k;
  const int V_dim = H_v * D_v;
  std::vector<std::vector<int32_t>> shapes = {
      {1, T, 2 * K_dim + V_dim},  // 0: conv_out
      {1, T, V_dim + 2 * H_v},    // 1: zba
      {H_v},                      // 2: a_log (const)
      {H_v},                      // 3: dt_bias (const)
      {D_v},                      // 4: norm_weight (const)
      {1, H_v, D_k, D_v},         // 5: initial_state
  };
  std::vector<std::string> names = {"conv_out", "zba",         "a_log",
                                    "dt_bias",  "norm_weight", "initial_state"};
  std::vector<uint32_t> tensor_buffers = {0, 0, 1, 2, 3, 0};
  std::vector<int32_t> sg_inputs = {0, 1, 5};
  std::vector<int32_t> op_inputs = {0, 1, 2, 3, 4, 5};

  if (has_valid_mask) {
    shapes.push_back({1, T});
    names.push_back("valid_mask");
    tensor_buffers.push_back(0);
    sg_inputs.push_back(6);
    op_inputs.push_back(6);
  }

  const int32_t out_idx = static_cast<int32_t>(shapes.size());
  shapes.push_back({1, T, V_dim});
  names.push_back("out");
  tensor_buffers.push_back(0);

  const int32_t final_state_idx = static_cast<int32_t>(shapes.size());
  shapes.push_back({1, H_v, D_k, D_v});
  names.push_back("final_state");
  tensor_buffers.push_back(0);

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  for (size_t i = 0; i < shapes.size(); ++i) {
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(shapes[i]), tflite::TensorType_FLOAT32,
        tensor_buffers[i], builder.CreateString(names[i])));
  }

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("mode", 0);
    fbb.String("state_dtype", "float32");
    fbb.Int("num_k_heads", H_k);
    fbb.Int("num_v_heads", H_v);
    fbb.Int("head_k_dim", D_k);
    fbb.Int("head_v_dim", D_v);
    fbb.Float("eps", eps);
    fbb.Float("norm_eps", norm_eps);
  });
  fbb.Finish();

  std::vector<int32_t> op_outputs = {out_idx, final_state_idx};
  auto op = tflite::CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector(op_inputs),
      builder.CreateVector(op_outputs), tflite::BuiltinOptions_NONE,
      /*builtin_options=*/0, builder.CreateVector(fbb.GetBuffer()),
      tflite::CustomOptionsFormat_FLEXBUFFERS);

  auto subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors), builder.CreateVector(sg_inputs),
      builder.CreateVector(op_outputs), builder.CreateVector({op}),
      builder.CreateString("main"));
  auto model =
      tflite::CreateModel(builder, TFLITE_SCHEMA_VERSION, opcodes_vec,
                          builder.CreateVector({subgraph}),
                          builder.CreateString("GatedDeltaUpdate block model"),
                          builder.CreateVector(buffers));
  builder.Finish(model, tflite::ModelIdentifier());

  const uint8_t* buf = builder.GetBufferPointer();
  return std::vector<uint8_t>(buf, buf + builder.GetSize());
}

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GATED_DELTA_UPDATE_TEST_UTIL_H_
