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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_TFLITE_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_TFLITE_BUILDER_H_

// A TFLite model under construction (tflite::ModelT), for the artificial
// models of the NVIDIA backend's end-to-end checks (qsa_artifact.cc,
// qwen4_artifact.cc): tensors, constants, builtin, custom and composite ops,
// and signatures.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tflite/schema/schema_generated.h"

namespace litert::nvidia::artifact {

using Shape = std::vector<int32_t>;

class Random {
 public:
  explicit Random(uint32_t seed) : state_(seed * 2654435761u + 7u) {}
  float Next() {  // [-1, 1)
    state_ = state_ * 1664525u + 1013904223u;
    return static_cast<float>(state_ >> 8) / 8388608.0f - 1.0f;
  }
  std::vector<float> Values(size_t count, float scale) {
    std::vector<float> values(count);
    for (auto& value : values) value = Next() * scale;
    return values;
  }

 private:
  uint32_t state_;
};

// A TFLite model under construction, one subgraph at a time.
class Builder {
 public:
  Builder() {
    model_.version = 3;
    model_.buffers.push_back(std::make_unique<tflite::BufferT>());
  }

  int32_t NewSubgraph(const std::string& name) {
    auto subgraph = std::make_unique<tflite::SubGraphT>();
    subgraph->name = name;
    model_.subgraphs.push_back(std::move(subgraph));
    current_ = static_cast<int32_t>(model_.subgraphs.size()) - 1;
    return current_;
  }
  void Select(int32_t subgraph) { current_ = subgraph; }
  int32_t Current() const { return current_; }
  tflite::SubGraphT& graph() { return *model_.subgraphs[current_]; }

  int32_t Tensor(const std::string& name, tflite::TensorType type,
                 const Shape& shape) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->name = name;
    tensor->type = type;
    tensor->shape = shape;
    tensor->buffer = 0;
    graph().tensors.push_back(std::move(tensor));
    return static_cast<int32_t>(graph().tensors.size()) - 1;
  }
  int32_t F32(const std::string& name, const Shape& shape) {
    return Tensor(name, tflite::TensorType_FLOAT32, shape);
  }
  int32_t I32(const std::string& name, const Shape& shape) {
    return Tensor(name, tflite::TensorType_INT32, shape);
  }
  int32_t Bool(const std::string& name, const Shape& shape) {
    return Tensor(name, tflite::TensorType_BOOL, shape);
  }
  // A tensor quantized per tensor.
  int32_t Quantized(const std::string& name, tflite::TensorType type,
                    const Shape& shape, float scale, int64_t zero_point) {
    const int32_t tensor = Tensor(name, type, shape);
    auto quantization = std::make_unique<tflite::QuantizationParametersT>();
    quantization->scale = {scale};
    quantization->zero_point = {zero_point};
    graph().tensors[tensor]->quantization = std::move(quantization);
    return tensor;
  }
  // An INT8 constant [rows, columns] quantized per row (symmetric), from
  // `values`: the weights of a FULLY_CONNECTED quantized per output channel.
  int32_t ConstInt8PerRow(const std::string& name, int rows, int columns,
                          const std::vector<float>& values) {
    std::vector<int8_t> q(values.size());
    std::vector<float> scales(rows);
    for (int r = 0; r < rows; ++r) {
      float largest = 0.0f;
      for (int c = 0; c < columns; ++c) {
        largest = std::max(
            largest, std::fabs(values[static_cast<size_t>(r) * columns + c]));
      }
      scales[r] = largest > 0.0f ? largest / 127.0f : 1.0f;
      for (int c = 0; c < columns; ++c) {
        const size_t i = static_cast<size_t>(r) * columns + c;
        q[i] = static_cast<int8_t>(std::lround(values[i] / scales[r]));
      }
    }
    const int32_t tensor =
        Constant(name, tflite::TensorType_INT8, {rows, columns}, q);
    auto quantization = std::make_unique<tflite::QuantizationParametersT>();
    quantization->scale = scales;
    quantization->zero_point.assign(rows, 0);
    quantization->quantized_dimension = 0;
    graph().tensors[tensor]->quantization = std::move(quantization);
    return tensor;
  }
  int32_t Input(int32_t tensor) {
    graph().inputs.push_back(tensor);
    return tensor;
  }
  void Output(int32_t tensor) { graph().outputs.push_back(tensor); }

  template <typename T>
  int32_t Constant(const std::string& name, tflite::TensorType type,
                   const Shape& shape, const std::vector<T>& values) {
    const int32_t tensor = Tensor(name, type, shape);
    auto buffer = std::make_unique<tflite::BufferT>();
    buffer->data.resize(values.size() * sizeof(T));
    std::memcpy(buffer->data.data(), values.data(), buffer->data.size());
    model_.buffers.push_back(std::move(buffer));
    graph().tensors[tensor]->buffer =
        static_cast<uint32_t>(model_.buffers.size()) - 1;
    return tensor;
  }
  int32_t ConstF32(const std::string& name, const Shape& shape,
                   const std::vector<float>& values) {
    return Constant(name, tflite::TensorType_FLOAT32, shape, values);
  }
  int32_t ConstI32(const std::string& name, const Shape& shape,
                   const std::vector<int32_t>& values) {
    return Constant(name, tflite::TensorType_INT32, shape, values);
  }
  int32_t ScalarI32(const std::string& name, int32_t value) {
    return ConstI32(name, {}, {value});
  }
  int32_t OneI32(const std::string& name, int32_t value) {
    return ConstI32(name, {1}, {value});
  }
  int32_t ScalarF32(const std::string& name, float value) {
    return ConstF32(name, {}, {value});
  }
  int32_t Iota(const std::string& name, int32_t count) {
    std::vector<int32_t> values(count);
    for (int32_t i = 0; i < count; ++i) values[i] = i;
    return ConstI32(name, {count}, values);
  }

  tflite::OperatorT& Op(tflite::BuiltinOperator code,
                        std::vector<int32_t> inputs,
                        std::vector<int32_t> outputs) {
    size_t index = 0;
    for (; index < model_.operator_codes.size(); ++index) {
      if (model_.operator_codes[index]->builtin_code == code &&
          code != tflite::BuiltinOperator_CUSTOM) {
        break;
      }
    }
    if (index == model_.operator_codes.size()) {
      auto op_code = std::make_unique<tflite::OperatorCodeT>();
      op_code->builtin_code = code;
      op_code->deprecated_builtin_code = static_cast<int8_t>(
          code < tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
              ? code
              : tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
      op_code->version = 1;
      model_.operator_codes.push_back(std::move(op_code));
    }
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = static_cast<uint32_t>(index);
    op->inputs = std::move(inputs);
    op->outputs = std::move(outputs);
    graph().operators.push_back(std::move(op));
    return *graph().operators.back();
  }

  // A custom op (builtin code CUSTOM) named `name`.
  tflite::OperatorT& Custom(const std::string& name,
                            std::vector<int32_t> inputs,
                            std::vector<int32_t> outputs) {
    size_t index = 0;
    for (; index < model_.operator_codes.size(); ++index) {
      const auto& code = *model_.operator_codes[index];
      if (code.builtin_code == tflite::BuiltinOperator_CUSTOM &&
          code.custom_code == name) {
        break;
      }
    }
    if (index == model_.operator_codes.size()) {
      auto op_code = std::make_unique<tflite::OperatorCodeT>();
      op_code->builtin_code = tflite::BuiltinOperator_CUSTOM;
      op_code->deprecated_builtin_code =
          static_cast<int8_t>(tflite::BuiltinOperator_CUSTOM);
      op_code->custom_code = name;
      op_code->version = 1;
      model_.operator_codes.push_back(std::move(op_code));
    }
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = static_cast<uint32_t>(index);
    op->inputs = std::move(inputs);
    op->outputs = std::move(outputs);
    graph().operators.push_back(std::move(op));
    return *graph().operators.back();
  }

  // Ops whose result is a new tensor of the given type and shape.
  int32_t Unary(tflite::BuiltinOperator code, int32_t x,
                tflite::TensorType type, const Shape& shape) {
    const int32_t out = Tensor(Name(), type, shape);
    Op(code, {x}, {out});
    return out;
  }
  int32_t Binary(tflite::BuiltinOperator code, int32_t a, int32_t b,
                 tflite::TensorType type, const Shape& shape) {
    const int32_t out = Tensor(Name(), type, shape);
    auto& op = Op(code, {a, b}, {out});
    switch (code) {
      case tflite::BuiltinOperator_ADD: {
        tflite::AddOptionsT options;
        op.builtin_options.Set(options);
        break;
      }
      case tflite::BuiltinOperator_SUB: {
        tflite::SubOptionsT options;
        op.builtin_options.Set(options);
        break;
      }
      case tflite::BuiltinOperator_MUL: {
        tflite::MulOptionsT options;
        op.builtin_options.Set(options);
        break;
      }
      case tflite::BuiltinOperator_DIV: {
        tflite::DivOptionsT options;
        op.builtin_options.Set(options);
        break;
      }
      default:
        break;
    }
    return out;
  }
  int32_t AddF(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_ADD, a, b, tflite::TensorType_FLOAT32,
                  s);
  }
  int32_t MulF(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_MUL, a, b, tflite::TensorType_FLOAT32,
                  s);
  }
  int32_t AddI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_ADD, a, b, tflite::TensorType_INT32,
                  s);
  }
  int32_t SubI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_SUB, a, b, tflite::TensorType_INT32,
                  s);
  }
  int32_t MulI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_MUL, a, b, tflite::TensorType_INT32,
                  s);
  }
  int32_t MaxI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_MAXIMUM, a, b,
                  tflite::TensorType_INT32, s);
  }
  int32_t MinI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_MINIMUM, a, b,
                  tflite::TensorType_INT32, s);
  }
  int32_t FloorDivI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_FLOOR_DIV, a, b,
                  tflite::TensorType_INT32, s);
  }
  int32_t Compare(tflite::BuiltinOperator code, int32_t a, int32_t b,
                  const Shape& s) {
    return Binary(code, a, b, tflite::TensorType_BOOL, s);
  }
  int32_t And(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_LOGICAL_AND, a, b,
                  tflite::TensorType_BOOL, s);
  }
  int32_t Select(int32_t condition, int32_t a, int32_t b,
                 tflite::TensorType type, const Shape& s) {
    const int32_t out = Tensor(Name(), type, s);
    Op(tflite::BuiltinOperator_SELECT_V2, {condition, a, b}, {out});
    return out;
  }
  int32_t Reshape(int32_t x, const Shape& shape) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, shape);
    const int32_t shape_tensor =
        ConstI32(Name(), {static_cast<int32_t>(shape.size())},
                 std::vector<int32_t>(shape));
    auto& op = Op(tflite::BuiltinOperator_RESHAPE, {x, shape_tensor}, {out});
    tflite::ReshapeOptionsT options;
    options.new_shape = shape;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t Transpose(int32_t x, const std::vector<int32_t>& perm,
                    const Shape& shape) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, shape);
    const int32_t perm_tensor =
        ConstI32(Name(), {static_cast<int32_t>(perm.size())}, perm);
    Op(tflite::BuiltinOperator_TRANSPOSE, {x, perm_tensor}, {out});
    return out;
  }
  int32_t Slice(int32_t x, const std::vector<int32_t>& begin,
                const Shape& size) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, size);
    const int32_t b =
        ConstI32(Name(), {static_cast<int32_t>(begin.size())}, begin);
    const int32_t s = ConstI32(Name(), {static_cast<int32_t>(size.size())},
                               std::vector<int32_t>(size));
    Op(tflite::BuiltinOperator_SLICE, {x, b, s}, {out});
    return out;
  }
  int32_t Concat(const std::vector<int32_t>& parts, int32_t axis,
                 const Shape& shape) {
    const auto type = graph().tensors[parts[0]]->type;
    const int32_t out = Tensor(Name(), type, shape);
    auto& op = Op(tflite::BuiltinOperator_CONCATENATION, parts, {out});
    tflite::ConcatenationOptionsT options;
    options.axis = axis;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t Gather(int32_t params, int32_t indices, int32_t axis,
                 const Shape& shape) {
    const auto type = graph().tensors[params]->type;
    const int32_t out = Tensor(Name(), type, shape);
    auto& op = Op(tflite::BuiltinOperator_GATHER, {params, indices}, {out});
    tflite::GatherOptionsT options;
    options.axis = axis;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t Reduce(tflite::BuiltinOperator code, int32_t x,
                 const std::vector<int32_t>& axes, bool keep_dims,
                 const Shape& shape) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, shape);
    const int32_t axes_tensor =
        ConstI32(Name(), {static_cast<int32_t>(axes.size())}, axes);
    auto& op = Op(code, {x, axes_tensor}, {out});
    tflite::ReducerOptionsT options;
    options.keep_dims = keep_dims;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t Cast(int32_t x, tflite::TensorType type) {
    const int32_t out = Tensor(Name(), type, graph().tensors[x]->shape);
    auto& op = Op(tflite::BuiltinOperator_CAST, {x}, {out});
    tflite::CastOptionsT options;
    options.in_data_type = graph().tensors[x]->type;
    options.out_data_type = type;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t BatchMatMul(int32_t a, int32_t b, bool adj_y, const Shape& shape) {
    const int32_t out = F32(Name(), shape);
    auto& op = Op(tflite::BuiltinOperator_BATCH_MATMUL, {a, b}, {out});
    tflite::BatchMatMulOptionsT options;
    options.adj_y = adj_y;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t FullyConnected(int32_t x, int32_t weights, const Shape& shape) {
    const int32_t out = F32(Name(), shape);
    auto& op =
        Op(tflite::BuiltinOperator_FULLY_CONNECTED, {x, weights, -1}, {out});
    tflite::FullyConnectedOptionsT options;
    op.builtin_options.Set(options);
    return out;
  }
  // QUANTIZE of a float `x` into a new tensor quantized like that.
  int32_t Quantize(int32_t x, tflite::TensorType type, float scale,
                   int64_t zero_point) {
    const int32_t out =
        Quantized(Name(), type, graph().tensors[x]->shape, scale, zero_point);
    Op(tflite::BuiltinOperator_QUANTIZE, {x}, {out});
    return out;
  }
  int32_t Dequantize(int32_t x) {
    const int32_t out = F32(Name(), graph().tensors[x]->shape);
    Op(tflite::BuiltinOperator_DEQUANTIZE, {x}, {out});
    return out;
  }
  int32_t Softmax(int32_t x) {
    const int32_t out = F32(Name(), graph().tensors[x]->shape);
    auto& op = Op(tflite::BuiltinOperator_SOFTMAX, {x}, {out});
    tflite::SoftmaxOptionsT options;
    options.beta = 1.0f;
    op.builtin_options.Set(options);
    return out;
  }
  // c * tanh(x / c).
  int32_t Softcap(int32_t x, float cap, const Shape& s) {
    const int32_t divided =
        Binary(tflite::BuiltinOperator_DIV, x, ConstF32(Name(), {1}, {cap}),
               tflite::TensorType_FLOAT32, s);
    const int32_t tanh = Unary(tflite::BuiltinOperator_TANH, divided,
                               tflite::TensorType_FLOAT32, s);
    return MulF(tanh, ConstF32(Name(), {1}, {cap}), s);
  }
  int32_t Tile(int32_t x, const std::vector<int32_t>& multiples,
               const Shape& shape) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, shape);
    const int32_t m =
        ConstI32(Name(), {static_cast<int32_t>(multiples.size())}, multiples);
    Op(tflite::BuiltinOperator_TILE, {x, m}, {out});
    return out;
  }

  void Composite(const std::string& name, int32_t decomposition,
                 const std::function<void(flexbuffers::Builder*)>& attributes,
                 std::vector<int32_t> inputs, std::vector<int32_t> outputs) {
    auto& op = Op(tflite::BuiltinOperator_STABLEHLO_COMPOSITE,
                  std::move(inputs), std::move(outputs));
    flexbuffers::Builder fbb;
    const auto map = fbb.StartMap();
    attributes(&fbb);
    fbb.EndMap(map);
    fbb.Finish();
    tflite::StableHLOCompositeOptionsT options;
    options.name = name;
    options.decomposition_subgraph_index = decomposition;
    options.composite_attributes = fbb.GetBuffer();
    options.composite_attributes_format =
        tflite::CustomOptionsFormat_FLEXBUFFERS;
    op.builtin_options_2.Set(std::move(options));
  }

  void Signature(const std::string& key, int32_t subgraph,
                 const std::vector<std::pair<std::string, int32_t>>& inputs,
                 const std::vector<std::pair<std::string, int32_t>>& outputs) {
    auto signature = std::make_unique<tflite::SignatureDefT>();
    signature->signature_key = key;
    signature->subgraph_index = static_cast<uint32_t>(subgraph);
    for (const auto& [name, tensor] : inputs) {
      auto map = std::make_unique<tflite::TensorMapT>();
      map->name = name;
      map->tensor_index = static_cast<uint32_t>(tensor);
      signature->inputs.push_back(std::move(map));
    }
    for (const auto& [name, tensor] : outputs) {
      auto map = std::make_unique<tflite::TensorMapT>();
      map->name = name;
      map->tensor_index = static_cast<uint32_t>(tensor);
      signature->outputs.push_back(std::move(map));
    }
    model_.signature_defs.push_back(std::move(signature));
  }

  std::string Pack() {
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model_));
    return std::string(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize());
  }

 private:
  std::string Name() { return absl::StrCat("t", next_name_++); }

  tflite::ModelT model_;
  int32_t current_ = -1;
  int next_name_ = 0;
};

}  // namespace litert::nvidia::artifact

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_TFLITE_BUILDER_H_
