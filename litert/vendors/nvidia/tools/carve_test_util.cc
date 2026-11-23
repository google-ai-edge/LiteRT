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

#include "litert/vendors/nvidia/tools/carve_test_util.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tflite/schema/schema_generated.h"

namespace litert::tools::testing {
namespace {

using Shape = std::vector<int32_t>;

class Builder {
 public:
  explicit Builder(const FactorizedMlpModelOptions& options)
      : options_(options) {
    model_.version = 3;
    model_.description = "carve test model";
    // Buffer 0 is the empty buffer of tensors without data.
    model_.buffers.push_back(std::make_unique<tflite::BufferT>());
    model_.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
    graph().name = "main";
  }

  std::string Build() {
    const int m = options_.tokens;
    const int f = 2 * options_.half_features;
    const int32_t x = Tensor("x", tflite::TensorType_FLOAT32, {1, m, f});
    int32_t stream = Tensor("stream0", tflite::TensorType_FLOAT16, {1, m, f});
    Op(tflite::BuiltinOperator_CAST, 5, {x}, {stream});
    for (int k = 0; k < options_.num_mlps; ++k) {
      const int32_t mlp = Mlp(k, stream);
      const int32_t next = Tensor(absl::StrCat("stream", k + 1),
                                  tflite::TensorType_FLOAT16, {1, m, f});
      Op(tflite::BuiltinOperator_ADD, 6, {stream, mlp}, {next})
          .builtin_options.Set(tflite::AddOptionsT());
      stream = next;
    }
    const int32_t y = Tensor("y", tflite::TensorType_FLOAT32, {1, m, f});
    Op(tflite::BuiltinOperator_CAST, 5, {stream}, {y});
    graph().inputs = {x};
    graph().outputs = {y};

    auto signature = std::make_unique<tflite::SignatureDefT>();
    signature->signature_key = "main";
    signature->subgraph_index = 0;
    for (const auto& [name, tensor, list] :
         {std::make_tuple("x", x, &signature->inputs),
          std::make_tuple("y", y, &signature->outputs)}) {
      auto map = std::make_unique<tflite::TensorMapT>();
      map->name = name;
      map->tensor_index = tensor;
      list->push_back(std::move(map));
    }
    model_.signature_defs.push_back(std::move(signature));

    if (options_.mutate) options_.mutate(model_);
    return options_.appended_buffers ? PackWithAppendedBuffers() : Pack();
  }

 private:
  tflite::SubGraphT& graph() { return *model_.subgraphs[0]; }

  int32_t Tensor(std::string name, tflite::TensorType type, Shape shape) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->name = std::move(name);
    tensor->type = type;
    tensor->shape = std::move(shape);
    tensor->has_rank = true;
    graph().tensors.push_back(std::move(tensor));
    return static_cast<int32_t>(graph().tensors.size() - 1);
  }

  // An INT8 activation: one scale, zero point 0.
  int32_t Quantized(std::string name, Shape shape, float scale) {
    const int32_t index =
        Tensor(std::move(name), tflite::TensorType_INT8, std::move(shape));
    auto quantization = std::make_unique<tflite::QuantizationParametersT>();
    quantization->scale = {scale};
    quantization->zero_point = {0};
    graph().tensors[index]->quantization = std::move(quantization);
    return index;
  }

  int32_t Constant(std::string name, tflite::TensorType type, Shape shape,
                   std::vector<uint8_t> data) {
    const int32_t index = Tensor(std::move(name), type, std::move(shape));
    auto buffer = std::make_unique<tflite::BufferT>();
    buffer->data = std::move(data);
    model_.buffers.push_back(std::move(buffer));
    graph().tensors[index]->buffer =
        static_cast<uint32_t>(model_.buffers.size() - 1);
    return index;
  }

  template <class T>
  static std::vector<uint8_t> Bytes(const std::vector<T>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
  }

  // A shape or index constant. Equal ones are one tensor, as in converted
  // models, so the MLPs share them with each other and with ops around them.
  int32_t Ints(const std::vector<int32_t>& values) {
    auto [it, inserted] = ints_.try_emplace(values, 0);
    if (inserted) {
      it->second =
          Constant(absl::StrCat("ints", ints_.size()), tflite::TensorType_INT32,
                   {static_cast<int32_t>(values.size())}, Bytes(values));
    }
    return it->second;
  }

  // Centered INT2 weights: 2-bit codes, and a scale and a zero point of -0.5
  // for each block of `block_shape`.
  int32_t Weights(const std::string& name, Shape shape, Shape block_shape) {
    size_t elements = 1;
    Shape parameter_shape;
    for (size_t d = 0; d < shape.size(); ++d) {
      elements *= shape[d];
      parameter_shape.push_back(shape[d] / block_shape[d]);
    }
    size_t parameters = 1;
    for (const int32_t d : parameter_shape) parameters *= d;
    std::vector<float> scales(parameters);
    for (auto& scale : scales) {
      scale = std::uniform_real_distribution<float>(0.2f, 0.6f)(random_);
    }
    std::vector<uint8_t> codes((elements + 3) / 4);
    for (auto& byte : codes) byte = static_cast<uint8_t>(random_());

    const int32_t scales_index =
        Constant(name + "_scales", tflite::TensorType_FLOAT32, parameter_shape,
                 Bytes(scales));
    const int32_t zero_points_index =
        Constant(name + "_zero_points", tflite::TensorType_FLOAT32,
                 parameter_shape, Bytes(std::vector<float>(parameters, -0.5f)));
    const int32_t index = Constant(name, tflite::TensorType_INT2,
                                   std::move(shape), std::move(codes));
    tflite::BlockwiseQuantizationT blockwise;
    blockwise.scales = scales_index;
    blockwise.zero_points = zero_points_index;
    blockwise.block_shape = std::move(block_shape);
    auto quantization = std::make_unique<tflite::QuantizationParametersT>();
    quantization->details.Set(std::move(blockwise));
    graph().tensors[index]->quantization = std::move(quantization);
    return index;
  }

  tflite::OperatorT& Op(tflite::BuiltinOperator code, int version,
                        std::vector<int32_t> inputs,
                        std::vector<int32_t> outputs) {
    size_t code_index = 0;
    for (; code_index < model_.operator_codes.size(); ++code_index) {
      if (model_.operator_codes[code_index]->builtin_code == code) break;
    }
    if (code_index == model_.operator_codes.size()) {
      auto operator_code = std::make_unique<tflite::OperatorCodeT>();
      operator_code->builtin_code = code;
      operator_code->deprecated_builtin_code = static_cast<int8_t>(
          code < tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
              ? code
              : tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
      operator_code->version = version;
      model_.operator_codes.push_back(std::move(operator_code));
    }
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = static_cast<uint32_t>(code_index);
    op->inputs = std::move(inputs);
    op->outputs = std::move(outputs);
    graph().operators.push_back(std::move(op));
    return *graph().operators.back();
  }

  int32_t Reshape(const std::string& name, int32_t input, Shape shape) {
    const auto& source = *graph().tensors[input];
    const int32_t output = Tensor(name, source.type, shape);
    if (source.quantization != nullptr) {
      graph().tensors[output]->quantization =
          std::make_unique<tflite::QuantizationParametersT>(
              *source.quantization);
    }
    Op(tflite::BuiltinOperator_RESHAPE, 1, {input, Ints(shape)}, {output});
    return output;
  }

  int32_t Slice(const std::string& name, int32_t input, Shape begin, Shape size,
                float scale) {
    const int32_t output = Quantized(name, size, scale);
    Op(tflite::BuiltinOperator_SLICE, 8, {input, Ints(begin), Ints(size)},
       {output});
    return output;
  }

  int32_t FullyConnected(const std::string& name, int32_t input,
                         int32_t weights, Shape shape, float scale) {
    const int32_t output = Quantized(name, std::move(shape), scale);
    tflite::FullyConnectedOptionsT options;
    options.keep_num_dims = true;
    // The model has no bias: the third input is left out.
    Op(tflite::BuiltinOperator_FULLY_CONNECTED, 17, {input, weights, -1},
       {output})
        .builtin_options.Set(std::move(options));
    return output;
  }

  int32_t Concatenation(const std::string& name, int32_t a, int32_t b, int axis,
                        Shape shape, float scale) {
    const int32_t output = Quantized(name, std::move(shape), scale);
    tflite::ConcatenationOptionsT options;
    options.axis = axis;
    Op(tflite::BuiltinOperator_CONCATENATION, 6, {a, b}, {output})
        .builtin_options.Set(std::move(options));
    return output;
  }

  int32_t BatchMatMul(const std::string& name, int32_t input, int32_t weights,
                      Shape shape, float scale) {
    const int32_t output = Quantized(name, std::move(shape), scale);
    Op(tflite::BuiltinOperator_BATCH_MATMUL, 5, {input, weights}, {output})
        .builtin_options.Set(tflite::BatchMatMulOptionsT());
    return output;
  }

  // One factorized MLP on the F16 tensor `x` [1, M, F]; returns its F16 result.
  int32_t Mlp(int k, int32_t x) {
    const int m = options_.tokens;
    const int h = options_.half_features;
    const int f = 2 * h;
    const int g = options_.groups;
    const int c = options_.group_features;
    const int u = g * c;
    const int kk = 2 * c;
    const int n = options_.group_hidden;
    const int d = g * n;
    // Each stage has its own scale, and each MLP its own set. With inputs in
    // [-1, 1] and the weights of Weights() they keep every stage at tens of
    // steps, so that the signal neither dies nor saturates on the way.
    const float s_in = 0.010f + 0.001f * k;
    const float s_up1 = 0.021f + 0.001f * k;
    const float s_up2 = 0.030f + 0.001f * k;
    const float s_act = 0.019f + 0.001f * k;
    const float s_down1 = 0.015f + 0.001f * k;
    const float s_out = 0.023f + 0.001f * k;
    auto name = [k](const char* suffix) {
      return absl::StrCat("mlp", k, "/", suffix);
    };
    const auto f16 = tflite::TensorType_FLOAT16;
    const bool rank4 = options_.rank4_block_weights;

    const int32_t w_up1a = Weights(name("w_up1a"), {u, h}, {1, h});
    const int32_t w_up1b = Weights(name("w_up1b"), {u, h}, {1, h});
    const int32_t w_up2 =
        rank4 ? Weights(name("w_up2"), {1, g, kk, n}, {1, 1, kk, 1})
              : Weights(name("w_up2"), {g, kk, n}, {1, kk, 1});
    const int32_t w_down1 =
        rank4 ? Weights(name("w_down1"), {1, g, n, kk}, {1, 1, n, 1})
              : Weights(name("w_down1"), {g, n, kk}, {1, n, 1});
    const int32_t w_down2a = Weights(name("w_down2a"), {h, u}, {1, u});
    const int32_t w_down2b = Weights(name("w_down2b"), {h, u}, {1, u});

    const int32_t x2 = Reshape(name("x2"), x, {m, f});
    const int32_t q0 = Quantized(name("q0"), {m, f}, s_in);
    Op(tflite::BuiltinOperator_QUANTIZE, 4, {x2}, {q0});
    const int32_t h0 = Slice(name("h0"), q0, {0, 0}, {m, h}, s_in);
    const int32_t u0 = FullyConnected(name("u0"), h0, w_up1a, {m, u}, s_up1);
    const int32_t r0 = Reshape(name("r0"), u0, {m, g, 1, c});
    const int32_t h1 = Slice(name("h1"), q0, {0, h}, {m, h}, s_in);
    const int32_t u1 = FullyConnected(name("u1"), h1, w_up1b, {m, u}, s_up1);
    const int32_t r1 = Reshape(name("r1"), u1, {m, g, 1, c});
    const int32_t c0 =
        Concatenation(name("c0"), r0, r1, 2, {m, g, 2, c}, s_up1);
    const int32_t r2 = Reshape(name("r2"), c0, {m, g, 1, kk});
    const int32_t m0 = BatchMatMul(name("m0"), r2, w_up2, {m, g, 1, n}, s_up2);
    const int32_t r3 = Reshape(name("r3"), m0, {m, d});
    const int32_t d0 = Tensor(name("d0"), f16, {m, d});
    Op(tflite::BuiltinOperator_DEQUANTIZE, 6, {r3}, {d0});
    const int32_t s0 = Tensor(name("s0"), f16, {m, d});
    Op(tflite::BuiltinOperator_LOGISTIC, 1, {d0}, {s0});
    const int32_t a0 = Tensor(name("a0"), f16, {m, d});
    Op(tflite::BuiltinOperator_MUL, 8, {d0, s0}, {a0})
        .builtin_options.Set(tflite::MulOptionsT());
    const int32_t q1 = Quantized(name("q1"), {m, d}, s_act);
    Op(tflite::BuiltinOperator_QUANTIZE, 4, {a0}, {q1});
    const int32_t r4 = Reshape(name("r4"), q1, {m, g, 1, n});
    const int32_t m1 =
        BatchMatMul(name("m1"), r4, w_down1, {m, g, 1, kk}, s_down1);
    const int32_t r5 = Reshape(name("r5"), m1, {m, g, 2, c});
    const int32_t p0 =
        Slice(name("p0"), r5, {0, 0, 0, 0}, {m, g, 1, c}, s_down1);
    const int32_t r6 = Reshape(name("r6"), p0, {m, u});
    const int32_t f0 = FullyConnected(name("f0"), r6, w_down2a, {m, h}, s_out);
    const int32_t p1 =
        Slice(name("p1"), r5, {0, 0, 1, 0}, {m, g, 1, c}, s_down1);
    const int32_t r7 = Reshape(name("r7"), p1, {m, u});
    const int32_t f1 = FullyConnected(name("f1"), r7, w_down2b, {m, h}, s_out);
    const int32_t c1 = Concatenation(name("c1"), f0, f1, 1, {m, f}, s_out);
    const int32_t r8 = Reshape(name("r8"), c1, {1, m, f});
    const int32_t y = Tensor(name("y"), f16, {1, m, f});
    Op(tflite::BuiltinOperator_DEQUANTIZE, 6, {r8}, {y});
    return y;
  }

  std::string Pack() const {
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model_));
    return std::string(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize());
  }

  // Moves the constants behind the flatbuffer. Equal contents are stored once,
  // which several buffers then share (the zero point tensors of centered INT2
  // weights do).
  std::string PackWithAppendedBuffers() {
    std::vector<std::vector<uint8_t>> data(model_.buffers.size());
    for (size_t i = 0; i < model_.buffers.size(); ++i) {
      auto& buffer = *model_.buffers[i];
      if (buffer.data.size() <= options_.keep_inline_up_to) continue;
      data[i] = std::move(buffer.data);
      buffer.data.clear();
      // Any offset above 1 keeps the field in the flatbuffer; the real one
      // is known once the flatbuffer's size is.
      buffer.offset = 2;
      buffer.size = data[i].size();
    }
    const size_t flatbuffer_size = Pack().size();
    std::string appended;
    std::map<std::vector<uint8_t>, uint64_t> offsets;
    for (size_t i = 0; i < model_.buffers.size(); ++i) {
      if (data[i].empty()) continue;
      auto [it, inserted] = offsets.try_emplace(data[i], 0);
      if (inserted) {
        appended.resize((appended.size() + 15) / 16 * 16);
        it->second = (flatbuffer_size + 15) / 16 * 16 + appended.size();
        appended.append(reinterpret_cast<const char*>(data[i].data()),
                        data[i].size());
      }
      model_.buffers[i]->offset = it->second;
    }
    std::string bytes = Pack();
    bytes.resize((flatbuffer_size + 15) / 16 * 16);
    bytes += appended;
    // Kernels read past the last constant.
    bytes.resize(bytes.size() + 128);
    return bytes;
  }

  const FactorizedMlpModelOptions& options_;
  tflite::ModelT model_;
  std::map<std::vector<int32_t>, int32_t> ints_;
  std::mt19937 random_{20261114};
};

}  // namespace

std::string BuildFactorizedMlpModel(const FactorizedMlpModelOptions& options) {
  return Builder(options).Build();
}

std::string ToBytes(const CarvedModel& model) {
  const litert::BufferRef<uint8_t>& flatbuffer = model.flatbuffer;
  std::string bytes(flatbuffer.StrData(), flatbuffer.Size());
  bytes.append(model.gap, char{0});
  bytes.append(model.appended.StrData(), model.appended.Size());
  return bytes;
}

std::string Mutated(const std::string& bytes,
                    const std::function<void(tflite::ModelT&)>& mutate) {
  std::unique_ptr<tflite::ModelT> model(
      tflite::GetModel(bytes.data())->UnPack());
  mutate(*model);
  auto carved = SerializeCarvedModel(*model, AsBuffer(bytes));
  return carved.ok() ? ToBytes(*carved) : std::string();
}

}  // namespace litert::tools::testing
