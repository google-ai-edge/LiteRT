// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_STUB_TFLITE_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_STUB_TFLITE_CONTEXT_H_

#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/util.h"

namespace litert::ml_drift {

// StubTfLiteContext is a TfLiteContext which has 3 nodes as the followings.
// dummyAdd -> target op -> dummyAdd
class StubTfLiteContext : public TfLiteContext {
 public:
  StubTfLiteContext(const int builtin_code, const int op_version,
                    const int input_offset, const int num_inputs,
                    const std::vector<int>& shape)
      : TfLiteContext({0}) {
    // Stub execution plan
    exec_plan_ = TfLiteIntArrayCreate(3);
    for (int i = 0; i < 3; ++i) exec_plan_->data[i] = i;

    int tensor_no = 0;
    std::memset(nodes_, 0, sizeof(nodes_));
    std::memset(registrations_, 0, sizeof(registrations_));

    // Node 0, dummyAdd
    nodes_[0].inputs = TfLiteIntArrayCreate(1);
    nodes_[0].inputs->data[0] = tensor_no++;
    nodes_[0].outputs = TfLiteIntArrayCreate(1);
    nodes_[0].outputs->data[0] = tensor_no;
    nodes_[0].builtin_data = nullptr;

    // Node 1, target op
    nodes_[1].inputs = TfLiteIntArrayCreate(num_inputs + input_offset);
    for (int i = 0; i < input_offset; ++i) {
      nodes_[1].inputs->data[i] = -1;
    }
    for (int i = 0; i < num_inputs; ++i) {
      nodes_[1].inputs->data[i + input_offset] = tensor_no++;
    }
    nodes_[1].outputs = TfLiteIntArrayCreate(1);
    nodes_[1].outputs->data[0] = tensor_no;
    nodes_[1].builtin_data = operator new(1024);
    std::memset(nodes_[1].builtin_data, 0, 1024);

    // Node 2, dummyAdd
    nodes_[2].inputs = TfLiteIntArrayCreate(1);
    nodes_[2].inputs->data[0] = tensor_no++;
    nodes_[2].outputs = TfLiteIntArrayCreate(1);
    nodes_[2].outputs->data[0] = tensor_no++;
    nodes_[2].builtin_data = nullptr;

    int data_size = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<int>());

    // Create tensors of float32 with specified dimension
    tensors_.resize(tensor_no);
    for (size_t i = 0; i < tensors_.size(); ++i) {
      std::memset(&tensors_[i], 0, sizeof(tensors_[i]));
      tensors_[i].buffer_handle = kTfLiteNullBufferHandle;
      tensors_[i].type = kTfLiteFloat32;
      tensors_[i].dims = tflite::ConvertVectorToTfLiteIntArray(shape);
      tensors_[i].data.f = new float[data_size]();
      tensors_[i].bytes = data_size * sizeof(float);
      tensors_[i].type = kTfLiteFloat32;
      tensors_[i].allocation_type = kTfLiteArenaRw;
    }
    tensors = tensors_.data();
    tensors_size = tensors_.size();

    // Create registrations
    registrations_[0].builtin_code = kTfLiteBuiltinAdd;
    registrations_[1].builtin_code = builtin_code;
    registrations_[1].version = op_version;
    registrations_[2].builtin_code = kTfLiteBuiltinAdd;

    this->GetExecutionPlan = StubGetExecutionPlan;
    this->GetNodeAndRegistration = StubGetNodeAndRegistration;
  }

  // Constructor for original interface, default 0 offset
  StubTfLiteContext(const int builtin_code, const int op_version,
                    const int num_inputs, const std::vector<int>& shape)
      : StubTfLiteContext(builtin_code, op_version, 0, num_inputs, shape) {}

  StubTfLiteContext(const int builtin_code, const int op_version,
                    const int num_inputs,
                    const std::vector<::ml_drift::BHWC>& input_shapes)
      : StubTfLiteContext(builtin_code, op_version, num_inputs,
                          std::vector({1, 1, 1, 1})) {
    ABSL_CHECK(!input_shapes.empty());

    // Set tensor 0 shape to input_shapes[0]
    ChangeTensorShape(0, {input_shapes[0].b, input_shapes[0].h,
                          input_shapes[0].w, input_shapes[0].c});

    // Set tensor 1 to tensor n shape to input_shapes[1] to input_shapes[n]
    for (size_t i = 0; i < input_shapes.size(); ++i) {
      ChangeTensorShape(i + 1, {input_shapes[i].b, input_shapes[i].h,
                                input_shapes[i].w, input_shapes[i].c});
    }

    // Set tensor n + 1 shape to input_shapes[n]
    ChangeTensorShape(input_shapes.size(),
                      {input_shapes.back().b, input_shapes.back().h,
                       input_shapes.back().w, input_shapes.back().c});
  }

  void ChangeTensorShape(int id, const std::vector<int>& shape) {
    int data_size = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<int>());

    TfLiteIntArrayFree(tensors_[id].dims);
    delete[] tensors_[id].data.f;
    tensors_[id].dims = tflite::ConvertVectorToTfLiteIntArray(shape);
    tensors_[id].data.f = new float[data_size]();
    tensors_[id].bytes = data_size * sizeof(float);
  }

  void QuantizeTensor(int id, TfLiteType type) {
    tensors_[id].type = type;
    tensors_[id].quantization.type = kTfLiteAffineQuantization;
    TfLiteAffineQuantization* quant_params = new TfLiteAffineQuantization();
    quant_params->scale = TfLiteFloatArrayCreate(1);
    tensors_[id].quantization.params = reinterpret_cast<void*>(quant_params);
  }

  void SetTensorType(int id, TfLiteType type,
                     TfLiteAllocationType allocation_type) {
    tensors_[id].type = type;
    tensors_[id].allocation_type = allocation_type;
  }

  // Makes input tensor with tensor_id of the main node quantized.
  void MakeTensorQuantized(int main_node_input_id) {
    int id = nodes_[1].inputs->data[main_node_input_id];
    ChangeTensorShape(id, {1, 1});
    QuantizeTensor(id, kTfLiteInt8);
  }

  ~StubTfLiteContext() {
    for (auto& node : nodes_) {
      TfLiteIntArrayFree(node.inputs);
      TfLiteIntArrayFree(node.outputs);
      if (node.builtin_data) {
        operator delete(node.builtin_data);
      }
    }
    for (auto& tensor : tensors_) {
      TfLiteIntArrayFree(tensor.dims);
      delete[] tensor.data.f;
      if (tensor.quantization.params) {
        if (tensor.quantization.type == kTfLiteAffineQuantization) {
          TfLiteAffineQuantization* quant_params =
              reinterpret_cast<TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          TfLiteFloatArrayFree(quant_params->scale);
          delete quant_params;
        } else if (tensor.quantization.type == kTfLiteBlockwiseQuantization) {
          TfLiteBlockwiseQuantization* quant_params =
              reinterpret_cast<TfLiteBlockwiseQuantization*>(
                  tensor.quantization.params);
          delete quant_params;
        } else {
          ABSL_CHECK(false)
              << "Unsupported quantization type with non-null params: "
              << static_cast<int>(tensor.quantization.type);
        }
      }
    }
    TfLiteIntArrayFree(exec_plan_);
  }

  TfLiteIntArray* exec_plan() const { return exec_plan_; }
  TfLiteNode* node() { return &nodes_[1]; }
  TfLiteRegistration* registration() { return &registrations_[1]; }
  TfLiteNode* node(int node_index) { return &nodes_[node_index]; }
  TfLiteRegistration* registration(int reg_index) {
    return &registrations_[reg_index];
  }
  TfLiteTensor* tensor(int tensor_index) { return &tensors_[tensor_index]; }

 private:
  static TfLiteStatus StubGetExecutionPlan(TfLiteContext* context,
                                           TfLiteIntArray** execution_plan) {
    StubTfLiteContext* stub = reinterpret_cast<StubTfLiteContext*>(context);
    *execution_plan = stub->exec_plan();
    return kTfLiteOk;
  }

  static TfLiteStatus StubGetNodeAndRegistration(
      TfLiteContext* context, int node_index, TfLiteNode** node,
      TfLiteRegistration** registration) {
    StubTfLiteContext* stub = reinterpret_cast<StubTfLiteContext*>(context);
    *node = stub->node(node_index);
    *registration = stub->registration(node_index);
    return kTfLiteOk;
  }

  TfLiteIntArray* exec_plan_;
  TfLiteNode nodes_[3];
  TfLiteRegistration registrations_[3];
  std::vector<TfLiteTensor> tensors_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_STUB_TFLITE_CONTEXT_H_
