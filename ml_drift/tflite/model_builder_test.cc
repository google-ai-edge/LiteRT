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

#include "third_party/odml/litert/ml_drift/tflite/model_builder.h"

#include <stddef.h>

#include <any>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/model_builder_internal.h"
#include "third_party/odml/litert/ml_drift/tflite/object_reader.h"
#include "third_party/odml/litert/ml_drift/tflite/operation_parser.h"
#include "third_party/odml/litert/ml_drift/tflite/shared_const_tensor_map.h"
#include "third_party/odml/litert/ml_drift/tflite/stub_tflite_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/subgraph.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift {
namespace {

using absl_testing::StatusIs;
using testing::_;
using testing::HasSubstr;

class DelegatedInterpreter {
 public:
  explicit DelegatedInterpreter(int num_nodes) {
    exec_plan_ = TfLiteIntArrayCreate(num_nodes);
  }
  virtual ~DelegatedInterpreter() {
    TfLiteIntArrayFree(exec_plan_);
    for (auto params : delegate_params_) {
      TfLiteIntArrayFree(params.nodes_to_replace);
      TfLiteIntArrayFree(params.input_tensors);
      TfLiteIntArrayFree(params.output_tensors);
    }
  }

  // Get the TfLiteContext to be mocked for swapping out functions that have to
  // be called inside delegate (i.e. in delegate kernel mode).
  TfLiteContext* context() { return interpreter_.primary_subgraph().context(); }

  // node(int) and registration(int) are used to implement
  // GetNodeAndRegistration.  We can't implement those using
  //   TfLiteContext *context = interpreter_.primary_subgraph().context();
  //   context->GetNodeAndRegistration(context, &node, &registration);
  // here, because calling GetNodeAndRegistration from within it's own
  // implementation would lead to an infinite loop.
  // Instead, we just call node_and_registration and use a const_cast.
  // These const_casts are a bit ugly, but I think less ugly than exposing
  // the private GetNodeAndRegistration method in Subgraph as public,
  // or making this class a friend of Subgraph.
  TfLiteNode* node(int index) {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration =
        interpreter_.primary_subgraph().node_and_registration(index);
    return const_cast<TfLiteNode*>(&node_and_registration->first);
  }
  TfLiteRegistration* registration(int index) {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration =
        interpreter_.primary_subgraph().node_and_registration(index);
    return const_cast<TfLiteRegistration*>(&node_and_registration->second);
  }

  TfLiteIntArray* exec_plan() {
    // This simulates how TFLite's GetExecutionPlan invalidates previous
    // output before returning new data.
    const int num_nodes = exec_plan_->size;
    TfLiteIntArray* new_array = TfLiteIntArrayCreate(num_nodes);
    std::memcpy(new_array->data, exec_plan_->data, num_nodes * sizeof(int32_t));
    TfLiteIntArrayFree(exec_plan_);
    exec_plan_ = new_array;
    return exec_plan_;
  }
  TfLiteDelegateParams* add_delegate_params() {
    delegate_params_.push_back(TfLiteDelegateParams());
    return &delegate_params_.back();
  }
  TfLiteDelegateParams* delegate_params() { return &delegate_params_.front(); }
  int num_delegate_params() { return delegate_params_.size(); }
  tflite::Interpreter& interpreter() { return interpreter_; }

 protected:
  tflite::Interpreter interpreter_;

 private:
  // The manually-set execution plan for this delegated interpreter.
  TfLiteIntArray* exec_plan_ = nullptr;

  // The TfLiteDelegateParams object that's manually populated inside the mocked
  // TfLiteContext::PreviewDelegatePartitioning.
  std::vector<TfLiteDelegateParams> delegate_params_;
};

// TODO: b/350098405 - Fix this test when shared tensors are supported.
TEST(DISABLED_ModelBuilderTest, SharedTensorsTest) {
  DelegatedInterpreter interpreter = DelegatedInterpreter(0);
  ::ml_drift::GraphFloat32 graph;
  SharedConstTensorsMap shared_tensors;
  TfLiteContext* context = interpreter.context();
  EXPECT_NE(context, nullptr);

  TfLiteDelegateParams* params = interpreter.add_delegate_params();
  params->nodes_to_replace = TfLiteIntArrayCreate(1);
  params->nodes_to_replace->data[0] = 2;
  params->input_tensors = TfLiteIntArrayCreate(2);
  params->input_tensors->data[0] = 1;
  params->input_tensors->data[1] = 3;
  params->output_tensors = TfLiteIntArrayCreate(1);
  params->output_tensors->data[0] = 4;

  const TfLiteRegistration reg = {nullptr, nullptr, nullptr,
                                  nullptr, nullptr, kTfLiteBuiltinDequantize};
  EXPECT_EQ(interpreter.interpreter().AddNodeWithParameters(
                /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                /*init_data_size=*/0, /*builtin_data=*/nullptr,
                /*registration=*/&reg),
            kTfLiteError);
  EXPECT_FALSE(BuildFinalModel(context, params, /*options=*/{}, &graph,
                               /*quant_conversion_map=*/nullptr,
                               &shared_tensors)
                   .ok());
  TfLiteIntArrayFree(params->nodes_to_replace);
  TfLiteIntArrayFree(params->input_tensors);
  TfLiteIntArrayFree(params->output_tensors);
}

void* alloc_builtin_data(TfLiteBuiltinOperator op) {
  switch (op) {
    case kTfLiteBuiltinAdd:
      return static_cast<void*>(new TfLiteAddParams());
    case kTfLiteBuiltinMul:
      return static_cast<void*>(new TfLiteMulParams());
    case kTfLiteBuiltinSub:
      return static_cast<void*>(new TfLiteSubParams());
    case kTfLiteBuiltinDiv:
      return static_cast<void*>(new TfLiteDivParams());
    case kTfLiteBuiltinSplit: {
      TfLiteSplitParams* params = new TfLiteSplitParams();
      params->num_splits = 2;
      return static_cast<void*>(params);
    }
    default:
      return malloc(sizeof(int));
  }
}

class InterpreterFp16 : public DelegatedInterpreter {
 public:
  explicit InterpreterFp16(TfLiteBuiltinOperator op,
                           bool const_dequantize_inputs = true)
      : DelegatedInterpreter(3) {
    void* builtin_data = alloc_builtin_data(op);

    EXPECT_EQ(interpreter_.AddTensors(5), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 2}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({4}), kTfLiteOk);

    // Add a Dequantize Node.
    const TfLiteRegistration reg_dequant0 = {
        nullptr, nullptr, nullptr, nullptr, nullptr, kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant0),
              kTfLiteOk);

    // Add a Dequantize Node.
    const TfLiteRegistration reg_dequant1 = {
        nullptr, nullptr, nullptr, nullptr, nullptr, kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{2}, /*outputs=*/{3}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant1),
              kTfLiteOk);

    // Add a node that GPU delegate can parse.
    const TfLiteRegistration reg_op0 = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int(1));
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        op};

    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{1, 3}, /*outputs=*/{4}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_op0),
              kTfLiteOk);

    // Set inputs to Dequantize node to the fp16 type, and outputs
    // to fp32 type.
    const std::vector<int> dims = {1};
    TfLiteQuantization quantization;
    quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            0, TfLiteType::kTfLiteFloat16, "t0", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            2, TfLiteType::kTfLiteFloat16, "t2", dims, quantization, false),
        kTfLiteOk);
    if (const_dequantize_inputs) {
      // This simulates the dequantize inputs being constants in the graph.
      // If this is not true, FP16GraphPartitionHelper should not consider the
      // corresponding DEQUANTIZE ops.
      auto* tensor0 = interpreter_.tensor(0);
      auto* tensor2 = interpreter_.tensor(2);
      tensor0->allocation_type = kTfLiteMmapRo;
      tensor2->allocation_type = kTfLiteMmapRo;
    }
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteFloat32, "t1", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, quantization, false),
        kTfLiteOk);

    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            4, TfLiteType::kTfLiteFloat32, "t4", dims, quantization, false),
        kTfLiteOk);

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
    exec_plan()->data[2] = 2;
  }
};

// **NOTE**: we have several interpreter instances created at global scope to
// test *exactly* the GetOpsToReplace function alone, and not the sequence of
// function calls that includes GetOpsToReplace when calling
// ModifyGraphWithDelegate. A TfLiteContext is needed to test GetOpsToReplace,
// but TfLiteContexts intentionally make it difficult to call certain functions
// in a non-delegate context (see tensorflow/lite/subgraph/subgraph.cc for
// details) We create our own GetExecutionPlan, GetNodeAndRegistration and
// PreviewDelegatePartitioning lambdas inside each test, but we can't use local
// captures without changing the function signature. Therefore, this test data
// lives at global scope in order to be accessible inside the lambda.

InterpreterFp16* interpreter_fp16_add_op =
    new InterpreterFp16(kTfLiteBuiltinAdd);

// TODO: crbug.com/435537262 - This test depends on the incorrect premise that
// the delegate supports passing two constant tensors as inputs to Add.
TEST(ModelBuilderTest, DISABLED_GetOpsToReplaceAcceptsFp16DequantizeNodes) {
  // Before pruning, the graph has three nodes:
  //
  //   t0 (FP16) -> DequantNode -> t1 (FP32) -> Add -> t4
  //   t2 (FP16) -> DequantNode -> t3 (FP32) --/
  //
  // OpsToReplace should choose all three nodes for replacement, and
  // the graph on the GPU will look like this (no Dequants):
  //
  //   t0 (FP16) --> Add -> t4
  //   t2 (FP16) --/
  //
  TfLiteContext* context = interpreter_fp16_add_op->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp16_add_op->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_fp16_add_op->node(node_index);
    *registration = interpreter_fp16_add_op->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // The partitioner should accept only the Add op initially.
        EXPECT_EQ(nodes_to_replace->size, 1);
        // Single partition output.
        auto params = interpreter_fp16_add_op->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 2;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 3;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 4;

        *partition_params_array = interpreter_fp16_add_op->delegate_params();
        *num_partitions = interpreter_fp16_add_op->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // Ensure all nodes are delegated, and the ADD op has FP16 inputs.
  EXPECT_EQ(ops_to_replace->size, 3);
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  context->GetNodeAndRegistration(context, /**node_id**/ 2, &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  TfLiteIntArrayFree(ops_to_replace);
}

InterpreterFp16* interpreter_fp16_non_constant =
    new InterpreterFp16(kTfLiteBuiltinAdd, /*const_dequantize_inputs=*/false);

// Same as GetOpsToReplaceAcceptsFp16DequantizeNodes, but the DEQUANTIZE inputs
// are not constant. As a result, we don't allow the delegate to accept them.
TEST(ModelBuilderTest, GetOpsToReplaceRejectsNonConstantFp16DequantizeNodes) {
  TfLiteContext* context = interpreter_fp16_non_constant->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp16_non_constant->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_fp16_non_constant->node(node_index);
    *registration = interpreter_fp16_non_constant->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // The partitioner should accept only the Add op initially.
        EXPECT_EQ(nodes_to_replace->size, 1);
        // Single partition output.
        auto params = interpreter_fp16_non_constant->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 2;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 3;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 4;

        *partition_params_array =
            interpreter_fp16_non_constant->delegate_params();
        *num_partitions = interpreter_fp16_non_constant->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // Only ADD is delegated, with FP32 (dequantized) inputs.
  EXPECT_EQ(ops_to_replace->size, 1);
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  context->GetNodeAndRegistration(context, ops_to_replace->data[0], &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat32);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat32);
  TfLiteIntArrayFree(ops_to_replace);
}

InterpreterFp16* interpreter_fp16_gt_op =
    new InterpreterFp16(kTfLiteBuiltinGreater);

TEST(ModelBuilderTest, GetOpsToReplaceRejectsFp16DequantizeNodes) {
  // Before pruning, the graph has three nodes:
  //
  //   t0 (FP16) -> DequantNode -> t1 (FP32) -> Greater Op -> t4
  //   t2 (FP16) -> DequantNode -> t3 (FP32) --/
  //
  // Because there is no GPU equivalent for the Greater op, we don't choose any
  // nodes.

  TfLiteContext* context = interpreter_fp16_gt_op->context();
  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp16_gt_op->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_fp16_gt_op->node(node_index);
    *registration = interpreter_fp16_gt_op->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // No selected nodes.
        EXPECT_EQ(nodes_to_replace->size, 0);
        *partition_params_array = nullptr;
        *num_partitions = 0;
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // No nodes were found to replace.
  EXPECT_EQ(ops_to_replace->size, 0);
  // Inputs to Greater op are still fp32.
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  const int kGreaterOpIndex = 2;
  context->GetNodeAndRegistration(context, kGreaterOpIndex, &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat32);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat32);
  TfLiteIntArrayFree(ops_to_replace);
}

class InterpreterFp32 : public DelegatedInterpreter {
 public:
  InterpreterFp32() : DelegatedInterpreter(2) {
    void* builtin_data = alloc_builtin_data(kTfLiteBuiltinAdd);
    EXPECT_EQ(interpreter_.AddTensors(4), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 2}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({3}), kTfLiteOk);

    // Add a Dequantize Node with uint8 input.
    const TfLiteRegistration reg_dequant0 = {/*init=*/nullptr,
                                             /*free=*/nullptr,
                                             /*prepare=*/nullptr,
                                             /*invoke=*/nullptr,
                                             /*profiling_string=*/nullptr,
                                             kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant0),
              kTfLiteOk);

    // Add a node that GPU delegate can parse.
    const TfLiteRegistration reg_add0 = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int(1));
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        kTfLiteBuiltinAdd};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{1, 2}, /*outputs=*/{3}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_add0),
              kTfLiteOk);

    const std::vector<int> dims = {1};
    TfLiteQuantization quantization;
    quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  0, TfLiteType::kTfLiteUInt8, "t0", dims, quantization, false),
              kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteFloat32, "t1", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            2, TfLiteType::kTfLiteFloat32, "t2", dims, quantization, false),
        kTfLiteOk);

    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, quantization, false),
        kTfLiteOk);

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
  }
};

InterpreterFp32* interpreter_fp32 = new InterpreterFp32();

TEST(ModelBuilderTest, GetOpsToReplaceDoesNotPruneUint8) {
  // A graph with a Dequant node with uint8 input is not pruned. As this op is
  // currently not supported on the GPU. Therefore, the Dequant op will be
  // scheduled to run on the CPU while the remaining supported op Add on the
  // GPU.
  //
  //   t0 (uint8) --> Dequant --> t1 (FP32) --> Add -> t3
  //                              t2 (FP32) --/
  //
  TfLiteContext* context = interpreter_fp32->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp32->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_fp32->node(node_index);
    *registration = interpreter_fp32->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        auto params = interpreter_fp32->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 1;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 2;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 3;

        *partition_params_array = interpreter_fp32->delegate_params();
        *num_partitions = interpreter_fp32->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // As the Dequant op is not pruned and the ADD op could run on GPU, we have
  // 1 partition.
  EXPECT_EQ(ops_to_replace->size, 1);
  // ADD at index 1.
  EXPECT_EQ(1, ops_to_replace->data[0]);

  TfLiteIntArrayFree(ops_to_replace);
}

class Interpreter2Fp32 : public DelegatedInterpreter {
 public:
  Interpreter2Fp32() : DelegatedInterpreter(4) {
    void* builtin_data = alloc_builtin_data(kTfLiteBuiltinAdd);
    EXPECT_EQ(interpreter_.AddTensors(8), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 2, 4, 6}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({7}), kTfLiteOk);

    // Add a Dequantize Node with uint8 input.
    const TfLiteRegistration reg_dequant = {/*init=*/nullptr,
                                            /*free=*/nullptr,
                                            /*prepare=*/nullptr,
                                            /*invoke=*/nullptr,
                                            /*profiling_string=*/nullptr,
                                            kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant),
              kTfLiteOk);

    // Add an ADD node that GPU delegate can parse.
    const TfLiteRegistration reg_add0 = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int(1));
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        kTfLiteBuiltinAdd};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{1, 2}, /*outputs=*/{3}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_add0),
              kTfLiteOk);

    // Add a Pack Node that GPU delegate doesn't support
    const TfLiteRegistration reg_pack = {/*init=*/nullptr,
                                         /*free=*/nullptr,
                                         /*prepare=*/nullptr,
                                         /*invoke=*/nullptr,
                                         /*profiling_string=*/nullptr,
                                         kTfLiteBuiltinPack};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{3, 4}, /*outputs=*/{5}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_pack),
              kTfLiteOk);

    const TfLiteRegistration reg_add1 = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int[2]);
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        kTfLiteBuiltinAdd};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{5, 6}, /*outputs=*/{7}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_add1),
              kTfLiteOk);

    std::vector<int> dims = {1};
    TfLiteQuantization quantization;
    quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  0, TfLiteType::kTfLiteUInt8, "t0", dims, quantization, false),
              kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteFloat32, "t1", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            2, TfLiteType::kTfLiteFloat32, "t2", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            4, TfLiteType::kTfLiteFloat32, "t4", dims, quantization, false),
        kTfLiteOk);

    dims.push_back(2);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            5, TfLiteType::kTfLiteFloat32, "t5", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            6, TfLiteType::kTfLiteFloat32, "t6", dims, quantization, false),
        kTfLiteOk);

    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            7, TfLiteType::kTfLiteFloat32, "t7", dims, quantization, false),
        kTfLiteOk);

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
    exec_plan()->data[2] = 2;
    exec_plan()->data[3] = 3;
  }
};

Interpreter2Fp32* interpreter2_fp32 = new Interpreter2Fp32();

TEST(ModelBuilderTest, GetOpsToReplaceMultiplePartitions) {
  // A graph with a Dequant node with uint8 input, a Pack node are not pruned.
  // As these ops are currently not supported on the GPU, they will be scheduled
  // to run on the CPU while the remaining supported op Add on the GPU.
  //
  //   t0 (uint8) -> Dequant(0) -> t1 (FP32) -> Add(1) -> t3 (FP32) -> PACK (2)
  //                               t2 (FP32) -/           t4 (FP32) -/
  //   PACK (2) -> t5 (FP32) -> Add(3) -> t7
  //            -> t6 (FP32) -/
  //
  TfLiteContext* context = interpreter2_fp32->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter2_fp32->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter2_fp32->node(node_index);
    *registration = interpreter2_fp32->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        auto params = interpreter2_fp32->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 1;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 2;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 3;

        params = interpreter2_fp32->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 3;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 5;
        params->input_tensors->data[1] = 6;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 7;

        *partition_params_array = interpreter2_fp32->delegate_params();
        *num_partitions = interpreter2_fp32->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(
      context, /*allow_quant_ops=*/false, /*max_delegated_partitions*/ 2);

  // As the Dequant op is not pruned and the ADD op could run on GPU, we have
  // 2 partitions with an ADD each (op #1 and op #3).
  ASSERT_EQ(ops_to_replace->size, 2);
  EXPECT_THAT(absl::MakeConstSpan(ops_to_replace->data, 2),
              testing::UnorderedElementsAre(1, 3));

  TfLiteIntArrayFree(ops_to_replace);
}

class InterpreterMultiNode : public DelegatedInterpreter {
 public:
  explicit InterpreterMultiNode(bool both_ops_supported = true)
      : DelegatedInterpreter(5) {
    void* builtin_data = alloc_builtin_data(kTfLiteBuiltinAdd);
    EXPECT_EQ(interpreter_.AddTensors(8), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 1, 2}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({6, 7}), kTfLiteOk);

    // Add 3 Dequantize Nodes with float16 input.
    for (int i = 0; i < 3; ++i) {
      const TfLiteRegistration reg_dequant = {/*init=*/nullptr,
                                              /*free=*/nullptr,
                                              /*prepare=*/nullptr,
                                              /*invoke=*/nullptr,
                                              /*profiling_string=*/nullptr,
                                              kTfLiteBuiltinDequantize};
      EXPECT_EQ(interpreter_.AddNodeWithParameters(
                    /*inputs=*/{i}, /*outputs=*/{i + 3}, /*init_data=*/nullptr,
                    /*init_data_size=*/0, /*builtin_data=*/nullptr,
                    /*registration=*/&reg_dequant),
                kTfLiteOk);
    }

    if (both_ops_supported) {
      // Add 2 ADD ops.
      const TfLiteRegistration reg_add0 = {
          [](TfLiteContext* context, const char* buffer, size_t length) {
            return reinterpret_cast<void*>(new int(1));
          },
          [](TfLiteContext* context, void* buffer) {
            delete reinterpret_cast<int*>(buffer);
          },
          nullptr,
          nullptr,
          nullptr,
          kTfLiteBuiltinAdd};
      EXPECT_EQ(interpreter_.AddNodeWithParameters(
                    /*inputs=*/{4, 5}, /*outputs=*/{7}, /*init_data=*/nullptr,
                    /*init_data_size=*/0,
                    /*builtin_data=*/builtin_data,
                    /*registration=*/&reg_add0),
                kTfLiteOk);

      const TfLiteRegistration reg_add1 = {
          [](TfLiteContext* context, const char* buffer, size_t length) {
            return reinterpret_cast<void*>(new int(1));
          },
          [](TfLiteContext* context, void* buffer) {
            delete reinterpret_cast<int*>(buffer);
          },
          nullptr,
          nullptr,
          nullptr,
          kTfLiteBuiltinAdd};
      EXPECT_EQ(interpreter_.AddNodeWithParameters(
                    /*inputs=*/{3, 4}, /*outputs=*/{6}, /*init_data=*/nullptr,
                    /*init_data_size=*/0,
                    /*builtin_data=*/builtin_data,
                    /*registration=*/&reg_add1),
                kTfLiteOk);
    } else {
      // Add the GREATER op node that GPU delegate doesn't support.
      const TfLiteRegistration reg_greater = {
          [](TfLiteContext* context, const char* buffer, size_t length) {
            return reinterpret_cast<void*>(new int(1));
          },
          [](TfLiteContext* context, void* buffer) {
            delete reinterpret_cast<int*>(buffer);
          },
          nullptr,
          nullptr,
          nullptr,
          kTfLiteBuiltinGreater};
      EXPECT_EQ(interpreter_.AddNodeWithParameters(
                    /*inputs=*/{3, 4}, /*outputs=*/{6}, /*init_data=*/nullptr,
                    /*init_data_size=*/0,
                    /*builtin_data=*/builtin_data,
                    /*registration=*/&reg_greater),
                kTfLiteOk);

      // Add the ADD op node that GPU delegate supports.
      const TfLiteRegistration reg_add0 = {
          [](TfLiteContext* context, const char* buffer, size_t length) {
            return reinterpret_cast<void*>(new int(1));
          },
          [](TfLiteContext* context, void* buffer) {
            delete reinterpret_cast<int*>(buffer);
          },
          nullptr,
          nullptr,
          nullptr,
          kTfLiteBuiltinAdd};
      EXPECT_EQ(interpreter_.AddNodeWithParameters(
                    /*inputs=*/{4, 5}, /*outputs=*/{7}, /*init_data=*/nullptr,
                    /*init_data_size=*/0,
                    /*builtin_data=*/builtin_data,
                    /*registration=*/&reg_add0),
                kTfLiteOk);
    }
    const std::vector<int> dims = {1};
    TfLiteQuantization quantization;
    quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            0, TfLiteType::kTfLiteFloat16, "t0", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteFloat16, "t1", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            2, TfLiteType::kTfLiteFloat16, "t2", dims, quantization, false),
        kTfLiteOk);
    // Simulate DEQUANTIZE inputs being constants.
    auto* tensor0 = interpreter_.tensor(0);
    auto* tensor1 = interpreter_.tensor(1);
    auto* tensor2 = interpreter_.tensor(2);
    tensor0->allocation_type = kTfLiteMmapRo;
    tensor1->allocation_type = kTfLiteMmapRo;
    tensor2->allocation_type = kTfLiteMmapRo;
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            4, TfLiteType::kTfLiteFloat32, "t4", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            5, TfLiteType::kTfLiteFloat32, "t5", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            6, TfLiteType::kTfLiteFloat32, "t6", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            7, TfLiteType::kTfLiteFloat32, "t7", dims, quantization, false),
        kTfLiteOk);

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
    exec_plan()->data[2] = 2;
    exec_plan()->data[3] = 3;
    exec_plan()->data[4] = 4;
  }
};

InterpreterMultiNode* interpreter_mn =
    new InterpreterMultiNode(/*both_ops_supported*/ false);

// TODO: crbug.com/435537262 - This test depends on the incorrect premise that
// the delegate supports passing two constant tensors as inputs to Add.
TEST(ModelBuilderTest,
     DISABLED_GetOpsToReplaceSelectsCorrectFp16Nodes_SingleDelegatedPartition) {
  // A graph with three Dequant nodes feeding two ops, 'Add' and 'Greater'.
  // 'Add' can be replaced by the GPU delegate, but 'Greater' can not.
  //   t0 (FP16) --> Dequant(0) --> t3 (FP32) --> Greater(3) -> t6
  //   t1 (FP16) --> Dequant(1) --> t4 (FP32) --/
  //                                          --\
  //   t3 (FP16) --> Dequant(2) --> t5 (FP32) --> Add(4) -> t7
  //
  //  OpsToReplace should ONLY accept 'Add'.
  TfLiteContext* context = interpreter_mn->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_mn->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_mn->node(node_index);
    *registration = interpreter_mn->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // The FP16GraphPartitioner should only mark the ADD op as accepted.
        EXPECT_EQ(nodes_to_replace->size, 1);
        EXPECT_EQ(nodes_to_replace->data[0], 4);
        // Single partition.
        auto params = interpreter_mn->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 4;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 3;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 7;

        *partition_params_array = interpreter_mn->delegate_params();
        *num_partitions = interpreter_mn->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  EXPECT_EQ(ops_to_replace->size, 1);
  // Op at index 4 is the Add op.
  EXPECT_EQ(ops_to_replace->data[0], 4);
  // Verify that Add op has fp16 inputs.
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  context->GetNodeAndRegistration(context, ops_to_replace->data[0], &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  TfLiteIntArrayFree(ops_to_replace);
}

InterpreterMultiNode* interpreter_mn2 =
    new InterpreterMultiNode(/*both_ops_supported*/ true);

// TODO: crbug.com/435537262 - This test depends on the incorrect premise that
// the delegate supports passing two constant tensors as inputs to Add.
TEST(ModelBuilderTest,
     DISABLED_GetOpsToReplaceSelectsCorrectFp16Nodes_MultiDelegatePartitions) {
  // A graph with three Dequant nodes feeding two Add ops.
  //   t0 (FP16) --> Dequant(0) --> t3 (FP32) --> Add(3) -> t6
  //   t1 (FP16) --> Dequant(1) --> t4 (FP32) --/
  //                                          --\
  //   t3 (FP16) --> Dequant(2) --> t5 (FP32) --> Add(4) -> t7
  //
  // In this test case, we purposely partition Add(3) & Add(4) into different
  // partitions from the runtime. However, since all non-DEQUANT ops are
  // delegated, the partitioner suggests delegating the DEQUANTs too.

  TfLiteContext* context = interpreter_mn2->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_mn2->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_mn2->node(node_index);
    *registration = interpreter_mn2->registration(node_index);
    return kTfLiteOk;
  };

  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // The FP16GraphPartitioner should only mark both ADD ops as accepted.
        EXPECT_EQ(nodes_to_replace->size, 2);
        EXPECT_EQ(nodes_to_replace->data[0], 3);
        EXPECT_EQ(nodes_to_replace->data[1], 4);
        // Technically, both ADD ops should end up in the same partition.
        // But we put them in different partitions to test post-processing with
        // DEQUANTIZE nodes.
        // First partition with Add(3).
        auto params = interpreter_mn2->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 3;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 3;
        params->input_tensors->data[1] = 4;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 6;
        // Second partition with Add(4).
        params = interpreter_mn2->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 4;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 4;
        params->input_tensors->data[1] = 5;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 7;

        *partition_params_array = interpreter_mn2->delegate_params();
        *num_partitions = interpreter_mn2->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(
      context, /*allow_quant_ops*/ false, /*max_delegated_partitions*/ 2);

  // All ops should be selected.
  EXPECT_EQ(ops_to_replace->size, 5);

  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  // Verify that both Add ops have fp16 inputs.
  context->GetNodeAndRegistration(context, /**node_index**/ 3, &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  context->GetNodeAndRegistration(context, /**node_index**/ 4, &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  TfLiteIntArrayFree(ops_to_replace);
}

// Adds the pattern:
//
// float -> QUANTIZE -> ADD -> DEQUANTIZE -> float
// float -> QUANTIZE ----^
//
// The tensors between the QUANTIZE & DEQUANTIZE nodes are int8.
class InterpreterQuantized : public DelegatedInterpreter {
 public:
  InterpreterQuantized() : DelegatedInterpreter(4) {
    void* builtin_data = alloc_builtin_data(kTfLiteBuiltinAdd);
    EXPECT_EQ(interpreter_.AddTensors(6), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 3}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({5}), kTfLiteOk);

    // QUANTIZE 1
    const TfLiteRegistration reg_quant0 = {/*init=*/nullptr,
                                           /*free=*/nullptr,
                                           /*prepare=*/nullptr,
                                           /*invoke=*/nullptr,
                                           /*profiling_string=*/nullptr,
                                           kTfLiteBuiltinQuantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_quant0),
              kTfLiteOk);

    // QUANTIZE 2
    const TfLiteRegistration reg_quant1 = {/*init=*/nullptr,
                                           /*free=*/nullptr,
                                           /*prepare=*/nullptr,
                                           /*invoke=*/nullptr,
                                           /*profiling_string=*/nullptr,
                                           kTfLiteBuiltinQuantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{3}, /*outputs=*/{2}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_quant1),
              kTfLiteOk);

    // ADD
    const TfLiteRegistration reg_add0 = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int(1));
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        kTfLiteBuiltinAdd};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{1, 2}, /*outputs=*/{4}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_add0),
              kTfLiteOk);

    // DEQUANTIZE
    const TfLiteRegistration reg_dequant0 = {/*init=*/nullptr,
                                             /*free=*/nullptr,
                                             /*prepare=*/nullptr,
                                             /*invoke=*/nullptr,
                                             /*profiling_string=*/nullptr,
                                             kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{4}, /*outputs=*/{5}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant0),
              kTfLiteOk);

    const std::vector<int> dims = {1, 3, 3, 2};

    // Input & output tensors are floating-point.
    TfLiteQuantization no_quantization;
    no_quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            0, TfLiteType::kTfLiteFloat32, "t0", dims, no_quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, no_quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            5, TfLiteType::kTfLiteFloat32, "t5", dims, no_quantization, false),
        kTfLiteOk);
    // Other tensors are int8.
    float scale = 0.5f;
    int32_t zero_point = 12;
    TfLiteQuantization rw_quantization;
    rw_quantization.type = kTfLiteAffineQuantization;
    auto* rw_affine_quantization = static_cast<TfLiteAffineQuantization*>(
        malloc(sizeof(TfLiteAffineQuantization)));
    rw_affine_quantization->scale = TfLiteFloatArrayCreate(1);
    rw_affine_quantization->zero_point = TfLiteIntArrayCreate(1);
    rw_affine_quantization->scale->data[0] = scale;
    rw_affine_quantization->zero_point->data[0] = zero_point;
    rw_quantization.params = rw_affine_quantization;
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteInt8, "t1", dims, rw_quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            2, TfLiteType::kTfLiteInt8, "t2", dims, rw_quantization, false),
        kTfLiteOk);

    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            4, TfLiteType::kTfLiteInt8, "t4", dims, rw_quantization, false),
        kTfLiteOk);

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
    exec_plan()->data[2] = 2;
    exec_plan()->data[3] = 3;
  }
};

class InterpreterSimpleAdd : public DelegatedInterpreter {
 public:
  InterpreterSimpleAdd() : DelegatedInterpreter(1) {
    void* builtin_data = alloc_builtin_data(kTfLiteBuiltinAdd);
    EXPECT_EQ(interpreter_.AddTensors(3), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 1}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({2}), kTfLiteOk);

    const TfLiteRegistration reg_add = {nullptr, nullptr, nullptr,
                                        nullptr, nullptr, kTfLiteBuiltinAdd};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0, 1}, /*outputs=*/{2}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, builtin_data,
                  /*registration=*/&reg_add),
              kTfLiteOk);

    const std::vector<int> dims = {1, 1, 1, 1};
    TfLiteQuantization quantization;
    quantization.type = kTfLiteNoQuantization;
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                    i, TfLiteType::kTfLiteFloat32, "t", dims, quantization),
                kTfLiteOk);
    }
    exec_plan()->data[0] = 0;
  }
};

InterpreterSimpleAdd* interpreter_simple_add = new InterpreterSimpleAdd();

TEST(ModelBuilderTest, BuildModel_SimpleAdd) {
  TfLiteContext* context = interpreter_simple_add->context();

  ::ml_drift::GraphFloat32 graph;
  TfLiteDelegateParams params;
  params.nodes_to_replace = TfLiteIntArrayCreate(1);
  params.nodes_to_replace->data[0] = 0;
  params.input_tensors = TfLiteIntArrayCreate(2);
  params.input_tensors->data[0] = 0;
  params.input_tensors->data[1] = 1;
  params.output_tensors = TfLiteIntArrayCreate(1);
  params.output_tensors->data[0] = 2;

  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_simple_add->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_simple_add->node(node_index);
    *registration = interpreter_simple_add->registration(node_index);
    return kTfLiteOk;
  };

  EXPECT_TRUE(BuildModel(context, &params, /*options=*/{}, &graph).ok());
  EXPECT_EQ(graph.nodes().size(), 1);
  EXPECT_EQ(
      ::ml_drift::OperationTypeFromString(graph.nodes()[0]->operation.type),
      ::ml_drift::OperationType::ADD);

  TfLiteIntArrayFree(params.nodes_to_replace);
  TfLiteIntArrayFree(params.input_tensors);
  TfLiteIntArrayFree(params.output_tensors);
}

class InterpreterEmbeddingLookupBlockwise : public DelegatedInterpreter {
 public:
  InterpreterEmbeddingLookupBlockwise() : DelegatedInterpreter(1) {
    EXPECT_EQ(interpreter_.AddTensors(4), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({3}), kTfLiteOk);

    const TfLiteRegistration reg_op = {nullptr, nullptr,
                                       nullptr, nullptr,
                                       nullptr, kTfLiteBuiltinEmbeddingLookup};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0, 1}, /*outputs=*/{3}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_op),
              kTfLiteOk);

    const std::vector<int> indices_dims = {1, 10};
    const std::vector<int> weights_dims = {100, 64};
    const std::vector<int> scale_dims = {100, 2};
    const std::vector<int> output_dims = {1, 10, 64};

    TfLiteQuantization no_quantization;
    no_quantization.type = kTfLiteNoQuantization;

    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  0, TfLiteType::kTfLiteInt32, "indices", indices_dims,
                  no_quantization, false),
              kTfLiteOk);

    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  2, TfLiteType::kTfLiteFloat32, "scale", scale_dims,
                  no_quantization, false),
              kTfLiteOk);
    interpreter_.tensor(2)->allocation_type = kTfLiteMmapRo;
    interpreter_.tensor(2)->bytes = 100 * 2 * sizeof(float);
    interpreter_.tensor(2)->data.raw =
        static_cast<char*>(malloc(interpreter_.tensor(2)->bytes));

    TfLiteQuantization blockwise_quantization;
    blockwise_quantization.type = kTfLiteBlockwiseQuantization;
    auto* blockwise_params = static_cast<TfLiteBlockwiseQuantization*>(
        malloc(sizeof(TfLiteBlockwiseQuantization)));
    blockwise_params->scale = 2;
    blockwise_params->zero_point = -1;
    blockwise_params->blocksize = 32;
    blockwise_quantization.params = blockwise_params;

    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  1, TfLiteType::kTfLiteInt8, "weights", weights_dims,
                  blockwise_quantization, false),
              kTfLiteOk);
    interpreter_.tensor(1)->allocation_type = kTfLiteMmapRo;
    interpreter_.tensor(1)->bytes = 100 * 64;
    interpreter_.tensor(1)->data.raw =
        static_cast<char*>(malloc(interpreter_.tensor(1)->bytes));

    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  3, TfLiteType::kTfLiteFloat32, "output", output_dims,
                  no_quantization, false),
              kTfLiteOk);

    exec_plan()->data[0] = 0;
  }

  ~InterpreterEmbeddingLookupBlockwise() override {
    free(interpreter_.tensor(2)->data.raw);
    free(interpreter_.tensor(1)->data.raw);
  }
};

TEST(ModelBuilderTest, BuildModel_EmbeddingLookupBlockwiseQuantized) {
  static auto* interpreter = new InterpreterEmbeddingLookupBlockwise();
  TfLiteContext* context = interpreter->context();

  ::ml_drift::GraphFloat32 graph;
  TfLiteDelegateParams params;
  params.nodes_to_replace = TfLiteIntArrayCreate(1);
  params.nodes_to_replace->data[0] = 0;
  params.input_tensors = TfLiteIntArrayCreate(1);
  params.input_tensors->data[0] = 0;
  params.output_tensors = TfLiteIntArrayCreate(1);
  params.output_tensors->data[0] = 3;

  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter->node(node_index);
    *registration = interpreter->registration(node_index);
    return kTfLiteOk;
  };

  SharedConstTensorsMap shared_tensors;
  TensorIndexToBufferIdMap tensor_to_buffer_id_map;
  tensor_to_buffer_id_map[1] =
      100;  // weights tensor (index 1) is shared with buffer ID 100

  EXPECT_TRUE(BuildFinalModel(context, &params, /*options=*/{}, &graph,
                              /*quant_conversion_map=*/nullptr, &shared_tensors,
                              &tensor_to_buffer_id_map,
                              /*tensor_to_external_buffer_id_map=*/nullptr)
                  .ok());

  ASSERT_EQ(graph.nodes().size(), 1);
  EXPECT_EQ(
      ::ml_drift::OperationTypeFromString(graph.nodes()[0]->operation.type),
      ::ml_drift::OperationType::EMBEDDING_LOOKUP);

  auto attr = std::any_cast<::ml_drift::EmbeddingLookupAttributes>(
      graph.nodes()[0]->operation.attributes);
  EXPECT_EQ(attr.weights_type,
            ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt8);
  EXPECT_EQ(attr.original_weights_shape.o, 100);
  EXPECT_EQ(attr.original_weights_shape.i, 64);
  EXPECT_EQ(attr.scale_zp_shape.o, 100);
  EXPECT_EQ(attr.scale_zp_shape.i, 2);  // 64 / 32 = 2

  ASSERT_EQ(shared_tensors.size(), 1);
  auto shared_tensor_info = shared_tensors.begin()->second;
  EXPECT_EQ(shared_tensor_info.tflite_tensor_id, 1);
  EXPECT_EQ(shared_tensor_info.global_id, 100);

  TfLiteIntArrayFree(params.nodes_to_replace);
  TfLiteIntArrayFree(params.input_tensors);
  TfLiteIntArrayFree(params.output_tensors);

  delete interpreter;
  interpreter = nullptr;
}

class InterpreterBroadcastMul : public DelegatedInterpreter {
 public:
  InterpreterBroadcastMul() : DelegatedInterpreter(1) {
    void* builtin_data = alloc_builtin_data(kTfLiteBuiltinMul);
    EXPECT_EQ(interpreter_.AddTensors(3), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 1}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({2}), kTfLiteOk);

    const TfLiteRegistration reg_mul = {nullptr, nullptr, nullptr,
                                        nullptr, nullptr, kTfLiteBuiltinMul};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0, 1}, /*outputs=*/{2}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, builtin_data,
                  /*registration=*/&reg_mul),
              kTfLiteOk);

    const std::vector<int> dims0 = {8};
    const std::vector<int> dims1 = {5, 1};
    const std::vector<int> dims2 = {5, 8};
    TfLiteQuantization quantization;
    quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  0, TfLiteType::kTfLiteFloat32, "t0", dims0, quantization),
              kTfLiteOk);
    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  1, TfLiteType::kTfLiteFloat32, "t1", dims1, quantization),
              kTfLiteOk);
    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  2, TfLiteType::kTfLiteFloat32, "t2", dims2, quantization),
              kTfLiteOk);
    exec_plan()->data[0] = 0;
  }
};

InterpreterBroadcastMul* interpreter_broadcast_mul =
    new InterpreterBroadcastMul();

TEST(ModelBuilderTest, BuildModel_BroadcastMul) {
  TfLiteContext* context = interpreter_broadcast_mul->context();

  ::ml_drift::GraphFloat32 graph;
  TfLiteDelegateParams params;
  params.nodes_to_replace = TfLiteIntArrayCreate(1);
  params.nodes_to_replace->data[0] = 0;
  params.input_tensors = TfLiteIntArrayCreate(2);
  params.input_tensors->data[0] = 0;
  params.input_tensors->data[1] = 1;
  params.output_tensors = TfLiteIntArrayCreate(1);
  params.output_tensors->data[0] = 2;

  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_broadcast_mul->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_broadcast_mul->node(node_index);
    *registration = interpreter_broadcast_mul->registration(node_index);
    return kTfLiteOk;
  };

  EXPECT_TRUE(BuildModel(context, &params, /*options=*/{}, &graph).ok());
  EXPECT_EQ(graph.nodes().size(), 4);
  EXPECT_EQ(
      ::ml_drift::OperationTypeFromString(graph.nodes()[0]->operation.type),
      ::ml_drift::OperationType::RESHAPE);
  EXPECT_EQ(
      ::ml_drift::OperationTypeFromString(graph.nodes()[1]->operation.type),
      ::ml_drift::OperationType::RESHAPE);
  EXPECT_EQ(
      ::ml_drift::OperationTypeFromString(graph.nodes()[2]->operation.type),
      ::ml_drift::OperationType::MUL);
  EXPECT_EQ(
      ::ml_drift::OperationTypeFromString(graph.nodes()[3]->operation.type),
      ::ml_drift::OperationType::RESHAPE);

  const auto& reshape_attr0 =
      std::any_cast<const ::ml_drift::ReshapeAttributes&>(
          graph.nodes()[0]->operation.attributes);
  EXPECT_EQ(reshape_attr0.new_shape, ::ml_drift::BHWC(1, 1, 1, 8));

  const auto& reshape_attr1 =
      std::any_cast<const ::ml_drift::ReshapeAttributes&>(
          graph.nodes()[1]->operation.attributes);
  EXPECT_EQ(reshape_attr1.new_shape, ::ml_drift::BHWC(1, 1, 5, 1));

  // Verifies that the reshape nodes consume the correct unswapped input
  // tensors. This ensures AddOpWithBroadcastReshape correctly matches the
  // incoming Value* containers with their corresponding broadcast reshape
  // dimensions, preventing dimension scrambling and silent tensor corruption.
  EXPECT_EQ(graph.FindInputs(graph.nodes()[0]->id)[0]->tensor.shape,
            ::ml_drift::BHWC(8, 1, 1, 1));
  EXPECT_EQ(graph.FindInputs(graph.nodes()[1]->id)[0]->tensor.shape,
            ::ml_drift::BHWC(5, 1, 1, 1));

  TfLiteIntArrayFree(params.nodes_to_replace);
  TfLiteIntArrayFree(params.input_tensors);
  TfLiteIntArrayFree(params.output_tensors);
}

TEST(ModelBuilderTest, GetOpsToReplace_SimpleAdd) {
  TfLiteContext* context = interpreter_simple_add->context();

  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_simple_add->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_simple_add->node(node_index);
    *registration = interpreter_simple_add->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        if (nodes_to_replace->size == 0) {
          *num_partitions = 0;
          return kTfLiteOk;
        }
        auto params = interpreter_simple_add->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 0;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 0;
        params->input_tensors->data[1] = 1;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 2;

        *partition_params_array = interpreter_simple_add->delegate_params();
        *num_partitions = interpreter_simple_add->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
  EXPECT_EQ(ops_to_replace->size, 1);
  EXPECT_EQ(ops_to_replace->data[0], 0);

  TfLiteIntArrayFree(ops_to_replace);
}

InterpreterQuantized* interpreter_quant = new InterpreterQuantized();
TEST(ModelBuilderTest, GetOpsToReplace_AllowQuantOps) {
  TfLiteContext* context = interpreter_quant->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_quant->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_quant->node(node_index);
    *registration = interpreter_quant->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        if (nodes_to_replace->size == 0) {
          *num_partitions = 0;
          return kTfLiteOk;
        } else if (nodes_to_replace->size == 4) {
          auto params = interpreter_quant->add_delegate_params();
          params->nodes_to_replace = TfLiteIntArrayCreate(4);
          params->nodes_to_replace->data[0] = 0;
          params->nodes_to_replace->data[1] = 1;
          params->nodes_to_replace->data[2] = 2;
          params->nodes_to_replace->data[3] = 3;
          params->input_tensors = TfLiteIntArrayCreate(2);
          params->input_tensors->data[0] = 0;
          params->input_tensors->data[1] = 3;
          params->output_tensors = TfLiteIntArrayCreate(1);
          params->output_tensors->data[0] = 5;

          *partition_params_array = interpreter_quant->delegate_params();
          *num_partitions = interpreter_quant->num_delegate_params();
          return kTfLiteOk;
        } else {
          // Shouldn't happen!
          return kTfLiteError;
        }
      };

  TfLiteIntArray* ops_to_replace =
      GetOpsToReplace(context, /**allow_quant_ops=*/true);
  // If we allow quant ops, all ops should get delegated.
  EXPECT_EQ(ops_to_replace->size, 4);

  TfLiteIntArray* ops_to_replace_without_quant =
      GetOpsToReplace(context, /**allow_quant_ops=*/false);
  // No ops should be accepted.
  EXPECT_EQ(ops_to_replace_without_quant->size, 0);

  TfLiteIntArrayFree(ops_to_replace);
  TfLiteIntArrayFree(ops_to_replace_without_quant);
}

InterpreterFp16* interpreter_fp16_split_op =
    new InterpreterFp16(kTfLiteBuiltinSplit);

TEST(ModelBuilderTest, GetOpsToReplaceAcceptsSplitOpCl) {
  // Before pruning, the graph has three nodes:
  //
  //   t0 (FP16) -> DequantNode -> t1 (FP32) -> Split -> t4
  //   t2 (FP16) -> DequantNode -> t3 (FP32) --/
  //
  // OpsToReplace should choose all three nodes for replacement, and
  // the graph on the GPU will look like this (no Dequants):
  //
  //   t0 (FP16) --> Split -> t4
  //   t2 (FP16) --/
  //
  TfLiteContext* context = interpreter_fp16_split_op->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp16_split_op->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_fp16_split_op->node(node_index);
    *registration = interpreter_fp16_split_op->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // The partitioner should accept only the Add op initially.
        EXPECT_EQ(nodes_to_replace->size, 1);
        // Single partition output.
        auto params = interpreter_fp16_split_op->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 2;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 3;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 4;

        *partition_params_array = interpreter_fp16_split_op->delegate_params();
        *num_partitions = interpreter_fp16_split_op->num_delegate_params();
        return kTfLiteOk;
      };

  context->tensors[0].data.i32 = new int[1];
  context->tensors[0].data.i32[0] = 0;

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  delete[] context->tensors[0].data.i32;
  context->tensors[0].data.i32 = nullptr;

  // Ensure all nodes are delegated, and the SPLIT op has FP16 inputs.
  EXPECT_EQ(ops_to_replace->size, 3);
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  context->GetNodeAndRegistration(context, /**node_id**/ 2, &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  TfLiteIntArrayFree(ops_to_replace);
}

InterpreterFp16* interpreter_fp16_split_op2 =
    new InterpreterFp16(kTfLiteBuiltinSplit);
TEST(ModelBuilderTest, GetOpsToReplaceRejectsSplitOpGl) {
  // Same graph as that in the test case `GetOpsToReplaceAcceptsSplitOpCl`,
  // while OpenCL is not available when calling GetOpsToReplace.
  // OpenGL does not support SPLIT op, so we don't choose any nodes.

  TfLiteContext* context = interpreter_fp16_split_op2->context();
  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp16_split_op2->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_fp16_split_op2->node(node_index);
    *registration = interpreter_fp16_split_op2->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // No selected nodes.
        EXPECT_EQ(nodes_to_replace->size, 0);
        *partition_params_array = nullptr;
        *num_partitions = 0;
        return kTfLiteOk;
      };
  absl::flat_hash_set<TfLiteBuiltinOperator> excluded_ops = {
      kTfLiteBuiltinSplit};
  TfLiteIntArray* ops_to_replace =
      GetOpsToReplace(context, /*allow_quant_ops=*/false,
                      /*max_delegated_partitions=*/1, &excluded_ops);

  // No nodes were found to replace.
  EXPECT_EQ(ops_to_replace->size, 0);
  // Inputs to Split op are still fp32.
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  context->GetNodeAndRegistration(context, /**node_id**/ 2, &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat32);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat32);
  TfLiteIntArrayFree(ops_to_replace);
}

TEST(AddOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinAdd,
      /*op_version=*/7,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid (version 2)
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinAdd,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid (version 6)
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinAdd,
      /*op_version=*/6,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid Data Type - IsTwoArgumentOperationWithConst: Float16
  context->SetTensorType(2, kTfLiteFloat16, kTfLiteMmapRo);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid Data Type - IsTwoArgumentOperationWithConst: Int32
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid Data Type - IsTwoArgumentOperationWithConst: Int16
  context->SetTensorType(2, kTfLiteInt16, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid Data Type - IsTwoArgumentOperationWithConst: Int8
  context->SetTensorType(2, kTfLiteInt8, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ArgMaxOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinArgMax,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinArgMax,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  // Axis must be constant
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(BatchMatMulOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinBatchMatmul,
      /*op_version=*/1,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinBatchMatmul,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // invalid input tensor
  context->node()->inputs->data[0] = kTfLiteOptionalTensor;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->inputs->data[0] = 0;
}

TEST(BitcastOperationParserTest, TestIsSupported) {
  // Invalid num inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinBitcast,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 2, 4}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Test for equal bits
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinBitcast,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 2, 4}));
  context->tensor(1)->type = kTfLiteInt32;
  context->tensor(1)->dims->size = 3;
  context->tensor(1)->dims->data[0] = 1;
  context->tensor(1)->dims->data[1] = 2;
  context->tensor(1)->dims->data[2] = 4;
  context->tensor(2)->type = kTfLiteFloat32;
  context->tensor(2)->dims->size = 3;
  context->tensor(2)->dims->data[0] = 1;
  context->tensor(2)->dims->data[1] = 2;
  context->tensor(2)->dims->data[2] = 5;  // Incorrect shape
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->tensor(2)->dims->data[2] = 4;  // correct shape
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Input elem size > output elem size
  // Incorrect dim size
  context->tensor(2)->type = kTfLiteInt16;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Incorrect final type
  context->tensor(1)->dims->size = 2;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Correct
  context->tensor(2)->dims->data[2] = 2;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Input elem size < output elem size
  // Incorrect dim size (both have dims=(1, 2))
  context->tensor(1)->type = kTfLiteInt8;
  context->tensor(2)->dims->size = 2;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Correct dim size, incorrect val
  context->tensor(2)->dims->size = 1;
  context->tensor(2)->dims->data[0] = 2;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Correct
  context->tensor(2)->dims->data[0] = 1;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(BroadcastInDimOperationParserTest, TestIsSupported) {
  // 1 runtime, 1 const tensor required as inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinStablehloBroadcastInDim,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({4, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Valid inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinStablehloBroadcastInDim,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));

  context->ChangeTensorShape(2, {4});
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(2)->data.i32[0] = 0;
  context->tensor(2)->data.i32[1] = 1;
  context->tensor(2)->data.i32[2] = 2;
  context->tensor(2)->data.i32[3] = 3;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // invalid non-unique indices
  context->tensor(2)->data.i32[2] = 0;
  context->tensor(2)->data.i32[3] = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->tensor(2)->data.i32[2] = 2;
  context->tensor(2)->data.i32[3] = 3;

  // Invalid non-const indices
  context->SetTensorType(2, kTfLiteInt32, kTfLiteArenaRw);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid indices type. Expect i32
  context->SetTensorType(2, kTfLiteInt16, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid indices dims. Expect 1D
  context->ChangeTensorShape(2, {4, 1});
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(2)->data.i32[0] = 0;
  context->tensor(2)->data.i32[1] = 1;
  context->tensor(2)->data.i32[2] = 2;
  context->tensor(2)->data.i32[3] = 3;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Expect rank(input) == indices.size()
  context->ChangeTensorShape(2, {3});
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(2)->data.i32[0] = 0;
  context->tensor(2)->data.i32[1] = 1;
  context->tensor(2)->data.i32[2] = 2;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Back to valid
  context->ChangeTensorShape(2, {4});
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(2)->data.i32[0] = 0;
  context->tensor(2)->data.i32[1] = 1;
  context->tensor(2)->data.i32[2] = 2;
  context->tensor(2)->data.i32[3] = 3;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // invalid input tensor
  context->node()->inputs->data[0] = kTfLiteOptionalTensor;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->inputs->data[0] = 0;
}

TEST(CastOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCast,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCast,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));

  context->SetTensorType(1, kTfLiteFloat32, kTfLiteArenaRw);
  context->SetTensorType(2, kTfLiteInt32, kTfLiteArenaRw);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->SetTensorType(1, kTfLiteInt32, kTfLiteArenaRw);
  context->SetTensorType(2, kTfLiteFloat32, kTfLiteArenaRw);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->SetTensorType(1, kTfLiteInt8, kTfLiteArenaRw);
  context->SetTensorType(2, kTfLiteFloat32, kTfLiteArenaRw);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->SetTensorType(1, kTfLiteBool, kTfLiteArenaRw);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ClampOperationsParserTest, TestIsSupported) {
  // Always valid
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReluN1To1,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ConcatenationOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinConcatenation,
      /*op_version=*/7,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid (version 2)
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinConcatenation,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid (version 6)
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinConcatenation,
      /*op_version=*/6,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // invalid RetrieveBuiltinData
  void* builtin_data = context->node()->builtin_data;
  context->node()->builtin_data = nullptr;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->builtin_data = builtin_data;
}

class Conv2DOperationParserTest : public testing::Test {
 public:
  void SetUp() override {
    // Setup the context.
    context_ = std::make_unique<StubTfLiteContext>(
        kTfLiteBuiltinConv2d,
        /*op_version=*/5,
        /*num_inputs=*/2,
        /*shape=*/std::vector<int>({1, 1, 1, 1}));
    TfLiteConvParams* const tf_options =
        static_cast<TfLiteConvParams*>(context_->node()->builtin_data);
    tf_options->stride_width = 1;
    tf_options->stride_height = 1;
    tf_options->dilation_width_factor = 1;
    tf_options->dilation_height_factor = 1;
    tf_options->activation = kTfLiteActRelu;

    parser_ = NewOperationParser(context_->node(), context_->registration());

    // If the op isn't supported, this test is moot.
    ASSERT_TRUE(parser_
                    ->IsSupported(context_.get(), context_->node(),
                                  context_->registration())
                    .ok());
  }

 protected:
  std::unique_ptr<StubTfLiteContext> context_;
  ::ml_drift::GraphFloat32 graph_;
  std::unique_ptr<TFLiteOperationParser> parser_;
  absl::flat_hash_map<int, ::ml_drift::Value*> tensor_to_value_;
};

TEST_F(Conv2DOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinConv2d,
      /*op_version=*/6,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid strides and dilation
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinConv2d,
      /*op_version=*/5,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteConvParams* tf_options =
      static_cast<TfLiteConvParams*>(context->node()->builtin_data);
  tf_options->stride_width = 0;
  tf_options->stride_height = 0;
  tf_options->dilation_width_factor = 0;
  tf_options->dilation_height_factor = 0;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid dilation
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->dilation_width_factor = 0;
  tf_options->dilation_height_factor = 0;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid activation
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->dilation_width_factor = 1;
  tf_options->dilation_height_factor = 1;
  tf_options->activation = kTfLiteActSignBit;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->dilation_width_factor = 1;
  tf_options->dilation_height_factor = 1;
  tf_options->activation = kTfLiteActRelu;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST_F(Conv2DOperationParserTest, ParseWithSharedTensorsWorks) {
  constexpr int kLocalId = 2;
  constexpr int kGlobalId = 0;
  TensorIndexToBufferIdMap tensor_to_buffer_id_map({{kLocalId, kGlobalId}});
  SharedConstTensorsMap shared_tensor_map;
  ObjectReader reader(
      &graph_, context_.get(), context_->node(), &tensor_to_value_,
      /*quant_conversion_map=*/nullptr, &tensor_to_buffer_id_map,
      /*tensor_to_external_buffer_id_map=*/nullptr, &shared_tensor_map);
  parser_->Parse(context_->node(), context_->registration(), &graph_, &reader);
  ASSERT_EQ(graph_.inputs().size(), 2);
  const ::ml_drift::ValueId kWeightsValueId = graph_.inputs()[1]->id;

  ASSERT_THAT(shared_tensor_map.size(), 1);
  ASSERT_TRUE(shared_tensor_map.contains(kWeightsValueId));
  const SharedTfliteTensor kExpected{.tflite_tensor_id = kLocalId,
                                     .global_id = kGlobalId,
                                     .dequant_forced = false};
  EXPECT_EQ(shared_tensor_map.at(kWeightsValueId), kExpected);
}

TEST(DepthwiseConvolutionOperationParserTest, TestIsSupported) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDepthwiseConv2d,
      /*op_version=*/7,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDepthwiseConv2d,
      /*op_version=*/6,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteDepthwiseConvParams* tf_options =
      static_cast<TfLiteDepthwiseConvParams*>(context->node()->builtin_data);
  // Invalid strides and dilation
  tf_options->stride_width = 0;
  tf_options->stride_height = 0;
  tf_options->dilation_width_factor = 0;
  tf_options->dilation_height_factor = 0;
  tf_options->depth_multiplier = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid dilation
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->dilation_width_factor = 0;
  tf_options->dilation_height_factor = 0;
  tf_options->depth_multiplier = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid activation
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->dilation_width_factor = 1;
  tf_options->dilation_height_factor = 1;
  tf_options->depth_multiplier = 1;
  tf_options->activation = kTfLiteActSignBit;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid depth_multiplier
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->dilation_width_factor = 1;
  tf_options->dilation_height_factor = 1;
  tf_options->depth_multiplier = 0;
  tf_options->activation = kTfLiteActRelu;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->dilation_width_factor = 1;
  tf_options->dilation_height_factor = 1;
  tf_options->depth_multiplier = 1;
  tf_options->activation = kTfLiteActRelu;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // invalid input tensor
  context->node()->inputs->data[0] = kTfLiteOptionalTensor;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->inputs->data[0] = 0;

  // invalid RetrieveBuiltinData
  void* builtin_data = context->node()->builtin_data;
  context->node()->builtin_data = nullptr;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->builtin_data = builtin_data;

  // invalid number of inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDepthwiseConv2d,
      /*op_version=*/6,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(DepthToSpaceOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDepthToSpace,
      /*op_version=*/1,
      /*num_inputs=*/0,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDepthToSpace,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteDepthToSpaceParams* d2s_params =
      static_cast<TfLiteDepthToSpaceParams*>(context->node()->builtin_data);
  // Invalid block_size
  d2s_params->block_size = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  d2s_params->block_size = 2;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // invalid RetrieveBuiltinData
  void* builtin_data = context->node()->builtin_data;
  context->node()->builtin_data = nullptr;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->builtin_data = builtin_data;
}

TEST(DequantizeOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDequantize,
      /*op_version=*/4,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration(),
                                   /*allow_quant_ops=*/true);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDequantize,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid input type
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDequantize,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(1)->type = kTfLiteInt16;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid without Density support
  context->tensor(1)->type = kTfLiteInt8;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(LogicalElementwiseOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinEqual,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid consumer
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinEqual,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid amount of operands
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // invalid RetrieveBuiltinData
  void* builtin_data = context->node()->builtin_data;
  context->node()->builtin_data = nullptr;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->builtin_data = builtin_data;
}

TEST(ArithmeticUnaryElementwiseOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinAbs,
      /*op_version=*/3,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinAbs,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // invalid input tensor
  context->node()->inputs->data[0] = kTfLiteOptionalTensor;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->inputs->data[0] = 0;
}

TEST(ArithmeticBinaryElementwiseOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDiv,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDiv,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

class FullyConnectedOperationParserTest : public testing::Test {
  void SetUp() override {
    context_ = std::make_unique<StubTfLiteContext>(
        kTfLiteBuiltinFullyConnected,
        /*op_version=*/9,
        /*num_inputs=*/2,
        /*shape=*/std::vector<int>({1, 1, 1, 1}));
    parser_ = NewOperationParser(context_->node(), context_->registration());

    ASSERT_TRUE(parser_
                    ->IsSupported(context_.get(), context_->node(),
                                  context_->registration())
                    .ok());
  }

 protected:
  void QuantizeTensor(int main_node_input_id) {
    context_->MakeTensorQuantized(main_node_input_id);
  }

  std::unique_ptr<StubTfLiteContext> context_;
  ::ml_drift::GraphFloat32 graph_;
  std::unique_ptr<TFLiteOperationParser> parser_;
  absl::flat_hash_map<int, ::ml_drift::Value*> tensor_to_value_;
};

TEST_F(FullyConnectedOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinFullyConnected,
      /*op_version=*/9,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid weights_format
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinFullyConnected,
      /*op_version=*/9,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteFullyConnectedParams* tf_options =
      static_cast<TfLiteFullyConnectedParams*>(context->node()->builtin_data);
  tf_options->weights_format =
      kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid keep_num_dims
  tf_options->weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
  tf_options->keep_num_dims = true;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid keep_num_dims
  context->tensor(1)->dims->size = 3;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST_F(FullyConnectedOperationParserTest, ParseWithSharedTensorsWorks) {
  constexpr int kLocalId = 2;
  constexpr int kGlobalId = 0;
  TensorIndexToBufferIdMap tensor_to_buffer_id_map({{kLocalId, kGlobalId}});
  SharedConstTensorsMap shared_tensor_map;
  ObjectReader reader(
      &graph_, context_.get(), context_->node(), &tensor_to_value_,
      /*quant_conversion_map=*/nullptr, &tensor_to_buffer_id_map,
      /*tensor_to_external_buffer_id_map=*/nullptr, &shared_tensor_map);
  parser_->Parse(context_->node(), context_->registration(), &graph_, &reader);

  ASSERT_EQ(graph_.inputs().size(), 2);
  const ::ml_drift::ValueId kWeightsValueId = graph_.inputs()[1]->id;

  ASSERT_EQ(shared_tensor_map.size(), 1);
  ASSERT_TRUE(shared_tensor_map.contains(kWeightsValueId));

  const SharedTfliteTensor kExpected{.tflite_tensor_id = kLocalId,
                                     .global_id = kGlobalId,
                                     .dequant_forced = false};
  EXPECT_EQ(shared_tensor_map.at(kWeightsValueId), kExpected);
}

TEST_F(FullyConnectedOperationParserTest,
       ParseWithSharedQuantizedTensorsWorks) {
  this->QuantizeTensor(/*main_node_input_id=*/1);
  constexpr int kLocalId = 2;
  constexpr int kGlobalId = 0;
  TensorIndexToBufferIdMap tensor_to_buffer_id_map({{kLocalId, kGlobalId}});
  SharedConstTensorsMap shared_tensor_map;
  ObjectReader reader(
      &graph_, context_.get(), context_->node(), &tensor_to_value_,
      /*quant_conversion_map=*/nullptr, &tensor_to_buffer_id_map,
      /*tensor_to_external_buffer_id_map=*/nullptr, &shared_tensor_map);
  parser_->Parse(context_->node(), context_->registration(), &graph_, &reader);

  ASSERT_EQ(graph_.inputs().size(), 2);
  const ::ml_drift::ValueId kWeightsValueId = graph_.inputs()[1]->id;

  ASSERT_EQ(shared_tensor_map.size(), 1);
  ASSERT_TRUE(shared_tensor_map.contains(kWeightsValueId));

  const SharedTfliteTensor kExpected{.tflite_tensor_id = kLocalId,
                                     .global_id = kGlobalId,
                                     .dequant_forced = false};
  EXPECT_EQ(shared_tensor_map.at(kWeightsValueId), kExpected);

  EXPECT_EQ(graph_.nodes()[0]->operation.attributes.type(),
            typeid(::ml_drift::FullyConnectedInt8Attributes));
}

TEST_F(FullyConnectedOperationParserTest,
       ParseWithSharedBlockwiseQuantizedTensorsWorks) {
  context_->ChangeTensorShape(1, std::vector<int>({1, 1, 1, 64}));
  context_->ChangeTensorShape(2, std::vector<int>({8, 1, 1, 64}));
  context_->ChangeTensorShape(3, std::vector<int>({1, 1, 1, 8}));

  const int kWeightsTensorId = 2;
  TfLiteTensor* weights_tensor = context_->tensor(kWeightsTensorId);
  weights_tensor->type = kTfLiteInt8;
  weights_tensor->quantization.type = kTfLiteBlockwiseQuantization;
  auto quant_params = std::make_unique<TfLiteBlockwiseQuantization>();
  quant_params->scale = -1;
  quant_params->zero_point = -1;
  quant_params->blocksize = 32;
  quant_params->quantized_dimension = 0;
  weights_tensor->quantization.params = quant_params.release();

  constexpr int kLocalId = 2;
  constexpr int kGlobalId = 0;
  TensorIndexToBufferIdMap tensor_to_buffer_id_map({{kLocalId, kGlobalId}});
  SharedConstTensorsMap shared_tensor_map;
  ObjectReader reader(
      &graph_, context_.get(), context_->node(), &tensor_to_value_,
      /*quant_conversion_map=*/nullptr, &tensor_to_buffer_id_map,
      /*tensor_to_external_buffer_id_map=*/nullptr, &shared_tensor_map);
  parser_->Parse(context_->node(), context_->registration(), &graph_, &reader);

  ASSERT_EQ(graph_.nodes().size(), 1);
  EXPECT_EQ(graph_.nodes()[0]->operation.attributes.type(),
            typeid(::ml_drift::FullyConnectedInt8Attributes));

  const auto& attr =
      std::any_cast<const ::ml_drift::FullyConnectedInt8Attributes&>(
          graph_.nodes()[0]->operation.attributes);

  EXPECT_EQ(attr.scale.shape.o, 8);
  EXPECT_EQ(attr.scale.shape.h, 1);
  EXPECT_EQ(attr.scale.shape.w, 1);
  EXPECT_EQ(attr.scale.shape.i, 2);  // 64 / 32
}

TEST_F(FullyConnectedOperationParserTest, ParseSrc3dDst2d) {
  context_->ChangeTensorShape(1, std::vector<int>({1, 16, 4}));
  context_->ChangeTensorShape(2, std::vector<int>({8, 1, 1, 4}));
  context_->ChangeTensorShape(3, std::vector<int>({16, 8}));

  ObjectReader reader(&graph_, context_.get(), context_->node(),
                      &tensor_to_value_,
                      /*quant_conversion_map=*/nullptr,
                      /*tensor_to_buffer_id_map=*/nullptr,
                      /*shared_tensor_map=*/nullptr);
  parser_->Parse(context_->node(), context_->registration(), &graph_, &reader);

  ASSERT_EQ(graph_.nodes().size(), 2);  // added reshape node for output
  ASSERT_EQ(
      ::ml_drift::OperationTypeFromString(graph_.nodes()[0]->operation.type),
      ::ml_drift::OperationType::FULLY_CONNECTED);
  ASSERT_EQ(
      ::ml_drift::OperationTypeFromString(graph_.nodes()[1]->operation.type),
      ::ml_drift::OperationType::RESHAPE);
  auto fc_input = graph_.FindInputs(graph_.nodes()[0]->id)[0];
  ASSERT_EQ(fc_input->tensor.shape, ::ml_drift::BHWC(1, 1, 16, 4));
  auto fc_output = graph_.FindOutputs(graph_.nodes()[0]->id)[0];
  ASSERT_EQ(fc_output->tensor.shape, ::ml_drift::BHWC(1, 1, 16, 8));
  ASSERT_EQ(graph_.inputs()[0]->tensor.shape, ::ml_drift::BHWC(1, 1, 16, 4));
  ASSERT_EQ(graph_.outputs()[0]->tensor.shape, ::ml_drift::BHWC(16, 1, 1, 8));
}

TEST(GatherOperationParserTest, TestNumInputs) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/1, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  std::unique_ptr<TFLiteOperationParser> parser =
      NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid num inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/3, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(2)->dims->size = 1;
  context->tensor(2)->type = kTfLiteInt32;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(GatherOperationParserTest, TestValueTensor) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/2, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  std::unique_ptr<TFLiteOperationParser> parser =
      NewOperationParser(context->node(), context->registration());
  // INVALID with constant value tensor
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/2, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(2)->dims->size = 1;
  context->tensor(2)->type = kTfLiteFloat32;
  context->tensor(1)->allocation_type = kTfLiteMmapRo;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // VALID
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/2, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(2)->dims->size = 1;
  context->tensor(2)->type = kTfLiteInt32;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(GatherOperationParserTest, TestIndicesTensor) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/2, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  std::unique_ptr<TFLiteOperationParser> parser =
      NewOperationParser(context->node(), context->registration());
  // Need 1D indices
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/2, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(2)->dims->size = 2;
  context->tensor(2)->type = kTfLiteInt32;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Need int32 indices
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/2, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(2)->dims->size = 1;
  context->tensor(2)->type = kTfLiteFloat32;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // VALID with constant indices
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/2, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(2)->dims->size = 1;
  context->tensor(2)->type = kTfLiteInt32;
  context->tensor(2)->allocation_type = kTfLiteMmapRo;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(GatherOperationParserTest, ValidConstantValueTensor) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinGather,
      /*op_version=*/1,
      /*num_inputs=*/2, /*shape=*/std::vector<int>({1, 1, 1, 1}));
  std::unique_ptr<TFLiteOperationParser> parser =
      NewOperationParser(context->node(), context->registration());
  context->tensor(1)->allocation_type = kTfLiteMmapRo;  // constant value tensor
  context->tensor(2)->dims->size = 1;
  context->tensor(2)->type = kTfLiteInt32;
  context->tensor(2)->allocation_type = kTfLiteArenaRw;  // runtime indices
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(HardSwishOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinHardSwish,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinHardSwish,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(LSTMOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinLstm,
                                          /*op_version=*/5,
                                          /*num_inputs=*/24,
                                          /*shape=*/std::vector<int>({1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid num_inputs for kTfLiteLSTMFullKernel
  context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinLstm,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1,
                                          /*shape=*/std::vector<int>({1, 1}));
  TfLiteLSTMParams* tf_options =
      static_cast<TfLiteLSTMParams*>(context->node()->builtin_data);
  tf_options->kernel_type = kTfLiteLSTMFullKernel;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid activation for kTfLiteLSTMFullKernel
  context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinLstm,
                                          /*op_version=*/1,
                                          /*num_inputs=*/24,
                                          /*shape=*/std::vector<int>({1, 1}));
  tf_options = static_cast<TfLiteLSTMParams*>(context->node()->builtin_data);
  tf_options->kernel_type = kTfLiteLSTMFullKernel;
  tf_options->activation = kTfLiteActRelu;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->activation = kTfLiteActSigmoid;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(MulOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMul,
      /*op_version=*/9,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid (version 8)
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMul,
      /*op_version=*/8,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid activation
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMul,
      /*op_version=*/4,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteMulParams* tf_options =
      static_cast<TfLiteMulParams*>(context->node()->builtin_data);
  tf_options->activation = kTfLiteActSignBit;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid activation
  tf_options->activation = kTfLiteActSignBit;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid activation
  tf_options->activation = kTfLiteActSigmoid;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid dims (first_has_smaller_dim && second_has_smaller_dim) broadcasting
  // scenario.
  context->tensor(1)->dims->data[0] = 256;
  context->tensor(2)->dims->data[1] = 256;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(MulOperationParserTest, TestIsSupportedFailed) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMul,
      /*op_version=*/4,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  std::unique_ptr<TFLiteOperationParser> parser =
      NewOperationParser(context->node(), context->registration());
  context->ChangeTensorShape(1, {2});
  context->ChangeTensorShape(2, {1, 3});
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(PackOperationParserTest, TestIsSupported) {
  // Always pass
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinPack,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(PadOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinPad,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({4, 2, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinPad,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({4, 2, 1, 1}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinPad,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({4, 2, 1, 1}));
  context->tensor(2)->allocation_type = kTfLiteMmapRo;  // Treat input2 as const
  // Invalid padding dimension 4d
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid padding dimension 4x2
  context->tensor(2)->dims->size = 2;
  context->tensor(2)->dims->data[0] = 4;
  context->tensor(2)->dims->data[1] = 2;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  //   padding dimension 4x1
  context->tensor(2)->dims->size = 2;
  context->tensor(2)->dims->data[0] = 4;
  context->tensor(2)->dims->data[1] = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(Padv2OperationParserTest, TestIsSupported) {
  // Valid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinPad,
      /*op_version=*/3,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({4, 2, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinPad,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({4, 2, 1, 1}));
  context->tensor(2)->allocation_type =
      kTfLiteMmapRo;  // Treat padding vals as const

  // Valid padding dimension 4x2
  context->tensor(2)->dims->size = 2;
  context->tensor(2)->dims->data[0] = 4;
  context->tensor(2)->dims->data[1] = 2;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->tensor(3)->allocation_type =
      kTfLiteMmapRo;  // Treat const_value as const

  // Valid const values
  context->tensor(3)->dims->size = 0;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(MirrorPadOperationParserTest, TestIsSupported) {
  // Invalid mirror pad mode
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMirrorPad,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteMirrorPaddingParams* tf_options =
      static_cast<TfLiteMirrorPaddingParams*>(context->node()->builtin_data);
  tf_options->mode = kTfLiteMirrorPaddingSymmetric;
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid op_version
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMirrorPad,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  tf_options =
      static_cast<TfLiteMirrorPaddingParams*>(context->node()->builtin_data);
  tf_options->mode = kTfLiteMirrorPaddingReflect;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMirrorPad,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  tf_options =
      static_cast<TfLiteMirrorPaddingParams*>(context->node()->builtin_data);
  tf_options->mode = kTfLiteMirrorPaddingReflect;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMirrorPad,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  tf_options =
      static_cast<TfLiteMirrorPaddingParams*>(context->node()->builtin_data);
  tf_options->mode = kTfLiteMirrorPaddingReflect;
  context->tensor(2)->allocation_type = kTfLiteMmapRo;  // Treat input2 as const
  // Invalid padding dimension 4d
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid padding dimension 4x2
  context->ChangeTensorShape(2, {4, 2});
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  //   padding dimension 4x1
  context->tensor(2)->dims->data[0] = 4;
  context->tensor(2)->dims->data[1] = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(AveragePooling2DOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinAveragePool2d,
      /*op_version=*/3,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinAveragePool2d,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLitePoolParams* tf_options =
      static_cast<TfLitePoolParams*>(context->node()->builtin_data);

  // Invalid filter and stride
  tf_options->filter_height = 0;
  tf_options->filter_width = 0;
  tf_options->stride_width = 0;
  tf_options->stride_height = 0;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid filter
  tf_options->filter_height = 0;
  tf_options->filter_width = 0;
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;

  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid activation
  tf_options->filter_height = 1;
  tf_options->filter_width = 1;
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->activation = kTfLiteActSignBit;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->filter_height = 1;
  tf_options->filter_width = 1;
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->activation = kTfLiteActTanh;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(MaxPooling2DOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMaxPool2d,
      /*op_version=*/3,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMaxPool2d,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLitePoolParams* tf_options =
      static_cast<TfLitePoolParams*>(context->node()->builtin_data);

  // Invalid filter and stride
  tf_options->filter_height = 0;
  tf_options->filter_width = 0;
  tf_options->stride_width = 0;
  tf_options->stride_height = 0;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid filter
  tf_options->filter_height = 0;
  tf_options->filter_width = 0;
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;

  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid activation
  tf_options->filter_height = 1;
  tf_options->filter_width = 1;
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->activation = kTfLiteActSignBit;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->filter_height = 1;
  tf_options->filter_width = 1;
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  tf_options->activation = kTfLiteActTanh;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(CustomMaxPooling2DOperationParserTest, TestIsSupported) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCustom,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->registration()->custom_name = "MaxPoolingWithArgmax2D";
  TfLitePoolParams tf_options;
  context->node()->custom_initial_data = &tf_options;
  TfLiteIntArrayFree(context->node()->outputs);
  // To make the op node has two outputs
  context->node()->outputs = TfLiteIntArrayCreate(2);
  context->node()->outputs->data[0] = 2;
  context->node()->outputs->data[1] = 3;
  auto parser = NewOperationParser(context->node(), context->registration());

  // Invalid filter and stride
  tf_options.filter_height = 0;
  tf_options.filter_width = 0;
  tf_options.stride_width = 0;
  tf_options.stride_height = 0;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid filter
  tf_options.filter_height = 0;
  tf_options.filter_width = 0;
  tf_options.stride_width = 1;
  tf_options.stride_height = 1;

  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid activation
  tf_options.filter_height = 1;
  tf_options.filter_width = 1;
  tf_options.stride_width = 1;
  tf_options.stride_height = 1;
  tf_options.activation = kTfLiteActSignBit;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options.filter_height = 1;
  tf_options.filter_width = 1;
  tf_options.stride_width = 1;
  tf_options.stride_height = 1;
  tf_options.activation = kTfLiteActTanh;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ReduceMaxOperationParserTest, TestIsSupported) {
  // Non constant axes tensor
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReduceMax,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Valid
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid axes tensor type
  context->SetTensorType(2, kTfLiteFloat32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ReduceMinOperationParserTest, TestIsSupported) {
  // Non constant axes tensor
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReduceMin,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Valid
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid axes tensor type
  context->SetTensorType(2, kTfLiteFloat32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ReduceProductOperationParserTest, TestIsSupported) {
  // Non constant axes tensor
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReduceProd,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Valid
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid axes tensor type
  context->SetTensorType(2, kTfLiteFloat32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(QuantizeOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinQuantize,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration(),
                                   /*allow_quant_ops=*/true);
  context->QuantizeTensor(2, kTfLiteInt8);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinQuantize,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->QuantizeTensor(2, kTfLiteInt8);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinQuantize,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->QuantizeTensor(2, kTfLiteInt8);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ReLUOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinRelu,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinRelu,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ReLU6OperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinRelu6,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinRelu6,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(LeakyReLUOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinLeakyRelu,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinLeakyRelu,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(RemainderOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinStablehloRemainder,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinStablehloRemainder,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ResamplerOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  std::vector<::ml_drift::BHWC> input_shapes = {::ml_drift::BHWC(1, 1, 1, 1)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shapes);
  context->registration()->custom_name = "Resampler";
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  input_shapes.push_back(::ml_drift::BHWC(1, 1, 1, 2));
  context = std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                                /*op_version=*/1,
                                                /*num_inputs=*/2, input_shapes);
  context->registration()->custom_name = "Resampler";
  const int warp_shape_id = context->node()->inputs->data[1];
  context->tensors[warp_shape_id].dims->data[3] = 2;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ReshapeOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReshape,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReshape,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReshape,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(Resize2DBilinearOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinResizeBilinear,
      /*op_version=*/4,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinResizeBilinear,
      /*op_version=*/3,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid: if half_pixel_centers is True, align_corners must be False
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinResizeBilinear,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteResizeBilinearParams* tf_options =
      static_cast<TfLiteResizeBilinearParams*>(context->node()->builtin_data);
  tf_options->half_pixel_centers = true;
  tf_options->align_corners = true;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->half_pixel_centers = true;
  tf_options->align_corners = false;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->half_pixel_centers = false;
  tf_options->align_corners = true;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->half_pixel_centers = false;
  tf_options->align_corners = false;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(Resize2DNearestNeighborOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinResizeNearestNeighbor,
      /*op_version=*/4,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinResizeNearestNeighbor,
      /*op_version=*/3,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinResizeNearestNeighbor,
      /*op_version=*/3,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteResizeNearestNeighborParams* tf_options =
      static_cast<TfLiteResizeNearestNeighborParams*>(
          context->node()->builtin_data);
  tf_options->half_pixel_centers = true;
  tf_options->align_corners = true;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->half_pixel_centers = true;
  tf_options->align_corners = false;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->half_pixel_centers = false;
  tf_options->align_corners = true;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->half_pixel_centers = false;
  tf_options->align_corners = false;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ReverseOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReverseV2,
      /*op_version=*/1,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid axis tensor
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReverseV2,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(2)->allocation_type = kTfLiteArenaRw;
  context->tensor(2)->type = kTfLiteInt32;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid axis tensor type
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReverseV2,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(2)->allocation_type = kTfLiteMmapRo;  // Treat axis as const
  context->tensor(2)->type = kTfLiteFloat32;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinReverseV2,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->tensor(2)->allocation_type = kTfLiteMmapRo;  // Treat axis as const
  context->tensor(2)->type = kTfLiteInt32;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(PositionalEmbeddingParserTest, TestIsSupported) {
  // Invalid num inputs
  std::vector<::ml_drift::BHWC> input_shapes = {::ml_drift::BHWC(1, 1, 8, 256)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shapes);
  context->registration()->custom_name =
      "custom_call.absolute_positional_embedding";

  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid num inputs, need w dim to match
  input_shapes.push_back(::ml_drift::BHWC(1, 1, 1, 256));
  context = std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                                /*op_version=*/1,
                                                /*num_inputs=*/2, input_shapes);
  context->registration()->custom_name =
      "custom_call.absolute_positional_embedding";

  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  input_shapes[1].w = 8;
  context = std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                                /*op_version=*/1,
                                                /*num_inputs=*/2, input_shapes);
  context->registration()->custom_name =
      "custom_call.absolute_positional_embedding";
  parser = NewOperationParser(context->node(), context->registration());
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(RoPEParserTest, TestIsSupported) {
  // Invalid num inputs
  std::vector<::ml_drift::BHWC> input_shapes = {::ml_drift::BHWC(1, 1, 8, 256)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shapes);
  context->registration()->custom_name =
      "custom_call.rotary_positional_embedding";

  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid num inputs, need w dim to match
  input_shapes.push_back(::ml_drift::BHWC(1, 1, 1, 256));
  context = std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                                /*op_version=*/1,
                                                /*num_inputs=*/2, input_shapes);
  context->registration()->custom_name =
      "custom_call.rotary_positional_embedding";

  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  input_shapes[1].w = 8;
  context = std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                                /*op_version=*/1,
                                                /*num_inputs=*/2, input_shapes);
  context->registration()->custom_name =
      "custom_call.rotary_positional_embedding";
  parser = NewOperationParser(context->node(), context->registration());
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // If 3 inputs, then pos tensor is index 2. Also require 2 outputs
  input_shapes.push_back(::ml_drift::BHWC(1, 1, 8, 256));
  context = std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                                /*op_version=*/1,
                                                /*num_inputs=*/3, input_shapes);
  context->registration()->custom_name =
      "custom_call.rotary_positional_embedding";
  parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid (3 inputs, 2 outputs)
  context = std::make_unique<StubTfLiteContext>(kTfLiteBuiltinCustom,
                                                /*op_version=*/1,
                                                /*num_inputs=*/3, input_shapes);
  TfLiteIntArrayFree(context->node()->outputs);
  context->node()->outputs = TfLiteIntArrayCreate(2);
  context->node()->outputs->data[0] = 0;  // dummy data
  context->node()->outputs->data[1] = 0;  // dummy data
  context->registration()->custom_name =
      "custom_call.rotary_positional_embedding";
  parser = NewOperationParser(context->node(), context->registration());
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ScaledDotProductAttentionParserTest, TestIsSupportedNoMask) {
  std::vector<::ml_drift::BHWC> input_shapes = {
      ::ml_drift::BHWC(1, 1, 8, 256), ::ml_drift::BHWC(1, 1024, 1, 256),
      ::ml_drift::BHWC(1, 1024, 1, 256)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/3, input_shapes);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.scaled_dot_product_attention";

  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ScaledDotProductAttentionParserTest, TestIsSupportedWithMask) {
  std::vector<::ml_drift::BHWC> input_shapes = {
      ::ml_drift::BHWC(1, 1, 8, 256), ::ml_drift::BHWC(1, 1024, 1, 256),
      ::ml_drift::BHWC(1, 1024, 1, 256), ::ml_drift::BHWC(1, 1, 1, 1024)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/4, input_shapes);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.scaled_dot_product_attention";
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(ScaledDotProductAttentionParserTest, TestIsSupportedWithConstantMask) {
  std::vector<::ml_drift::BHWC> input_shapes = {
      ::ml_drift::BHWC(1, 1, 8, 256), ::ml_drift::BHWC(1, 1024, 1, 256),
      ::ml_drift::BHWC(1, 1024, 1, 256), ::ml_drift::BHWC(1, 1, 1, 1024)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/4, input_shapes);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.scaled_dot_product_attention";
  context->tensor(4)->allocation_type =
      kTfLiteMmapRo;  // Treat Mask vals as const
  auto parser = NewOperationParser(context->node(), context->registration());
  ASSERT_OK(parser->IsSupported(context.get(), context->node(),
                                context->registration()));

  ::ml_drift::GraphFloat32 graph;
  absl::flat_hash_map<int, ::ml_drift::Value*> tensor_to_value;
  ObjectReader reader(&graph, context.get(), context->node(), &tensor_to_value,
                      /*quant_conversion_map=*/nullptr,
                      /*tensor_to_buffer_id_map=*/nullptr,
                      /*shared_tensor_map=*/nullptr);
  parser->Parse(context->node(), context->registration(), &graph, &reader);

  auto sdpa_node = graph.nodes()[0];
  EXPECT_EQ(sdpa_node->operation.type, "scaled_dot_product_attention");
  EXPECT_EQ(graph.FindInputs(sdpa_node->id).size(), 4);
}

TEST(ScaledDotProductAttentionParserTest, TestIsSupportedIncorrectMask) {
  std::vector<::ml_drift::BHWC> input_shapes = {
      ::ml_drift::BHWC(1, 1, 8, 256), ::ml_drift::BHWC(1, 1024, 1, 256),
      ::ml_drift::BHWC(1, 1024, 1, 256), ::ml_drift::BHWC(1, 1, 1, 1023)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/4, input_shapes);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.scaled_dot_product_attention";
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(SoftmaxOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSoftmax,
      /*op_version=*/5,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSoftmax,
      /*op_version=*/2,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid beta
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSoftmax,
      /*op_version=*/4,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteSoftmaxParams* tf_options =
      static_cast<TfLiteSoftmaxParams*>(context->node()->builtin_data);
  tf_options->beta = 2;
  // Valid
  tf_options->beta = 1;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(SplitOperationParserTest, TestIsSupported) {
  // Always valid
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSplit,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  TfLiteSplitParams* tf_options =
      static_cast<TfLiteSplitParams*>(context->node()->builtin_data);
  tf_options->num_splits = 1;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(SplitVOperationParserTest, TestIsSupported) {
  // Always valid
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSplitV,
      /*op_version=*/1,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->SetTensorType(1, kTfLiteInt32, kTfLiteMmapRo);
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  auto parser = NewOperationParser(context->node(), context->registration());
  TfLiteSplitVParams* tf_options =
      static_cast<TfLiteSplitVParams*>(context->node()->builtin_data);
  tf_options->num_splits = 1;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(TileOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTile,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTile,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(TransposeConvBuiltinOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTransposeConv,
      /*op_version=*/4,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTransposeConv,
      /*op_version=*/3,
      /*input_offset=*/1,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid stride
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTransposeConv,
      /*op_version=*/3,
      /*input_offset=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  TfLiteTransposeConvParams* tf_options =
      static_cast<TfLiteTransposeConvParams*>(context->node()->builtin_data);
  tf_options->stride_width = 0;
  tf_options->stride_height = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options->stride_width = 1;
  tf_options->stride_height = 1;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(TransposeConvCustomOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCustom,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->registration()->custom_name = "Convolution2DTransposeBias";
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // No custom_initial_data
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCustom,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->registration()->custom_name = "Convolution2DTransposeBias";
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid stride
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCustom,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->registration()->custom_name = "Convolution2DTransposeBias";
  TfLiteTransposeConvParams tf_options;
  context->node()->custom_initial_data = &tf_options;
  tf_options.stride_width = 0;
  tf_options.stride_height = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options.stride_width = 1;
  tf_options.stride_height = 1;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(SpaceToDepthOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSpaceToDepth,
      /*op_version=*/2,
      /*num_inputs=*/0,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  TfLiteSpaceToDepthParams* params =
      static_cast<TfLiteSpaceToDepthParams*>(context->node()->builtin_data);
  params->block_size = 2;
  context->node(1)->builtin_data = params;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSpaceToDepth,
      /*op_version=*/2,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  params =
      static_cast<TfLiteSpaceToDepthParams*>(context->node()->builtin_data);
  params->block_size = 2;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid block_size
  params->block_size = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  params->block_size = 2;

  // Invalid RetrieveData
  void* builtin_data = context->node()->builtin_data;
  context->node()->builtin_data = nullptr;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->builtin_data = builtin_data;
}

TEST(TransposeOperationParserTest, TestIsSupported) {
  // Invalid op_version
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTranspose,
      /*op_version=*/10,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 4}));
  auto parser = NewOperationParser(context->node(), context->registration());
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Valid (version 9)
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTranspose,
      /*op_version=*/9,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 4}));
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid num_inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTranspose,
      /*op_version=*/5,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 4}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid runtime inputs
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTranspose,
      /*op_version=*/5,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 4}));
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // IValid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinTranspose,
      /*op_version=*/5,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 4}));
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(Unpooling2DOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCustom,
      /*op_version=*/1,
      /*num_inputs=*/1,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->registration()->custom_name = "MaxUnpooling2D";
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // No custom_initial_data
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCustom,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->registration()->custom_name = "MaxUnpooling2D";
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid filter and stride
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCustom,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->registration()->custom_name = "MaxUnpooling2D";
  TfLitePoolParams tf_options;
  context->node()->custom_initial_data = &tf_options;

  tf_options.filter_height = 0;
  tf_options.filter_width = 0;
  tf_options.stride_width = 0;
  tf_options.stride_height = 0;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid filter
  tf_options.filter_height = 0;
  tf_options.filter_width = 1;
  tf_options.stride_width = 1;
  tf_options.stride_height = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid stride
  tf_options.filter_height = 1;
  tf_options.filter_width = 1;
  tf_options.stride_width = 0;
  tf_options.stride_height = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  tf_options.filter_height = 1;
  tf_options.filter_width = 1;
  tf_options.stride_width = 1;
  tf_options.stride_height = 1;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(MeanOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMean,
      /*op_version=*/1,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Invalid axis tensor
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMean,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->SetTensorType(2, kTfLiteInt32, kTfLiteArenaRw);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid axis tensor type
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMean,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->SetTensorType(2, kTfLiteFloat32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // Valid
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinMean,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(CumsumOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCumsum,
      /*op_version=*/1,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->SetTensorType(2, kTfLiteFloat32, kTfLiteMmapRo);
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinCumsum,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  // bad axes
  context->SetTensorType(2, kTfLiteFloat32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // bad input
  context->SetTensorType(1, kTfLiteInt32, kTfLiteMmapRo);
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->SetTensorType(1, kTfLiteFloat32, kTfLiteMmapRo);
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(2)->allocation_type = kTfLiteMmapRo;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(OneHotOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinOneHot,
      /*op_version=*/1,
      /*num_inputs=*/4,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  auto status = parser->IsSupported(context.get(), context->node(),
                                    context->registration());

  context->tensor(1)->dims->data[1] = 2;
  context->tensor(1)->dims->data[2] = 2;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->tensor(1)->type = kTfLiteInt32;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  TfLiteOneHotParams* params =
      static_cast<TfLiteOneHotParams*>(context->node()->builtin_data);
  params->axis = -1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->tensor(1)->dims->data[1] = 1;
  context->tensor(1)->dims->data[2] = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->ChangeTensorShape(3, {1});
  context->SetTensorType(3, kTfLiteFloat32, kTfLiteMmapRo);
  context->ChangeTensorShape(4, {1});
  context->SetTensorType(4, kTfLiteFloat32, kTfLiteMmapRo);
  params->axis =
      context->tensor(1)->dims->data[context->tensor(1)->dims->size - 1];
  context->node(1)->builtin_data = params;

  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->tensor(1)->dims->data[0] = 2;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(SelectV2OperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSelectV2,
      /*op_version=*/1,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 2, 1, 4}));
  auto parser = NewOperationParser(context->node(), context->registration());
  auto status = parser->IsSupported(context.get(), context->node(),
                                    context->registration());

  // Input is (1, 2, 3, 4)
  context->ChangeTensorShape(4, {1, 2, 3, 4});
  context->SetTensorType(1, kTfLiteInt32, kTfLiteMmapRo);
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->SetTensorType(3, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->SetTensorType(1, kTfLiteFloat32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->SetTensorType(2, kTfLiteFloat32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->SetTensorType(3, kTfLiteFloat32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context->ChangeTensorShape(2, {1});
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  for (int i = 0; i < context->tensor(4)->dims->size; ++i) {
    context->tensor(3)->dims->data[i] = context->tensor(4)->dims->data[i];
  }
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(SliceOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSlice,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Invalid op_version
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSlice,
      /*op_version=*/9,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->ChangeTensorShape(2, {4});
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(2)->data.i32[0] = 1;
  context->tensor(2)->data.i32[1] = 1;
  context->tensor(2)->data.i32[2] = 1;
  context->tensor(2)->data.i32[3] = 1;
  context->ChangeTensorShape(3, {4});
  context->SetTensorType(3, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(3)->data.i32[0] = 1;
  context->tensor(3)->data.i32[1] = 1;
  context->tensor(3)->data.i32[2] = 1;
  context->tensor(3)->data.i32[3] = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // Valid (version 8)
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSlice,
      /*op_version=*/8,
      /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->ChangeTensorShape(2, {4});
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(2)->data.i32[0] = 1;
  context->tensor(2)->data.i32[1] = 1;
  context->tensor(2)->data.i32[2] = 1;
  context->tensor(2)->data.i32[3] = 1;
  context->ChangeTensorShape(3, {4});
  context->SetTensorType(3, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(3)->data.i32[0] = 1;
  context->tensor(3)->data.i32[1] = 1;
  context->tensor(3)->data.i32[2] = 1;
  context->tensor(3)->data.i32[3] = 1;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(StridedSliceOperationParserTest, TestIsSupported) {
  // Invalid num_inputs
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinStridedSlice,
      /*op_version=*/1,
      /*num_inputs=*/4,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinSlice,
      /*op_version=*/1,
      /*num_inputs=*/4,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));

  TfLiteStridedSliceParams* params =
      static_cast<TfLiteStridedSliceParams*>(context->node()->builtin_data);
  context->ChangeTensorShape(2, {4});
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(2)->data.i32[0] = 0;
  context->tensor(2)->data.i32[1] = 0;
  context->tensor(2)->data.i32[2] = 0;
  context->tensor(2)->data.i32[3] = 0;
  context->ChangeTensorShape(3, {4});
  context->SetTensorType(3, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(3)->data.i32[0] = 1;
  context->tensor(3)->data.i32[1] = 1;
  context->tensor(3)->data.i32[2] = 1;
  context->tensor(3)->data.i32[3] = 1;
  context->ChangeTensorShape(4, {4});
  context->SetTensorType(4, kTfLiteInt32, kTfLiteMmapRo);
  context->tensor(4)->data.i32[0] = 1;
  context->tensor(4)->data.i32[1] = 1;
  context->tensor(4)->data.i32[2] = 1;
  context->tensor(4)->data.i32[3] = 1;
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  /* not supported mask type */
  params->ellipsis_mask = 1;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  params->ellipsis_mask = 0;

  /* bad ends */
  context->tensor(3)->data.i32[0] = 0;
  context->tensor(3)->data.i32[1] = 0;
  context->tensor(3)->data.i32[2] = 0;
  context->tensor(3)->data.i32[3] = 0;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(DynamicUpdateSliceOperationParserTest, TestIsSupportedFails) {
  // Invalid num_inputs.
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDynamicUpdateSlice, /*op_version=*/1, /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDynamicUpdateSlice, /*op_version=*/1, /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->SetTensorType(1, kTfLiteInt32, kTfLiteMmapRo);
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);
  context->ChangeTensorShape(3, {0});
  context->SetTensorType(3, kTfLiteInt32, kTfLiteArenaRw);
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());

  // invalid input tensor
  context->node()->inputs->data[0] = kTfLiteOptionalTensor;
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context->node()->inputs->data[0] = 0;

  // start_indices is not runtime input
  context->SetTensorType(3, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  // different dims of operand and update_slice
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDynamicUpdateSlice, /*op_version=*/1, /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->ChangeTensorShape(1, {1});
  context->ChangeTensorShape(2, {1, 1, 1, 2});
  context->ChangeTensorShape(3, {1});
  context->SetTensorType(1, kTfLiteFloat32, kTfLiteMmapRo);
  context->SetTensorType(2, kTfLiteFloat32, kTfLiteMmapRo);
  context->SetTensorType(3, kTfLiteInt32, kTfLiteMmapRo);
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(DynamicSliceOperationParserTest, TestIsSupportedSucceeds) {
  // Valid num_inputs.
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinDynamicUpdateSlice, /*op_version=*/1, /*num_inputs=*/3,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->ChangeTensorShape(3, {1});
  context->SetTensorType(3, kTfLiteInt32, kTfLiteArenaRw);
  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_TRUE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

TEST(EmbeddingLookupOperationParserTest, TestIsSupportedFailed) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinEmbeddingLookup,
      /*op_version=*/1,
      /*num_inputs=*/4,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  auto parser = NewOperationParser(context->node(), context->registration());
  // Invalid num_inputs
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinEmbeddingLookup,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->ChangeTensorShape(1, {1, 2});
  // Invalid ids_spec shape
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
  context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinEmbeddingLookup,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));
  context->ChangeTensorShape(1, {1});
  context->ChangeTensorShape(2, {1});
  // Invalid value_spec shape
  EXPECT_FALSE(
      parser
          ->IsSupported(context.get(), context->node(), context->registration())
          .ok());
}

enum class NormType { kGroupNorm, kLayerNorm };

struct CompositeLayerOrGroupNormParserInput {
  NormType norm_type;
  bool add_sub_type;
};

class CompositeLayerOrGroupNormParserTest
    : public ::testing::TestWithParam<CompositeLayerOrGroupNormParserInput> {
 protected:
  static constexpr int kSubTypeGroupNorm = 0;
  static constexpr int kSubTypeLayerNorm = 1;

  NormType norm_type() { return GetParam().norm_type; }
  bool add_sub_type() { return GetParam().add_sub_type; }

  // Adds reduction axes to the flexbuffer builder:
  //   GroupNorm w/  sub_type: {1, 2, ..., channel_axis}
  //   GroupNorm w/o sub_type: {channel_axis}
  //   LayerNorm w/  sub_type: {channel_axis}
  //   LayerNorm w/o sub_type: <no reduction axes>
  void AddReductionAxes(flexbuffers::Builder& fbb, int channel_axis) {
    if (add_sub_type()) {
      fbb.Int("sub_type", norm_type() == NormType::kGroupNorm
                              ? kSubTypeGroupNorm
                              : kSubTypeLayerNorm);
    }
    if (norm_type() == NormType::kGroupNorm) {
      std::vector<int> reduction_axes;
      if (add_sub_type()) {
        for (int n = 1; n <= channel_axis; ++n) {
          reduction_axes.push_back(n);
        }
      } else {
        reduction_axes = {channel_axis};
      }
      fbb.Map("_TENSOR_V1_reduction_axes", [&]() {
        fbb.Vector("TENSOR_DATA", [&]() {
          for (int axis : reduction_axes) {
            fbb.Add(axis);
          }
        });
      });
    }

    if (norm_type() == NormType::kLayerNorm && add_sub_type()) {
      fbb.Map("_TENSOR_V1_reduction_axes", [&]() {
        fbb.Vector("TENSOR_DATA", [&]() { fbb.Add(channel_axis); });
      });
    }
  }
};

TEST_P(CompositeLayerOrGroupNormParserTest, TestIsSupported4DSucceeds) {
  std::vector<::ml_drift::BHWC> input_shapes = {::ml_drift::BHWC(1, 2, 3, 4)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shapes);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.group_norm";
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1e-5);
    fbb.Int("channel_axis", 3);  // last axis
    AddReductionAxes(fbb, 3);
  });
  fbb.Finish();
  composite_params->attributes = fbb.GetBuffer().data();
  composite_params->attributes_size = fbb.GetSize();

  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_OK(parser->IsSupported(context.get(), context->node(),
                                context->registration()));
}

TEST_P(CompositeLayerOrGroupNormParserTest, TestIsSupported3DSucceeds) {
  std::vector<int> input_shape = {1, 2, 3};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shape);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.group_norm";
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1e-5);
    fbb.Int("channel_axis", 2);  // last axis
    AddReductionAxes(fbb, 2);
  });
  fbb.Finish();
  composite_params->attributes = fbb.GetBuffer().data();
  composite_params->attributes_size = fbb.GetSize();

  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_OK(parser->IsSupported(context.get(), context->node(),
                                context->registration()));
}

TEST_P(CompositeLayerOrGroupNormParserTest, TestIsSupported5DFails) {
  std::vector<int> input_shape = {1, 1, 1, 1, 1};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shape);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.group_norm";
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1e-5);
    fbb.Int("channel_axis", 3);  // last axis
    AddReductionAxes(fbb, 3);
  });
  fbb.Finish();
  composite_params->attributes = fbb.GetBuffer().data();
  composite_params->attributes_size = fbb.GetSize();

  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_THAT(parser->IsSupported(context.get(), context->node(),
                                  context->registration()),
              StatusIs(_, HasSubstr("Norm has bad input tensor dims")));
}

TEST_P(CompositeLayerOrGroupNormParserTest,
       TestIsSupported4DWithBadChannelAxis) {
  std::vector<::ml_drift::BHWC> input_shapes = {::ml_drift::BHWC(1, 2, 3, 4)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shapes);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.group_norm";
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1e-5);
    fbb.Int("channel_axis", 1);  // bad channel axis (not last axis 3)
    AddReductionAxes(fbb, 3);
  });
  fbb.Finish();
  composite_params->attributes = fbb.GetBuffer().data();
  composite_params->attributes_size = fbb.GetSize();

  auto parser = NewOperationParser(context->node(), context->registration());
  absl::Status status = parser->IsSupported(context.get(), context->node(),
                                            context->registration());
  if (norm_type() == NormType::kGroupNorm) {
    EXPECT_THAT(
        status,
        StatusIs(_, HasSubstr("Only channel-last tensor is supported")));
  } else {
    // LayerNorm does not use channel_axis.
    EXPECT_OK(status);
  }
}

TEST_P(CompositeLayerOrGroupNormParserTest,
       TestIsSupported4DWithBadReductionAxes) {
  std::vector<::ml_drift::BHWC> input_shapes = {::ml_drift::BHWC(1, 2, 3, 4)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shapes);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.group_norm";
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1e-5);
    fbb.Int("channel_axis", 3);
    AddReductionAxes(fbb, 2);  // Bad reduction axes, should be 3.
  });
  fbb.Finish();
  composite_params->attributes = fbb.GetBuffer().data();
  composite_params->attributes_size = fbb.GetSize();

  auto parser = NewOperationParser(context->node(), context->registration());
  absl::Status status = parser->IsSupported(context.get(), context->node(),
                                            context->registration());
  if (norm_type() == NormType::kGroupNorm || add_sub_type()) {
    EXPECT_THAT(status, StatusIs(_, HasSubstr("unexpected reduction axes")));
  } else {
    // LayerNorm does not use channel_axis.
    EXPECT_OK(status);
  }
}

TEST_P(CompositeLayerOrGroupNormParserTest, TestIsSupportedMissingAttributes) {
  std::vector<::ml_drift::BHWC> input_shapes = {::ml_drift::BHWC(1, 1, 1, 1)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shapes);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.group_norm";
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    // Missing num_groups and epsilon
    fbb.Int("channel_axis", 3);  // last axis
    AddReductionAxes(fbb, 3);
  });
  fbb.Finish();
  composite_params->attributes = fbb.GetBuffer().data();
  composite_params->attributes_size = fbb.GetSize();

  auto parser = NewOperationParser(context->node(), context->registration());
  EXPECT_THAT(parser->IsSupported(context.get(), context->node(),
                                  context->registration()),
              StatusIs(_, HasSubstr("Norm is missing")));
}

TEST_P(CompositeLayerOrGroupNormParserTest, TestParse) {
  std::vector<::ml_drift::BHWC> input_shapes = {::ml_drift::BHWC(1, 1, 1, 32)};
  auto context =
      std::make_unique<StubTfLiteContext>(kTfLiteBuiltinStablehloComposite,
                                          /*op_version=*/1,
                                          /*num_inputs=*/1, input_shapes);
  TfLiteStablehloCompositeParams* composite_params =
      static_cast<TfLiteStablehloCompositeParams*>(
          context->node()->builtin_data);
  composite_params->name = "odml.group_norm";
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1e-5f);
    fbb.Int("channel_axis", 3);  // last axis
    AddReductionAxes(fbb, 3);
  });
  fbb.Finish();
  composite_params->attributes = fbb.GetBuffer().data();
  composite_params->attributes_size = fbb.GetSize();

  auto parser = NewOperationParser(context->node(), context->registration());
  ASSERT_OK(parser->IsSupported(context.get(), context->node(),
                                context->registration()));

  ::ml_drift::GraphFloat32 graph;
  absl::flat_hash_map<int, ::ml_drift::Value*> tensor_to_value;
  ObjectReader reader(&graph, context.get(), context->node(), &tensor_to_value,
                      /*quant_conversion_map=*/nullptr,
                      /*tensor_to_buffer_id_map=*/nullptr,
                      /*shared_tensor_map=*/nullptr);
  parser->Parse(context->node(), context->registration(), &graph, &reader);

  auto group_norm_node = graph.nodes()[0];
  if (norm_type() == NormType::kGroupNorm) {
    EXPECT_EQ(group_norm_node->operation.type, "group_norm");
    EXPECT_EQ(graph.FindInputs(group_norm_node->id).size(), 1);
    const auto& attr = std::any_cast<::ml_drift::GroupNormAttributes>(
        group_norm_node->operation.attributes);
    EXPECT_EQ(attr.groups, 32);
    EXPECT_NEAR(attr.epsilon, 1e-5, 1e-6);
  } else {
    EXPECT_EQ(group_norm_node->operation.type, "layer_norm");
    EXPECT_EQ(graph.FindInputs(group_norm_node->id).size(), 1);
    const auto& attr = std::any_cast<::ml_drift::LayerNormAttributes>(
        group_norm_node->operation.attributes);
    EXPECT_NEAR(attr.epsilon, 1e-5, 1e-6);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CompositeLayerOrGroupNormParserTest, CompositeLayerOrGroupNormParserTest,
    ::testing::Values(
        CompositeLayerOrGroupNormParserInput{.norm_type = NormType::kGroupNorm,
                                             .add_sub_type = false},
        CompositeLayerOrGroupNormParserInput{.norm_type = NormType::kLayerNorm,
                                             .add_sub_type = false},
        CompositeLayerOrGroupNormParserInput{.norm_type = NormType::kGroupNorm,
                                             .add_sub_type = true},
        CompositeLayerOrGroupNormParserInput{.norm_type = NormType::kLayerNorm,
                                             .add_sub_type = true}));

TEST(ModelBuilderTest, CheckIfSupportedNode_ArgMax) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinArgMax,
      /*op_version=*/1,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));

  // Axis must be constant
  context->SetTensorType(2, kTfLiteInt32, kTfLiteMmapRo);

  EXPECT_TRUE(CheckIfSupportedNode(context.get(), context->node(),
                                   context->registration())
                  .ok());
}

TEST(ModelBuilderTest, CheckIfSupportedNode_FullyConnected) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinFullyConnected,
      /*op_version=*/9,
      /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));

  EXPECT_TRUE(CheckIfSupportedNode(context.get(), context->node(),
                                   context->registration())
                  .ok());
}

}  // namespace
}  // namespace litert::ml_drift
