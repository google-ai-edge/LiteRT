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

#include "litert/test/generators/reference_evaluator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/concatenation.h"
#include "litert/core/model/ops/matmul.h"
#include "litert/core/model/ops/simple_binary.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/core/model/ops/transpose.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/generators/common.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/types/half.h"

namespace litert::testing {

ReferenceEvaluator::ReferenceEvaluator() { RegisterStandardOps(); }

ReferenceEvaluator ReferenceEvaluator::Create() { return ReferenceEvaluator(); }

void ReferenceEvaluator::RegisterOp(LiteRtOpCode op_code,
                                    OpKernelHandler handler) {
  registry_[op_code] = std::move(handler);
}

Expected<size_t> ReferenceEvaluator::TensorData::NumElements() const {
  size_t num_elements = 1;
  for (int32_t d : dimensions) {
    if (d <= 0) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   absl::StrFormat("Invalid non-positive dimension %d", d));
    }
    num_elements *= static_cast<size_t>(d);
  }
  return num_elements;
}

Expected<void> ReferenceEvaluator::TensorData::AssignData(const void* data,
                                                          size_t num_elements) {
  if (num_elements == 0) {
    f32_data.clear();
    i32_data.clear();
    return {};
  }
  if (data == nullptr) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Null data pointer provided to TensorData::AssignData");
  }
  if (element_type == kLiteRtElementTypeFloat32) {
    const float* ptr = static_cast<const float*>(data);
    f32_data.assign(ptr, ptr + num_elements);
  } else if (element_type == kLiteRtElementTypeFloat16) {
    const auto* ptr = static_cast<const tflite::half*>(data);
    f32_data.resize(num_elements);
    for (size_t j = 0; j < num_elements; ++j) {
      f32_data[j] = static_cast<float>(ptr[j]);
    }
  } else if (element_type == kLiteRtElementTypeInt32) {
    const int32_t* ptr = static_cast<const int32_t*>(data);
    i32_data.assign(ptr, ptr + num_elements);
  } else {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported element type in ReferenceEvaluator::TensorData");
  }
  return {};
}

Expected<void> ReferenceEvaluator::TensorData::AssignData(const void* data) {
  LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements());
  return AssignData(data, num_elements);
}

Expected<void> ReferenceEvaluator::TensorData::CopyTo(
    SimpleBuffer& out_buf) const {
  if (out_buf.Type().ElementType() == ElementType::Float32) {
    auto span = out_buf.Span<float>();
    if (span.size() != f32_data.size()) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Output float buffer size mismatch");
    }
    std::copy(f32_data.begin(), f32_data.end(), span.begin());
  } else if (out_buf.Type().ElementType() == ElementType::Float16) {
    auto span = out_buf.Span<tflite::half>();
    if (span.size() != f32_data.size()) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Output half buffer size mismatch");
    }
    for (size_t j = 0; j < f32_data.size(); ++j) {
      span[j] = tflite::half(f32_data[j]);
    }
  } else if (out_buf.Type().ElementType() == ElementType::Int32) {
    auto span = out_buf.Span<int32_t>();
    if (span.size() != i32_data.size()) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Output int32 buffer size mismatch");
    }
    std::copy(i32_data.begin(), i32_data.end(), span.begin());
  } else {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported output element type in ReferenceEvaluator");
  }
  return {};
}

Expected<void> ReferenceEvaluator::Evaluate(const LiteRtSubgraphT& subgraph,
                                            const VarBuffers& inputs,
                                            VarBuffers& outputs) const {
  if (inputs.size() < subgraph.Inputs().size()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 absl::StrFormat("Expected at least %d inputs, got %d",
                                 subgraph.Inputs().size(), inputs.size()));
  }
  if (outputs.size() < subgraph.Outputs().size()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 absl::StrFormat("Expected at least %d outputs, got %d",
                                 subgraph.Outputs().size(), outputs.size()));
  }

  TensorEnv tensor_env;

  // 1. Initialize subgraph inputs from VarBuffers.
  for (size_t i = 0; i < subgraph.Inputs().size(); ++i) {
    const LiteRtTensorT* in_tensor = subgraph.Inputs()[i];
    const auto& in_buf = inputs[i];
    TensorData tdata;
    const auto& dims = in_buf.Type().Layout().Dimensions();
    tdata.dimensions.assign(dims.begin(), dims.end());
    tdata.element_type =
        static_cast<LiteRtElementType>(in_buf.Type().ElementType());

    LITERT_RETURN_IF_ERROR(tdata.AssignData(in_buf.Data().Data()));
    tensor_env[in_tensor] = std::move(tdata);
  }

  // 2. Initialize constant weight tensors in the subgraph.
  for (const auto* tensor : subgraph.Tensors()) {
    if (tensor->Weights().Buffer().Size() == 0) continue;
    auto [it, inserted] = tensor_env.try_emplace(tensor);
    if (!inserted) continue;

    TensorData& tdata = it->second;
    if (tensor->Type().first == kLiteRtRankedTensorType) {
      const auto& layout = tensor->Type().second.ranked_tensor_type.layout;
      tdata.dimensions.assign(layout.dimensions,
                              layout.dimensions + layout.rank);
      tdata.element_type =
          tensor->Type().second.ranked_tensor_type.element_type;
    }
    const auto& weights = tensor->Weights().Buffer();
    LITERT_RETURN_IF_ERROR(tdata.AssignData(weights.Data()));
  }

  // 3. Execute operations in topological order.
  for (const auto* op : subgraph.Ops()) {
    LITERT_RETURN_IF_ERROR(ExecuteOp(*op, tensor_env));
  }

  // 4. Copy results to outputs.
  for (size_t i = 0; i < subgraph.Outputs().size(); ++i) {
    const auto* out_tensor = subgraph.Outputs()[i];
    auto it = tensor_env.find(out_tensor);
    if (it == tensor_env.end()) {
      return Error(kLiteRtStatusErrorNotFound,
                   absl::StrFormat("Output tensor %s not found in environment",
                                   out_tensor->Name()));
    }
    LITERT_RETURN_IF_ERROR(it->second.CopyTo(outputs[i]));
  }

  return {};
}

Expected<void> ReferenceEvaluator::EvaluateComposite(
    const LiteRtModelT& model, const VarBuffers& inputs,
    VarBuffers& outputs) const {
  if (model.Subgraphs().empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument, "Model has no subgraphs");
  }
  const auto& main_subgraph = *model.Subgraphs()[0];
  const LiteRtOpT* composite_op = nullptr;
  for (const auto* op : main_subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      composite_op = op;
      break;
    }
  }
  if (!composite_op) {
    return Error(kLiteRtStatusErrorNotFound,
                 "No composite op found in subgraph 0");
  }

  const auto& opts2 = ::litert::internal::GetTflOptions2(*composite_op);
  const auto* comp_opts = opts2.AsStableHLOCompositeOptions();
  if (!comp_opts) {
    return Error(kLiteRtStatusErrorNotFound,
                 "Composite op missing StableHLOCompositeOptions");
  }
  int32_t decomp_index = comp_opts->decomposition_subgraph_index;
  if (decomp_index < 0 ||
      decomp_index >= static_cast<int32_t>(model.Subgraphs().size())) {
    return Error(
        kLiteRtStatusErrorIndexOOB,
        absl::StrFormat(
            "Invalid decomposition subgraph index %d (model has %d subgraphs)",
            decomp_index, model.Subgraphs().size()));
  }

  const auto& decomp_subgraph = *model.Subgraphs()[decomp_index];
  return Evaluate(decomp_subgraph, inputs, outputs);
}

Expected<void> ReferenceEvaluator::EvaluateSubgraph(
    const LiteRtSubgraphT& subgraph, const VarBuffers& inputs,
    VarBuffers& outputs) {
  static const absl::NoDestructor<ReferenceEvaluator> default_evaluator;
  return default_evaluator->Evaluate(subgraph, inputs, outputs);
}

Expected<void> ReferenceEvaluator::EvaluateCompositeReference(
    const LiteRtModelT& model, const VarBuffers& inputs, VarBuffers& outputs) {
  static const absl::NoDestructor<ReferenceEvaluator> default_evaluator;
  return default_evaluator->EvaluateComposite(model, inputs, outputs);
}

void ReferenceEvaluator::RegisterStandardOps() {
  RegisterOp(kLiteRtOpCodeTflBatchMatmul,
             [](const LiteRtOpT& op, const TensorEnv& env,
                TensorData& out) -> Expected<void> {
               const auto& lhs = env.at(op.Inputs()[0]);
               const auto& rhs = env.at(op.Inputs()[1]);
               const auto& opts = litert::internal::GetTflOptions(op);
               const auto* bmm_opts = opts.AsBatchMatMulOptions();
               bool adj_x = bmm_opts ? bmm_opts->adj_x : false;
               bool adj_y = bmm_opts ? bmm_opts->adj_y : false;

               litert::internal::ReferenceBatchMatmul(
                   lhs.f32_data.data(), lhs.dimensions.data(),
                   lhs.dimensions.size(), rhs.f32_data.data(),
                   rhs.dimensions.data(), rhs.dimensions.size(),
                   out.f32_data.data(), out.dimensions.data(),
                   out.dimensions.size(), adj_x, adj_y);
               return {};
             });

  // Elementwise binary operations (Add, Mul, Div, Sub).
  auto MakeBinaryHandler = [](auto binary_op, auto get_faf) -> OpKernelHandler {
    return [binary_op, get_faf](const LiteRtOpT& op, const TensorEnv& env,
                                TensorData& out) -> Expected<void> {
      const auto& in1 = env.at(op.Inputs()[0]);
      const auto& in2 = env.at(op.Inputs()[1]);
      litert::internal::ReferenceBinaryGeneric(
          in1.f32_data.data(), in1.dimensions.data(), in1.dimensions.size(),
          in2.f32_data.data(), in2.dimensions.data(), in2.dimensions.size(),
          out.f32_data.data(), out.dimensions.data(), out.dimensions.size(),
          binary_op);
      litert::internal::ApplyActivation(out.f32_data.data(),
                                        out.f32_data.size(), get_faf(op));
      return {};
    };
  };

  RegisterOp(kLiteRtOpCodeTflAdd,
             MakeBinaryHandler(std::plus<float>(), [](const LiteRtOpT& op) {
               const auto* opts =
                   litert::internal::GetTflOptions(op).AsAddOptions();
               return opts ? opts->fused_activation_function
                           : tflite::ActivationFunctionType_NONE;
             }));

  RegisterOp(
      kLiteRtOpCodeTflMul,
      MakeBinaryHandler(std::multiplies<float>(), [](const LiteRtOpT& op) {
        const auto* opts = litert::internal::GetTflOptions(op).AsMulOptions();
        return opts ? opts->fused_activation_function
                    : tflite::ActivationFunctionType_NONE;
      }));

  RegisterOp(kLiteRtOpCodeTflDiv,
             MakeBinaryHandler(std::divides<float>(), [](const LiteRtOpT& op) {
               const auto* opts =
                   litert::internal::GetTflOptions(op).AsDivOptions();
               return opts ? opts->fused_activation_function
                           : tflite::ActivationFunctionType_NONE;
             }));

  RegisterOp(kLiteRtOpCodeTflSub,
             MakeBinaryHandler(std::minus<float>(), [](const LiteRtOpT& op) {
               const auto* opts =
                   litert::internal::GetTflOptions(op).AsSubOptions();
               return opts ? opts->fused_activation_function
                           : tflite::ActivationFunctionType_NONE;
             }));

  RegisterOp(kLiteRtOpCodeTflSoftmax,
             [](const LiteRtOpT& op, const TensorEnv& env,
                TensorData& out) -> Expected<void> {
               const auto& in = env.at(op.Inputs()[0]);
               const auto& opts = litert::internal::GetTflOptions(op);
               const auto* sm_opts = opts.AsSoftmaxOptions();
               float beta = sm_opts ? sm_opts->beta : 1.0f;

               int depth = out.dimensions.empty() ? 1 : out.dimensions.back();
               int batch = depth > 0
                               ? static_cast<int>(out.f32_data.size() / depth)
                               : 1;

               litert::internal::ReferenceSoftmax(
                   in.f32_data.data(), out.f32_data.data(), batch, depth, beta);
               return {};
             });

  RegisterOp(kLiteRtOpCodeTflTanh,
             [](const LiteRtOpT& op, const TensorEnv& env,
                TensorData& out) -> Expected<void> {
               const auto& in = env.at(op.Inputs()[0]);
               litert::internal::ReferenceTanh(
                   in.f32_data.data(), in.f32_data.size(), out.f32_data.data());
               return {};
             });

  RegisterOp(kLiteRtOpCodeTflReshape,
             [](const LiteRtOpT& op, const TensorEnv& env,
                TensorData& out) -> Expected<void> {
               const auto& in = env.at(op.Inputs()[0]);
               out.f32_data = in.f32_data;
               out.i32_data = in.i32_data;
               return {};
             });

  RegisterOp(kLiteRtOpCodeTflTranspose,
             [](const LiteRtOpT& op, const TensorEnv& env,
                TensorData& out) -> Expected<void> {
               const auto& in = env.at(op.Inputs()[0]);
               const auto& perm = env.at(op.Inputs()[1]);
               litert::internal::ReferenceTranspose(
                   in.f32_data.data(), in.dimensions.data(),
                   perm.i32_data.data(), in.dimensions.size(),
                   out.f32_data.data());
               return {};
             });

  RegisterOp(kLiteRtOpCodeTflConcatenation,
             [](const LiteRtOpT& op, const TensorEnv& env,
                TensorData& out) -> Expected<void> {
               std::vector<const float*> in_ptrs;
               std::vector<litert::internal::Dims> in_dims;
               in_ptrs.reserve(op.Inputs().size());
               in_dims.reserve(op.Inputs().size());

               for (const auto* in_tensor : op.Inputs()) {
                 const auto& in_data = env.at(in_tensor);
                 in_ptrs.push_back(in_data.f32_data.data());
                 in_dims.push_back(litert::internal::Dims(
                     in_data.dimensions.begin(), in_data.dimensions.end()));
               }

               const auto& opts = litert::internal::GetTflOptions(op);
               const auto* concat_opts = opts.AsConcatenationOptions();
               int axis = concat_opts ? concat_opts->axis : 0;
               tflite::ActivationFunctionType faf =
                   concat_opts ? concat_opts->fused_activation_function
                               : tflite::ActivationFunctionType_NONE;

               absl::Span<const float* const> in_ptrs_span(in_ptrs.data(),
                                                           in_ptrs.size());
               litert::internal::ReferenceConcatenation<float>(
                   in_ptrs_span, absl::MakeSpan(in_dims), out.f32_data.data(),
                   axis, faf);
               return {};
             });
}

Expected<void> ReferenceEvaluator::ExecuteOp(const LiteRtOpT& op,
                                             TensorEnv& env) const {
  auto it = registry_.find(op.OpCode());
  if (it == registry_.end()) {
    return Error(kLiteRtStatusErrorUnsupported,
                 absl::StrFormat("Op %d not supported by ReferenceEvaluator",
                                 op.OpCode()));
  }

  TensorData out;
  if (op.NumOutputs() > 0 && op.Outputs()[0] != nullptr &&
      op.Outputs()[0]->Type().first == kLiteRtRankedTensorType) {
    const auto& layout =
        op.Outputs()[0]->Type().second.ranked_tensor_type.layout;
    out.dimensions.assign(layout.dimensions, layout.dimensions + layout.rank);
    out.element_type =
        op.Outputs()[0]->Type().second.ranked_tensor_type.element_type;
    LITERT_ASSIGN_OR_RETURN(size_t out_elements, out.NumElements());
    out.f32_data.resize(out_elements);
  }

  LITERT_RETURN_IF_ERROR(it->second(op, env, out));
  if (op.NumOutputs() > 0 && op.Outputs()[0] != nullptr) {
    env[op.Outputs()[0]] = std::move(out);
  }
  return {};
}

}  // namespace litert::testing
