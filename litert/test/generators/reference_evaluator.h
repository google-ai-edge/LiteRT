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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REFERENCE_EVALUATOR_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REFERENCE_EVALUATOR_H_

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

// An evaluator that executes a LiteRtSubgraphT or the decomposition of a
// composite operation using an extensible registry of reference operation
// kernels.
class ReferenceEvaluator {
 public:
  struct TensorData {
    LiteRtElementType element_type = kLiteRtElementTypeNone;
    std::vector<int32_t> dimensions;
    std::vector<float> f32_data;
    std::vector<int32_t> i32_data;
  };

  using TensorEnv = absl::flat_hash_map<const LiteRtTensorT*, TensorData>;

  // Function signature for executing an operation kernel.
  // `out` is pre-allocated with the shape of `op.Outputs()[0]` (if ranked).
  using OpKernelHandler = std::function<Expected<void>(
      const LiteRtOpT& op, const TensorEnv& env, TensorData& out)>;

  ReferenceEvaluator() { RegisterStandardOps(); }

  // Registers an op handler for a given op code.
  void RegisterOp(LiteRtOpCode op_code, OpKernelHandler handler) {
    registry_[op_code] = std::move(handler);
  }

  // Evaluates an arbitrary LiteRtSubgraphT using registered reference
  // operations.
  Expected<void> Evaluate(const LiteRtSubgraphT& subgraph,
                          const VarBuffers& inputs, VarBuffers& outputs) const {
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

      if (in_buf.Type().ElementType() == ElementType::Float32) {
        auto span = in_buf.Span<float>();
        tdata.f32_data.assign(span.begin(), span.end());
      } else if (in_buf.Type().ElementType() == ElementType::Float16) {
        auto span = in_buf.Span<tflite::half>();
        tdata.f32_data.resize(span.size());
        for (size_t j = 0; j < span.size(); ++j) {
          tdata.f32_data[j] = static_cast<float>(span[j]);
        }
      } else if (in_buf.Type().ElementType() == ElementType::Int32) {
        auto span = in_buf.Span<int32_t>();
        tdata.i32_data.assign(span.begin(), span.end());
      }
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
      size_t num_elements = 1;
      for (int32_t d : tdata.dimensions) {
        num_elements *= d;
      }
      const auto& weights = tensor->Weights().Buffer();
      if (tdata.element_type == kLiteRtElementTypeFloat32) {
        const float* ptr = reinterpret_cast<const float*>(weights.Data());
        tdata.f32_data.assign(ptr, ptr + num_elements);
      } else if (tdata.element_type == kLiteRtElementTypeFloat16) {
        const auto* ptr =
            reinterpret_cast<const tflite::half*>(weights.Data());
        tdata.f32_data.resize(num_elements);
        for (size_t j = 0; j < num_elements; ++j) {
          tdata.f32_data[j] = static_cast<float>(ptr[j]);
        }
      } else if (tdata.element_type == kLiteRtElementTypeInt32) {
        const int32_t* ptr = reinterpret_cast<const int32_t*>(weights.Data());
        tdata.i32_data.assign(ptr, ptr + num_elements);
      }
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
        return Error(
            kLiteRtStatusErrorNotFound,
            absl::StrFormat("Output tensor %s not found in environment",
                            out_tensor->Name()));
      }
      auto& out_buf = outputs[i];
      if (out_buf.Type().ElementType() == ElementType::Float32) {
        auto span = out_buf.Span<float>();
        if (span.size() != it->second.f32_data.size()) {
          return Error(kLiteRtStatusErrorRuntimeFailure,
                       "Output float buffer size mismatch");
        }
        std::copy(it->second.f32_data.begin(), it->second.f32_data.end(),
                  span.begin());
      } else if (out_buf.Type().ElementType() == ElementType::Float16) {
        auto span = out_buf.Span<tflite::half>();
        if (span.size() != it->second.f32_data.size()) {
          return Error(kLiteRtStatusErrorRuntimeFailure,
                       "Output half buffer size mismatch");
        }
        for (size_t j = 0; j < it->second.f32_data.size(); ++j) {
          span[j] = tflite::half(it->second.f32_data[j]);
        }
      } else if (out_buf.Type().ElementType() == ElementType::Int32) {
        auto span = out_buf.Span<int32_t>();
        if (span.size() != it->second.i32_data.size()) {
          return Error(kLiteRtStatusErrorRuntimeFailure,
                       "Output int32 buffer size mismatch");
        }
        std::copy(it->second.i32_data.begin(), it->second.i32_data.end(),
                  span.begin());
      } else {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported output element type in ReferenceEvaluator");
      }
    }

    return {};
  }

  // Evaluates the decomposition subgraph of a composite op inside a
  // LiteRtModelT.
  Expected<void> EvaluateComposite(const LiteRtModelT& model,
                                   const VarBuffers& inputs,
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
          absl::StrFormat("Invalid decomposition subgraph index %d (model has "
                          "%d subgraphs)",
                          decomp_index, model.Subgraphs().size()));
    }

    const auto& decomp_subgraph = *model.Subgraphs()[decomp_index];
    return Evaluate(decomp_subgraph, inputs, outputs);
  }

  // Static convenience wrappers using default registered operations.
  static Expected<void> EvaluateSubgraph(const LiteRtSubgraphT& subgraph,
                                         const VarBuffers& inputs,
                                         VarBuffers& outputs) {
    static const absl::NoDestructor<ReferenceEvaluator> default_evaluator;
    return default_evaluator->Evaluate(subgraph, inputs, outputs);
  }

  static Expected<void> EvaluateCompositeReference(const LiteRtModelT& model,
                                                   const VarBuffers& inputs,
                                                   VarBuffers& outputs) {
    static const absl::NoDestructor<ReferenceEvaluator> default_evaluator;
    return default_evaluator->EvaluateComposite(model, inputs, outputs);
  }

 private:
  void RegisterStandardOps() {
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
    auto MakeBinaryHandler = [](auto binary_op,
                                auto get_faf) -> OpKernelHandler {
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

    RegisterOp(
        kLiteRtOpCodeTflDiv,
        MakeBinaryHandler(std::divides<float>(), [](const LiteRtOpT& op) {
          const auto* opts = litert::internal::GetTflOptions(op).AsDivOptions();
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

    RegisterOp(
        kLiteRtOpCodeTflSoftmax,
        [](const LiteRtOpT& op, const TensorEnv& env,
           TensorData& out) -> Expected<void> {
          const auto& in = env.at(op.Inputs()[0]);
          const auto& opts = litert::internal::GetTflOptions(op);
          const auto* sm_opts = opts.AsSoftmaxOptions();
          float beta = sm_opts ? sm_opts->beta : 1.0f;

          int depth = out.dimensions.empty() ? 1 : out.dimensions.back();
          int batch =
              depth > 0 ? static_cast<int>(out.f32_data.size() / depth) : 1;

          litert::internal::ReferenceSoftmax(
              in.f32_data.data(), out.f32_data.data(), batch, depth, beta);
          return {};
        });

    RegisterOp(kLiteRtOpCodeTflTanh,
               [](const LiteRtOpT& op, const TensorEnv& env,
                  TensorData& out) -> Expected<void> {
                 const auto& in = env.at(op.Inputs()[0]);
                 litert::internal::ReferenceTanh(in.f32_data.data(),
                                                 in.f32_data.size(),
                                                 out.f32_data.data());
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

  Expected<void> ExecuteOp(const LiteRtOpT& op, TensorEnv& env) const {
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
      size_t out_elements = 1;
      for (int32_t d : out.dimensions) out_elements *= d;
      out.f32_data.resize(out_elements);
    }

    LITERT_RETURN_IF_ERROR(it->second(op, env, out));
    if (op.NumOutputs() > 0 && op.Outputs()[0] != nullptr) {
      env[op.Outputs()[0]] = std::move(out);
    }
    return {};
  }

  absl::flat_hash_map<LiteRtOpCode, OpKernelHandler> registry_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REFERENCE_EVALUATOR_H_
