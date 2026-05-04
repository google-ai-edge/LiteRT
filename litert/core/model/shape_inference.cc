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

#include "litert/core/model/shape_inference.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/broadcast_to.h"
#include "litert/core/model/ops/concatenation.h"
#include "litert/core/model/ops/convolution.h"
#include "litert/core/model/ops/gather.h"
#include "litert/core/model/ops/matmul.h"
#include "litert/core/model/ops/pack.h"
#include "litert/core/model/ops/pad.h"
#include "litert/core/model/ops/pooling.h"
#include "litert/core/model/ops/reductions.h"
#include "litert/core/model/ops/reshape.h"
#include "litert/core/model/ops/select.h"
#include "litert/core/model/ops/shape.h"
#include "litert/core/model/ops/simple_binary.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/core/model/ops/slice.h"
#include "litert/core/model/ops/spatial.h"
#include "litert/core/model/ops/topk.h"
#include "litert/core/model/ops/transpose.h"
#include "litert/core/model/ops/unpack.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"

namespace litert::internal {

namespace {

// Concrete context for shape inference within a LiteRtModel graph.
class GraphShapeInferenceContext : public ShapeInferenceContext {
 public:
  GraphShapeInferenceContext(const LiteRtOpT& op,
                             const TensorDataMap& transient_data)
      : op_(op), transient_data_(transient_data) {}

  Dims GetInputShape(size_t index) const override {
    if (index >= op_.Inputs().size() || op_.Inputs()[index] == nullptr) {
      return {};
    }
    const auto& tensor = *op_.Inputs()[index];
    if (tensor.Type().first == kLiteRtRankedTensorType) {
      const auto& layout = tensor.Type().second.ranked_tensor_type.layout;
      return Dims(layout.dimensions, layout.dimensions + layout.rank);
    }
    return {};
  }

  absl::Span<const uint8_t> GetInputData(size_t index) const override {
    if (index >= op_.Inputs().size() || op_.Inputs()[index] == nullptr) {
      return {};
    }
    const auto& tensor = *op_.Inputs()[index];
    // Prioritize dynamic/transient data from previous ops in this pass.
    if (auto it = transient_data_.find(&tensor); it != transient_data_.end()) {
      return absl::MakeConstSpan(it->second);
    }
    // Fallback to static weights defined in the model.
    if (tensor.Weights().Buffer().Size() > 0) {
      auto weights = tensor.Weights().Buffer();
      return absl::MakeConstSpan(weights.Data(), weights.Size());
    }
    return {};
  }

  const TflOptions& GetOptions() const override { return GetTflOptions(op_); }

  LiteRtOpCode GetOpCode() const override { return op_.OpCode(); }

  const LiteRtOpT& GetOp() const { return op_; }

 private:
  const LiteRtOpT& op_;
  const TensorDataMap& transient_data_;
};

}  // namespace

ShapeInferenceEngine::ShapeInferenceEngine(LiteRtModelT* model)
    : model_(model) {
  RegisterStandardOps();
}

ShapeInferenceEngine::ShapeInferenceEngine() { RegisterStandardOps(); }

void ShapeInferenceEngine::RegisterStandardOps() {
  // Helper to bridge legacy inferrers into the new stateless system.
  auto AdaptToStatelessOpInferrer =
      [](OpShapeInferrer legacy_f) -> StatelessOpInferrer {
    return
        [legacy_f](const ShapeInferenceContext& ctx, InferenceResult& result) {
          const auto& graph_ctx =
              static_cast<const GraphShapeInferenceContext&>(ctx);
          std::vector<Dims> input_shapes;
          input_shapes.reserve(graph_ctx.GetOp().NumInputs());
          for (size_t i = 0; i < graph_ctx.GetOp().NumInputs(); ++i) {
            input_shapes.push_back(graph_ctx.GetInputShape(i));
          }
          if (result.output_shapes.size() < graph_ctx.GetOp().NumOutputs()) {
            result.output_shapes.resize(graph_ctx.GetOp().NumOutputs());
          }
          return legacy_f(graph_ctx.GetOp(), absl::MakeSpan(input_shapes),
                          result.output_shapes);
        };
  };

  RegisterInferrer(kLiteRtOpCodeTflAbs, AdaptToStatelessOpInferrer(InferAbs));
  RegisterInferrer(kLiteRtOpCodeTflCeil, AdaptToStatelessOpInferrer(InferCeil));
  RegisterInferrer(kLiteRtOpCodeTflCos, AdaptToStatelessOpInferrer(InferCos));
  RegisterInferrer(kLiteRtOpCodeTflDequantize,
                   AdaptToStatelessOpInferrer(InferDequantize));
  RegisterInferrer(kLiteRtOpCodeTflElu, AdaptToStatelessOpInferrer(InferElu));
  RegisterInferrer(kLiteRtOpCodeTflExp, AdaptToStatelessOpInferrer(InferExp));
  RegisterInferrer(kLiteRtOpCodeTflFloor,
                   AdaptToStatelessOpInferrer(InferFloor));
  RegisterInferrer(kLiteRtOpCodeTflGelu, AdaptToStatelessOpInferrer(InferGelu));
  RegisterInferrer(kLiteRtOpCodeTflHardSwish,
                   AdaptToStatelessOpInferrer(InferHardSwish));
  RegisterInferrer(kLiteRtOpCodeTflLeakyRelu,
                   AdaptToStatelessOpInferrer(InferLeakyRelu));
  RegisterInferrer(kLiteRtOpCodeTflLog, AdaptToStatelessOpInferrer(InferLog));
  RegisterInferrer(kLiteRtOpCodeTflLogicalNot,
                   AdaptToStatelessOpInferrer(InferLogicalNot));
  RegisterInferrer(kLiteRtOpCodeTflLogistic,
                   AdaptToStatelessOpInferrer(InferLogistic));
  RegisterInferrer(kLiteRtOpCodeTflNeg, AdaptToStatelessOpInferrer(InferNeg));
  RegisterInferrer(kLiteRtOpCodeTflQuantize,
                   AdaptToStatelessOpInferrer(InferQuantize));
  RegisterInferrer(kLiteRtOpCodeTflRelu, AdaptToStatelessOpInferrer(InferRelu));
  RegisterInferrer(kLiteRtOpCodeTflRelu0To1,
                   AdaptToStatelessOpInferrer(InferRelu0To1));
  RegisterInferrer(kLiteRtOpCodeTflRelu6,
                   AdaptToStatelessOpInferrer(InferRelu6));
  RegisterInferrer(kLiteRtOpCodeTflReluN1To1,
                   AdaptToStatelessOpInferrer(InferReluN1To1));
  RegisterInferrer(kLiteRtOpCodeTflRound,
                   AdaptToStatelessOpInferrer(InferRound));
  RegisterInferrer(kLiteRtOpCodeTflRsqrt,
                   AdaptToStatelessOpInferrer(InferRsqrt));
  RegisterInferrer(kLiteRtOpCodeTflSign, AdaptToStatelessOpInferrer(InferSign));
  RegisterInferrer(kLiteRtOpCodeTflSin, AdaptToStatelessOpInferrer(InferSin));
  RegisterInferrer(kLiteRtOpCodeTflSoftmax,
                   AdaptToStatelessOpInferrer(InferSoftmax));
  RegisterInferrer(kLiteRtOpCodeTflSqrt, AdaptToStatelessOpInferrer(InferSqrt));
  RegisterInferrer(kLiteRtOpCodeTflSquare,
                   AdaptToStatelessOpInferrer(InferSquare));
  RegisterInferrer(kLiteRtOpCodeTflTanh, AdaptToStatelessOpInferrer(InferTanh));
  RegisterInferrer(kLiteRtOpCodeTflEqual,
                   AdaptToStatelessOpInferrer(InferEqual));
  RegisterInferrer(kLiteRtOpCodeTflFloorDiv,
                   AdaptToStatelessOpInferrer(InferFloorDiv));
  RegisterInferrer(kLiteRtOpCodeTflGreater,
                   AdaptToStatelessOpInferrer(InferGreater));
  RegisterInferrer(kLiteRtOpCodeTflGreaterEqual,
                   AdaptToStatelessOpInferrer(InferGreaterEqual));
  RegisterInferrer(kLiteRtOpCodeTflLess, AdaptToStatelessOpInferrer(InferLess));
  RegisterInferrer(kLiteRtOpCodeTflLessEqual,
                   AdaptToStatelessOpInferrer(InferLessEqual));
  RegisterInferrer(kLiteRtOpCodeTflLogicalAnd,
                   AdaptToStatelessOpInferrer(InferLogicalAnd));
  RegisterInferrer(kLiteRtOpCodeTflLogicalOr,
                   AdaptToStatelessOpInferrer(InferLogicalOr));
  RegisterInferrer(kLiteRtOpCodeTflMaximum,
                   AdaptToStatelessOpInferrer(InferMaximum));
  RegisterInferrer(kLiteRtOpCodeTflMinimum,
                   AdaptToStatelessOpInferrer(InferMinimum));
  RegisterInferrer(kLiteRtOpCodeTflNotEqual,
                   AdaptToStatelessOpInferrer(InferNotEqual));
  RegisterInferrer(kLiteRtOpCodeTflPow, AdaptToStatelessOpInferrer(InferPow));
  RegisterInferrer(kLiteRtOpCodeTflPrelu,
                   AdaptToStatelessOpInferrer(InferPrelu));
  RegisterInferrer(kLiteRtOpCodeTflSquaredDifference,
                   AdaptToStatelessOpInferrer(InferSquaredDifference));
  RegisterInferrer(kLiteRtOpCodeTflAdd, AdaptToStatelessOpInferrer(InferAdd));
  RegisterInferrer(kLiteRtOpCodeTflArgMax,
                   AdaptToStatelessOpInferrer(InferArgMinMax));
  RegisterInferrer(kLiteRtOpCodeTflArgMin,
                   AdaptToStatelessOpInferrer(InferArgMinMax));
  RegisterInferrer(kLiteRtOpCodeTflAveragePool2d,
                   AdaptToStatelessOpInferrer(InferPool2D));
  RegisterInferrer(kLiteRtOpCodeTflBatchMatmul,
                   AdaptToStatelessOpInferrer(InferBatchMatmul));
  RegisterInferrer(kLiteRtOpCodeTflBroadcastTo,
                   AdaptToStatelessOpInferrer(InferBroadcastTo));
  RegisterInferrer(kLiteRtOpCodeTflCast, AdaptToStatelessOpInferrer(InferCast));
  RegisterInferrer(kLiteRtOpCodeTflConcatenation,
                   AdaptToStatelessOpInferrer(InferConcatenation));
  RegisterInferrer(kLiteRtOpCodeTflConv2d,
                   AdaptToStatelessOpInferrer(InferConv2D));
  RegisterInferrer(kLiteRtOpCodeTflConv3d,
                   AdaptToStatelessOpInferrer(InferConv3D));
  RegisterInferrer(kLiteRtOpCodeTflConv3dTranspose,
                   AdaptToStatelessOpInferrer(InferConv3DTranspose));
  RegisterInferrer(kLiteRtOpCodeTflDepthToSpace,
                   AdaptToStatelessOpInferrer(InferDepthToSpace));
  RegisterInferrer(kLiteRtOpCodeTflDepthwiseConv2d,
                   AdaptToStatelessOpInferrer(InferDepthwiseConv2D));
  RegisterInferrer(kLiteRtOpCodeTflDiv, AdaptToStatelessOpInferrer(InferDiv));
  RegisterInferrer(kLiteRtOpCodeTflDynamicUpdateSlice,
                   AdaptToStatelessOpInferrer(InferDynamicUpdateSlice));
  RegisterInferrer(kLiteRtOpCodeTflEmbeddingLookup,
                   AdaptToStatelessOpInferrer(InferEmbeddingLookup));
  RegisterInferrer(kLiteRtOpCodeTflFullyConnected,
                   AdaptToStatelessOpInferrer(InferFullyConnected));
  RegisterInferrer(kLiteRtOpCodeTflGather,
                   AdaptToStatelessOpInferrer(InferGather));
  RegisterInferrer(kLiteRtOpCodeTflGatherNd,
                   AdaptToStatelessOpInferrer(InferGatherNd));
  RegisterInferrer(kLiteRtOpCodeTflL2Pool2d,
                   AdaptToStatelessOpInferrer(InferPool2D));
  RegisterInferrer(kLiteRtOpCodeTflMaxPool2d,
                   AdaptToStatelessOpInferrer(InferPool2D));
  RegisterInferrer(kLiteRtOpCodeTflMean,
                   AdaptToStatelessOpInferrer(InferReduce));
  RegisterInferrer(kLiteRtOpCodeTflMirrorPad,
                   AdaptToStatelessOpInferrer(InferMirrorPad));
  RegisterInferrer(kLiteRtOpCodeTflMul, AdaptToStatelessOpInferrer(InferMul));
  RegisterInferrer(kLiteRtOpCodeTflPack, AdaptToStatelessOpInferrer(InferPack));
  RegisterInferrer(kLiteRtOpCodeTflPad, AdaptToStatelessOpInferrer(InferPad));
  RegisterInferrer(kLiteRtOpCodeTflPadv2,
                   AdaptToStatelessOpInferrer(InferPadv2));
  RegisterInferrer(kLiteRtOpCodeTflReduceAll,
                   AdaptToStatelessOpInferrer(InferReduce));
  RegisterInferrer(kLiteRtOpCodeTflReduceAny,
                   AdaptToStatelessOpInferrer(InferReduce));
  RegisterInferrer(kLiteRtOpCodeTflReduceMax,
                   AdaptToStatelessOpInferrer(InferReduce));
  RegisterInferrer(kLiteRtOpCodeTflReduceMin,
                   AdaptToStatelessOpInferrer(InferReduce));
  RegisterInferrer(kLiteRtOpCodeTflSum,
                   AdaptToStatelessOpInferrer(InferReduce));
  RegisterInferrer(kLiteRtOpCodeTflResizeBilinear,
                   AdaptToStatelessOpInferrer(InferResizeBilinear));
  RegisterInferrer(kLiteRtOpCodeTflResizeNearestNeighbor,
                   AdaptToStatelessOpInferrer(InferResizeNearestNeighbor));
  RegisterInferrer(kLiteRtOpCodeTflSelect,
                   AdaptToStatelessOpInferrer(InferSelect));
  RegisterInferrer(kLiteRtOpCodeTflSelectV2,
                   AdaptToStatelessOpInferrer(InferSelect));
  RegisterInferrer(kLiteRtOpCodeTflSlice,
                   AdaptToStatelessOpInferrer(InferSlice));
  RegisterInferrer(kLiteRtOpCodeTflSpaceToDepth,
                   AdaptToStatelessOpInferrer(InferSpaceToDepth));
  RegisterInferrer(kLiteRtOpCodeTflSub, AdaptToStatelessOpInferrer(InferSub));
  RegisterInferrer(kLiteRtOpCodeTflTranspose,
                   AdaptToStatelessOpInferrer(InferTranspose));
  RegisterInferrer(kLiteRtOpCodeTflTransposeConv,
                   AdaptToStatelessOpInferrer(InferTransposeConv));
  RegisterInferrer(kLiteRtOpCodeTflUnpack,
                   AdaptToStatelessOpInferrer(InferUnpack));
  RegisterInferrer(kLiteRtOpCodeTflCumsum,
                   AdaptToStatelessOpInferrer(InferCumsum));
  RegisterInferrer(kLiteRtOpCodeTflL2Normalization,
                   AdaptToStatelessOpInferrer(InferL2Normalization));
  RegisterInferrer(kLiteRtOpCodeTflReverseV2,
                   AdaptToStatelessOpInferrer(InferReverseV2));
  RegisterInferrer(kLiteRtOpCodeTflTopkV2,
                   AdaptToStatelessOpInferrer(InferTopKV2));

  // Native stateless inferrers.
  RegisterInferrer(kLiteRtOpCodeTflShape, InferShape);
  RegisterInferrer(kLiteRtOpCodeTflRank, InferRank);
  RegisterInferrer(kLiteRtOpCodeTflReshape, InferReshape);
}

void ShapeInferenceEngine::RegisterInferrer(LiteRtOpCode op_code,
                                            StatelessOpInferrer inferrer) {
  registry_[op_code] = std::move(inferrer);
}

LiteRtStatus ShapeInferenceEngine::InferShapes(bool validation_only,
                                               LiteRtOp* failing_op) {
  transient_data_.clear();
  if (!model_) {
    LITERT_LOG(LITERT_ERROR, "Model is null.");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  for (auto* subgraph : model_->Subgraphs()) {
    if (auto status =
            InferSubgraphShapes(subgraph, validation_only, failing_op);
        status != kLiteRtStatusOk) {
      return status;
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ShapeInferenceEngine::InferSubgraphShapes(
    LiteRtSubgraphT* subgraph, bool validation_only, LiteRtOp* failing_op) {
  transient_data_.clear();
  for (auto* op : subgraph->Ops()) {
    if (auto status = InferOpShapes(op, validation_only);
        status != kLiteRtStatusOk) {
      if (failing_op) *failing_op = op;
      return status;
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ShapeInferenceEngine::InferOpShapes(LiteRtOpT* op,
                                                 bool validation_only) {
  GraphShapeInferenceContext ctx(*op, transient_data_);
  InferenceResult result;
  result.output_shapes.resize(op->NumOutputs());

  if (auto status = InferOpShapes(ctx, result); status != kLiteRtStatusOk) {
    return status;
  }

  for (size_t i = 0; i < op->Outputs().size(); ++i) {
    if (i >= result.output_shapes.size() || op->Outputs()[i] == nullptr) {
      continue;
    }
    auto& tensor = op->Output(i);
    const auto& shape = result.output_shapes[i];

    if (validation_only) {
      if (tensor.Type().first == kLiteRtRankedTensorType) {
        const auto& existing_shape =
            tensor.Type().second.ranked_tensor_type.layout;
        if (existing_shape.rank != (int32_t)shape.size()) {
          LITERT_LOG(LITERT_ERROR,
                     "Rank mismatch for output tensor after shape inference.");
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
        for (size_t d = 0; d < shape.size(); ++d) {
          if (existing_shape.dimensions[d] != -1 && shape[d] != -1 &&
              existing_shape.dimensions[d] != shape[d]) {
            LITERT_LOG(LITERT_ERROR,
                       "Dimension mismatch for output tensor after shape "
                       "inference.");
            return kLiteRtStatusErrorShapeInferenceFailed;
          }
        }
      }
    } else {
      // For non-validation runs, ensure we don't accidentally resize constant
      // tensors.
      if (tensor.Weights().Buffer().Size() > 0) {
        if (tensor.Type().first == kLiteRtRankedTensorType) {
          const auto& existing_shape =
              tensor.Type().second.ranked_tensor_type.layout;
          if (existing_shape.rank != (int32_t)shape.size()) {
            return kLiteRtStatusErrorShapeInferenceFailed;
          }
          for (size_t d = 0; d < shape.size(); ++d) {
            if (existing_shape.dimensions[d] != shape[d]) {
              return kLiteRtStatusErrorShapeInferenceFailed;
            }
          }
        }
      }

      // Preserve element type while updating the shape.
      LiteRtElementType element_type = kLiteRtElementTypeNone;
      if (tensor.Type().first == kLiteRtRankedTensorType) {
        element_type = tensor.Type().second.ranked_tensor_type.element_type;
      } else if (tensor.Type().first == kLiteRtUnrankedTensorType) {
        element_type = tensor.Type().second.unranked_tensor_type.element_type;
      }

      if (element_type != kLiteRtElementTypeNone) {
        tensor.SetType(
            MakeRankedTensorType(element_type, absl::MakeSpan(shape)));
      }
    }

    // Always update transient data to allow downstream ops to resolve shapes.
    if (auto it = result.propagated_data.find(i);
        it != result.propagated_data.end()) {
      transient_data_[&tensor] = it->second;
      // If we are actually modifying the model, also update the tensor weights.
      if (!validation_only) {
        auto& buf = it->second;
        OwningBufferRef<uint8_t> new_buf(buf.size());
        std::memcpy(new_buf.Data(), buf.data(), buf.size());
        SetWeightsFromOwnedBuffer(tensor.Weights(), std::move(new_buf));
      }
    }
  }

  return kLiteRtStatusOk;
}

LiteRtStatus ShapeInferenceEngine::SpecializeSubgraph(
    LiteRtSubgraphT* subgraph, absl::Span<Dims> input_shapes,
    LiteRtSubgraphT** new_subgraph) {
  if (!model_ || !subgraph || !new_subgraph) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shapes.size() != subgraph->Inputs().size()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  auto& dest = model_->EmplaceSubgraph();
  *new_subgraph = &dest;

  absl::flat_hash_map<LiteRtTensorT*, LiteRtTensorT*> tensor_map;
  for (auto* tensor : subgraph->Tensors()) {
    tensor_map[tensor] = &MakeClone(dest, *tensor);
  }

  for (auto* input : subgraph->Inputs()) {
    dest.Inputs().push_back(tensor_map[input]);
  }
  for (auto* output : subgraph->Outputs()) {
    dest.Outputs().push_back(tensor_map[output]);
  }

  for (auto* op : subgraph->Ops()) {
    auto& new_op = MakeClone(dest, *op);
    for (auto* input : op->Inputs()) {
      AttachInput(tensor_map[input], new_op);
    }
    for (auto* output : op->Outputs()) {
      AttachOutput(tensor_map[output], new_op);
    }
  }

  for (size_t i = 0; i < input_shapes.size(); ++i) {
    auto& input_tensor = dest.Input(i);
    LiteRtElementType element_type = kLiteRtElementTypeNone;
    if (input_tensor.Type().first == kLiteRtRankedTensorType) {
      element_type = input_tensor.Type().second.ranked_tensor_type.element_type;
    } else if (input_tensor.Type().first == kLiteRtUnrankedTensorType) {
      element_type =
          input_tensor.Type().second.unranked_tensor_type.element_type;
    }

    if (element_type != kLiteRtElementTypeNone) {
      input_tensor.SetType(
          MakeRankedTensorType(element_type, absl::MakeSpan(input_shapes[i])));
    }
  }

  return InferSubgraphShapes(&dest);
}

LiteRtStatus ShapeInferenceEngine::InferOpShapes(
    const ShapeInferenceContext& ctx, InferenceResult& result) {
  auto it = registry_.find(ctx.GetOpCode());
  if (it == registry_.end()) {
    return kLiteRtStatusErrorUnsupportedOpShapeInferer;
  }
  return it->second(ctx, result);
}

}  // namespace litert::internal
