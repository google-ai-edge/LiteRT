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
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
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
#include "litert/core/model/ops/simple_binary.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/core/model/ops/slice.h"
#include "litert/core/model/ops/spatial.h"
#include "litert/core/model/ops/topk.h"
#include "litert/core/model/ops/transpose.h"
#include "litert/core/model/ops/unpack.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

ShapeInferenceEngine::ShapeInferenceEngine(LiteRtModelT* model)
    : model_(model) {
  RegisterStandardOps();
}

ShapeInferenceEngine::ShapeInferenceEngine() { RegisterStandardOps(); }

void ShapeInferenceEngine::RegisterStandardOps() {
  RegisterInferrer(kLiteRtOpCodeTflAbs, InferAbs);
  RegisterInferrer(kLiteRtOpCodeTflCeil, InferCeil);
  RegisterInferrer(kLiteRtOpCodeTflCos, InferCos);
  RegisterInferrer(kLiteRtOpCodeTflDequantize, InferDequantize);
  RegisterInferrer(kLiteRtOpCodeTflElu, InferElu);
  RegisterInferrer(kLiteRtOpCodeTflExp, InferExp);
  RegisterInferrer(kLiteRtOpCodeTflFloor, InferFloor);
  RegisterInferrer(kLiteRtOpCodeTflGelu, InferGelu);
  RegisterInferrer(kLiteRtOpCodeTflHardSwish, InferHardSwish);
  RegisterInferrer(kLiteRtOpCodeTflLeakyRelu, InferLeakyRelu);
  RegisterInferrer(kLiteRtOpCodeTflLog, InferLog);
  RegisterInferrer(kLiteRtOpCodeTflLogicalNot, InferLogicalNot);
  RegisterInferrer(kLiteRtOpCodeTflLogistic, InferLogistic);
  RegisterInferrer(kLiteRtOpCodeTflNeg, InferNeg);
  RegisterInferrer(kLiteRtOpCodeTflQuantize, InferQuantize);
  RegisterInferrer(kLiteRtOpCodeTflRelu, InferRelu);
  RegisterInferrer(kLiteRtOpCodeTflRelu0To1, InferRelu0To1);
  RegisterInferrer(kLiteRtOpCodeTflRelu6, InferRelu6);
  RegisterInferrer(kLiteRtOpCodeTflReluN1To1, InferReluN1To1);
  RegisterInferrer(kLiteRtOpCodeTflRound, InferRound);
  RegisterInferrer(kLiteRtOpCodeTflRsqrt, InferRsqrt);
  RegisterInferrer(kLiteRtOpCodeTflSign, InferSign);
  RegisterInferrer(kLiteRtOpCodeTflSin, InferSin);
  RegisterInferrer(kLiteRtOpCodeTflSoftmax, InferSoftmax);
  RegisterInferrer(kLiteRtOpCodeTflSqrt, InferSqrt);
  RegisterInferrer(kLiteRtOpCodeTflSquare, InferSquare);
  RegisterInferrer(kLiteRtOpCodeTflTanh, InferTanh);
  RegisterInferrer(kLiteRtOpCodeTflEqual, InferEqual);
  RegisterInferrer(kLiteRtOpCodeTflFloorDiv, InferFloorDiv);
  RegisterInferrer(kLiteRtOpCodeTflGreater, InferGreater);
  RegisterInferrer(kLiteRtOpCodeTflGreaterEqual, InferGreaterEqual);
  RegisterInferrer(kLiteRtOpCodeTflLess, InferLess);
  RegisterInferrer(kLiteRtOpCodeTflLessEqual, InferLessEqual);
  RegisterInferrer(kLiteRtOpCodeTflLogicalAnd, InferLogicalAnd);
  RegisterInferrer(kLiteRtOpCodeTflLogicalOr, InferLogicalOr);
  RegisterInferrer(kLiteRtOpCodeTflMaximum, InferMaximum);
  RegisterInferrer(kLiteRtOpCodeTflMinimum, InferMinimum);
  RegisterInferrer(kLiteRtOpCodeTflNotEqual, InferNotEqual);
  RegisterInferrer(kLiteRtOpCodeTflPow, InferPow);
  RegisterInferrer(kLiteRtOpCodeTflPrelu, InferPrelu);
  RegisterInferrer(kLiteRtOpCodeTflSquaredDifference, InferSquaredDifference);
  RegisterInferrer(kLiteRtOpCodeTflAdd, InferAdd);
  RegisterInferrer(kLiteRtOpCodeTflArgMax, InferArgMinMax);
  RegisterInferrer(kLiteRtOpCodeTflArgMin, InferArgMinMax);
  RegisterInferrer(kLiteRtOpCodeTflAveragePool2d, InferPool2D);
  RegisterInferrer(kLiteRtOpCodeTflBatchMatmul, InferBatchMatmul);
  RegisterInferrer(kLiteRtOpCodeTflBroadcastTo, InferBroadcastTo);
  RegisterInferrer(kLiteRtOpCodeTflCast, InferCast);
  RegisterInferrer(kLiteRtOpCodeTflConcatenation, InferConcatenation);
  RegisterInferrer(kLiteRtOpCodeTflConv2d, InferConv2D);
  RegisterInferrer(kLiteRtOpCodeTflConv3d, InferConv3D);
  RegisterInferrer(kLiteRtOpCodeTflConv3dTranspose, InferConv3DTranspose);
  RegisterInferrer(kLiteRtOpCodeTflDepthToSpace, InferDepthToSpace);
  RegisterInferrer(kLiteRtOpCodeTflDepthwiseConv2d, InferDepthwiseConv2D);
  RegisterInferrer(kLiteRtOpCodeTflDiv, InferDiv);
  RegisterInferrer(kLiteRtOpCodeTflDynamicUpdateSlice, InferDynamicUpdateSlice);
  RegisterInferrer(kLiteRtOpCodeTflEmbeddingLookup, InferEmbeddingLookup);
  RegisterInferrer(kLiteRtOpCodeTflFullyConnected, InferFullyConnected);
  RegisterInferrer(kLiteRtOpCodeTflGather, InferGather);
  RegisterInferrer(kLiteRtOpCodeTflGatherNd, InferGatherNd);
  RegisterInferrer(kLiteRtOpCodeTflL2Pool2d, InferPool2D);
  RegisterInferrer(kLiteRtOpCodeTflMaxPool2d, InferPool2D);
  RegisterInferrer(kLiteRtOpCodeTflMean, InferReduce);
  RegisterInferrer(kLiteRtOpCodeTflMirrorPad, InferMirrorPad);
  RegisterInferrer(kLiteRtOpCodeTflMul, InferMul);
  RegisterInferrer(kLiteRtOpCodeTflPack, InferPack);
  RegisterInferrer(kLiteRtOpCodeTflPad, InferPad);
  RegisterInferrer(kLiteRtOpCodeTflPadv2, InferPadv2);
  RegisterInferrer(kLiteRtOpCodeTflReduceAll, InferReduce);
  RegisterInferrer(kLiteRtOpCodeTflReduceAny, InferReduce);
  RegisterInferrer(kLiteRtOpCodeTflReduceMax, InferReduce);
  RegisterInferrer(kLiteRtOpCodeTflReduceMin, InferReduce);
  RegisterInferrer(kLiteRtOpCodeTflSum, InferReduce);
  RegisterInferrer(kLiteRtOpCodeTflReshape, InferReshape);
  RegisterInferrer(kLiteRtOpCodeTflResizeBilinear, InferResizeBilinear);
  RegisterInferrer(kLiteRtOpCodeTflResizeNearestNeighbor,
                   InferResizeNearestNeighbor);
  RegisterInferrer(kLiteRtOpCodeTflSelect, InferSelect);
  RegisterInferrer(kLiteRtOpCodeTflSelectV2, InferSelect);
  RegisterInferrer(kLiteRtOpCodeTflSlice, InferSlice);
  RegisterInferrer(kLiteRtOpCodeTflSpaceToDepth, InferSpaceToDepth);
  RegisterInferrer(kLiteRtOpCodeTflSub, InferSub);
  RegisterInferrer(kLiteRtOpCodeTflTranspose, InferTranspose);
  RegisterInferrer(kLiteRtOpCodeTflTransposeConv, InferTransposeConv);
  RegisterInferrer(kLiteRtOpCodeTflUnpack, InferUnpack);
  RegisterInferrer(kLiteRtOpCodeTflCumsum, InferCumsum);
  RegisterInferrer(kLiteRtOpCodeTflL2Normalization, InferL2Normalization);
  RegisterInferrer(kLiteRtOpCodeTflReverseV2, InferReverseV2);
  RegisterInferrer(kLiteRtOpCodeTflTopkV2, InferTopKV2);
}

void ShapeInferenceEngine::RegisterInferrer(LiteRtOpCode op_code,
                                            OpShapeInferrer inferrer) {
  registry_[op_code] = std::move(inferrer);
}

LiteRtStatus ShapeInferenceEngine::InferShapes(bool validation_only,
                                               LiteRtOp* failing_op) {
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
  std::vector<Dims> input_shapes;
  input_shapes.reserve(op->Inputs().size());

  for (auto& tensor : op->Inputs()) {
    if (tensor == nullptr) {
      input_shapes.push_back({});
      continue;
    }
    if (tensor->Type().first == kLiteRtRankedTensorType) {
      const auto& layout = tensor->Type().second.ranked_tensor_type.layout;
      Dims shape;
      shape.reserve(layout.rank);
      for (int i = 0; i < layout.rank; ++i) {
        shape.push_back(layout.dimensions[i]);
      }
      input_shapes.push_back(std::move(shape));
    } else {
      // Unranked or unknown.
      LITERT_LOG(LITERT_ERROR, "Unranked or unknown input tensor type for op.");
      return kLiteRtStatusErrorUnsupported;
    }
  }

  std::vector<Dims> output_shapes(op->NumOutputs());
  auto status = InferOpShapes(*op, absl::MakeSpan(input_shapes), output_shapes);
  if (status != kLiteRtStatusOk) {
    return status;
  }

  for (size_t i = 0; i < op->Outputs().size(); ++i) {
    if (i >= output_shapes.size()) break;
    if (op->Outputs()[i] == nullptr) continue;
    auto& tensor = op->Output(i);
    const auto& shape = output_shapes[i];

    if (validation_only) {
      if (tensor.Type().first == kLiteRtRankedTensorType) {
        const auto& existing_shape =
            tensor.Type().second.ranked_tensor_type.layout;
        if (existing_shape.rank != shape.size()) {
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
      continue;
    }
    // Ensure we are not changing the shape of a constant tensor.
    if (tensor.Weights().Buffer().Size() > 0) {
      if (tensor.Type().first == kLiteRtRankedTensorType) {
        const auto& existing_shape =
            tensor.Type().second.ranked_tensor_type.layout;
        if (existing_shape.rank != shape.size()) {
          LITERT_LOG(LITERT_ERROR,
                     "Rank mismatch for constant tensor after shape inference.")
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
        for (size_t d = 0; d < shape.size(); ++d) {
          if (existing_shape.dimensions[d] != shape[d]) {
            LITERT_LOG(LITERT_ERROR,
                       "Dimension mismatch for constant tensor after shape "
                       "inference.");
            return kLiteRtStatusErrorShapeInferenceFailed;
          }
        }
      }
    }

    // Preserve element type.
    LiteRtElementType element_type = kLiteRtElementTypeNone;
    if (tensor.Type().first == kLiteRtRankedTensorType) {
      element_type = tensor.Type().second.ranked_tensor_type.element_type;
    } else if (tensor.Type().first == kLiteRtUnrankedTensorType) {
      element_type = tensor.Type().second.unranked_tensor_type.element_type;
    }

    if (element_type != kLiteRtElementTypeNone) {
      tensor.SetType(MakeRankedTensorType(element_type, absl::MakeSpan(shape)));
    }
  }

  return kLiteRtStatusOk;
}

LiteRtStatus ShapeInferenceEngine::SpecializeSubgraph(
    LiteRtSubgraphT* subgraph, const std::vector<Dims>& input_shapes,
    LiteRtSubgraphT** new_subgraph) {
  if (!model_ || !subgraph || !new_subgraph) {
    LITERT_LOG(LITERT_ERROR, "Invalid arguments to SpecializeSubgraph.");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shapes.size() != subgraph->Inputs().size()) {
    LITERT_LOG(LITERT_ERROR,
               "Number of input shapes does not match number of inputs for "
               "subgraph.");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  auto& dest = model_->EmplaceSubgraph();
  *new_subgraph = &dest;

  absl::flat_hash_map<LiteRtTensorT*, LiteRtTensorT*> tensor_map;

  for (auto* tensor : subgraph->Tensors()) {
    auto& new_tensor = MakeClone(dest, *tensor);
    tensor_map[tensor] = &new_tensor;
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
    const LiteRtOpT& op, absl::Span<Dims> input_shapes,
    std::vector<Dims>& output_shapes) {
  auto it = registry_.find(op.OpCode());
  if (it == registry_.end()) {
    // No inferrer registered.
    return kLiteRtStatusErrorUnsupportedOpShapeInferer;
  }
  return it->second(op, input_shapes, output_shapes);
}

}  // namespace litert::internal
