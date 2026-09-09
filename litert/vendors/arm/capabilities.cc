// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/arm/capabilities.h"

#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"

namespace litert::arm {

// Operations currently accepted by the TOSA legalization flow.
bool IsSupportedOpCode(LiteRtOpCode op_code) {
  switch (op_code) {
    case kLiteRtOpCodeTflAbs:
    case kLiteRtOpCodeTflCeil:
    case kLiteRtOpCodeTflFloor:
    case kLiteRtOpCodeTflExp:
    case kLiteRtOpCodeTflLog:
    case kLiteRtOpCodeTflRsqrt:
    case kLiteRtOpCodeTflLogicalNot:
    case kLiteRtOpCodeTflCast:
    case kLiteRtOpCodeTflLogicalAnd:
    case kLiteRtOpCodeTflLogicalOr:
    case kLiteRtOpCodeTflBitwiseXor:
    case kLiteRtOpCodeTflPow:
    case kLiteRtOpCodeTflGelu:
    case kLiteRtOpCodeTflRelu:
    case kLiteRtOpCodeTflReluN1To1:
    case kLiteRtOpCodeTflRelu0To1:
    case kLiteRtOpCodeTflRelu6:
    case kLiteRtOpCodeTflEqual:
    case kLiteRtOpCodeTflNotEqual:
    case kLiteRtOpCodeTflGreater:
    case kLiteRtOpCodeTflGreaterEqual:
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflMul:
    case kLiteRtOpCodeTflSquare:
    case kLiteRtOpCodeTflSquaredDifference:
    case kLiteRtOpCodeTflSign:
    case kLiteRtOpCodeTflRound:
    case kLiteRtOpCodeTflDiv:
    case kLiteRtOpCodeTflMaximum:
    case kLiteRtOpCodeTflMinimum:
    case kLiteRtOpCodeTflFloorMod:
    case kLiteRtOpCodeTflFloorDiv:
    case kLiteRtOpCodeTflAddN:
    case kLiteRtOpCodeTflAveragePool2d:
    case kLiteRtOpCodeTflMaxPool2d:
    case kLiteRtOpCodeTflConcatenation:
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflRank:
    case kLiteRtOpCodeTflShape:
    case kLiteRtOpCodeTflExpandDims:
    case kLiteRtOpCodeTflSqueeze:
    case kLiteRtOpCodeTflFill:
    case kLiteRtOpCodeTflElu:
    case kLiteRtOpCodeTflSoftmax:
    case kLiteRtOpCodeTflLogSoftmax:
    case kLiteRtOpCodeTflSqrt:
    case kLiteRtOpCodeTflL2Normalization:
    case kLiteRtOpCodeTflReduceAll:
    case kLiteRtOpCodeTflReduceAny:
    case kLiteRtOpCodeTflReduceMax:
    case kLiteRtOpCodeTflReduceMin:
    case kLiteRtOpCodeTflMean:
    case kLiteRtOpCodeTflReduceProd:
    case kLiteRtOpCodeTflSum:
    case kLiteRtOpCodeTflConv2d:
    case kLiteRtOpCodeTflConv3d:
    case kLiteRtOpCodeTflTransposeConv:
    case kLiteRtOpCodeTflDepthwiseConv2d:
    case kLiteRtOpCodeTflFullyConnected:
    case kLiteRtOpCodeTflBatchMatmul:
    case kLiteRtOpCodeTflSplit:
    case kLiteRtOpCodeTflSplitV:
    case kLiteRtOpCodeTflPack:
    case kLiteRtOpCodeTflUnpack:
    case kLiteRtOpCodeTflTranspose:
    case kLiteRtOpCodeTflTile:
    case kLiteRtOpCodeTflSlice:
    case kLiteRtOpCodeTflStridedSlice:
    case kLiteRtOpCodeTflHardSwish:
    case kLiteRtOpCodeTflZerosLike:
    case kLiteRtOpCodeTflLess:
    case kLiteRtOpCodeTflLessEqual:
    case kLiteRtOpCodeTflPad:
    case kLiteRtOpCodeTflMirrorPad:
    case kLiteRtOpCodeTflPadv2:
    case kLiteRtOpCodeTflResizeBilinear:
    case kLiteRtOpCodeTflResizeNearestNeighbor:
    case kLiteRtOpCodeTflSelect:
    case kLiteRtOpCodeTflSelectV2:
    case kLiteRtOpCodeTflSpaceToBatchNd:
    case kLiteRtOpCodeTflBatchToSpaceNd:
    case kLiteRtOpCodeTflSpaceToDepth:
    case kLiteRtOpCodeTflDepthToSpace:
    case kLiteRtOpCodeTflBucketize:
    case kLiteRtOpCodeTflSin:
    case kLiteRtOpCodeTflCos:
    case kLiteRtOpCodeTflAtan2:
    case kLiteRtOpCodeTflLogistic:
    case kLiteRtOpCodeTflTanh:
    case kLiteRtOpCodeTflPrelu:
    case kLiteRtOpCodeTflLeakyRelu:
    case kLiteRtOpCodeTflNeg:
    case kLiteRtOpCodeTflReverseV2:
    case kLiteRtOpCodeTflQuantize:
    case kLiteRtOpCodeTflDequantize:
    case kLiteRtOpCodeTflGather:
    case kLiteRtOpCodeTflGatherNd:
    case kLiteRtOpCodeTflScatterNd:
    case kLiteRtOpCodeTflSparseToDense:
    case kLiteRtOpCodeTflOneHot:
    case kLiteRtOpCodeTflArgMax:
    case kLiteRtOpCodeTflArgMin:
    case kLiteRtOpCodeTflFakeQuant:
    case kLiteRtOpCodeTflWhile:
    case kLiteRtOpCodeTflReal:
    case kLiteRtOpCodeTflImag:
    case kLiteRtOpCodeTflRfft2d:
    case kLiteRtOpCodeTflBroadcastTo:
      return true;
    default:
      return false;
  }
}

bool IsSupportedType(ElementType type) {
  // TOSA PRO-INT and PRO-FLOAT profiles.
  return type == ElementType::Bool || type == ElementType::Int8 ||
         type == ElementType::UInt8 || type == ElementType::Int16 ||
         type == ElementType::UInt16 || type == ElementType::Int32 ||
         type == ElementType::UInt32 || type == ElementType::Float16 ||
         type == ElementType::Float32;
}

}  // namespace litert::arm
