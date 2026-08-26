// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/arm/capabilities.h"

#include <set>

namespace litert::arm {
namespace {

// Operations currently accepted by the TOSA legalization flow.
const std::set<LiteRtOpCode> kSupportedOps = {
    kLiteRtOpCodeTflAbs,
    kLiteRtOpCodeTflCeil,
    kLiteRtOpCodeTflFloor,
    kLiteRtOpCodeTflExp,
    kLiteRtOpCodeTflLog,
    kLiteRtOpCodeTflRsqrt,
    kLiteRtOpCodeTflLogicalNot,
    kLiteRtOpCodeTflCast,
    kLiteRtOpCodeTflLogicalAnd,
    kLiteRtOpCodeTflLogicalOr,
    kLiteRtOpCodeTflBitwiseXor,
    kLiteRtOpCodeTflPow,
    kLiteRtOpCodeTflGelu,
    kLiteRtOpCodeTflRelu,
    kLiteRtOpCodeTflReluN1To1,
    kLiteRtOpCodeTflRelu0To1,
    kLiteRtOpCodeTflRelu6,
    kLiteRtOpCodeTflEqual,
    kLiteRtOpCodeTflNotEqual,
    kLiteRtOpCodeTflGreater,
    kLiteRtOpCodeTflGreaterEqual,
    kLiteRtOpCodeTflAdd,
    kLiteRtOpCodeTflSub,
    kLiteRtOpCodeTflMul,
    kLiteRtOpCodeTflSquare,
    kLiteRtOpCodeTflSquaredDifference,
    kLiteRtOpCodeTflSign,
    kLiteRtOpCodeTflRound,
    kLiteRtOpCodeTflDiv,
    kLiteRtOpCodeTflMaximum,
    kLiteRtOpCodeTflMinimum,
    kLiteRtOpCodeTflFloorMod,
    kLiteRtOpCodeTflFloorDiv,
    kLiteRtOpCodeTflAddN,
    kLiteRtOpCodeTflAveragePool2d,
    kLiteRtOpCodeTflMaxPool2d,
    kLiteRtOpCodeTflConcatenation,
    kLiteRtOpCodeTflReshape,
    kLiteRtOpCodeTflRank,
    kLiteRtOpCodeTflShape,
    kLiteRtOpCodeTflExpandDims,
    kLiteRtOpCodeTflSqueeze,
    kLiteRtOpCodeTflFill,
    kLiteRtOpCodeTflElu,
    kLiteRtOpCodeTflSoftmax,
    kLiteRtOpCodeTflLogSoftmax,
    kLiteRtOpCodeTflSqrt,
    kLiteRtOpCodeTflL2Normalization,
    kLiteRtOpCodeTflReduceAll,
    kLiteRtOpCodeTflReduceAny,
    kLiteRtOpCodeTflReduceMax,
    kLiteRtOpCodeTflReduceMin,
    kLiteRtOpCodeTflMean,
    kLiteRtOpCodeTflReduceProd,
    kLiteRtOpCodeTflSum,
    kLiteRtOpCodeTflConv2d,
    kLiteRtOpCodeTflConv3d,
    kLiteRtOpCodeTflTransposeConv,
    kLiteRtOpCodeTflDepthwiseConv2d,
    kLiteRtOpCodeTflFullyConnected,
    kLiteRtOpCodeTflBatchMatmul,
    kLiteRtOpCodeTflSplit,
    kLiteRtOpCodeTflSplitV,
    kLiteRtOpCodeTflPack,
    kLiteRtOpCodeTflUnpack,
    kLiteRtOpCodeTflTranspose,
    kLiteRtOpCodeTflTile,
    kLiteRtOpCodeTflSlice,
    kLiteRtOpCodeTflStridedSlice,
    kLiteRtOpCodeTflHardSwish,
    kLiteRtOpCodeTflZerosLike,
    kLiteRtOpCodeTflLess,
    kLiteRtOpCodeTflLessEqual,
    kLiteRtOpCodeTflPad,
    kLiteRtOpCodeTflMirrorPad,
    kLiteRtOpCodeTflPadv2,
    kLiteRtOpCodeTflResizeBilinear,
    kLiteRtOpCodeTflResizeNearestNeighbor,
    kLiteRtOpCodeTflSelect,
    kLiteRtOpCodeTflSelectV2,
    kLiteRtOpCodeTflSpaceToBatchNd,
    kLiteRtOpCodeTflBatchToSpaceNd,
    kLiteRtOpCodeTflSpaceToDepth,
    kLiteRtOpCodeTflDepthToSpace,
    kLiteRtOpCodeTflBucketize,
    kLiteRtOpCodeTflSin,
    kLiteRtOpCodeTflCos,
    kLiteRtOpCodeTflAtan2,
    kLiteRtOpCodeTflLogistic,
    kLiteRtOpCodeTflTanh,
    kLiteRtOpCodeTflPrelu,
    kLiteRtOpCodeTflLeakyRelu,
    kLiteRtOpCodeTflNeg,
    kLiteRtOpCodeTflReverseV2,
    kLiteRtOpCodeTflQuantize,
    kLiteRtOpCodeTflDequantize,
    kLiteRtOpCodeTflGather,
    kLiteRtOpCodeTflGatherNd,
    kLiteRtOpCodeTflScatterNd,
    kLiteRtOpCodeTflSparseToDense,
    kLiteRtOpCodeTflOneHot,
    kLiteRtOpCodeTflArgMax,
    kLiteRtOpCodeTflArgMin,
    kLiteRtOpCodeTflFakeQuant,
    kLiteRtOpCodeTflWhile,
    kLiteRtOpCodeTflReal,
    kLiteRtOpCodeTflImag,
    kLiteRtOpCodeTflRfft2d,
    kLiteRtOpCodeTflBroadcastTo,
};

}  // namespace

bool IsSupportedOpCode(LiteRtOpCode op_code) {
  return kSupportedOps.find(op_code) != kSupportedOps.end();
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
