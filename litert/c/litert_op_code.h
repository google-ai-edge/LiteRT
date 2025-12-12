// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_LITERT_OP_CODE_H_
#define ODML_LITERT_LITERT_C_LITERT_OP_CODE_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  kLiteRtOpCodeTflAdd = 0,               // kTfLiteBuiltinAdd
  kLiteRtOpCodeTflAveragePool2d = 1,     // kTfLiteBuiltinAveragePool2d
  kLiteRtOpCodeTflConcatenation = 2,     // kTfLiteBuiltinConcatenation
  kLiteRtOpCodeTflConv2d = 3,            // kTfLiteBuiltinConv2d
  kLiteRtOpCodeTflDepthwiseConv2d = 4,   // kTfLiteBuiltinDepthwiseConv2d
  kLiteRtOpCodeTflDepthToSpace = 5,      // kTfLiteBuiltinDepthToSpace
  kLiteRtOpCodeTflDequantize = 6,        // kTfLiteBuiltinDequantize
  kLiteRtOpCodeTflEmbeddingLookup = 7,   // kTfLiteBuiltinEmbeddingLookup
  kLiteRtOpCodeTflFloor = 8,             // kTfLiteBuiltinFloor
  kLiteRtOpCodeTflFullyConnected = 9,    // kTfLiteBuiltinFullyConnected
  kLiteRtOpCodeTflHashtableLookup = 10,  // kTfLiteBuiltinHashtableLookup
  kLiteRtOpCodeTflL2Normalization = 11,  // kTfLiteBuiltinL2Normalization
  kLiteRtOpCodeTflL2Pool2d = 12,         // kTfLiteBuiltinL2Pool2d
  kLiteRtOpCodeTflLocalResponseNormalization =
      13,                         // kTfLiteBuiltinLocalResponseNormalization
  kLiteRtOpCodeTflLogistic = 14,  // kTfLiteBuiltinLogistic
  kLiteRtOpCodeTflLshProjection = 15,     // kTfLiteBuiltinLshProjection
  kLiteRtOpCodeTflLstm = 16,              // kTfLiteBuiltinLstm
  kLiteRtOpCodeTflMaxPool2d = 17,         // kTfLiteBuiltinMaxPool2d
  kLiteRtOpCodeTflMul = 18,               // kTfLiteBuiltinMul
  kLiteRtOpCodeTflRelu = 19,              // kTfLiteBuiltinRelu
  kLiteRtOpCodeTflReluN1To1 = 20,         // kTfLiteBuiltinReluN1To1
  kLiteRtOpCodeTflRelu6 = 21,             // kTfLiteBuiltinRelu6
  kLiteRtOpCodeTflReshape = 22,           // kTfLiteBuiltinReshape
  kLiteRtOpCodeTflResizeBilinear = 23,    // kTfLiteBuiltinResizeBilinear
  kLiteRtOpCodeTflRnn = 24,               // kTfLiteBuiltinRnn
  kLiteRtOpCodeTflSoftmax = 25,           // kTfLiteBuiltinSoftmax
  kLiteRtOpCodeTflSpaceToDepth = 26,      // kTfLiteBuiltinSpaceToDepth
  kLiteRtOpCodeTflSvdf = 27,              // kTfLiteBuiltinSvdf
  kLiteRtOpCodeTflTanh = 28,              // kTfLiteBuiltinTanh
  kLiteRtOpCodeTflConcatEmbeddings = 29,  // kTfLiteBuiltinConcatEmbeddings
  kLiteRtOpCodeTflSkipGram = 30,          // kTfLiteBuiltinSkipGram
  kLiteRtOpCodeTflCall = 31,              // kTfLiteBuiltinCall
  kLiteRtOpCodeTflCustom = 32,            // kTfLiteBuiltinCustom
  kLiteRtOpCodeTflEmbeddingLookupSparse =
      33,                    // kTfLiteBuiltinEmbeddingLookupSparse
  kLiteRtOpCodeTflPad = 34,  // kTfLiteBuiltinPad
  kLiteRtOpCodeTflUnidirectionalSequenceRnn =
      35,                       // kTfLiteBuiltinUnidirectionalSequenceRnn
  kLiteRtOpCodeTflGather = 36,  // kTfLiteBuiltinGather
  kLiteRtOpCodeTflBatchToSpaceNd = 37,  // kTfLiteBuiltinBatchToSpaceNd
  kLiteRtOpCodeTflSpaceToBatchNd = 38,  // kTfLiteBuiltinSpaceToBatchNd
  kLiteRtOpCodeTflTranspose = 39,       // kTfLiteBuiltinTranspose
  kLiteRtOpCodeTflMean = 40,            // kTfLiteBuiltinMean
  kLiteRtOpCodeTflSub = 41,             // kTfLiteBuiltinSub
  kLiteRtOpCodeTflDiv = 42,             // kTfLiteBuiltinDiv
  kLiteRtOpCodeTflSqueeze = 43,         // kTfLiteBuiltinSqueeze
  kLiteRtOpCodeTflUnidirectionalSequenceLstm =
      44,  // kTfLiteBuiltinUnidirectionalSequenceLstm
  kLiteRtOpCodeTflStridedSlice = 45,  // kTfLiteBuiltinStridedSlice
  kLiteRtOpCodeTflBidirectionalSequenceRnn =
      46,                           // kTfLiteBuiltinBidirectionalSequenceRnn
  kLiteRtOpCodeTflExp = 47,         // kTfLiteBuiltinExp
  kLiteRtOpCodeTflTopkV2 = 48,      // kTfLiteBuiltinTopkV2
  kLiteRtOpCodeTflSplit = 49,       // kTfLiteBuiltinSplit
  kLiteRtOpCodeTflLogSoftmax = 50,  // kTfLiteBuiltinLogSoftmax
  kLiteRtOpCodeTflDelegate = 51,    // kTfLiteBuiltinDelegate
  kLiteRtOpCodeTflBidirectionalSequenceLstm =
      52,                             // kTfLiteBuiltinBidirectionalSequenceLstm
  kLiteRtOpCodeTflCast = 53,          // kTfLiteBuiltinCast
  kLiteRtOpCodeTflPrelu = 54,         // kTfLiteBuiltinPrelu
  kLiteRtOpCodeTflMaximum = 55,       // kTfLiteBuiltinMaximum
  kLiteRtOpCodeTflArgMax = 56,        // kTfLiteBuiltinArgMax
  kLiteRtOpCodeTflMinimum = 57,       // kTfLiteBuiltinMinimum
  kLiteRtOpCodeTflLess = 58,          // kTfLiteBuiltinLess
  kLiteRtOpCodeTflNeg = 59,           // kTfLiteBuiltinNeg
  kLiteRtOpCodeTflPadv2 = 60,         // kTfLiteBuiltinPadv2
  kLiteRtOpCodeTflGreater = 61,       // kTfLiteBuiltinGreater
  kLiteRtOpCodeTflGreaterEqual = 62,  // kTfLiteBuiltinGreaterEqual
  kLiteRtOpCodeTflLessEqual = 63,     // kTfLiteBuiltinLessEqual
  kLiteRtOpCodeTflSelect = 64,        // kTfLiteBuiltinSelect
  kLiteRtOpCodeTflSlice = 65,         // kTfLiteBuiltinSlice
  kLiteRtOpCodeTflSin = 66,           // kTfLiteBuiltinSin
  kLiteRtOpCodeTflTransposeConv = 67,  // kTfLiteBuiltinTransposeConv
  kLiteRtOpCodeTflSparseToDense = 68,  // kTfLiteBuiltinSparseToDense
  kLiteRtOpCodeTflTile = 69,           // kTfLiteBuiltinTile
  kLiteRtOpCodeTflExpandDims = 70,     // kTfLiteBuiltinExpandDims
  kLiteRtOpCodeTflEqual = 71,          // kTfLiteBuiltinEqual
  kLiteRtOpCodeTflNotEqual = 72,       // kTfLiteBuiltinNotEqual
  kLiteRtOpCodeTflLog = 73,            // kTfLiteBuiltinLog
  kLiteRtOpCodeTflSum = 74,            // kTfLiteBuiltinSum
  kLiteRtOpCodeTflSqrt = 75,           // kTfLiteBuiltinSqrt
  kLiteRtOpCodeTflRsqrt = 76,          // kTfLiteBuiltinRsqrt
  kLiteRtOpCodeTflShape = 77,          // kTfLiteBuiltinShape
  kLiteRtOpCodeTflPow = 78,            // kTfLiteBuiltinPow
  kLiteRtOpCodeTflArgMin = 79,         // kTfLiteBuiltinArgMin
  kLiteRtOpCodeTflFakeQuant = 80,      // kTfLiteBuiltinFakeQuant
  kLiteRtOpCodeTflReduceProd = 81,     // kTfLiteBuiltinReduceProd
  kLiteRtOpCodeTflReduceMax = 82,      // kTfLiteBuiltinReduceMax
  kLiteRtOpCodeTflPack = 83,           // kTfLiteBuiltinPack
  kLiteRtOpCodeTflLogicalOr = 84,      // kTfLiteBuiltinLogicalOr
  kLiteRtOpCodeTflOneHot = 85,         // kTfLiteBuiltinOneHot
  kLiteRtOpCodeTflLogicalAnd = 86,     // kTfLiteBuiltinLogicalAnd
  kLiteRtOpCodeTflLogicalNot = 87,     // kTfLiteBuiltinLogicalNot
  kLiteRtOpCodeTflUnpack = 88,         // kTfLiteBuiltinUnpack
  kLiteRtOpCodeTflReduceMin = 89,      // kTfLiteBuiltinReduceMin
  kLiteRtOpCodeTflFloorDiv = 90,       // kTfLiteBuiltinFloorDiv
  kLiteRtOpCodeTflReduceAny = 91,      // kTfLiteBuiltinReduceAny
  kLiteRtOpCodeTflSquare = 92,         // kTfLiteBuiltinSquare
  kLiteRtOpCodeTflZerosLike = 93,      // kTfLiteBuiltinZerosLike
  kLiteRtOpCodeTflFill = 94,           // kTfLiteBuiltinFill
  kLiteRtOpCodeTflFloorMod = 95,       // kTfLiteBuiltinFloorMod
  kLiteRtOpCodeTflRange = 96,          // kTfLiteBuiltinRange
  kLiteRtOpCodeTflResizeNearestNeighbor =
      97,                          // kTfLiteBuiltinResizeNearestNeighbor
  kLiteRtOpCodeTflLeakyRelu = 98,  // kTfLiteBuiltinLeakyRelu
  kLiteRtOpCodeTflSquaredDifference = 99,  // kTfLiteBuiltinSquaredDifference
  kLiteRtOpCodeTflMirrorPad = 100,         // kTfLiteBuiltinMirrorPad
  kLiteRtOpCodeTflAbs = 101,               // kTfLiteBuiltinAbs
  kLiteRtOpCodeTflSplitV = 102,            // kTfLiteBuiltinSplitV
  kLiteRtOpCodeTflUnique = 103,            // kTfLiteBuiltinUnique
  kLiteRtOpCodeTflCeil = 104,              // kTfLiteBuiltinCeil
  kLiteRtOpCodeTflReverseV2 = 105,         // kTfLiteBuiltinReverseV2
  kLiteRtOpCodeTflAddN = 106,              // kTfLiteBuiltinAddN
  kLiteRtOpCodeTflGatherNd = 107,          // kTfLiteBuiltinGatherNd
  kLiteRtOpCodeTflCos = 108,               // kTfLiteBuiltinCos
  kLiteRtOpCodeTflWhere = 109,             // kTfLiteBuiltinWhere
  kLiteRtOpCodeTflRank = 110,              // kTfLiteBuiltinRank
  kLiteRtOpCodeTflElu = 111,               // kTfLiteBuiltinElu
  kLiteRtOpCodeTflReverseSequence = 112,   // kTfLiteBuiltinReverseSequence
  kLiteRtOpCodeTflMatrixDiag = 113,        // kTfLiteBuiltinMatrixDiag
  kLiteRtOpCodeTflQuantize = 114,          // kTfLiteBuiltinQuantize
  kLiteRtOpCodeTflMatrixSetDiag = 115,     // kTfLiteBuiltinMatrixSetDiag
  kLiteRtOpCodeTflRound = 116,             // kTfLiteBuiltinRound
  kLiteRtOpCodeTflHardSwish = 117,         // kTfLiteBuiltinHardSwish
  kLiteRtOpCodeTflIf = 118,                // kTfLiteBuiltinIf
  kLiteRtOpCodeTflWhile = 119,             // kTfLiteBuiltinWhile
  kLiteRtOpCodeTflNonMaxSuppressionV4 =
      120,  // kTfLiteBuiltinNonMaxSuppressionV4
  kLiteRtOpCodeTflNonMaxSuppressionV5 =
      121,                            // kTfLiteBuiltinNonMaxSuppressionV5
  kLiteRtOpCodeTflScatterNd = 122,    // kTfLiteBuiltinScatterNd
  kLiteRtOpCodeTflSelectV2 = 123,     // kTfLiteBuiltinSelectV2
  kLiteRtOpCodeTflDensify = 124,      // kTfLiteBuiltinDensify
  kLiteRtOpCodeTflSegmentSum = 125,   // kTfLiteBuiltinSegmentSum
  kLiteRtOpCodeTflBatchMatmul = 126,  // kTfLiteBuiltinBatchMatmul
  kLiteRtOpCodeTflPlaceholderForGreaterOpCodeTfls =
      127,                         // kTfLiteBuiltinPlaceholderForGreaterOpCodes
  kLiteRtOpCodeTflCumsum = 128,    // kTfLiteBuiltinCumsum
  kLiteRtOpCodeTflCallOnce = 129,  // kTfLiteBuiltinCallOnce
  kLiteRtOpCodeTflBroadcastTo = 130,      // kTfLiteBuiltinBroadcastTo
  kLiteRtOpCodeTflRfft2d = 131,           // kTfLiteBuiltinRfft2d
  kLiteRtOpCodeTflConv3d = 132,           // kTfLiteBuiltinConv3d
  kLiteRtOpCodeTflImag = 133,             // kTfLiteBuiltinImag
  kLiteRtOpCodeTflReal = 134,             // kTfLiteBuiltinReal
  kLiteRtOpCodeTflComplexAbs = 135,       // kTfLiteBuiltinComplexAbs
  kLiteRtOpCodeTflHashtable = 136,        // kTfLiteBuiltinHashtable
  kLiteRtOpCodeTflHashtableFind = 137,    // kTfLiteBuiltinHashtableFind
  kLiteRtOpCodeTflHashtableImport = 138,  // kTfLiteBuiltinHashtableImport
  kLiteRtOpCodeTflHashtableSize = 139,    // kTfLiteBuiltinHashtableSize
  kLiteRtOpCodeTflReduceAll = 140,        // kTfLiteBuiltinReduceAll
  kLiteRtOpCodeTflConv3dTranspose = 141,  // kTfLiteBuiltinConv3dTranspose
  kLiteRtOpCodeTflVarHandle = 142,        // kTfLiteBuiltinVarHandle
  kLiteRtOpCodeTflReadVariable = 143,     // kTfLiteBuiltinReadVariable
  kLiteRtOpCodeTflAssignVariable = 144,   // kTfLiteBuiltinAssignVariable
  kLiteRtOpCodeTflBroadcastArgs = 145,    // kTfLiteBuiltinBroadcastArgs
  kLiteRtOpCodeTflRandomStandardNormal =
      146,                              // kTfLiteBuiltinRandomStandardNormal
  kLiteRtOpCodeTflBucketize = 147,      // kTfLiteBuiltinBucketize
  kLiteRtOpCodeTflRandomUniform = 148,  // kTfLiteBuiltinRandomUniform
  kLiteRtOpCodeTflMultinomial = 149,    // kTfLiteBuiltinMultinomial
  kLiteRtOpCodeTflGelu = 150,           // kTfLiteBuiltinGelu
  kLiteRtOpCodeTflDynamicUpdateSlice = 151,  // kTfLiteBuiltinDynamicUpdateSlice
  kLiteRtOpCodeTflRelu0To1 = 152,            // kTfLiteBuiltinRelu0To1
  kLiteRtOpCodeTflUnsortedSegmentProd =
      153,  // kTfLiteBuiltinUnsortedSegmentProd
  kLiteRtOpCodeTflUnsortedSegmentMax = 154,  // kTfLiteBuiltinUnsortedSegmentMax
  kLiteRtOpCodeTflUnsortedSegmentSum = 155,  // kTfLiteBuiltinUnsortedSegmentSum
  kLiteRtOpCodeTflAtan2 = 156,               // kTfLiteBuiltinAtan2
  kLiteRtOpCodeTflUnsortedSegmentMin = 157,  // kTfLiteBuiltinUnsortedSegmentMin
  kLiteRtOpCodeTflSign = 158,                // kTfLiteBuiltinSign
  kLiteRtOpCodeTflBitcast = 159,             // kTfLiteBuiltinBitcast
  kLiteRtOpCodeTflBitwiseXor = 160,          // kTfLiteBuiltinBitwiseXor
  kLiteRtOpCodeTflRightShift = 161,          // kTfLiteBuiltinRightShift
  kLiteRtOpCodeShloLogistic = 162,           // kTfLiteBuiltinStablehloLogistic
  kLiteRtOpCodeShloAdd = 163,                // kTfLiteBuiltinStablehloAdd
  kLiteRtOpCodeShloDivide = 164,             // kTfLiteBuiltinStablehloDivide
  kLiteRtOpCodeShloMultiply = 165,           // kTfLiteBuiltinStablehloMultiply
  kLiteRtOpCodeShloMaximum = 166,            // kTfLiteBuiltinStablehloMaximum
  kLiteRtOpCodeShloReshape = 167,            // kTfLiteBuiltinStablehloReshape
  kLiteRtOpCodeShloClamp = 168,              // kTfLiteBuiltinStablehloClamp
  kLiteRtOpCodeShloConcatenate = 169,  // kTfLiteBuiltinStablehloConcatenate
  kLiteRtOpCodeShloBroadcastInDim =
      170,                              // kTfLiteBuiltinStablehloBroadcastInDim
  kLiteRtOpCodeShloConvolution = 171,   // kTfLiteBuiltinStablehloConvolution
  kLiteRtOpCodeShloSlice = 172,         // kTfLiteBuiltinStablehloSlice
  kLiteRtOpCodeShloCustomCall = 173,    // kTfLiteBuiltinStablehloCustomCall
  kLiteRtOpCodeShloReduce = 174,        // kTfLiteBuiltinStablehloReduce
  kLiteRtOpCodeShloAbs = 175,           // kTfLiteBuiltinStablehloAbs
  kLiteRtOpCodeShloAnd = 176,           // kTfLiteBuiltinStablehloAnd
  kLiteRtOpCodeShloCosine = 177,        // kTfLiteBuiltinStablehloCosine
  kLiteRtOpCodeShloExponential = 178,   // kTfLiteBuiltinStablehloExponential
  kLiteRtOpCodeShloFloor = 179,         // kTfLiteBuiltinStablehloFloor
  kLiteRtOpCodeShloLog = 180,           // kTfLiteBuiltinStablehloLog
  kLiteRtOpCodeShloMinimum = 181,       // kTfLiteBuiltinStablehloMinimum
  kLiteRtOpCodeShloNegate = 182,        // kTfLiteBuiltinStablehloNegate
  kLiteRtOpCodeShloOr = 183,            // kTfLiteBuiltinStablehloOr
  kLiteRtOpCodeShloPower = 184,         // kTfLiteBuiltinStablehloPower
  kLiteRtOpCodeShloRemainder = 185,     // kTfLiteBuiltinStablehloRemainder
  kLiteRtOpCodeShloRsqrt = 186,         // kTfLiteBuiltinStablehloRsqrt
  kLiteRtOpCodeShloSelect = 187,        // kTfLiteBuiltinStablehloSelect
  kLiteRtOpCodeShloSubtract = 188,      // kTfLiteBuiltinStablehloSubtract
  kLiteRtOpCodeShloTanh = 189,          // kTfLiteBuiltinStablehloTanh
  kLiteRtOpCodeShloScatter = 190,       // kTfLiteBuiltinStablehloScatter
  kLiteRtOpCodeShloCompare = 191,       // kTfLiteBuiltinStablehloCompare
  kLiteRtOpCodeShloConvert = 192,       // kTfLiteBuiltinStablehloConvert
  kLiteRtOpCodeShloDynamicSlice = 193,  // kTfLiteBuiltinStablehloDynamicSlice
  kLiteRtOpCodeShloDynamicUpdateSlice =
      194,                         // kTfLiteBuiltinStablehloDynamicUpdateSlice
  kLiteRtOpCodeShloPad = 195,      // kTfLiteBuiltinStablehloPad
  kLiteRtOpCodeShloIota = 196,     // kTfLiteBuiltinStablehloIota
  kLiteRtOpCodeShloGeneral = 197,  // kTfLiteBuiltinStablehloDotGeneral
  kLiteRtOpCodeShloWindow = 198,   // kTfLiteBuiltinStablehloReduceWindow
  kLiteRtOpCodeShloSort = 199,     // kTfLiteBuiltinStablehloSort
  kLiteRtOpCodeShloWhile = 200,    // kTfLiteBuiltinStablehloWhile
  kLiteRtOpCodeShloGather = 201,   // kTfLiteBuiltinStablehloGather
  kLiteRtOpCodeShloTranspose = 202,  // kTfLiteBuiltinStablehloTranspose
  kLiteRtOpCodeTflDilate = 203,      // kTfLiteBuiltinDilate
  kLiteRtOpCodeShloRngBitGenerator =
      204,                             // kTfLiteBuiltinStablehloRngBitGenerator
  kLiteRtOpCodeTflReduceWindow = 205,  // kTfLiteBuiltinReduceWindow
  kLiteRtOpCodeShloComposite = 206,    // kTfLiteBuiltinStablehloComposite
} LiteRtOpCode;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_OP_CODE_H_
