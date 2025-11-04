// Copyright 2025 Google LLC.
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

#include "litert/vendors/cc/namespace_heuristics.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"

namespace litert {
namespace {

// Calculates the Levenshtein edit distance between two strings with an optional
// threshold for early exit. Returns -1 if the distance exceeds the threshold.
int EditDistance(absl::string_view str1, absl::string_view str2,
                 const int threshold = std::numeric_limits<int>::max()) {
  absl::string_view shorter_str = (str1.length() < str2.length()) ? str1 : str2;
  absl::string_view longer_str = (str1.length() < str2.length()) ? str2 : str1;
  const int short_len = shorter_str.length();
  const int long_len = longer_str.length();

  std::vector<int> prev_row(short_len + 1);
  std::vector<int> curr_row(short_len + 1);

  // Initialize the first row with values from 0 to short_len.
  for (int j = 0; j <= short_len; ++j) {
    prev_row[j] = j;
  }

  for (int i = 1; i <= long_len; ++i) {
    curr_row[0] = i;  // Represents deletion of all characters in the prefix.
    int min_in_row = i;

    for (int j = 1; j <= short_len; ++j) {
      // If characters are the same, cost is 0, otherwise 1.
      const int cost = (longer_str[i - 1] == shorter_str[j - 1]) ? 0 : 1;
      // The edit distance is the minimum of:
      // Deletion: prev_row[j] + 1
      // Insertion: curr_row[j-1] + 1
      // Substitution: prev_row[j-1] + cost
      curr_row[j] = std::min(
          {prev_row[j] + 1, curr_row[j - 1] + 1, prev_row[j - 1] + cost});
      min_in_row = std::min(min_in_row, curr_row[j]);
    }

    // Swaps rows for the next iteration.
    prev_row.swap(curr_row);

    // Early exits if the minimum distance in the current row already exceeds
    // the threshold.
    if (min_in_row > threshold) {
      return -1;
    }
  }

  return prev_row[short_len];
}

// Preprocesses a candidate name for matching. It extracts the final path
// component (after the last '/'), removes non-alphabetic characters, and
// converts the result to lowercase.
std::string PreprocessCandidateName(absl::string_view name) {
  // Finds the start of the last path component.
  const size_t last_slash_pos = name.find_last_of('/');
  if (last_slash_pos != absl::string_view::npos) {
    name.remove_prefix(last_slash_pos + 1);
  }

  std::string result;
  result.reserve(name.length());

  // Builds the result string by appending only desired characters.
  for (const char c : name) {
    if (absl::ascii_isalpha(c)) {
      result += absl::ascii_tolower(c);
    }
  }
  return result;
}

}  // namespace

std::string TfliteNodeNamespaceHeuristic(
    absl::string_view op_name, absl::Span<const std::string> candidate_names) {
  if (candidate_names.empty()) {
    return "";
  }
  if (candidate_names.size() == 1) {
    return candidate_names.front();
  }

  const std::string processed_op_name =
      absl::StrReplaceAll(op_name, {{"_", ""}});

  std::string best_match = candidate_names.front();
  int min_distance = std::numeric_limits<int>::max();

  // A threshold to prune the search space. If the edit distance is
  // significantly larger than the op name length, the candidate is likely
  // irrelevant. The multiplier is a heuristic.
  constexpr int kDistanceThresholdMultiplier = 3;
  const int max_distance_threshold =
      kDistanceThresholdMultiplier * processed_op_name.length();

  // Iterates backwards, as later names in the candidate list might be more
  // relevant in some contexts (eg. more specific tensor names).
  for (auto it = candidate_names.rbegin(); it != candidate_names.rend(); ++it) {
    absl::string_view name = *it;
    const std::string processed_candidate = PreprocessCandidateName(name);

    if (processed_candidate.empty()) {
      continue;
    }

    // This allows for aggressive early termination.
    const int current_threshold =
        std::min(min_distance, max_distance_threshold);

    if (current_threshold == 0) break;

    const int cur_distance =
        EditDistance(processed_op_name, processed_candidate, current_threshold);

    // If EditDistance returned -1, it means the threshold was exceeded, so
    // this candidate is not better than our current best. We can skip it.
    if (cur_distance == -1) {
      continue;
    }

    if (cur_distance < min_distance) {
      min_distance = cur_distance;
      best_match = std::string(name);
    }
  }
  return best_match;
}

absl::string_view GetTfliteOpName(LiteRtOpCode op_code) {
  switch (op_code) {
    case kLiteRtOpCodeTflAdd:
      return "Add";
    case kLiteRtOpCodeTflAveragePool2d:
      return "AveragePool2d";
    case kLiteRtOpCodeTflConcatenation:
      return "Concatenation";
    case kLiteRtOpCodeTflConv2d:
      return "Conv2d";
    case kLiteRtOpCodeTflDepthwiseConv2d:
      return "DepthwiseConv2d";
    case kLiteRtOpCodeTflDepthToSpace:
      return "DepthToSpace";
    case kLiteRtOpCodeTflDequantize:
      return "Dequantize";
    case kLiteRtOpCodeTflEmbeddingLookup:
      return "EmbeddingLookup";
    case kLiteRtOpCodeTflFloor:
      return "Floor";
    case kLiteRtOpCodeTflFullyConnected:
      return "FullyConnected";
    case kLiteRtOpCodeTflHashtableLookup:
      return "HashtableLookup";
    case kLiteRtOpCodeTflL2Normalization:
      return "L2Normalization";
    case kLiteRtOpCodeTflL2Pool2d:
      return "L2Pool2d";
    case kLiteRtOpCodeTflLocalResponseNormalization:
      return "LocalResponseNormalization";
    case kLiteRtOpCodeTflLogistic:
      return "Logistic";
    case kLiteRtOpCodeTflLshProjection:
      return "LshProjection";
    case kLiteRtOpCodeTflLstm:
      return "Lstm";
    case kLiteRtOpCodeTflMaxPool2d:
      return "MaxPool2d";
    case kLiteRtOpCodeTflMul:
      return "Mul";
    case kLiteRtOpCodeTflRelu:
      return "Relu";
    case kLiteRtOpCodeTflReluN1To1:
      return "ReluN1To1";
    case kLiteRtOpCodeTflRelu6:
      return "Relu6";
    case kLiteRtOpCodeTflReshape:
      return "Reshape";
    case kLiteRtOpCodeTflResizeBilinear:
      return "ResizeBilinear";
    case kLiteRtOpCodeTflRnn:
      return "Rnn";
    case kLiteRtOpCodeTflSoftmax:
      return "Softmax";
    case kLiteRtOpCodeTflSpaceToDepth:
      return "SpaceToDepth";
    case kLiteRtOpCodeTflSvdf:
      return "Svdf";
    case kLiteRtOpCodeTflTanh:
      return "Tanh";
    case kLiteRtOpCodeTflConcatEmbeddings:
      return "ConcatEmbeddings";
    case kLiteRtOpCodeTflSkipGram:
      return "SkipGram";
    case kLiteRtOpCodeTflCall:
      return "Call";
    case kLiteRtOpCodeTflCustom:
      return "Custom";
    case kLiteRtOpCodeTflEmbeddingLookupSparse:
      return "EmbeddingLookupSparse";
    case kLiteRtOpCodeTflPad:
      return "Pad";
    case kLiteRtOpCodeTflUnidirectionalSequenceRnn:
      return "UnidirectionalSequenceRnn";
    case kLiteRtOpCodeTflGather:
      return "Gather";
    case kLiteRtOpCodeTflBatchToSpaceNd:
      return "BatchToSpaceNd";
    case kLiteRtOpCodeTflSpaceToBatchNd:
      return "SpaceToBatchNd";
    case kLiteRtOpCodeTflTranspose:
      return "Transpose";
    case kLiteRtOpCodeTflMean:
      return "Mean";
    case kLiteRtOpCodeTflSub:
      return "Sub";
    case kLiteRtOpCodeTflDiv:
      return "Div";
    case kLiteRtOpCodeTflSqueeze:
      return "Squeeze";
    case kLiteRtOpCodeTflUnidirectionalSequenceLstm:
      return "UnidirectionalSequenceLstm";
    case kLiteRtOpCodeTflStridedSlice:
      return "StridedSlice";
    case kLiteRtOpCodeTflBidirectionalSequenceRnn:
      return "BidirectionalSequenceRnn";
    case kLiteRtOpCodeTflExp:
      return "Exp";
    case kLiteRtOpCodeTflTopkV2:
      return "TopkV2";
    case kLiteRtOpCodeTflSplit:
      return "Split";
    case kLiteRtOpCodeTflLogSoftmax:
      return "LogSoftmax";
    case kLiteRtOpCodeTflDelegate:
      return "Delegate";
    case kLiteRtOpCodeTflBidirectionalSequenceLstm:
      return "BidirectionalSequenceLstm";
    case kLiteRtOpCodeTflCast:
      return "Cast";
    case kLiteRtOpCodeTflPrelu:
      return "Prelu";
    case kLiteRtOpCodeTflMaximum:
      return "Maximum";
    case kLiteRtOpCodeTflArgMax:
      return "ArgMax";
    case kLiteRtOpCodeTflMinimum:
      return "Minimum";
    case kLiteRtOpCodeTflLess:
      return "Less";
    case kLiteRtOpCodeTflNeg:
      return "Neg";
    case kLiteRtOpCodeTflPadv2:
      return "Padv2";
    case kLiteRtOpCodeTflGreater:
      return "Greater";
    case kLiteRtOpCodeTflGreaterEqual:
      return "GreaterEqual";
    case kLiteRtOpCodeTflLessEqual:
      return "LessEqual";
    case kLiteRtOpCodeTflSelect:
      return "Select";
    case kLiteRtOpCodeTflSlice:
      return "Slice";
    case kLiteRtOpCodeTflSin:
      return "Sin";
    case kLiteRtOpCodeTflTransposeConv:
      return "TransposeConv";
    case kLiteRtOpCodeTflSparseToDense:
      return "SparseToDense";
    case kLiteRtOpCodeTflTile:
      return "Tile";
    case kLiteRtOpCodeTflExpandDims:
      return "ExpandDims";
    case kLiteRtOpCodeTflEqual:
      return "Equal";
    case kLiteRtOpCodeTflNotEqual:
      return "NotEqual";
    case kLiteRtOpCodeTflLog:
      return "Log";
    case kLiteRtOpCodeTflSum:
      return "Sum";
    case kLiteRtOpCodeTflSqrt:
      return "Sqrt";
    case kLiteRtOpCodeTflRsqrt:
      return "Rsqrt";
    case kLiteRtOpCodeTflShape:
      return "Shape";
    case kLiteRtOpCodeTflPow:
      return "Pow";
    case kLiteRtOpCodeTflArgMin:
      return "ArgMin";
    case kLiteRtOpCodeTflFakeQuant:
      return "FakeQuant";
    case kLiteRtOpCodeTflReduceProd:
      return "ReduceProd";
    case kLiteRtOpCodeTflReduceMax:
      return "ReduceMax";
    case kLiteRtOpCodeTflPack:
      return "Pack";
    case kLiteRtOpCodeTflLogicalOr:
      return "LogicalOr";
    case kLiteRtOpCodeTflOneHot:
      return "OneHot";
    case kLiteRtOpCodeTflLogicalAnd:
      return "LogicalAnd";
    case kLiteRtOpCodeTflLogicalNot:
      return "LogicalNot";
    case kLiteRtOpCodeTflUnpack:
      return "Unpack";
    case kLiteRtOpCodeTflReduceMin:
      return "ReduceMin";
    case kLiteRtOpCodeTflFloorDiv:
      return "FloorDiv";
    case kLiteRtOpCodeTflReduceAny:
      return "ReduceAny";
    case kLiteRtOpCodeTflSquare:
      return "Square";
    case kLiteRtOpCodeTflZerosLike:
      return "ZerosLike";
    case kLiteRtOpCodeTflFill:
      return "Fill";
    case kLiteRtOpCodeTflFloorMod:
      return "FloorMod";
    case kLiteRtOpCodeTflRange:
      return "Range";
    case kLiteRtOpCodeTflResizeNearestNeighbor:
      return "ResizeNearestNeighbor";
    case kLiteRtOpCodeTflLeakyRelu:
      return "LeakyRelu";
    case kLiteRtOpCodeTflSquaredDifference:
      return "SquaredDifference";
    case kLiteRtOpCodeTflMirrorPad:
      return "MirrorPad";
    case kLiteRtOpCodeTflAbs:
      return "Abs";
    case kLiteRtOpCodeTflSplitV:
      return "SplitV";
    case kLiteRtOpCodeTflUnique:
      return "Unique";
    case kLiteRtOpCodeTflCeil:
      return "Ceil";
    case kLiteRtOpCodeTflReverseV2:
      return "ReverseV2";
    case kLiteRtOpCodeTflAddN:
      return "AddN";
    case kLiteRtOpCodeTflGatherNd:
      return "GatherNd";
    case kLiteRtOpCodeTflCos:
      return "Cos";
    case kLiteRtOpCodeTflWhere:
      return "Where";
    case kLiteRtOpCodeTflRank:
      return "Rank";
    case kLiteRtOpCodeTflElu:
      return "Elu";
    case kLiteRtOpCodeTflReverseSequence:
      return "ReverseSequence";
    case kLiteRtOpCodeTflMatrixDiag:
      return "MatrixDiag";
    case kLiteRtOpCodeTflQuantize:
      return "Quantize";
    case kLiteRtOpCodeTflMatrixSetDiag:
      return "MatrixSetDiag";
    case kLiteRtOpCodeTflRound:
      return "Round";
    case kLiteRtOpCodeTflHardSwish:
      return "HardSwish";
    case kLiteRtOpCodeTflIf:
      return "If";
    case kLiteRtOpCodeTflWhile:
      return "While";
    case kLiteRtOpCodeTflNonMaxSuppressionV4:
      return "NonMaxSuppressionV4";
    case kLiteRtOpCodeTflNonMaxSuppressionV5:
      return "NonMaxSuppressionV5";
    case kLiteRtOpCodeTflScatterNd:
      return "ScatterNd";
    case kLiteRtOpCodeTflSelectV2:
      return "SelectV2";
    case kLiteRtOpCodeTflDensify:
      return "Densify";
    case kLiteRtOpCodeTflSegmentSum:
      return "SegmentSum";
    case kLiteRtOpCodeTflBatchMatmul:
      return "BatchMatmul";
    case kLiteRtOpCodeTflPlaceholderForGreaterOpCodeTfls:
      return "PlaceholderForGreaterOpCodes";
    case kLiteRtOpCodeTflCumsum:
      return "Cumsum";
    case kLiteRtOpCodeTflCallOnce:
      return "CallOnce";
    case kLiteRtOpCodeTflBroadcastTo:
      return "BroadcastTo";
    case kLiteRtOpCodeTflRfft2d:
      return "Rfft2d";
    case kLiteRtOpCodeTflConv3d:
      return "Conv3d";
    case kLiteRtOpCodeTflImag:
      return "Imag";
    case kLiteRtOpCodeTflReal:
      return "Real";
    case kLiteRtOpCodeTflComplexAbs:
      return "ComplexAbs";
    case kLiteRtOpCodeTflHashtable:
      return "Hashtable";
    case kLiteRtOpCodeTflHashtableFind:
      return "HashtableFind";
    case kLiteRtOpCodeTflHashtableImport:
      return "HashtableImport";
    case kLiteRtOpCodeTflHashtableSize:
      return "HashtableSize";
    case kLiteRtOpCodeTflReduceAll:
      return "ReduceAll";
    case kLiteRtOpCodeTflConv3dTranspose:
      return "Conv3dTranspose";
    case kLiteRtOpCodeTflVarHandle:
      return "VarHandle";
    case kLiteRtOpCodeTflReadVariable:
      return "ReadVariable";
    case kLiteRtOpCodeTflAssignVariable:
      return "AssignVariable";
    case kLiteRtOpCodeTflBroadcastArgs:
      return "BroadcastArgs";
    case kLiteRtOpCodeTflRandomStandardNormal:
      return "RandomStandardNormal";
    case kLiteRtOpCodeTflBucketize:
      return "Bucketize";
    case kLiteRtOpCodeTflRandomUniform:
      return "RandomUniform";
    case kLiteRtOpCodeTflMultinomial:
      return "Multinomial";
    case kLiteRtOpCodeTflGelu:
      return "Gelu";
    case kLiteRtOpCodeTflDynamicUpdateSlice:
      return "DynamicUpdateSlice";
    case kLiteRtOpCodeTflRelu0To1:
      return "Relu0To1";
    case kLiteRtOpCodeTflUnsortedSegmentProd:
      return "UnsortedSegmentProd";
    case kLiteRtOpCodeTflUnsortedSegmentMax:
      return "UnsortedSegmentMax";
    case kLiteRtOpCodeTflUnsortedSegmentSum:
      return "UnsortedSegmentSum";
    case kLiteRtOpCodeTflAtan2:
      return "Atan2";
    case kLiteRtOpCodeTflUnsortedSegmentMin:
      return "UnsortedSegmentMin";
    case kLiteRtOpCodeTflSign:
      return "Sign";
    case kLiteRtOpCodeTflBitcast:
      return "Bitcast";
    case kLiteRtOpCodeTflBitwiseXor:
      return "BitwiseXor";
    case kLiteRtOpCodeTflRightShift:
      return "RightShift";
    case kLiteRtOpCodeShloLogistic:
      return "StablehloLogistic";
    case kLiteRtOpCodeShloAdd:
      return "StablehloAdd";
    case kLiteRtOpCodeShloDivide:
      return "StablehloDivide";
    case kLiteRtOpCodeShloMultiply:
      return "StablehloMultiply";
    case kLiteRtOpCodeShloMaximum:
      return "StablehloMaximum";
    case kLiteRtOpCodeShloReshape:
      return "StablehloReshape";
    case kLiteRtOpCodeShloClamp:
      return "StablehloClamp";
    case kLiteRtOpCodeShloConcatenate:
      return "StablehloConcatenate";
    case kLiteRtOpCodeShloBroadcastInDim:
      return "StablehloBroadcastInDim";
    case kLiteRtOpCodeShloConvolution:
      return "StablehloConvolution";
    case kLiteRtOpCodeShloSlice:
      return "StablehloSlice";
    case kLiteRtOpCodeShloCustomCall:
      return "StablehloCustomCall";
    case kLiteRtOpCodeShloReduce:
      return "StablehloReduce";
    case kLiteRtOpCodeShloAbs:
      return "StablehloAbs";
    case kLiteRtOpCodeShloAnd:
      return "StablehloAnd";
    case kLiteRtOpCodeShloCosine:
      return "StablehloCosine";
    case kLiteRtOpCodeShloExponential:
      return "StablehloExponential";
    case kLiteRtOpCodeShloFloor:
      return "StablehloFloor";
    case kLiteRtOpCodeShloLog:
      return "StablehloLog";
    case kLiteRtOpCodeShloMinimum:
      return "StablehloMinimum";
    case kLiteRtOpCodeShloNegate:
      return "StablehloNegate";
    case kLiteRtOpCodeShloOr:
      return "StablehloOr";
    case kLiteRtOpCodeShloPower:
      return "StablehloPower";
    case kLiteRtOpCodeShloRemainder:
      return "StablehloRemainder";
    case kLiteRtOpCodeShloRsqrt:
      return "StablehloRsqrt";
    case kLiteRtOpCodeShloSelect:
      return "StablehloSelect";
    case kLiteRtOpCodeShloSubtract:
      return "StablehloSubtract";
    case kLiteRtOpCodeShloTanh:
      return "StablehloTanh";
    case kLiteRtOpCodeShloScatter:
      return "StablehloScatter";
    case kLiteRtOpCodeShloCompare:
      return "StablehloCompare";
    case kLiteRtOpCodeShloConvert:
      return "StablehloConvert";
    case kLiteRtOpCodeShloDynamicSlice:
      return "StablehloDynamicSlice";
    case kLiteRtOpCodeShloDynamicUpdateSlice:
      return "StablehloDynamicUpdateSlice";
    case kLiteRtOpCodeShloPad:
      return "StablehloPad";
    case kLiteRtOpCodeShloIota:
      return "StablehloIota";
    case kLiteRtOpCodeShloGeneral:
      return "StablehloDotGeneral";
    case kLiteRtOpCodeShloWindow:
      return "StablehloReduceWindow";
    case kLiteRtOpCodeShloSort:
      return "StablehloSort";
    case kLiteRtOpCodeShloWhile:
      return "StablehloWhile";
    case kLiteRtOpCodeShloGather:
      return "StablehloGather";
    case kLiteRtOpCodeShloTranspose:
      return "StablehloTranspose";
    case kLiteRtOpCodeTflDilate:
      return "Dilate";
    case kLiteRtOpCodeShloRngBitGenerator:
      return "StablehloRngBitGenerator";
    case kLiteRtOpCodeTflReduceWindow:
      return "ReduceWindow";
    case kLiteRtOpCodeShloComposite:
      return "StablehloComposite";

    default:
      return "Unknown";
  }
}

}  // namespace litert
