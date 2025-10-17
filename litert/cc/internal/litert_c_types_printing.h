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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_C_TYPES_PRINTING_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_C_TYPES_PRINTING_H_

#include <string>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_logging.h"

// AbslStringify specializations for types in the litert c api.
// TODO: lukeboyer - Migrate code in tools/dump.h to leverage the abseil
// stringify framework.

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtElementType& type) {
  std::string dtype_str;
  switch (type) {
    case kLiteRtElementTypeInt32:
      dtype_str = "i32";
      break;
    case kLiteRtElementTypeInt64:
      dtype_str = "i64";
      break;
    case kLiteRtElementTypeFloat32:
      dtype_str = "f32";
      break;
    case kLiteRtElementTypeFloat64:
      dtype_str = "f64";
      break;
    case kLiteRtElementTypeInt16:
      dtype_str = "i16";
      break;
    case kLiteRtElementTypeInt8:
      dtype_str = "i8";
      break;
    case kLiteRtElementTypeUInt8:
      dtype_str = "u8";
      break;
    case kLiteRtElementTypeUInt16:
      dtype_str = "u16";
      break;
    case kLiteRtElementTypeUInt32:
      dtype_str = "u32";
      break;
    case kLiteRtElementTypeUInt64:
      dtype_str = "u64";
      break;
    case kLiteRtElementTypeBool:
      dtype_str = "i1";
      break;
    default:
      dtype_str = ::litert::kNoPrinterTag;
      break;
  }

  absl::Format(&sink, "%s", dtype_str);
}

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtLayout& layout) {
  absl::Format(
      &sink, "<%s>",
      absl::StrJoin(absl::MakeConstSpan(layout.dimensions, layout.rank), "x"));
}

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtRankedTensorType& type) {
  const auto& layout = type.layout;
  absl::Format(&sink, "%ud_%v%v", layout.rank, type.element_type, layout);
}

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtOpCode& code) {
  std::string op_code_str;
  switch (code) {
    case kLiteRtOpCodeTflAdd:
      op_code_str = "tfl.add";
      break;
    case kLiteRtOpCodeTflMul:
      op_code_str = "tfl.mul";
      break;
    case kLiteRtOpCodeTflCustom:
      op_code_str = "tfl.custom_op";
      break;
    case kLiteRtOpCodeTflSlice:
      op_code_str = "tfl.slice";
      break;
    case kLiteRtOpCodeTflDiv:
      op_code_str = "tfl.div";
      break;
    case kLiteRtOpCodeTflRsqrt:
      op_code_str = "tfl.rsqrt";
      break;
    case kLiteRtOpCodeTflTanh:
      op_code_str = "tfl.tanh";
      break;
    case kLiteRtOpCodeTflSub:
      op_code_str = "tfl.sub";
      break;
    case kLiteRtOpCodeTflReshape:
      op_code_str = "tfl.reshape";
      break;
    case kLiteRtOpCodeTflBatchMatmul:
      op_code_str = "tfl.batch_matmul";
      break;
    case kLiteRtOpCodeTflSum:
      op_code_str = "tfl.sum";
      break;
    case kLiteRtOpCodeTflConcatenation:
      op_code_str = "tfl.concatenation";
      break;
    case kLiteRtOpCodeTflSoftmax:
      op_code_str = "tfl.softmax";
      break;
    case kLiteRtOpCodeTflCast:
      op_code_str = "tfl.cast";
      break;
    case kLiteRtOpCodeTflTranspose:
      op_code_str = "tfl.transpose";
      break;
    case kLiteRtOpCodeTflSin:
      op_code_str = "tfl.sin";
      break;
    case kLiteRtOpCodeTflCos:
      op_code_str = "tfl.cos";
      break;
    case kLiteRtOpCodeTflSelect:
      op_code_str = "tfl.select";
      break;
    case kLiteRtOpCodeTflSelectV2:
      op_code_str = "tfl.select_v2";
      break;
    case kLiteRtOpCodeTflFullyConnected:
      op_code_str = "tfl.fully_connected";
      break;
    case kLiteRtOpCodeTflEmbeddingLookup:
      op_code_str = "tfl.embedding_lookup";
      break;
    case kLiteRtOpCodeTflLogicalAnd:
      op_code_str = "tfl.logical_and";
      break;
    case kLiteRtOpCodeTflLess:
      op_code_str = "tfl.less";
      break;
    case kLiteRtOpCodeTflGreater:
      op_code_str = "tfl.greater";
      break;
    case kLiteRtOpCodeTflGelu:
      op_code_str = "tfl.gelu";
      break;
    case kLiteRtOpCodeTflDynamicUpdateSlice:
      op_code_str = "tfl.dynamic_update_slice";
      break;
    case kLiteRtOpCodeTflPack:
      op_code_str = "tfl.pack";
      break;
    case kLiteRtOpCodeTflQuantize:
      op_code_str = "tfl.quantize";
      break;
    case kLiteRtOpCodeTflLeakyRelu:
      op_code_str = "tfl.leaky_relu";
      break;
    case kLiteRtOpCodeTflHardSwish:
      op_code_str = "tfl.hard_swish";
      break;
    case kLiteRtOpCodeTflAveragePool2d:
      op_code_str = "tfl.average_pool2d";
      break;
    case kLiteRtOpCodeTflMaxPool2d:
      op_code_str = "tfl.max_pool2d";
      break;
    case kLiteRtOpCodeTflDepthwiseConv2d:
      op_code_str = "tfl.depthwise_conv2d";
      break;
    case kLiteRtOpCodeTflSpaceToDepth:
      op_code_str = "tfl.space_to_depth";
      break;
    case kLiteRtOpCodeTflDepthToSpace:
      op_code_str = "tfl.depth_to_space";
      break;
    case kLiteRtOpCodeTflConv2d:
      op_code_str = "tfl.conv2d";
      break;
    case kLiteRtOpCodeTflResizeBilinear:
      op_code_str = "tfl.resize_bilinear";
      break;
    case kLiteRtOpCodeTflMinimum:
      op_code_str = "tfl.minimum";
      break;
    case kLiteRtOpCodeTflMaximum:
      op_code_str = "tfl.maximum";
      break;
    case kLiteRtOpCodeTflResizeNearestNeighbor:
      op_code_str = "tfl.resize_nearest_neighbor";
      break;
    case kLiteRtOpCodeTflRelu:
      op_code_str = "tfl.relu";
      break;
    case kLiteRtOpCodeTflRelu6:
      op_code_str = "tfl.relu6";
      break;
    case kLiteRtOpCodeTflLogistic:
      op_code_str = "tfl.logistic";
      break;
    case kLiteRtOpCodeTflFloorDiv:
      op_code_str = "tfl.floor_div";
      break;
    case kLiteRtOpCodeTflNotEqual:
      op_code_str = "tfl.not_equal";
      break;
    case kLiteRtOpCodeTflPad:
      op_code_str = "tfl.pad";
      break;
    case kLiteRtOpCodeTflPadv2:
      op_code_str = "tfl.pad_v2";
      break;
    case kLiteRtOpCodeTflGatherNd:
      op_code_str = "tfl.gather_nd";
      break;
    case kLiteRtOpCodeTflCumsum:
      op_code_str = "tfl.cumsum";
      break;
    default:
      op_code_str = ::litert::kNoPrinterTag;
      break;
  }
  absl::Format(&sink, "%s", op_code_str);
}

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_C_TYPES_PRINTING_H_
